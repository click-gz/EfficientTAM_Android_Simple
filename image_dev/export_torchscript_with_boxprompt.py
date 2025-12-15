import torch
import os
import sys
from pathlib import Path
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from efficient_track_anything.build_efficienttam import build_efficienttam

# 获取项目根目录（脚本在项目根目录）
script_dir = Path(__file__).parent
project_root = script_dir  # 脚本已经在项目根目录

# 切换到项目根目录，这样相对路径才能正确工作
original_cwd = os.getcwd()
os.chdir(project_root)

try:
    # 配置与权重路径（相对于项目根目录）
    config_path = 'efficienttam/efficienttam_ti_512x512.yaml'
    ckpt_path = 'checkpoints/efficienttam_ti_512x512.pt'
    config_dir = 'efficient_track_anything/configs'  # 相对路径

    device = torch.device('cpu')

    # 验证工作目录切换是否成功
    current_dir = os.getcwd()
    config_dir_full = os.path.join(current_dir, config_dir)
    if not os.path.exists(config_dir_full):
        raise RuntimeError(f"配置目录不存在: {config_dir_full}\n当前工作目录: {current_dir}\n项目根目录: {project_root}")

    # 确保 efficient_track_anything 在 Python 路径中
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # hydra 初始化 - 使用 initialize（与 export_torchscript_lite.py 一致）
    GlobalHydra.instance().clear()
    with initialize(config_path="efficient_track_anything/configs"):
        model = build_efficienttam(config_path, ckpt_path, device=device)
    
    # --- 核心修改：定义极简 Wrapper，移除所有Python逻辑分支 ---
    class TAMWrapperSimple(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, img, point_coords, point_labels):
            """
            接口极简：移除了 boxes 参数。
            所有的 Prompt（无论是点还是框）都必须在外部转换成 point_coords 和 point_labels 传入。
            
            Args:
                img: [1, 3, 512, 512], float
                point_coords: [1, N, 2]，像素坐标
                point_labels: [1, N]，int。标签含义：1=前景点, 0=背景点, 2=框左上角, 3=框右下角, -1=占位符
            """
            # 1. 强制类型转换 (Android 端传入时可能会有类型偏差，这里兜底)
            img = img.float()
            point_coords = point_coords.float()
            point_labels = point_labels.int()

            # 2. 图像编码
            backbone_out = self.base_model.image_encoder(img)
            
            # 3. 构造输入字典（直接使用传入的点坐标和标签，不做任何判断）
            point_inputs = {
                'point_coords': point_coords,
                'point_labels': point_labels
            }
            
            # 4. 推理 (multimask_output=False 适合移动端，只返回一个最佳 Mask)
            sam_outputs = self.base_model._forward_sam_heads(
                backbone_features=backbone_out['backbone_fpn'][-1],
                point_inputs=point_inputs,
                mask_inputs=None,
                high_res_features=None,
                multimask_output=False, 
            )
            
            high_res_masks = sam_outputs[4]  # [B,1,512,512]
            return high_res_masks

    # 导出流程
    wrapper = TAMWrapperSimple(model).to(device).eval()

    # Dummy 数据：模拟一个框（即2个点：左上角和右下角）
    dummy_img = torch.rand(1, 3, 512, 512)
    # 框的两个角点：[左上x, 左上y, 右下x, 右下y] -> reshape为 [1, 2, 2]
    dummy_coords = torch.tensor([[[100.0, 100.0], [400.0, 400.0]]], dtype=torch.float32)
    dummy_labels = torch.tensor([[2, 3]], dtype=torch.int32)  # 2=左上, 3=右下

    print("开始导出极简模型（移除所有Python逻辑分支）...")
    print(f"Dummy coords shape: {dummy_coords.shape}")
    print(f"Dummy labels shape: {dummy_labels.shape}")
    
    # Trace 只记录 3 个参数（移除了 boxes 参数）
    traced_model = torch.jit.trace(wrapper, (dummy_img, dummy_coords, dummy_labels))
    
    # 保存到当前目录（项目根目录）
    output_path = project_root / 'efficienttam_ti_512x512_point_and_box.pt'
    traced_model.save(str(output_path))
    print(f'TorchScript 导出完成：{output_path}')
    print('注意：新模型只接受 (img, point_coords, point_labels) 三个参数')
    print('框必须在客户端转换为两个点（标签2和3）后再传入')
finally:
    # 恢复原始工作目录
    os.chdir(original_cwd)
