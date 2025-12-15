"""
EfficientTAM 图片分割演示脚本

生成效果图用于 README 展示

使用方法:
    python demo_image_segmentation.py

输出:
    assets/demo_image_segmentation.png - 图片分割效果展示
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from matplotlib.patches import Rectangle

# 配置 (相对于脚本所在目录)
MODEL_PATH = 'efficienttam_ti_512x512_point_and_box.pt'  # 需要先导出模型到此目录
IMAGE_PATH = 'examples/truck.jpg'  # 示例图片
OUTPUT_DIR = Path('assets')


def box_to_points(box):
    """将框 [x1, y1, x2, y2] 转换为两个点"""
    x1, y1, x2, y2 = box[0, 0].item(), box[0, 1].item(), box[0, 2].item(), box[0, 3].item()
    point_coords = torch.tensor([[[x1, y1], [x2, y2]]], dtype=torch.float32)
    point_labels = torch.tensor([[2, 3]], dtype=torch.int32)
    return point_coords, point_labels


def main():
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    device = 'cpu'
    model = torch.jit.load(MODEL_PATH, map_location=device).eval()
    
    # 读入图片+预处理
    print(f"加载图片: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH).convert('RGB')
    original_size = img.size
    img_resized = img.resize((512, 512))
    
    img_np = np.asarray(img_resized) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # 测试1: 点提示
    print("测试点提示...")
    point_coords = torch.tensor([[[256.0, 256.0]]])  # 中心点
    point_labels = torch.tensor([[1]], dtype=torch.int32)
    
    with torch.no_grad():
        mask_point = model(img_tensor, point_coords, point_labels).squeeze().numpy()
    
    # 测试2: 框提示
    print("测试框提示...")
    box = torch.tensor([[150.0, 150.0, 350.0, 350.0]])
    box_point_coords, box_point_labels = box_to_points(box)
    
    with torch.no_grad():
        mask_box = model(img_tensor, box_point_coords, box_point_labels).squeeze().numpy()
    
    # 测试3: 点+框组合
    print("测试点+框组合...")
    point_coords_combo = torch.tensor([[[300.0, 300.0]]])
    point_labels_combo = torch.tensor([[1]], dtype=torch.int32)
    box_combo = torch.tensor([[150.0, 150.0, 350.0, 350.0]])
    
    box_point_coords_combo, box_point_labels_combo = box_to_points(box_combo)
    concat_coords = torch.cat([box_point_coords_combo, point_coords_combo], dim=1)
    concat_labels = torch.cat([box_point_labels_combo, point_labels_combo], dim=1)
    
    with torch.no_grad():
        mask_combo = model(img_tensor, concat_coords, concat_labels).squeeze().numpy()
    
    # 可视化
    print("生成效果图...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原图
    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 点提示
    axes[0, 1].imshow(img_resized)
    axes[0, 1].plot(point_coords[0, 0, 0].item(), point_coords[0, 0, 1].item(), 
                    'g*', markersize=20, markeredgewidth=2, markeredgecolor='white')
    mask_overlay = np.zeros((*mask_point.shape, 4))
    mask_overlay[mask_point > 0] = [0, 1, 0, 0.5]  # 绿色半透明
    axes[0, 1].imshow(mask_overlay)
    axes[0, 1].set_title('Point Prompt (Green Star)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 框提示
    axes[1, 0].imshow(img_resized)
    rect = Rectangle(
        (box[0, 0].item(), box[0, 1].item()),
        (box[0, 2] - box[0, 0]).item(),
        (box[0, 3] - box[0, 1]).item(),
        linewidth=3, edgecolor='red', facecolor='none'
    )
    axes[1, 0].add_patch(rect)
    mask_overlay = np.zeros((*mask_box.shape, 4))
    mask_overlay[mask_box > 0] = [0, 1, 0, 0.5]
    axes[1, 0].imshow(mask_overlay)
    axes[1, 0].set_title('Box Prompt (Red Box)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 点+框组合
    axes[1, 1].imshow(img_resized)
    rect_combo = Rectangle(
        (box_combo[0, 0].item(), box_combo[0, 1].item()),
        (box_combo[0, 2] - box_combo[0, 0]).item(),
        (box_combo[0, 3] - box_combo[0, 1]).item(),
        linewidth=3, edgecolor='blue', facecolor='none'
    )
    axes[1, 1].add_patch(rect_combo)
    axes[1, 1].plot(point_coords_combo[0, 0, 0].item(), point_coords_combo[0, 0, 1].item(), 
                    'g*', markersize=20, markeredgewidth=2, markeredgecolor='white')
    mask_overlay = np.zeros((*mask_combo.shape, 4))
    mask_overlay[mask_combo > 0] = [0, 1, 0, 0.5]
    axes[1, 1].imshow(mask_overlay)
    axes[1, 1].set_title('Point + Box Combined', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存
    output_path = OUTPUT_DIR / 'demo_image_segmentation.png'
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n效果图已保存: {output_path}")
    print(f"图片尺寸: {original_size} -> (512, 512)")


if __name__ == "__main__":
    main()
