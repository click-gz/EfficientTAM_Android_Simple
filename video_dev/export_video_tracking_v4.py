"""
EfficientTAM 视频跟踪 Android 导出脚本 V4

根据官方流程文档设计，关键改进：
1. 首帧输出 obj_ptr
2. 后续帧 Memory Attention 接收 maskmem_features + obj_ptr
3. Memory Encoder 使用原始 features（不是 pix_feat_with_mem）
4. 替换 scaled_dot_product_attention 为手动实现（Android 兼容）
"""

import torch
import torch.nn.functional as F
import math
import os
import sys
from pathlib import Path
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from efficient_track_anything.build_efficienttam import build_efficienttam


# ImageNet 标准化参数
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def manual_scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """
    手动实现 scaled_dot_product_attention，兼容 PyTorch Mobile
    """
    head_dim = q.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if is_causal:
        N, M = q.size(-2), k.size(-2)
        causal_mask = torch.triu(torch.ones(N, M, device=q.device, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
    
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    return torch.matmul(attn_weights, v)


# 保存原始函数并替换
_original_sdpa = F.scaled_dot_product_attention


def patch_attention_for_mobile():
    """替换为 Android 兼容的实现"""
    F.scaled_dot_product_attention = manual_scaled_dot_product_attention


def restore_attention():
    """恢复原始实现"""
    F.scaled_dot_product_attention = _original_sdpa


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        config_path = 'efficienttam/efficienttam_ti_512x512.yaml'
        ckpt_path = 'checkpoints/efficienttam_ti_512x512.pt'
        config_dir = 'efficient_track_anything/configs'
        
        device = torch.device('cpu')
        
        if not os.path.exists(os.path.join(os.getcwd(), config_dir)):
            raise RuntimeError(f"配置目录不存在: {config_dir}")
        
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # 替换 attention 为 Android 兼容版本
        patch_attention_for_mobile()
        print("已替换 scaled_dot_product_attention 为 Android 兼容实现")
        
        GlobalHydra.instance().clear()
        with initialize(config_path="efficient_track_anything/configs"):
            model = build_efficienttam(config_path, ckpt_path, device=device)
        
        model.eval()
        print("模型加载成功")
        print(f"  hidden_dim: {model.hidden_dim}")
        print(f"  mem_dim: {model.mem_dim}")
        print(f"  num_maskmem: {model.num_maskmem}")
        print(f"  use_obj_ptrs_in_encoder: {model.use_obj_ptrs_in_encoder}")
        
        # ============================================================
        # 首帧编码器 V4
        # 输出: mask, maskmem_features, maskmem_pos_enc, obj_ptr
        # ============================================================
        class FirstFrameEncoderV4(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.image_encoder = base_model.image_encoder
                self.sam_prompt_encoder = base_model.sam_prompt_encoder
                self.sam_mask_decoder = base_model.sam_mask_decoder
                self.memory_encoder = base_model.memory_encoder
                self.obj_ptr_proj = base_model.obj_ptr_proj
                
                self.hidden_dim = base_model.hidden_dim
                self.image_size = base_model.image_size
                
                # 首帧需要加上 no_mem_embed
                self.no_mem_embed = base_model.no_mem_embed
                
                # obj_ptr occlusion 处理相关
                self.pred_obj_scores = base_model.pred_obj_scores
                self.fixed_no_obj_ptr = base_model.fixed_no_obj_ptr
                self.no_obj_ptr = base_model.no_obj_ptr
                
                # Memory Encoder mask 预处理参数
                self.sigmoid_scale_for_mem_enc = base_model.sigmoid_scale_for_mem_enc
                self.sigmoid_bias_for_mem_enc = base_model.sigmoid_bias_for_mem_enc
                
                self.register_buffer('img_mean', IMG_MEAN)
                self.register_buffer('img_std', IMG_STD)
            
            def forward(self, img, point_coords, point_labels):
                """
                首帧处理
                
                Args:
                    img: [1, 3, 512, 512] 原始图像（0-255）
                    point_coords: [1, N, 2] 点坐标（512x512 空间）
                    point_labels: [1, N] 点标签
                
                Returns:
                    mask: [1, 1, 512, 512] 分割掩码 logits
                    maskmem_features: [1, 64, 32, 32] Memory 特征
                    maskmem_pos_enc: [1, 64, 32, 32] Memory 位置编码
                    obj_ptr: [1, 256] Object Pointer
                """
                # 图像预处理
                img = img.float() / 255.0
                img = (img - self.img_mean) / self.img_std
                
                # 图像编码
                backbone_out = self.image_encoder(img)
                features = backbone_out['backbone_fpn'][-1]  # [B, C, H, W] = [1, 256, 32, 32]
                
                B, C, H, W = features.shape
                
                # 首帧处理：features + no_mem_embed
                # no_mem_embed shape: [1, 1, 256]
                # 需要 reshape 到 [HW, B, C] 格式后再加
                features_flat = features.flatten(2).permute(2, 0, 1)  # [HW, B, C]
                features_with_no_mem = features_flat + self.no_mem_embed  # broadcast
                pix_feat_with_mem = features_with_no_mem.permute(1, 2, 0).view(B, C, H, W)
                
                # Prompt 编码
                sparse_emb, dense_emb = self.sam_prompt_encoder(
                    points=(point_coords.float(), point_labels.int()),
                    boxes=None,
                    masks=None,
                )
                
                # Mask 解码 - 首帧使用 multimask_output=True
                # 官方配置: multimask_output_in_sam=True, multimask_max_pt_num=1
                low_res_multimasks, ious, sam_output_tokens, obj_scores = self.sam_mask_decoder(
                    image_embeddings=pix_feat_with_mem,
                    image_pe=self.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,  # 首帧使用多 mask 输出
                    repeat_image=False,
                    high_res_features=None,
                )
                
                # 选择最佳 mask (IoU 最高的)
                B = low_res_multimasks.shape[0]
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=low_res_multimasks.device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                
                # 上采样到高分辨率
                high_res_masks = F.interpolate(
                    low_res_masks.float(),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                
                # 提取 obj_ptr（包含 occlusion 处理）
                # multimask_output=True 时，sam_output_tokens 形状是 [B, 3, 256]
                # 需要选择与最佳 mask 对应的 token
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]  # [B, 256]
                obj_ptr = self.obj_ptr_proj(sam_output_token)  # [B, 256]
                
                # Occlusion 处理: 根据 obj_scores 调整 obj_ptr
                if self.pred_obj_scores:
                    is_obj_appearing = (obj_scores > 0).float()
                    if self.fixed_no_obj_ptr:
                        obj_ptr = is_obj_appearing * obj_ptr
                    obj_ptr = obj_ptr + (1 - is_obj_appearing) * self.no_obj_ptr
                
                # Memory 编码 - 使用原始 features（不是 pix_feat_with_mem）
                # 官方: mask_for_mem = sigmoid(mask) * scale + bias
                mask_for_mem = torch.sigmoid(high_res_masks)
                mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc + self.sigmoid_bias_for_mem_enc
                maskmem_out = self.memory_encoder(
                    features, mask_for_mem, skip_mask_sigmoid=True
                )
                maskmem_features = maskmem_out["vision_features"]  # [B, 64, 32, 32]
                maskmem_pos_enc = maskmem_out["vision_pos_enc"][-1]  # [B, 64, 32, 32]
                
                return high_res_masks, maskmem_features, maskmem_pos_enc, obj_ptr
        
        # ============================================================
        # 跟踪器 V4
        # 输入: img, maskmem_features, maskmem_pos_enc, obj_ptr
        # 输出: mask, new_maskmem_features, new_maskmem_pos_enc, new_obj_ptr
        # ============================================================
        class FrameTrackerV4(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.image_encoder = base_model.image_encoder
                self.memory_attention = base_model.memory_attention
                self.sam_prompt_encoder = base_model.sam_prompt_encoder
                self.sam_mask_decoder = base_model.sam_mask_decoder
                self.memory_encoder = base_model.memory_encoder
                self.obj_ptr_proj = base_model.obj_ptr_proj
                
                self.hidden_dim = base_model.hidden_dim
                self.mem_dim = base_model.mem_dim
                self.image_size = base_model.image_size
                self.maskmem_tpos_enc = base_model.maskmem_tpos_enc
                self.num_maskmem = base_model.num_maskmem
                
                # obj_ptr 时序位置编码投影
                self.obj_ptr_tpos_proj = base_model.obj_ptr_tpos_proj
                
                # obj_ptr occlusion 处理相关
                self.pred_obj_scores = base_model.pred_obj_scores
                self.fixed_no_obj_ptr = base_model.fixed_no_obj_ptr
                self.no_obj_ptr = base_model.no_obj_ptr
                
                # Memory Encoder mask 预处理参数
                self.sigmoid_scale_for_mem_enc = base_model.sigmoid_scale_for_mem_enc
                self.sigmoid_bias_for_mem_enc = base_model.sigmoid_bias_for_mem_enc
                
                self.register_buffer('img_mean', IMG_MEAN)
                self.register_buffer('img_std', IMG_STD)
            
            def forward(self, img, maskmem_features, maskmem_pos_enc, obj_ptr):
                """
                跟踪后续帧
                
                Args:
                    img: [1, 3, 512, 512] 原始图像（0-255）
                    maskmem_features: [1, 64, 32, 32] 上一帧的 Memory 特征
                    maskmem_pos_enc: [1, 64, 32, 32] Memory 位置编码
                    obj_ptr: [1, 256] 上一帧的 Object Pointer
                
                Returns:
                    mask: [1, 1, 512, 512] 分割掩码 logits
                    new_maskmem_features: [1, 64, 32, 32] 新的 Memory 特征
                    new_maskmem_pos_enc: [1, 64, 32, 32] 新的 Memory 位置编码
                    new_obj_ptr: [1, 256] 新的 Object Pointer
                """
                # 图像预处理
                img = img.float() / 255.0
                img = (img - self.img_mean) / self.img_std
                
                # 图像编码
                backbone_out = self.image_encoder(img)
                features = backbone_out['backbone_fpn'][-1]  # [B, C, H, W]
                pos_enc = backbone_out['vision_pos_enc'][-1]  # [B, C, H, W]
                
                B, C, H, W = features.shape
                
                # 准备 Memory Attention 输入
                # 当前帧特征: [HW, B, C]
                curr_feats = features.flatten(2).permute(2, 0, 1)
                curr_pos = pos_enc.flatten(2).permute(2, 0, 1)
                
                # Memory 特征: [HW, B, mem_dim]
                mem_feats = maskmem_features.flatten(2).permute(2, 0, 1)
                mem_pos = maskmem_pos_enc.flatten(2).permute(2, 0, 1)
                # 添加时序位置编码 (t_pos=0 对应条件帧，索引 num_maskmem-1)
                mem_pos = mem_pos + self.maskmem_tpos_enc[self.num_maskmem - 1]
                
                # Object Pointer 处理
                # obj_ptr: [B, 256] -> 拆分成 4 个 tokens (因为 mem_dim=64 < hidden_dim=256)
                # 结果: [4, B, 64]
                obj_ptrs = obj_ptr.view(B, C // self.mem_dim, self.mem_dim)  # [B, 4, 64]
                obj_ptrs = obj_ptrs.permute(1, 0, 2)  # [4, B, 64]
                
                # obj_ptr 位置编码 (简化：使用零向量)
                obj_pos = torch.zeros(obj_ptrs.shape[0], B, self.mem_dim, device=img.device)
                
                # 拼接 memory 和 obj_ptr
                memory = torch.cat([mem_feats, obj_ptrs], dim=0)
                memory_pos = torch.cat([mem_pos, obj_pos], dim=0)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
                
                # Memory Attention
                # 注意：memory_attention 内部会处理列表格式，但 trace 时需要直接传 tensor
                # 官方代码: curr=current_vision_feats (列表), 内部取 curr[0]
                pix_feat_with_mem = self.memory_attention(
                    curr=curr_feats,  # 直接传 tensor，不是列表
                    curr_pos=curr_pos,
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=num_obj_ptr_tokens,
                )
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                
                # Mask 解码（无点提示）
                sam_point_coords = torch.zeros(B, 1, 2, device=img.device)
                sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=img.device)
                
                sparse_emb, dense_emb = self.sam_prompt_encoder(
                    points=(sam_point_coords, sam_point_labels),
                    boxes=None,
                    masks=None,
                )
                
                low_res_masks, ious, sam_output_tokens, obj_scores = self.sam_mask_decoder(
                    image_embeddings=pix_feat_with_mem,
                    image_pe=self.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=None,
                )
                
                high_res_masks = F.interpolate(
                    low_res_masks.float(),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                
                # 提取新的 obj_ptr（包含 occlusion 处理）
                sam_output_token = sam_output_tokens[:, 0]
                new_obj_ptr = self.obj_ptr_proj(sam_output_token)
                
                # Occlusion 处理
                if self.pred_obj_scores:
                    is_obj_appearing = (obj_scores > 0).float()
                    if self.fixed_no_obj_ptr:
                        new_obj_ptr = is_obj_appearing * new_obj_ptr
                    new_obj_ptr = new_obj_ptr + (1 - is_obj_appearing) * self.no_obj_ptr
                
                # Memory 编码 - 使用原始 features（不是 pix_feat_with_mem）
                # 官方: mask_for_mem = sigmoid(mask) * scale + bias
                mask_for_mem = torch.sigmoid(high_res_masks)
                mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc + self.sigmoid_bias_for_mem_enc
                maskmem_out = self.memory_encoder(
                    features, mask_for_mem, skip_mask_sigmoid=True
                )
                new_maskmem_features = maskmem_out["vision_features"]
                new_maskmem_pos_enc = maskmem_out["vision_pos_enc"][-1]
                
                return high_res_masks, new_maskmem_features, new_maskmem_pos_enc, new_obj_ptr
        
        # ============================================================
        # 导出
        # ============================================================
        print("\n开始导出模型 V4...")
        
        # Dummy 输入
        dummy_img = torch.randint(0, 256, (1, 3, 512, 512), dtype=torch.float32)
        dummy_point_coords = torch.tensor([[[256.0, 256.0]]], dtype=torch.float32)
        dummy_point_labels = torch.tensor([[1]], dtype=torch.int32)
        
        # 首帧编码器
        print("导出首帧编码器...")
        first_encoder = FirstFrameEncoderV4(model).eval()
        
        with torch.no_grad():
            mask, maskmem_features, maskmem_pos_enc, obj_ptr = first_encoder(
                dummy_img, dummy_point_coords, dummy_point_labels
            )
        print(f"  mask shape: {mask.shape}")
        print(f"  maskmem_features shape: {maskmem_features.shape}")
        print(f"  maskmem_pos_enc shape: {maskmem_pos_enc.shape}")
        print(f"  obj_ptr shape: {obj_ptr.shape}")
        
        traced_first = torch.jit.trace(
            first_encoder, (dummy_img, dummy_point_coords, dummy_point_labels)
        )
        first_path = project_root / 'efficienttam_video_first_v4.pt'
        traced_first.save(str(first_path))
        print(f"  保存到: {first_path}")
        
        # 跟踪器
        print("\n导出跟踪器...")
        tracker = FrameTrackerV4(model).eval()
        
        with torch.no_grad():
            new_mask, new_mem, new_pos, new_ptr = tracker(
                dummy_img, maskmem_features, maskmem_pos_enc, obj_ptr
            )
        print(f"  new_mask shape: {new_mask.shape}")
        print(f"  new_maskmem_features shape: {new_mem.shape}")
        print(f"  new_obj_ptr shape: {new_ptr.shape}")
        
        traced_track = torch.jit.trace(
            tracker, (dummy_img, maskmem_features, maskmem_pos_enc, obj_ptr)
        )
        track_path = project_root / 'efficienttam_video_track_v4.pt'
        traced_track.save(str(track_path))
        print(f"  保存到: {track_path}")
        
        # 导出 PyTorch Mobile 格式 (.ptl)
        print("\n转换为 PyTorch Mobile 格式...")
        from torch.utils.mobile_optimizer import optimize_for_mobile
        
        first_optimized = optimize_for_mobile(traced_first)
        first_ptl_path = project_root / 'efficienttam_video_first_v4.ptl'
        first_optimized._save_for_lite_interpreter(str(first_ptl_path))
        print(f"  首帧编码器: {first_ptl_path.name} ({first_ptl_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        tracker_optimized = optimize_for_mobile(traced_track)
        tracker_ptl_path = project_root / 'efficienttam_video_track_v4.ptl'
        tracker_optimized._save_for_lite_interpreter(str(tracker_ptl_path))
        print(f"  跟踪器: {tracker_ptl_path.name} ({tracker_ptl_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # 恢复原始 attention
        restore_attention()
        
        print("\n" + "="*60)
        print("V4 导出完成！")
        print("TorchScript 格式:")
        print(f"  1. {first_path.name} - 首帧编码器")
        print(f"  2. {track_path.name} - 跟踪器")
        print("PyTorch Mobile 格式 (Android):")
        print(f"  3. {first_ptl_path.name} - 首帧编码器")
        print(f"  4. {tracker_ptl_path.name} - 跟踪器")
        print("="*60)
        
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
