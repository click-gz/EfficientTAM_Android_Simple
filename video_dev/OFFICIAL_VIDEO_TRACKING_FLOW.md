# EfficientTAM 官方视频分割完整流程分析

## 1. 涉及的核心文件

| 文件 | 作用 |
|------|------|
| `notebooks/example_video.py` | 使用示例 |
| `efficient_track_anything/build_efficienttam.py` | 模型构建 |
| `efficient_track_anything/efficienttam_video_predictor.py` | 视频预测器（用户接口） |
| `efficient_track_anything/modeling/efficienttam_base.py` | 核心模型逻辑 |
| `efficient_track_anything/modeling/memory_attention.py` | Memory Attention 模块 |
| `efficient_track_anything/modeling/memory_encoder.py` | Memory Encoder 模块 |
| `efficient_track_anything/utils/misc.py` | 图像加载和预处理 |

---

## 2. 使用流程（用户视角）

```python
# 1. 构建预测器
predictor = build_efficienttam_video_predictor(model_cfg, checkpoint, device=device)

# 2. 初始化状态（加载视频帧）
inference_state = predictor.init_state(video_path=video_dir)

# 3. 在首帧添加点提示
_, obj_ids, masks = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=np.array([[200, 300]]),
    labels=np.array([1]),  # 1=前景
)

# 4. 传播到所有帧
for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
    # 处理每帧的分割结果
    pass
```

---

## 3. 内部流程详解

### 3.1 init_state() - 初始化状态

**文件**: `efficienttam_video_predictor.py:48-106`

```
输入: video_path (视频帧目录)
输出: inference_state (字典)
```

**关键步骤**:
1. 调用 `load_video_frames()` 加载所有视频帧
   - 图像 resize 到 512x512
   - 应用 ImageNet 标准化: `(img - mean) / std`
   - mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
2. 初始化 `inference_state` 字典，包含:
   - `images`: 预处理后的视频帧列表
   - `video_height`, `video_width`: 原始视频尺寸
   - `output_dict_per_obj`: 每个对象的输出字典
   - `cached_features`: 缓存的图像特征
3. 预热：调用 `_get_image_feature()` 缓存第 0 帧的特征

---

### 3.2 add_new_points_or_box() - 添加点/框提示

**文件**: `efficienttam_video_predictor.py:170-302`

```
输入: inference_state, frame_idx, obj_id, points, labels
输出: frame_idx, obj_ids, video_res_masks
```

**关键步骤**:

1. **坐标转换** (line 221-226):
   ```python
   if normalize_coords:
       points = points / [video_W, video_H]  # 归一化到 [0,1]
   points = points * self.image_size  # 缩放到 512
   ```

2. **调用 `_run_single_frame_inference()`** (line 272-287):
   - `is_init_cond_frame=True` (首帧)
   - `run_mem_encoder=False` (暂不编码 memory)

3. **存储输出到临时字典** (line 289):
   ```python
   obj_temp_output_dict["cond_frame_outputs"][frame_idx] = current_out
   ```

---

### 3.3 _run_single_frame_inference() - 单帧推理

**文件**: `efficienttam_video_predictor.py:746-812`

```
输入: inference_state, output_dict, frame_idx, is_init_cond_frame, point_inputs, ...
输出: compact_current_out, pred_masks_gpu
```

**关键步骤**:

1. **获取图像特征** (line 761-767):
   ```python
   (_, _, current_vision_feats, current_vision_pos_embeds, feat_sizes) = 
       self._get_image_feature(inference_state, frame_idx, batch_size)
   ```

2. **调用 `track_step()`** (line 771-784):
   ```python
   current_out = self.track_step(
       frame_idx=frame_idx,
       is_init_cond_frame=is_init_cond_frame,
       current_vision_feats=current_vision_feats,
       current_vision_pos_embeds=current_vision_pos_embeds,
       feat_sizes=feat_sizes,
       point_inputs=point_inputs,
       ...
   )
   ```

---

### 3.4 track_step() - 核心跟踪步骤

**文件**: `efficienttam_base.py:817-882`

```
输入: frame_idx, is_init_cond_frame, current_vision_feats, point_inputs, output_dict, ...
输出: current_out (包含 pred_masks, maskmem_features, obj_ptr 等)
```

**关键步骤**:

1. **调用 `_track_step()`** 获取 SAM 输出
2. **调用 `_encode_memory_in_output()`** 编码 memory（如果 run_mem_encoder=True）

---

### 3.5 _track_step() - 跟踪步骤核心

**文件**: `efficienttam_base.py:731-790`

**关键步骤**:

1. **准备 memory conditioned features** (line 764-773):
   ```python
   pix_feat = self._prepare_memory_conditioned_features(
       frame_idx=frame_idx,
       is_init_cond_frame=is_init_cond_frame,
       current_vision_feats=current_vision_feats[-1:],
       current_vision_pos_embeds=current_vision_pos_embeds[-1:],
       ...
   )
   ```

2. **调用 SAM heads** (line 782-788):
   ```python
   sam_outputs = self._forward_sam_heads(
       backbone_features=pix_feat,
       point_inputs=point_inputs,
       ...
   )
   ```

---

### 3.6 _prepare_memory_conditioned_features() - 准备带 memory 的特征

**文件**: `efficienttam_base.py:500-679`

**这是最关键的函数！**

#### 情况 1: 首帧 (is_init_cond_frame=True)

```python
if self.directly_add_no_mem_embed:  # 默认为 True
    # 直接将 no_mem_embed 加到特征上
    pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
    pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
    return pix_feat_with_mem
```

**关键**: 首帧不经过 Memory Attention，只是简单地加上 `no_mem_embed`

#### 情况 2: 后续帧 (is_init_cond_frame=False)

1. **收集 memory** (line 527-587):
   - 从 `cond_frame_outputs` 收集条件帧的 memory
   - 从 `non_cond_frame_outputs` 收集最近 `num_maskmem-1` 帧的 memory
   - 每个 memory 加上时序位置编码: `maskmem_tpos_enc[num_maskmem - t_pos - 1]`

2. **Memory Attention** (line 670-678):
   ```python
   pix_feat_with_mem = self.memory_attention(
       curr=current_vision_feats,  # 列表格式，内部取 [0]
       curr_pos=current_vision_pos_embeds,  # 列表格式，内部取 [0]
       memory=memory,  # [seq_len, B, mem_dim]
       memory_pos=memory_pos_embed,  # [seq_len, B, mem_dim]
       num_obj_ptr_tokens=num_obj_ptr_tokens,
   )
   ```
   
   **Memory Attention 内部处理**:
   - 输入 `curr` 如果是列表，会取 `curr[0]`
   - `pos_enc_at_input=True` 时: `output = curr + 0.1 * curr_pos`
   - `batch_first=True` 时会做 transpose
   - `num_obj_ptr_tokens` 用于 RoPE 排除（obj_ptr 不参与位置编码）
   
   **维度差异处理**（重要！）:
   - `curr` 维度: `[HW, B, 256]` (hidden_dim)
   - `memory` 维度: `[seq, B, 64]` (mem_dim)
   - Cross Attention 内部有投影层处理维度差异:
     - `q_proj`: Linear(256 → 256)
     - `k_proj`: Linear(64 → 256)  ← 将 memory 投影到 256 维
     - `v_proj`: Linear(64 → 256)  ← 将 memory 投影到 256 维

---

### 3.7 propagate_in_video() - 视频传播

**文件**: `efficienttam_video_predictor.py:555-639`

**关键步骤**:

1. **调用 `propagate_in_video_preflight()`** (line 563):
   - 运行 memory encoder 编码首帧的 memory
   - 将临时输出合并到正式输出字典

2. **遍历每一帧** (line 592-639):
   - 条件帧: 直接使用已有输出
   - 非条件帧: 调用 `_run_single_frame_inference()` 进行跟踪
     - `is_init_cond_frame=False`
     - `run_mem_encoder=True`

---

### 3.8 _encode_new_memory() - 编码新 memory

**文件**: `efficienttam_base.py:681-729`

```
输入: current_vision_feats, pred_masks_high_res, ...
输出: maskmem_features, maskmem_pos_enc
```

**关键步骤**:

1. **获取原始视觉特征** (line 694):
   ```python
   pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
   ```
   **注意**: 使用的是**原始视觉特征**，不是加了 no_mem_embed 的特征！

2. **处理 mask** (line 703-713):
   ```python
   if binarize:
       mask_for_mem = (pred_masks_high_res > 0).float()
   else:
       mask_for_mem = torch.sigmoid(pred_masks_high_res)
   ```

3. **调用 memory encoder** (line 714-718):
   ```python
   maskmem_out = self.memory_encoder(
       pix_feat, mask_for_mem, skip_mask_sigmoid=True
   )
   ```

---

## 4. 数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           init_state()                                   │
│  video_path → load_video_frames() → images (预处理后的帧列表)            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      add_new_points_or_box()                             │
│  points → 坐标转换 → _run_single_frame_inference()                       │
│                              ↓                                           │
│                    _get_image_feature()                                  │
│                    images[0] → image_encoder → features, pos_enc         │
│                              ↓                                           │
│                         track_step()                                     │
│                              ↓                                           │
│              _prepare_memory_conditioned_features()                      │
│              (首帧: features + no_mem_embed)                             │
│                              ↓                                           │
│                    _forward_sam_heads()                                  │
│              pix_feat + point_inputs → mask                              │
│                              ↓                                           │
│              存储到 temp_output_dict["cond_frame_outputs"][0]            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       propagate_in_video()                               │
│                              ↓                                           │
│              propagate_in_video_preflight()                              │
│              运行 memory encoder 编码首帧 memory                         │
│              temp_output → output_dict["cond_frame_outputs"]             │
│                              ↓                                           │
│              for frame_idx in range(num_frames):                         │
│                  if frame_idx in cond_frame_outputs:                     │
│                      直接使用已有输出                                    │
│                  else:                                                   │
│                      _run_single_frame_inference()                       │
│                              ↓                                           │
│                    _get_image_feature()                                  │
│                    images[frame_idx] → features, pos_enc                 │
│                              ↓                                           │
│                         track_step()                                     │
│                              ↓                                           │
│              _prepare_memory_conditioned_features()                      │
│              (后续帧: Memory Attention)                                  │
│                  收集之前帧的 maskmem_features + obj_ptr                 │
│                  memory_attention(curr, memory+obj_ptr) → pix_feat_with_mem │
│                              ↓                                           │
│                    _forward_sam_heads()                                  │
│              pix_feat_with_mem → mask                                    │
│                              ↓                                           │
│                    _encode_memory_in_output()                            │
│              调用 _encode_new_memory(current_vision_feats, mask)         │
│              **使用原始 features，不是 pix_feat_with_mem！**             │
│                              ↓                                           │
│              存储到 output_dict["non_cond_frame_outputs"][frame_idx]     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `image_size` | 512 | 模型输入图像尺寸 |
| `hidden_dim` | 256 | 特征维度 |
| `mem_dim` | 64 | Memory 维度 |
| `num_maskmem` | 7 | Memory 帧数 |
| `directly_add_no_mem_embed` | True | 首帧直接加 no_mem_embed |
| `use_high_res_features_in_sam` | False | 是否使用高分辨率特征 |
| `num_feature_levels` | 1 | 特征层级数（只用最后一层） |
| `binarize_mask_from_pts_for_mem_enc` | False | Memory 编码时是否二值化 mask |
| `use_obj_ptrs_in_encoder` | True | 是否在 Memory Attention 中使用 object pointers |
| `max_obj_ptrs_in_encoder` | 16 | 最大 object pointer 数量 |
| `backbone_stride` | 16 | 骨干网络下采样倍数 |
| `sam_image_embedding_size` | 32 | SAM 特征图尺寸 (512/16=32) |
| `no_mem_embed` shape | [1, 1, 256] | 无记忆嵌入的形状 |
| `memory_attention.pos_enc_at_input` | True | Memory Attention 输入加位置编码 |
| `memory_attention.batch_first` | True | Memory Attention 使用 batch first |
| `memory_attention.d_model` | 256 | Memory Attention 模型维度 |
| `pred_obj_scores` | True | 是否预测对象分数 |
| `soft_no_obj_ptr` | False | 是否使用软 no_obj_ptr |
| `fixed_no_obj_ptr` | True | 是否固定 no_obj_ptr |
| `sigmoid_scale_for_mem_enc` | 20.0 | Memory Encoder mask 缩放因子 |
| `sigmoid_bias_for_mem_enc` | -10.0 | Memory Encoder mask 偏置 |
| `multimask_output_in_sam` | True | SAM 是否输出多个 mask |
| `multimask_output_for_tracking` | True | 跟踪时是否使用多 mask |
| `multimask_min_pt_num` | 0 | 多 mask 最小点数 |
| `multimask_max_pt_num` | 1 | 多 mask 最大点数 |

---

## 6. Object Pointers (obj_ptr)

**重要**: 当 `use_obj_ptrs_in_encoder=True` 时，后续帧的 Memory Attention 不仅使用 `maskmem_features`，还会使用 **object pointers**。

### 6.1 Object Pointer 来源

- 从 `_forward_sam_heads()` 的输出中提取
- 通过 `obj_ptr_proj` (MLP) 投影 SAM output token
- **重要**: 还需要处理 occlusion（遮挡）逻辑：
  ```python
  obj_ptr = self.obj_ptr_proj(sam_output_token)
  if self.pred_obj_scores:  # True
      lambda_is_obj_appearing = is_obj_appearing.float()  # soft_no_obj_ptr=False
      if self.fixed_no_obj_ptr:  # True
          obj_ptr = lambda_is_obj_appearing * obj_ptr
      obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
  ```
- 存储在 `current_out["obj_ptr"]` 中
- **维度**: `[B, 256]` (hidden_dim)

### 6.2 Object Pointer 在 Memory Attention 中的使用

```python
# 在 _prepare_memory_conditioned_features() 中
if self.use_obj_ptrs_in_encoder:
    # 收集条件帧和非条件帧的 obj_ptr
    pos_and_ptrs = [(t_diff, out["obj_ptr"]) for t, out in ...]
    
    # 当 mem_dim < hidden_dim 时，拆分 obj_ptr
    # mem_dim=64, hidden_dim=256, 所以每个 obj_ptr 拆分成 4 个 tokens
    if self.mem_dim < C:
        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
    
    # 添加到 memory 中
    to_cat_memory.append(obj_ptrs)
    to_cat_memory_pos_embed.append(obj_pos)
    num_obj_ptr_tokens = obj_ptrs.shape[0]
```

### 6.3 对导出的影响

由于 `use_obj_ptrs_in_encoder=True`，导出模型时需要：
1. 首帧输出 `obj_ptr`
2. 后续帧的 Memory Attention 需要接收 `obj_ptr` 作为输入

---

## 7. Memory 时序位置编码

```python
# maskmem_tpos_enc shape: [7, 1, 1, 64]
# t_pos=0 (条件帧): 索引 6
# t_pos=1 (最近帧): 索引 5
# ...
# t_pos=6 (最远帧): 索引 0
maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
```

---

## 8. 图像特征提取流程

### 8.1 forward_image()

**文件**: `efficienttam_base.py:470-482`

```python
def forward_image(self, img_batch: torch.Tensor):
    backbone_out = self.image_encoder(img_batch)
    if self.use_high_res_features_in_sam:
        # 预计算 level 0 和 level 1 的投影特征
        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
    return backbone_out
```

### 8.2 _prepare_backbone_features()

**文件**: `efficienttam_base.py:484-498`

```python
def _prepare_backbone_features(self, backbone_out):
    feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
    vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
    feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
    # flatten NxCxHxW to HWxNxC
    vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
    vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
    return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
```

### 8.3 _get_image_feature() 返回值

**文件**: `efficienttam_video_predictor.py:713-744`

```python
features = self._prepare_backbone_features(expanded_backbone_out)
features = (expanded_image,) + features
return features
# 返回: (image, backbone_out, vision_feats, vision_pos_embeds, feat_sizes)
```

### 8.4 关键维度说明

| 变量 | 维度 | 说明 |
|------|------|------|
| `current_vision_feats[-1]` | `[HW, B, C]` = `[1024, 1, 256]` | flatten 后的特征 (32*32=1024) |
| `current_vision_pos_embeds[-1]` | `[HW, B, C]` = `[1024, 1, 256]` | flatten 后的位置编码 |
| `feat_sizes[-1]` | `(32, 32)` | 特征图尺寸 |
| `pix_feat` (BCHW) | `[B, C, H, W]` = `[1, 256, 32, 32]` | reshape 后的特征 |
| `maskmem_features` | `[B, 64, 32, 32]` | Memory 特征 (mem_dim=64) |
| `maskmem_pos_enc` | **列表** `[tensor]` | Memory 位置编码，取 `[-1]` 得到 `[B, 64, 32, 32]` |
| `obj_ptr` | `[B, 256]` | Object Pointer |

### 8.5 Memory Attention 输入准备

```python
# maskmem_features: [B, 64, 32, 32] -> [HW, B, 64] = [1024, 1, 64]
feats = maskmem_features.flatten(2).permute(2, 0, 1)

# maskmem_pos_enc[-1]: [B, 64, 32, 32] -> [HW, B, 64] = [1024, 1, 64]
maskmem_enc = maskmem_pos_enc[-1].flatten(2).permute(2, 0, 1)

# 添加时序位置编码
maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]

# obj_ptr: [B, 256] -> 拆分成 4 个 tokens [4, B, 64]
obj_ptrs = obj_ptr.reshape(-1, B, C // self.mem_dim, self.mem_dim)
obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)

# 拼接
memory = torch.cat([feats, obj_ptrs], dim=0)  # [1024+4, B, 64]
memory_pos = torch.cat([maskmem_enc, obj_pos], dim=0)
```

---

## 9. 导出 TorchScript 的关键点

1. **首帧处理**:
   - 图像编码: `image_encoder(img)` → features
   - 特征处理: `features + no_mem_embed` → pix_feat_with_mem
   - SAM 解码: `sam_prompt_encoder + sam_mask_decoder` → mask, **obj_ptr**
   - Memory 编码: `memory_encoder(原始 features, mask)` → maskmem_features
   - **输出**: mask, maskmem_features, maskmem_pos_enc, **obj_ptr**

2. **后续帧处理**:
   - 图像编码: `image_encoder(img)` → features
   - Memory Attention: `memory_attention(features, maskmem_features + obj_ptr)` → pix_feat_with_mem
   - SAM 解码: `sam_mask_decoder(pix_feat_with_mem)` → mask, **new_obj_ptr**
   - Memory 编码: `memory_encoder(原始 features, mask)` → new_maskmem_features
   - **输出**: mask, new_maskmem_features, new_maskmem_pos_enc, **new_obj_ptr**

3. **关键点**:
   - **Memory Encoder 始终使用原始 `current_vision_feats`**，不是经过 memory attention 的特征！
   - **Object Pointer (obj_ptr) 必须传递**：`use_obj_ptrs_in_encoder=True`
   - 首帧的 `obj_ptr` 需要传递给后续帧的 Memory Attention
   - `_encode_new_memory()` 内部使用 `current_vision_feats[-1]` 作为 `pix_feat`
