# EfficientTAM 官方图片分割完整流程分析

## 1. 涉及的核心文件

| 文件 | 作用 |
|------|------|
| `notebooks/example_image.py` | 使用示例 |
| `efficient_track_anything/build_efficienttam.py` | 模型构建 |
| `efficient_track_anything/efficienttam_image_predictor.py` | 图片预测器（用户接口） |
| `efficient_track_anything/modeling/efficienttam_base.py` | 核心模型逻辑 |
| `efficient_track_anything/modeling/sam/prompt_encoder.py` | Prompt 编码器 |
| `efficient_track_anything/modeling/sam/mask_decoder.py` | Mask 解码器 |
| `efficient_track_anything/utils/transforms.py` | 图像预处理和坐标转换 |

---

## 2. 使用流程（用户视角）

```python
# 1. 构建预测器
predictor = build_efficienttam_image_predictor(model_cfg, checkpoint, device=device)

# 2. 设置图像（计算图像特征）
predictor.set_image(image)

# 3. 使用点提示进行分割
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),  # 1=前景, 0=背景
    multimask_output=True
)

# 4. 使用框提示进行分割
masks, scores, logits = predictor.predict(
    box=np.array([x1, y1, x2, y2]),
    multimask_output=True
)

# 5. 使用点+框组合提示
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),
    box=np.array([x1, y1, x2, y2]),
    multimask_output=False  # 多提示时建议关闭多mask输出
)
```

---

## 3. 内部流程详解

### 3.1 EfficientTAMImagePredictor 初始化

**文件**: `efficienttam_image_predictor.py:20-73`

```python
class EfficientTAMImagePredictor:
    def __init__(self, efficienttam_model, mask_threshold=0.0, ...):
        self.model = efficienttam_model
        self._transforms = EfficientTAMTransforms(resolution=self.model.image_size)
        
        # 预测器状态
        self._is_image_set = False
        self._features = None      # 缓存的图像特征
        self._orig_hw = None       # 原始图像尺寸
        
        # 特征图尺寸（512x512 输入时）
        self._bb_feat_sizes = [
            (128, 128),  # level 0
            (64, 64),    # level 1
            (32, 32),    # level 2 (最低分辨率)
        ]
```

**关键配置**:
- `mask_threshold`: mask 二值化阈值，默认 0.0
- `image_size`: 模型输入尺寸，512
- `_bb_feat_sizes`: 骨干网络输出的特征图尺寸

---

### 3.2 set_image() - 设置图像并计算特征

**文件**: `efficienttam_image_predictor.py:91-135`

```
输入: image (np.ndarray 或 PIL.Image)
输出: 无（特征缓存到 self._features）
```

**关键步骤**:

1. **记录原始尺寸** (line 107-113):
   ```python
   if isinstance(image, np.ndarray):
       self._orig_hw = [image.shape[:2]]  # (H, W)
   elif isinstance(image, Image):
       w, h = image.size
       self._orig_hw = [(h, w)]
   ```

2. **图像预处理** (line 116-117):
   ```python
   input_image = self._transforms(image)  # Resize + Normalize
   input_image = input_image[None, ...].to(self.device)  # 添加 batch 维度
   ```
   
   **预处理细节** (EfficientTAMTransforms):
   - Resize 到 512×512
   - 转换为 float32，值范围 0-1
   - ImageNet 标准化: `(img - mean) / std`
     - mean = [0.485, 0.456, 0.406]
     - std = [0.229, 0.224, 0.225]

3. **图像编码** (line 123-124):
   ```python
   backbone_out = self.model.forward_image(input_image)
   _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
   ```

4. **添加 no_mem_embed** (line 126-127):
   ```python
   if self.model.directly_add_no_mem_embed:
       vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
   ```
   **关键**: 图片分割模式下，直接将 `no_mem_embed` 加到最低分辨率特征上

5. **特征重塑和缓存** (line 129-133):
   ```python
   feats = [
       feat.permute(1, 2, 0).view(1, -1, *feat_size)
       for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
   ][::-1]
   self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
   ```
   
   **特征结构**:
   - `image_embed`: [1, 256, 32, 32] - 最低分辨率特征
   - `high_res_feats`: [[1, 256, 128, 128], [1, 256, 64, 64]] - 高分辨率特征

---

### 3.3 predict() - 执行分割预测

**文件**: `efficienttam_image_predictor.py:243-309`

```
输入:
  - point_coords: [N, 2] 点坐标（可选）
  - point_labels: [N] 点标签（可选）
  - box: [4] 框坐标 [x1, y1, x2, y2]（可选）
  - mask_input: [1, H, W] 上一次的 mask（可选）
  - multimask_output: 是否输出多个 mask
  - normalize_coords: 是否归一化坐标

输出:
  - masks: [C, H, W] 分割 mask
  - iou_predictions: [C] IoU 预测分数
  - low_res_masks: [C, 128, 128] 低分辨率 mask logits
```

**关键步骤**:

1. **准备提示** (line 293-295):
   ```python
   mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
       point_coords, point_labels, box, mask_input, normalize_coords
   )
   ```

2. **执行预测** (line 297-304):
   ```python
   masks, iou_predictions, low_res_masks = self._predict(
       unnorm_coords, labels, unnorm_box, mask_input, multimask_output
   )
   ```

3. **转换为 numpy** (line 306-309):
   ```python
   masks_np = masks.squeeze(0).float().detach().cpu().numpy()
   iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
   low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
   ```

---

### 3.4 _prep_prompts() - 准备提示输入

**文件**: `efficienttam_image_predictor.py:311-340`

**关键步骤**:

1. **点坐标转换** (line 316-328):
   ```python
   if point_coords is not None:
       point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
       unnorm_coords = self._transforms.transform_coords(
           point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
       )
       labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
       if len(unnorm_coords.shape) == 2:
           unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
   ```
   
   **坐标转换公式**:
   ```python
   # 如果 normalize_coords=True（默认）
   coords = coords / [orig_W, orig_H]  # 归一化到 [0, 1]
   coords = coords * image_size        # 缩放到 512
   ```

2. **框坐标转换** (line 329-333):
   ```python
   if box is not None:
       box = torch.as_tensor(box, dtype=torch.float, device=self.device)
       unnorm_box = self._transforms.transform_boxes(
           box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
       )  # 输出 shape: [B, 2, 2]
   ```

3. **Mask 输入处理** (line 334-339):
   ```python
   if mask_logits is not None:
       mask_input = torch.as_tensor(mask_logits, dtype=torch.float, device=self.device)
       if len(mask_input.shape) == 3:
           mask_input = mask_input[None, :, :, :]
   ```

---

### 3.5 _predict() - 核心预测逻辑

**文件**: `efficienttam_image_predictor.py:342-444`

**关键步骤**:

1. **合并点和框提示** (line 393-410):
   ```python
   if point_coords is not None:
       concat_points = (point_coords, point_labels)
   else:
       concat_points = None

   if boxes is not None:
       # 框转换为两个点（左上角和右下角）
       box_coords = boxes.reshape(-1, 2, 2)
       box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
       box_labels = box_labels.repeat(boxes.size(0), 1)
       
       # 合并框和点
       if concat_points is not None:
           concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
           concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
           concat_points = (concat_coords, concat_labels)
       else:
           concat_points = (box_coords, box_labels)
   ```
   
   **重要**: 框被转换为两个点，标签分别为 2（左上角）和 3（右下角）

2. **Prompt Encoder** (line 412-416):
   ```python
   sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
       points=concat_points,
       boxes=None,  # 框已经转换为点了
       masks=mask_input,
   )
   ```

3. **Mask Decoder** (line 426-434):
   ```python
   high_res_features = [
       feat_level[img_idx].unsqueeze(0)
       for feat_level in self._features["high_res_feats"]
   ]
   low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
       image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
       image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
       sparse_prompt_embeddings=sparse_embeddings,
       dense_prompt_embeddings=dense_embeddings,
       multimask_output=multimask_output,
       repeat_image=batched_mode,
       high_res_features=high_res_features,
   )
   ```

4. **Mask 后处理** (line 437-443):
   ```python
   # 上采样到原始图像尺寸
   masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx])
   low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
   if not return_logits:
       masks = masks > self.mask_threshold  # 默认阈值 0.0
   ```

---

## 4. Prompt Encoder 详解

**文件**: `efficient_track_anything/modeling/sam/prompt_encoder.py`

### 4.1 点嵌入 (_embed_points)

```python
def _embed_points(self, points, labels, pad):
    points = points + 0.5  # 偏移到像素中心
    
    # 位置编码
    point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
    
    # 根据标签添加不同的嵌入
    # label = -1: 填充点 → not_a_point_embed
    # label = 0:  背景点 → point_embeddings[0]
    # label = 1:  前景点 → point_embeddings[1]
    # label = 2:  框左上角 → point_embeddings[2]
    # label = 3:  框右下角 → point_embeddings[3]
```

**点标签含义**:

| 标签 | 含义 | 嵌入 |
|------|------|------|
| -1 | 填充点（无效） | `not_a_point_embed` |
| 0 | 背景点（负样本） | `point_embeddings[0]` |
| 1 | 前景点（正样本） | `point_embeddings[1]` |
| 2 | 框左上角 | `point_embeddings[2]` |
| 3 | 框右下角 | `point_embeddings[3]` |

### 4.2 框嵌入 (_embed_boxes)

```python
def _embed_boxes(self, boxes):
    boxes = boxes + 0.5  # 偏移到像素中心
    coords = boxes.reshape(-1, 2, 2)  # [B, 2, 2]
    
    # 位置编码
    corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
    
    # 添加角点嵌入
    corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # 左上角
    corner_embedding[:, 1, :] += self.point_embeddings[3].weight  # 右下角
```

**注意**: 框嵌入使用与点标签 2、3 相同的嵌入权重

### 4.3 Mask 嵌入 (_embed_masks)

```python
def _embed_masks(self, masks):
    # 下采样 mask: [B, 1, 512, 512] → [B, 256, 32, 32]
    mask_embedding = self.mask_downscaling(masks)
    return mask_embedding
```

### 4.4 forward() 输出

```python
def forward(self, points, boxes, masks):
    # 稀疏嵌入：点和框
    sparse_embeddings = []
    if points is not None:
        point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
        sparse_embeddings.append(point_embeddings)
    if boxes is not None:
        box_embeddings = self._embed_boxes(boxes)
        sparse_embeddings.append(box_embeddings)
    sparse_embeddings = torch.cat(sparse_embeddings, dim=1)  # [B, N, 256]
    
    # 密集嵌入：mask 或 no_mask_embed
    if masks is not None:
        dense_embeddings = self._embed_masks(masks)
    else:
        dense_embeddings = self.no_mask_embed  # [B, 256, 32, 32]
    
    return sparse_embeddings, dense_embeddings
```

---

## 5. Mask Decoder 详解

**文件**: `efficient_track_anything/modeling/sam/mask_decoder.py`

### 5.1 输入

| 参数 | 形状 | 说明 |
|------|------|------|
| `image_embeddings` | [B, 256, 32, 32] | 图像特征 |
| `image_pe` | [1, 256, 32, 32] | 图像位置编码 |
| `sparse_prompt_embeddings` | [B, N, 256] | 点/框稀疏嵌入 |
| `dense_prompt_embeddings` | [B, 256, 32, 32] | mask 密集嵌入 |
| `multimask_output` | bool | 是否输出多个 mask |

### 5.2 内部流程

```python
def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, 
            dense_prompt_embeddings, multimask_output, ...):
    
    # 1. 准备 tokens
    output_tokens = torch.cat([self.iou_token, self.mask_tokens], dim=1)
    # iou_token: [1, 256] - IoU 预测 token
    # mask_tokens: [4, 256] - 4 个 mask tokens (1 single + 3 multi)
    
    tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
    
    # 2. 图像特征 + 密集嵌入
    src = image_embeddings + dense_prompt_embeddings
    pos_src = image_pe
    
    # 3. Two-Way Transformer
    hs, src = self.transformer(src, pos_src, tokens)
    # hs: [B, N_tokens, 256] - 更新后的 tokens
    # src: [B, 32*32, 256] - 更新后的图像特征
    
    # 4. 生成 mask
    iou_token_out = hs[:, 0, :]  # IoU token
    mask_tokens_out = hs[:, 1:5, :]  # 4 个 mask tokens
    
    # 上采样图像特征
    src = src.transpose(1, 2).view(B, C, H, W)
    upscaled_embedding = self.output_upscaling(src)  # [B, 32, 128, 128]
    
    # 生成 mask
    hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
    masks = (hyper_in @ upscaled_embedding.flatten(2)).view(B, -1, H*4, W*4)
    # masks: [B, 4, 128, 128]
    
    # 5. IoU 预测
    iou_pred = self.iou_prediction_head(iou_token_out)
    # iou_pred: [B, 4]
    
    # 6. 选择输出
    if multimask_output:
        return masks[:, 1:], iou_pred[:, 1:]  # 3 个 mask
    else:
        return masks[:, 0:1], iou_pred[:, 0:1]  # 1 个 mask
```

### 5.3 输出

| 输出 | 形状 | 说明 |
|------|------|------|
| `low_res_masks` | [B, C, 128, 128] | 低分辨率 mask logits |
| `iou_predictions` | [B, C] | IoU 预测分数 |
| `sam_output_tokens` | [B, C, 256] | SAM 输出 tokens |
| `object_score_logits` | [B, 1] | 对象存在分数 |

其中 C = 3（multimask_output=True）或 C = 1（multimask_output=False）

---

## 6. _forward_sam_heads() 详解

**文件**: `efficienttam_base.py:260-416`

这是 EfficientTAMBase 中调用 SAM 头部的核心函数。

### 6.1 输入处理

```python
def _forward_sam_heads(self, backbone_features, point_inputs=None, 
                       mask_inputs=None, high_res_features=None, multimask_output=False):
    B = backbone_features.size(0)
    device = backbone_features.device
    
    # a) 处理点提示
    if point_inputs is not None:
        sam_point_coords = point_inputs["point_coords"]
        sam_point_labels = point_inputs["point_labels"]
    else:
        # 如果没有点，添加一个空点（标签 -1）
        sam_point_coords = torch.zeros(B, 1, 2, device=device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
    
    # b) 处理 mask 提示
    if mask_inputs is not None:
        # 下采样到 prompt encoder 期望的尺寸
        sam_mask_prompt = F.interpolate(mask_inputs, size=self.sam_prompt_encoder.mask_input_size)
    else:
        sam_mask_prompt = None
```

### 6.2 Prompt Encoder 调用

```python
sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
    points=(sam_point_coords, sam_point_labels),
    boxes=None,
    masks=sam_mask_prompt,
)
```

### 6.3 Mask Decoder 调用

```python
(low_res_multimasks, ious, sam_output_tokens, object_score_logits) = self.sam_mask_decoder(
    image_embeddings=backbone_features,
    image_pe=self.sam_prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=multimask_output,
    repeat_image=False,
    high_res_features=high_res_features,
)
```

### 6.4 后处理

```python
# 对象存在判断
if self.pred_obj_scores:
    is_obj_appearing = object_score_logits > 0
    low_res_multimasks = torch.where(
        is_obj_appearing[:, None, None],
        low_res_multimasks,
        NO_OBJ_SCORE,  # -1024.0
    )

# 上采样到原始尺寸
high_res_multimasks = F.interpolate(
    low_res_multimasks,
    size=(self.image_size, self.image_size),  # 512x512
    mode="bilinear",
    align_corners=False,
)

# 选择最佳 mask
if multimask_output:
    best_iou_inds = torch.argmax(ious, dim=-1)
    low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
    high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
else:
    low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

# 提取 object pointer
obj_ptr = self.obj_ptr_proj(sam_output_token)
```

### 6.5 输出

```python
return (
    low_res_multimasks,   # [B, M, 128, 128] - 所有 mask
    high_res_multimasks,  # [B, M, 512, 512] - 所有 mask（上采样）
    ious,                 # [B, M] - IoU 预测
    low_res_masks,        # [B, 1, 128, 128] - 最佳 mask
    high_res_masks,       # [B, 1, 512, 512] - 最佳 mask（上采样）
    obj_ptr,              # [B, 256] - object pointer
    object_score_logits,  # [B, 1] - 对象存在分数
)
```

---

## 7. 数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           set_image()                                    │
│  image (H, W, 3) ──────────────────────────────────────────────────────┐│
│         │                                                               ││
│         ▼                                                               ││
│  ┌─────────────────┐                                                    ││
│  │ EfficientTAM    │                                                    ││
│  │ Transforms      │                                                    ││
│  │ - Resize 512    │                                                    ││
│  │ - Normalize     │                                                    ││
│  └────────┬────────┘                                                    ││
│           ▼                                                             ││
│  input_image [1, 3, 512, 512]                                           ││
│           │                                                             ││
│           ▼                                                             ││
│  ┌─────────────────┐                                                    ││
│  │ Image Encoder   │                                                    ││
│  │ (EfficientViT)  │                                                    ││
│  └────────┬────────┘                                                    ││
│           ▼                                                             ││
│  backbone_out:                                                          ││
│    backbone_fpn: [level0, level1, level2]                               ││
│    vision_pos_enc: [pos0, pos1, pos2]                                   ││
│           │                                                             ││
│           ▼                                                             ││
│  ┌─────────────────┐                                                    ││
│  │ _prepare_       │                                                    ││
│  │ backbone_       │                                                    ││
│  │ features()      │                                                    ││
│  └────────┬────────┘                                                    ││
│           ▼                                                             ││
│  vision_feats[-1] + no_mem_embed                                        ││
│           │                                                             ││
│           ▼                                                             ││
│  self._features = {                                                     ││
│      "image_embed": [1, 256, 32, 32],                                   ││
│      "high_res_feats": [[1, 256, 128, 128], [1, 256, 64, 64]]           ││
│  }                                                                      ││
└─────────────────────────────────────────────────────────────────────────┘│
                                                                           │
┌─────────────────────────────────────────────────────────────────────────┐│
│                            predict()                                     ││
│  point_coords, point_labels, box ───────────────────────────────────────┐│
│         │                                                               ││
│         ▼                                                               ││
│  ┌─────────────────┐                                                    ││
│  │ _prep_prompts() │                                                    ││
│  │ - 坐标转换      │                                                    ││
│  │ - 框转点        │                                                    ││
│  └────────┬────────┘                                                    ││
│           ▼                                                             ││
│  unnorm_coords [1, N, 2], labels [1, N], unnorm_box [1, 2, 2]           ││
│           │                                                             ││
│           ▼                                                             ││
│  ┌─────────────────┐                                                    ││
│  │ _predict()      │                                                    ││
│  │                 │                                                    ││
│  │  ┌───────────┐  │                                                    ││
│  │  │ 合并点+框 │  │  box → 2 points (labels 2, 3)                      ││
│  │  └─────┬─────┘  │                                                    ││
│  │        ▼        │                                                    ││
│  │  concat_points = (coords [1, N+2, 2], labels [1, N+2])               ││
│  │        │        │                                                    ││
│  │        ▼        │                                                    ││
│  │  ┌───────────────────┐                                               ││
│  │  │ sam_prompt_encoder│                                               ││
│  │  │ - _embed_points() │                                               ││
│  │  │ - no_mask_embed   │                                               ││
│  │  └─────────┬─────────┘                                               ││
│  │            ▼                                                         ││
│  │  sparse_embeddings [1, N+2, 256]                                     ││
│  │  dense_embeddings [1, 256, 32, 32]                                   ││
│  │            │                                                         ││
│  │            ▼                                                         ││
│  │  ┌───────────────────┐                                               ││
│  │  │ sam_mask_decoder  │                                               ││
│  │  │ - TwoWayTransformer│                                              ││
│  │  │ - output_upscaling│                                               ││
│  │  │ - hypernetworks   │                                               ││
│  │  └─────────┬─────────┘                                               ││
│  │            ▼                                                         ││
│  │  low_res_masks [1, C, 128, 128]                                      ││
│  │  iou_predictions [1, C]                                              ││
│  │            │                                                         ││
│  │            ▼                                                         ││
│  │  ┌───────────────────┐                                               ││
│  │  │ postprocess_masks │                                               ││
│  │  │ - 上采样到原始尺寸│                                               ││
│  │  │ - 阈值化 (>0)     │                                               ││
│  │  └─────────┬─────────┘                                               ││
│  │            ▼                                                         ││
│  │  masks [C, orig_H, orig_W]                                           ││
│  └────────────┘                                                         ││
└─────────────────────────────────────────────────────────────────────────┘│
```

---

## 8. 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `image_size` | 512 | 模型输入图像尺寸 |
| `hidden_dim` | 256 | 特征维度 |
| `backbone_stride` | 16 | 骨干网络下采样倍数 |
| `sam_image_embedding_size` | 32 | SAM 特征图尺寸 (512/16=32) |
| `mask_threshold` | 0.0 | mask 二值化阈值 |
| `num_multimask_outputs` | 3 | 多 mask 输出数量 |
| `directly_add_no_mem_embed` | True | 直接添加 no_mem_embed |
| `use_high_res_features_in_sam` | False | 是否使用高分辨率特征 |
| `pred_obj_scores` | True | 是否预测对象存在分数 |
| `NO_OBJ_SCORE` | -1024.0 | 无对象时的 mask 值 |

---

## 9. 关键维度说明

| 变量 | 维度 | 说明 |
|------|------|------|
| `input_image` | [1, 3, 512, 512] | 预处理后的输入图像 |
| `backbone_out["backbone_fpn"][-1]` | [1, 256, 32, 32] | 最低分辨率特征 |
| `vision_feats[-1]` | [1024, 1, 256] | flatten 后的特征 (32*32=1024) |
| `image_embed` | [1, 256, 32, 32] | 缓存的图像嵌入 |
| `sparse_embeddings` | [1, N, 256] | 点/框稀疏嵌入 |
| `dense_embeddings` | [1, 256, 32, 32] | mask 密集嵌入 |
| `low_res_masks` | [1, C, 128, 128] | 低分辨率 mask |
| `high_res_masks` | [1, C, 512, 512] | 高分辨率 mask |
| `iou_predictions` | [1, C] | IoU 预测分数 |
| `obj_ptr` | [1, 256] | object pointer |

---

## 10. 导出 TorchScript 的关键点

### 10.1 Android 导出版本的简化

```python
class TAMWrapperSimple(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, img, point_coords, point_labels):
        # 1. 图像编码
        backbone_out = self.base_model.image_encoder(img)
        
        # 2. 构造点输入（框已在外部转换为点）
        point_inputs = {
            'point_coords': point_coords,
            'point_labels': point_labels
        }
        
        # 3. SAM 推理
        sam_outputs = self.base_model._forward_sam_heads(
            backbone_features=backbone_out['backbone_fpn'][-1],
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=None,
            multimask_output=False,
        )
        
        # 4. 返回高分辨率 mask
        high_res_masks = sam_outputs[4]  # [B, 1, 512, 512]
        return high_res_masks
```

### 10.2 关键简化

| 原始版本 | Android 版本 | 原因 |
|----------|--------------|------|
| 分离的 set_image + predict | 单次 forward | 简化接口 |
| 支持框参数 | 框转换为点 | 统一输入格式 |
| 多 mask 输出 | 单 mask 输出 | 减少计算 |
| 特征缓存 | 无缓存 | 简化状态管理 |
| mask 提示 | 不支持 | 简化接口 |

### 10.3 框转点的标准约定

```python
# 框 [x1, y1, x2, y2] 转换为两个点
point_coords = [[x1, y1], [x2, y2]]  # shape: [1, 2, 2]
point_labels = [2, 3]                 # 2=左上角, 3=右下角
```

---

## 11. 与视频分割的区别

| 方面 | 图片分割 | 视频分割 |
|------|----------|----------|
| **Memory** | 无 | 需要维护 memory 状态 |
| **no_mem_embed** | 直接加到特征上 | 首帧加，后续帧用 Memory Attention |
| **Memory Attention** | 不使用 | 后续帧必须使用 |
| **Memory Encoder** | 不使用 | 每帧编码 memory |
| **Object Pointer** | 生成但不使用 | 传递给后续帧 |
| **特征缓存** | 支持（set_image） | 不缓存（每帧重新编码） |
| **状态管理** | 无状态 | 有状态（memory） |

---

## 12. 总结

### 12.1 图片分割核心流程

1. **set_image()**: 图像预处理 → 图像编码 → 特征缓存
2. **predict()**: 提示准备 → Prompt Encoder → Mask Decoder → 后处理

### 12.2 关键技术点

- **框转点**: 框被转换为两个点（标签 2, 3），与点使用相同的编码路径
- **no_mem_embed**: 图片模式下直接加到特征上，表示"无记忆"
- **多 mask 输出**: 默认输出 3 个候选 mask，选择 IoU 最高的
- **特征缓存**: set_image 后缓存特征，支持多次 predict

### 12.3 Android 导出要点

- 统一接口：框转换为点，只需 3 个参数
- 单 mask 输出：`multimask_output=False`
- 无特征缓存：每次推理重新编码图像
- 预处理外置：ImageNet 标准化在 Android 端完成
