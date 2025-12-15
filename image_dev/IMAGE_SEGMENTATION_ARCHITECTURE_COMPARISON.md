# EfficientTAM 图片分割 Android 部署架构对比

本文档详细说明 Android 图片分割版本与原始 EfficientTAM 的架构差异。

---

## 1. 整体架构对比

### 1.1 原始 EfficientTAM（Python）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          EfficientTAMPredictor                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         set_image()                              │    │
│  │  - 图像预处理：Resize + ImageNet 标准化                          │    │
│  │  - 图像编码：image_encoder(img) → backbone_out                   │    │
│  │  - 缓存特征：存储到 predictor 内部状态                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                          predict()                               │    │
│  │  - 支持多种输入：point_coords, point_labels, box, mask_input    │    │
│  │  - Prompt Encoder：编码点/框/mask 提示                           │    │
│  │  - Mask Decoder：生成分割 mask                                   │    │
│  │  - 多 mask 输出：返回 3 个候选 mask + scores                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     后处理（可选）                                │    │
│  │  - 多次迭代：使用上一次的 mask 作为输入                          │    │
│  │  - 多对象：为每个对象单独调用 predict()                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Android 部署版本

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       EfficientTAMPointPrompt                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       初始化 (init)                              │    │
│  │  - 加载 TorchScript 模型                                         │    │
│  │  - 模型包含完整流程：编码 + 解码                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │               predictWithPoints() / predictWithBox()             │    │
│  │  - 图像预处理：Resize + ImageNet 标准化                          │    │
│  │  - 坐标转换：图片坐标 → 512 空间                                 │    │
│  │  - 单次推理：model.forward(img, coords, labels)                  │    │
│  │  - 单 mask 输出：直接返回最佳 mask                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 关键差异对比

### 2.1 模型结构

| 方面 | 原始版本 | Android 版本 |
|------|----------|--------------|
| **模型数量** | 分离的编码器和解码器 | 1 个端到端模型 |
| **模型文件** | `efficienttam_ti.pt` (checkpoint) | `efficienttam_ti_512x512_point_and_box.pt` |
| **模型格式** | PyTorch state_dict | TorchScript (.pt) |
| **模型大小** | ~24 MB | ~24 MB |
| **特征缓存** | 支持（set_image 后缓存） | 不支持（每次重新编码） |

### 2.2 输入接口

| 方面 | 原始版本 | Android 版本 |
|------|----------|--------------|
| **图像输入** | 任意尺寸，内部 resize | 任意尺寸，内部 resize 到 512 |
| **点提示** | `point_coords` + `point_labels` | `point_coords` + `point_labels` |
| **框提示** | 单独的 `box` 参数 | 转换为 2 个点（标签 2, 3） |
| **mask 提示** | 支持 `mask_input` | 不支持 |
| **参数数量** | 4+ 个可选参数 | 固定 3 个参数 |

**原始版本调用**:
```python
masks, scores, logits = predictor.predict(
    point_coords=points,
    point_labels=labels,
    box=box,           # 单独的框参数
    mask_input=None,
    multimask_output=True
)
```

**Android 版本调用**:
```kotlin
// 点提示
val mask = model.forward(img, pointCoords, pointLabels)

// 框提示（框转换为两个点）
val boxCoords = floatArrayOf(x1, y1, x2, y2)  // 两个点
val boxLabels = longArrayOf(2L, 3L)           // 左上角=2, 右下角=3
val mask = model.forward(img, boxCoords, boxLabels)
```

### 2.3 图像预处理

| 方面 | 原始版本 | Android 版本 |
|------|----------|--------------|
| **输入尺寸** | 1024×1024 (SAM) 或 512×512 | 512×512 |
| **值范围** | 0-1 (归一化) | 0-1 (归一化) |
| **标准化** | ImageNet (mean/std) | ImageNet (mean/std) |
| **通道顺序** | RGB, CHW | RGB, CHW |
| **预处理位置** | Python 外部 | Kotlin 外部 |

**预处理代码对比**:

```python
# 原始版本 (Python)
img = img / 255.0
img = (img - mean) / std  # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

```kotlin
// Android 版本 (Kotlin)
val r = ((pixel shr 16) and 0xFF) / 255f
normalized[c][y][x] = (r - MEAN[c]) / STD[c]
```

### 2.4 输出处理

| 方面 | 原始版本 | Android 版本 |
|------|----------|--------------|
| **输出数量** | 3 个候选 mask | 1 个最佳 mask |
| **输出类型** | logits + scores | 仅 logits |
| **后处理** | 选择最高分 mask | 直接使用 |
| **阈值** | `mask > 0` | `mask > 0` |

---

## 3. 模型内部结构对比

### 3.1 原始版本内部流程

```
输入:
  - image: [H, W, 3]
  - point_coords: [N, 2] (可选)
  - point_labels: [N] (可选)
  - box: [4] (可选)
  - mask_input: [1, 1, 256, 256] (可选)

内部流程:
  1. set_image():
     - Resize 到目标尺寸
     - ImageNet 标准化
     - image_encoder(img) → image_embedding
     - 缓存 image_embedding

  2. predict():
     - prompt_encoder(points, boxes, masks) → sparse_embeddings, dense_embeddings
     - mask_decoder(image_embedding, sparse_embeddings, dense_embeddings)
       → masks [3, H, W], iou_predictions [3]
     - 选择最高分 mask

输出:
  - masks: [3, H, W] 或 [1, H, W]
  - scores: [3] 或 [1]
  - logits: [3, 256, 256] 或 [1, 256, 256]
```

### 3.2 Android 版本内部流程

```
输入:
  - img: [1, 3, 512, 512]      # RGB 图像，ImageNet 标准化后
  - point_coords: [1, N, 2]    # 点坐标，512 空间
  - point_labels: [1, N]       # 点标签

内部流程 (TAMWrapperSimple.forward):
  1. 图像编码:
     backbone_out = image_encoder(img)
     features = backbone_out['backbone_fpn'][-1]

  2. 构造提示:
     point_inputs = {
         'point_coords': point_coords,
         'point_labels': point_labels
     }

  3. SAM 解码:
     sam_outputs = _forward_sam_heads(
         backbone_features=features,
         point_inputs=point_inputs,
         mask_inputs=None,
         high_res_features=None,
         multimask_output=False
     )
     high_res_masks = sam_outputs[4]  # [1, 1, 512, 512]

输出:
  - mask: [1, 1, 512, 512]     # 分割 mask logits
```

---

## 4. 简化与优化

### 4.1 已简化的功能

| 功能 | 原始版本 | Android 版本 | 原因 |
|------|----------|--------------|------|
| **特征缓存** | 支持 | 不支持 | 简化状态管理 |
| **多 mask 输出** | 3 个候选 | 1 个最佳 | 减少计算和传输 |
| **mask 提示** | 支持 | 不支持 | 简化接口 |
| **框参数** | 单独参数 | 转换为点 | 统一接口 |
| **迭代优化** | 支持 | 不支持 | 简化流程 |
| **多对象** | 支持 | 单对象 | 简化实现 |

### 4.2 保留的核心功能

| 功能 | 说明 |
|------|------|
| **Image Encoder** | 完整的 EfficientViT 骨干网络 |
| **Prompt Encoder** | 完整的点/框编码 |
| **Mask Decoder** | 完整的 mask 解码器 |
| **点提示** | 前景点(1) + 背景点(0) |
| **框提示** | 左上角(2) + 右下角(3) |

### 4.3 Android 特定优化

| 优化 | 说明 |
|------|------|
| **TorchScript** | 使用 `torch.jit.trace` 导出 |
| **单 mask 输出** | `multimask_output=False` |
| **统一接口** | 框转换为点，只需 3 个参数 |
| **协程推理** | 使用 Kotlin Coroutines 后台执行 |

---

## 5. 框提示的特殊处理

### 5.1 原始版本

```python
# 框作为单独参数传入
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=np.array([x1, y1, x2, y2]),  # [4]
    multimask_output=True
)
```

### 5.2 Android 版本

```kotlin
// 框转换为两个点
// 左上角: (x1, y1) 标签 2
// 右下角: (x2, y2) 标签 3

val pointCoords = floatArrayOf(x1, y1, x2, y2)  // shape: [1, 2, 2]
val pointLabels = longArrayOf(2L, 3L)           // shape: [1, 2]

val mask = model.forward(img, pointCoords, pointLabels)
```

### 5.3 为什么这样设计？

1. **统一接口**: 点和框使用相同的输入格式，简化模型导出
2. **SAM 标准**: 标签 2 和 3 是 SAM 的标准约定
3. **减少分支**: 模型内部不需要判断是点还是框
4. **TorchScript 兼容**: 避免 Python 动态逻辑

---

## 6. 精度与性能对比

### 6.1 精度影响

| 因素 | 影响 | 说明 |
|------|------|------|
| **单 mask 输出** | 轻微下降 | 某些情况下非最优 mask |
| **无迭代优化** | 轻微下降 | 边缘可能不够精细 |
| **预处理一致** | 无影响 | 完全相同的标准化 |

### 6.2 性能数据（参考）

| 设备 | 推理延迟 | 说明 |
|------|----------|------|
| Snapdragon 8 Gen 2 | ~150ms | 高端旗舰 |
| Snapdragon 888 | ~250ms | 中高端 |
| Snapdragon 765G | ~400ms | 中端 |

*注：图片分割比视频跟踪快，因为不需要 Memory Attention*

---

## 7. 数据流对比图

### 7.1 原始版本数据流

```
图片 ──────────────────────────────────────────────────────────────────┐
     │                                                                  │
     ▼                                                                  │
┌─────────────────────────────────────────────────────────────────────┐ │
│                         set_image()                                 │ │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │ │
│  │ Resize      │ → │ Normalize   │ → │ Encoder     │ → embedding   │ │
│  └─────────────┘   └─────────────┘   └─────────────┘      ↓        │ │
│                                                       [缓存]        │ │
└─────────────────────────────────────────────────────────────────────┘ │
                                                            ↓           │
┌─────────────────────────────────────────────────────────────────────┐ │
│                          predict()                                  │ │
│  点/框/mask                                                         │ │
│      ↓                                                              │ │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │ │
│  │ Prompt Enc  │ → │ Mask Dec    │ → │ 选择最优    │ → mask        │ │
│  └─────────────┘   └─────────────┘   └─────────────┘               │ │
│                                              ↓                      │ │
│                                    masks[3], scores[3]              │ │
└─────────────────────────────────────────────────────────────────────┘ │
```

### 7.2 Android 版本数据流

```
图片 + 点/框 ──────────────────────────────────────────────────────────┐
     │                                                                  │
     ▼                                                                  │
┌─────────────────────────────────────────────────────────────────────┐ │
│              predictWithPoints() / predictWithBox()                 │ │
│                                                                     │ │
│  ┌─────────────┐   ┌─────────────┐                                 │ │
│  │ Bitmap →    │   │ 坐标转换    │                                 │ │
│  │ Tensor      │   │ → 512 空间  │                                 │ │
│  └─────────────┘   └─────────────┘                                 │ │
│         ↓                 ↓                                         │ │
│  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │                    model.forward()                           │   │ │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │   │ │
│  │  │ Encoder │ → │ Prompt  │ → │ Decoder │ → │ Mask    │      │   │ │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘      │   │ │
│  └─────────────────────────────────────────────────────────────┘   │ │
│                                      ↓                              │ │
│                              mask [1, 1, 512, 512]                  │ │
└─────────────────────────────────────────────────────────────────────┘ │
```

---

## 8. 与视频分割版本的对比

| 方面 | 图片分割 | 视频分割 |
|------|----------|----------|
| **模型数量** | 1 个 | 2 个（首帧 + 跟踪） |
| **Memory** | 无 | 需要维护 memory 状态 |
| **推理速度** | 更快 | 较慢（Memory Attention） |
| **状态管理** | 无状态 | 有状态（memory） |
| **适用场景** | 单张图片 | 视频序列 |

---

## 9. 总结

### 9.1 主要差异

1. **端到端模型**: 原始分离的编码器/解码器 → Android 单一模型
2. **统一接口**: 框转换为点，只需 3 个参数
3. **单 mask 输出**: 3 个候选 → 1 个最佳
4. **无特征缓存**: 每次推理重新编码图像

### 9.2 适用场景

| 场景 | 推荐版本 |
|------|----------|
| 高精度分割 | 原始 Python 版本 |
| 多对象分割 | 原始 Python 版本 |
| 移动端实时分割 | Android 版本 |
| 简单交互式分割 | Android 版本 |

### 9.3 未来优化方向

- [ ] 支持多对象分割
- [ ] 添加特征缓存（同一图片多次分割）
- [ ] 支持 mask 提示（迭代优化）
- [ ] NNAPI/GPU 加速
- [ ] 模型量化 (INT8)
