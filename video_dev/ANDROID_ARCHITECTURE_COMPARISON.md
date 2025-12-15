# EfficientTAM Android 部署架构对比

本文档详细说明 Android 部署版本与原始 EfficientTAM 的架构差异。

---

## 1. 整体架构对比

### 1.1 原始 EfficientTAM（Python）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EfficientTAMVideoPredictor                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         init_state()                             │    │
│  │  - 加载所有视频帧到内存                                          │    │
│  │  - 预处理：Resize + ImageNet 标准化                              │    │
│  │  - 初始化 inference_state 字典                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    add_new_points_or_box()                       │    │
│  │  - 坐标归一化 + 缩放到 512                                       │    │
│  │  - _get_image_feature() 提取图像特征                             │    │
│  │  - _prepare_memory_conditioned_features()                        │    │
│  │      └─ 首帧：features + no_mem_embed                            │    │
│  │  - _forward_sam_heads() 生成 mask                                │    │
│  │  - 存储到 temp_output_dict                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     propagate_in_video()                         │    │
│  │  - propagate_in_video_preflight()                                │    │
│  │      └─ 运行 memory_encoder 编码首帧 memory                      │    │
│  │  - 遍历每一帧：                                                  │    │
│  │      - _get_image_feature()                                      │    │
│  │      - _prepare_memory_conditioned_features()                    │    │
│  │          └─ 后续帧：memory_attention(features, memory+obj_ptr)   │    │
│  │      - _forward_sam_heads()                                      │    │
│  │      - _encode_new_memory()                                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Android 部署版本

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EfficientTAMSegmenter                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       initialize()                               │    │
│  │  - 加载 firstEncoder.ptl                                         │    │
│  │  - 加载 tracker.ptl                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    processFirstFrame()                           │    │
│  │  - bitmapToTensor(): Resize + 0-255 值范围                       │    │
│  │  - 坐标转换到 512 空间                                           │    │
│  │  - firstEncoder.forward(img, points, labels)                     │    │
│  │  - 保存 memory 状态                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      processFrame()                              │    │
│  │  - bitmapToTensor()                                              │    │
│  │  - tracker.forward(img, maskmem_features, maskmem_pos_enc, obj_ptr) │
│  │  - 更新 memory 状态                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 关键差异对比

### 2.1 模型拆分

| 方面 | 原始版本 | Android 版本 |
|------|----------|--------------|
| **模型数量** | 1 个完整模型 | 2 个独立模型 |
| **模型文件** | `efficienttam_ti.pt` (checkpoint) | `efficienttam_video_first_v4.ptl` + `efficienttam_video_track_v4.ptl` |
| **模型格式** | PyTorch state_dict | TorchScript Lite (.ptl) |
| **模型大小** | ~24 MB | ~46 MB + ~70 MB |

**为什么拆分？**
- 首帧需要处理用户点击的点坐标，后续帧不需要
- 首帧不需要 Memory Attention，后续帧需要
- 拆分后可以针对不同场景优化

### 2.2 图像预处理

| 方面 | 原始版本 | Android 版本 |
|------|----------|--------------|
| **输入尺寸** | 512×512 | 512×512 |
| **值范围** | 0-1 (归一化) | 0-255 |
| **标准化** | ImageNet (mean/std) | 无（模型内部处理） |
| **通道顺序** | RGB, CHW | RGB, CHW |

**原始版本预处理**:
```python
img = img / 255.0
img = (img - mean) / std  # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

**Android 版本预处理**:
```java
// 直接使用 0-255 值，模型内部包含标准化层
floatData[c * 512 * 512 + y * 512 + x] = pixelValue;  // 0-255
```

### 2.3 Memory 管理

| 方面 | 原始版本 | Android 版本 |
|------|----------|--------------|
| **Memory 帧数** | 最多 7 帧 (num_maskmem=7) | 仅 1 帧（最近帧） |
| **条件帧** | 单独存储，始终保留 | 合并到单帧 memory |
| **时序编码** | maskmem_tpos_enc[7] | 简化为固定编码 |
| **Object Pointer** | 多帧累积 | 仅最近帧 |

**原始版本 Memory 结构**:
```python
memory = concat([
    cond_frame_memory,      # 条件帧（首帧）
    recent_frame_memories,  # 最近 6 帧
    obj_ptrs                # 所有帧的 object pointers
])
```

**Android 版本 Memory 结构**:
```java
// 仅保存最近一帧的 memory
Tensor maskmemFeatures;   // [1, 64, 32, 32]
Tensor maskmemPosEnc;     // [1, 64, 32, 32]
Tensor objPtr;            // [1, 256]
```

### 2.4 推理流程

| 步骤 | 原始版本 | Android 版本 |
|------|----------|--------------|
| **首帧** | `add_new_points_or_box()` → 临时存储 | `processFirstFrame()` → 直接输出 |
| **Memory 编码** | `propagate_in_video_preflight()` 延迟编码 | 首帧模型内部直接编码 |
| **后续帧** | `propagate_in_video()` 批量处理 | `processFrame()` 逐帧处理 |
| **Memory 更新** | 累积多帧 memory | 替换为最新帧 memory |

---

## 3. 模型内部结构对比

### 3.1 首帧编码器 (FirstFrameEncoderV4)

```
输入:
  - img: [1, 3, 512, 512]      # RGB 图像，0-255
  - point_coords: [1, N, 2]    # 点坐标，512 空间
  - point_labels: [1, N]       # 点标签，1=前景

内部流程:
  1. 图像标准化: (img / 255.0 - mean) / std
  2. 图像编码: image_encoder(img) → backbone_out
  3. 特征准备: backbone_out["backbone_fpn"][-1] → vision_feats
  4. 添加 no_mem_embed: pix_feat = vision_feats + no_mem_embed
  5. SAM 解码: sam_prompt_encoder + sam_mask_decoder → mask, obj_ptr
  6. Memory 编码: memory_encoder(原始 vision_feats, mask) → maskmem

输出:
  - mask: [1, 1, 512, 512]           # 分割 mask logits
  - maskmem_features: [1, 64, 32, 32] # Memory 特征
  - maskmem_pos_enc: [1, 64, 32, 32]  # Memory 位置编码
  - obj_ptr: [1, 256]                 # Object Pointer
```

### 3.2 跟踪器 (FrameTrackerV4)

```
输入:
  - img: [1, 3, 512, 512]             # RGB 图像，0-255
  - maskmem_features: [1, 64, 32, 32] # 上一帧 Memory 特征
  - maskmem_pos_enc: [1, 64, 32, 32]  # Memory 位置编码
  - obj_ptr: [1, 256]                 # 上一帧 Object Pointer

内部流程:
  1. 图像标准化 + 编码
  2. Memory 准备:
     - maskmem_features → [HW, B, 64]
     - obj_ptr 拆分为 4 个 tokens → [4, B, 64]
     - 添加时序位置编码
  3. Memory Attention: memory_attention(vision_feats, memory+obj_ptr)
  4. SAM 解码: sam_mask_decoder(pix_feat_with_mem) → mask, new_obj_ptr
  5. Memory 编码: memory_encoder(原始 vision_feats, mask) → new_maskmem

输出:
  - mask: [1, 1, 512, 512]               # 分割 mask logits
  - new_maskmem_features: [1, 64, 32, 32] # 新 Memory 特征
  - new_maskmem_pos_enc: [1, 64, 32, 32]  # 新 Memory 位置编码
  - new_obj_ptr: [1, 256]                 # 新 Object Pointer
```

---

## 4. 简化与优化

### 4.1 已简化的功能

| 功能 | 原始版本 | Android 版本 | 原因 |
|------|----------|--------------|------|
| **多对象跟踪** | 支持多个 obj_id | 仅单对象 | 简化 memory 管理 |
| **多帧 Memory** | 7 帧累积 | 1 帧滑动窗口 | 减少内存占用 |
| **双向传播** | 支持前向/后向 | 仅前向 | 实时场景不需要 |
| **多 mask 输出** | 3 个候选 mask | 1 个最佳 mask | 减少计算量 |
| **高分辨率特征** | 可选 FPN 多层 | 仅最后一层 | EfficientTAM-Ti 只有 1 层 |
| **框提示** | 支持 box prompt | 仅点提示 | 简化交互 |

### 4.2 保留的核心功能

| 功能 | 说明 |
|------|------|
| **Image Encoder** | 完整的 EfficientViT 骨干网络 |
| **Memory Attention** | 完整的 cross-attention 机制 |
| **SAM Decoder** | 完整的 mask 解码器 |
| **Memory Encoder** | 完整的 memory 编码器 |
| **Object Pointer** | 完整的对象表示 |

### 4.3 Android 特定优化

| 优化 | 说明 |
|------|------|
| **TorchScript Lite** | 使用 `_save_for_lite_interpreter` 减小模型体积 |
| **算子替换** | 手动实现 `scaled_dot_product_attention` 兼容旧版 PyTorch |
| **内存复用** | Tensor 对象复用，避免频繁 GC |
| **异步推理** | 使用 ExecutorService 后台线程 |

---

## 5. 精度与性能对比

### 5.1 精度影响

| 因素 | 影响 | 说明 |
|------|------|------|
| **单帧 Memory** | 轻微下降 | 长视频可能累积误差 |
| **预处理差异** | 无影响 | 模型内部标准化 |
| **单对象限制** | 无影响 | 单对象场景完全一致 |

### 5.2 性能数据（参考）

| 设备 | 首帧延迟 | 跟踪帧延迟 | FPS |
|------|----------|------------|-----|
| Snapdragon 8 Gen 2 | ~500ms | ~200ms | ~5 |
| Snapdragon 888 | ~800ms | ~350ms | ~3 |
| Snapdragon 765G | ~1500ms | ~600ms | ~1.5 |

*注：实际性能取决于设备、图像复杂度等因素*

---

## 6. 数据流对比图

### 6.1 原始版本数据流

```
视频帧列表 ──────────────────────────────────────────────────────────────┐
     │                                                                    │
     ▼                                                                    │
┌─────────────────┐                                                       │
│  init_state()   │ ← 预加载所有帧                                        │
└────────┬────────┘                                                       │
         │                                                                │
         ▼                                                                │
┌─────────────────────────────────────────────────────────────────────┐   │
│                    add_new_points_or_box()                          │   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │   │
│  │ get_feature │ → │ + no_mem   │ → │ SAM decode  │ → mask₀       │   │
│  └─────────────┘   └─────────────┘   └─────────────┘               │   │
│                                              ↓                      │   │
│                                    temp_output_dict                 │   │
└─────────────────────────────────────────────────────────────────────┘   │
                                               ↓                          │
┌─────────────────────────────────────────────────────────────────────┐   │
│                     propagate_in_video()                            │   │
│  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │ preflight: memory_encoder(首帧) → cond_frame_memory         │   │   │
│  └─────────────────────────────────────────────────────────────┘   │   │
│                              ↓                                      │   │
│  for frame_idx in range(1, N):                                     │   │
│    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │   │
│    │ get_feature │ → │ mem_attn   │ → │ SAM decode  │ → maskᵢ    │   │
│    └─────────────┘   └─────────────┘   └─────────────┘            │   │
│                              ↓                                      │   │
│                    memory_encoder → 累积到 memory bank             │   │
└─────────────────────────────────────────────────────────────────────┘   │
```

### 6.2 Android 版本数据流

```
实时视频帧 ──────────────────────────────────────────────────────────────┐
     │                                                                    │
     ▼                                                                    │
┌─────────────────────────────────────────────────────────────────────┐   │
│                     processFirstFrame()                             │   │
│  ┌─────────────┐   ┌─────────────────────────────────────────────┐ │   │
│  │ Bitmap →    │   │           firstEncoder.forward()            │ │   │
│  │ Tensor      │ → │  img_enc → +no_mem → SAM → mem_enc         │ │   │
│  └─────────────┘   └─────────────────────────────────────────────┘ │   │
│                                      ↓                              │   │
│                    mask₀, maskmem, pos_enc, obj_ptr                │   │
│                              ↓ (保存状态)                           │   │
└─────────────────────────────────────────────────────────────────────┘   │
                                                                          │
┌─────────────────────────────────────────────────────────────────────┐   │
│                       processFrame() [循环]                         │   │
│  ┌─────────────┐   ┌─────────────────────────────────────────────┐ │   │
│  │ Bitmap →    │   │            tracker.forward()                │ │   │
│  │ Tensor      │ → │  img_enc → mem_attn → SAM → mem_enc        │ │   │
│  └─────────────┘   └─────────────────────────────────────────────┘ │   │
│         ↑                            ↓                              │   │
│         │          maskᵢ, new_maskmem, new_pos_enc, new_obj_ptr    │   │
│         │                    ↓ (更新状态)                           │   │
│         └────────────────────┘                                      │   │
└─────────────────────────────────────────────────────────────────────┘   │
```

---

## 7. 总结

### 7.1 主要差异

1. **模型拆分**: 原始单模型 → Android 双模型（首帧 + 跟踪）
2. **Memory 简化**: 7 帧累积 → 1 帧滑动窗口
3. **预处理内置**: 标准化移入模型内部
4. **实时优化**: 逐帧处理替代批量处理

### 7.2 适用场景

| 场景 | 推荐版本 |
|------|----------|
| 离线视频分析 | 原始 Python 版本 |
| 长视频高精度 | 原始 Python 版本 |
| 实时移动端 | Android 版本 |
| 短视频快速分割 | Android 版本 |

### 7.3 未来优化方向

- [ ] 支持多对象跟踪
- [ ] 增加 Memory 帧数（2-3 帧）
- [ ] NNAPI/GPU 加速
- [ ] 模型量化 (INT8)
- [ ] 动态分辨率支持
