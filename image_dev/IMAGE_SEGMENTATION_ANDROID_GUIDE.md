# EfficientTAM 图片分割 Android 开发指南

本文档详细介绍如何在 Android 应用中集成 EfficientTAM 图片分割功能。支持点选（前景/背景）和框选两种交互方式。

---

## 目录

1. [项目概述](#1-项目概述)
2. [环境配置](#2-环境配置)
3. [项目结构](#3-项目结构)
4. [模型文件说明](#4-模型文件说明)
5. [核心代码实现](#5-核心代码实现)
6. [完整示例代码](#6-完整示例代码)
7. [性能优化建议](#7-性能优化建议)
8. [常见问题](#8-常见问题)

---

## 1. 项目概述

### 1.1 功能描述

- 加载静态图片进行分割
- **点选模式**: 用户点击选择前景点（绿色）或背景点（红色）
- **框选模式**: 用户拖拽绘制矩形框选择目标区域
- 实时推理：每次交互立即触发分割
- 可视化结果：半透明绿色掩码叠加显示

### 1.2 模型架构

EfficientTAM 图片分割使用单一模型，支持点和框两种提示方式：

| 模型 | 文件 | 用途 | 输入 | 输出 |
|---|---|---|---|---|
| 图片分割器 | `efficienttam_ti_512x512_point_and_box.pt` | 点选/框选分割 | 图像 + 点坐标 + 标签 | mask |

### 1.3 提示类型

| 标签值 | 含义 | 用途 |
|--------|------|------|
| `1` | 前景点 | 用户想要分割的区域 |
| `0` | 背景点 | 用户不想要的区域 |
| `2` | 框左上角 | 框选的左上角点 |
| `3` | 框右下角 | 框选的右下角点 |

### 1.4 处理流程

```
用户点击/框选 → 坐标转换 → 模型推理 → mask 可视化
                    ↓
              点: [x, y] + 标签 1/0
              框: [x1, y1], [x2, y2] + 标签 2, 3
```

---

## 2. 环境配置

### 2.1 Gradle 依赖

在 `app/build.gradle.kts` 中添加：

```kotlin
android {
    defaultConfig {
        minSdk = 24
        
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }
    
    // 防止压缩 .pt 文件
    androidResources {
        noCompress += "pt"
    }
}

dependencies {
    // PyTorch Android Lite 2.1.0（支持 scaled_dot_product_attention）
    implementation("org.pytorch:pytorch_android_lite:2.1.0")
    implementation("org.pytorch:pytorch_android_torchvision_lite:2.1.0")
    
    // 协程支持
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
    
    // 其他
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
}
```

### 2.2 权限配置

在 `AndroidManifest.xml` 中添加：

```xml
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

### 2.3 Assets 文件

将以下文件放入 `app/src/main/assets/` 目录：

```
assets/
├── efficienttam_ti_512x512_point_and_box.pt    (~24 MB)
└── sample_image.jpg                             (测试图片)
```

---

## 3. 项目结构

```
app/src/main/
├── java/com/example/myapplication/
│   ├── MainActivity.kt                 # 主界面
│   ├── EfficientTAMPointPrompt.kt      # 模型封装类
│   ├── BoxSelectionView.kt             # 框选自定义 View
│   ├── ImageUtils.kt                   # 图像预处理
│   ├── TensorImageUtils.kt             # Tensor 转换
│   ├── CoordinateUtils.kt              # 坐标转换
│   └── MaskUtils.kt                    # Mask 可视化
├── res/
│   ├── layout/
│   │   └── activity_main.xml           # 主界面布局
│   └── values/
│       └── strings.xml
└── assets/
    ├── efficienttam_ti_512x512_point_and_box.pt
    └── sample_image.jpg
```

---

## 4. 模型文件说明

### 4.1 模型输入输出

**输入参数：**

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `img` | `[1, 3, 512, 512]` | Float | RGB图像，ImageNet 标准化后 |
| `point_coords` | `[1, N, 2]` | Float | 点坐标 (x, y)，512空间 |
| `point_labels` | `[1, N]` | Long | 点标签：1=前景, 0=背景, 2=框左上, 3=框右下 |

**输出参数：**

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `mask` | `[1, 1, 512, 512]` | Float | 分割 mask logits (>0 为前景) |

### 4.2 坐标转换

用户点击的屏幕坐标需要经过两级转换：

```kotlin
// 第一级：屏幕坐标 → 图片坐标
val imageCoords = CoordinateUtils.viewToImageCoordinates(
    touchX, touchY, imageView, bitmap
)

// 第二级：图片坐标 → 模型坐标 (512x512)
val modelX = imageCoords.first * 512f / bitmap.width
val modelY = imageCoords.second * 512f / bitmap.height
```

### 4.3 图像预处理

```kotlin
// 1. Resize 到 512x512
val resized = Bitmap.createScaledBitmap(bitmap, 512, 512, true)

// 2. 转换为 float 数组 [C, H, W]
// 3. ImageNet 标准化
val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
val std = floatArrayOf(0.229f, 0.224f, 0.225f)
normalized[c][y][x] = (pixel[c] / 255f - mean[c]) / std[c]
```

### 4.4 Mask 后处理

```kotlin
// mask > 0 表示前景（输出是 logits）
val isForeground = maskData[y * 512 + x] > 0f
```

---

## 5. 核心代码实现

### 5.1 图像预处理工具类

```kotlin
// TensorImageUtils.kt
package com.example.myapplication

import android.graphics.Bitmap
import org.pytorch.Tensor

object TensorImageUtils {
    private const val MODEL_SIZE = 512
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun bitmapToFloat32Tensor(bitmap: Bitmap): Tensor {
        val resized = Bitmap.createScaledBitmap(bitmap, MODEL_SIZE, MODEL_SIZE, true)
        val pixels = IntArray(MODEL_SIZE * MODEL_SIZE)
        resized.getPixels(pixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)

        val floatData = FloatArray(3 * MODEL_SIZE * MODEL_SIZE)

        for (y in 0 until MODEL_SIZE) {
            for (x in 0 until MODEL_SIZE) {
                val pixel = pixels[y * MODEL_SIZE + x]

                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                // ImageNet 标准化
                floatData[0 * MODEL_SIZE * MODEL_SIZE + y * MODEL_SIZE + x] = (r - MEAN[0]) / STD[0]
                floatData[1 * MODEL_SIZE * MODEL_SIZE + y * MODEL_SIZE + x] = (g - MEAN[1]) / STD[1]
                floatData[2 * MODEL_SIZE * MODEL_SIZE + y * MODEL_SIZE + x] = (b - MEAN[2]) / STD[2]
            }
        }

        if (resized != bitmap) {
            resized.recycle()
        }

        return Tensor.fromBlob(floatData, longArrayOf(1, 3, MODEL_SIZE.toLong(), MODEL_SIZE.toLong()))
    }
}
```

### 5.2 坐标转换工具类

```kotlin
// CoordinateUtils.kt
package com.example.myapplication

import android.graphics.Bitmap
import android.widget.ImageView

object CoordinateUtils {
    
    /**
     * 将 ImageView 触摸坐标转换为图片像素坐标
     */
    fun viewToImageCoordinates(
        touchX: Float,
        touchY: Float,
        imageView: ImageView,
        bitmap: Bitmap
    ): Pair<Float, Float>? {
        val imageMatrix = imageView.imageMatrix
        val values = FloatArray(9)
        imageMatrix.getValues(values)

        val scaleX = values[0]
        val scaleY = values[4]
        val transX = values[2]
        val transY = values[5]

        val imageX = (touchX - transX) / scaleX
        val imageY = (touchY - transY) / scaleY

        // 检查是否在图片范围内
        if (imageX < 0 || imageX >= bitmap.width || imageY < 0 || imageY >= bitmap.height) {
            return null
        }

        return Pair(imageX, imageY)
    }

    /**
     * 将图片坐标转换为模型坐标 (512x512)
     */
    fun imageToModelCoordinates(
        imageX: Float,
        imageY: Float,
        imageWidth: Int,
        imageHeight: Int
    ): Pair<Float, Float> {
        val modelX = imageX * 512f / imageWidth
        val modelY = imageY * 512f / imageHeight
        return Pair(modelX, modelY)
    }
}
```

### 5.3 EfficientTAM 分割器封装

```kotlin
// EfficientTAMPointPrompt.kt
package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class EfficientTAMPointPrompt(private val context: Context) {
    private val module: Module
    private val modelInputSize = 512f

    init {
        val modelPath = assetFilePath(context, "efficienttam_ti_512x512_point_and_box.pt")
        module = Module.load(modelPath)
    }

    /**
     * 使用点提示进行分割
     * 
     * @param bitmap 原始图片
     * @param points 点坐标列表（图片坐标系）
     * @param labels 点标签列表（1=前景, 0=背景）
     * @return mask 数据 (512*512)
     */
    fun predictWithPoints(
        bitmap: Bitmap,
        points: List<Pair<Float, Float>>,
        labels: List<Int>
    ): FloatArray? {
        if (points.isEmpty() || points.size != labels.size) {
            return null
        }

        try {
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap)

            val scaleX = modelInputSize / bitmap.width
            val scaleY = modelInputSize / bitmap.height

            // 转换坐标到模型空间
            val numPoints = points.size
            val pointCoordsArray = FloatArray(numPoints * 2)
            points.forEachIndexed { index, (x, y) ->
                pointCoordsArray[index * 2] = x * scaleX
                pointCoordsArray[index * 2 + 1] = y * scaleY
            }
            val pointCoordsTensor = Tensor.fromBlob(
                pointCoordsArray,
                longArrayOf(1, numPoints.toLong(), 2)
            )

            // 标签使用 long 类型
            val pointLabelsArray = labels.map { it.toLong() }.toLongArray()
            val pointLabelsTensor = Tensor.fromBlob(
                pointLabelsArray,
                longArrayOf(1, numPoints.toLong())
            )

            // 模型推理
            val outputTensor = module.forward(
                IValue.from(inputTensor),
                IValue.from(pointCoordsTensor),
                IValue.from(pointLabelsTensor)
            ).toTensor()

            return outputTensor.dataAsFloatArray

        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    /**
     * 使用框提示进行分割
     * 框会被转换为两个点（左上角标签2，右下角标签3）
     * 
     * @param bitmap 原始图片
     * @param box 框坐标（图片坐标系）
     * @return mask 数据 (512*512)
     */
    fun predictWithBox(bitmap: Bitmap, box: RectF): FloatArray? {
        try {
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap)

            val scaleX = modelInputSize / bitmap.width
            val scaleY = modelInputSize / bitmap.height

            // 转换框坐标到模型空间
            val left = minOf(box.left, box.right) * scaleX
            val top = minOf(box.top, box.bottom) * scaleY
            val right = maxOf(box.left, box.right) * scaleX
            val bottom = maxOf(box.top, box.bottom) * scaleY

            // Clamp 到有效范围
            val clampedLeft = left.coerceIn(0f, modelInputSize)
            val clampedTop = top.coerceIn(0f, modelInputSize)
            val clampedRight = right.coerceIn(0f, modelInputSize)
            val clampedBottom = bottom.coerceIn(0f, modelInputSize)

            // 框转换为两个点
            val pointsBuffer = floatArrayOf(
                clampedLeft, clampedTop,     // 左上角
                clampedRight, clampedBottom  // 右下角
            )
            val pointCoordsTensor = Tensor.fromBlob(pointsBuffer, longArrayOf(1, 2, 2))

            // 标签: 2=左上角, 3=右下角
            val labelsBuffer = longArrayOf(2L, 3L)
            val pointLabelsTensor = Tensor.fromBlob(labelsBuffer, longArrayOf(1, 2))

            // 模型推理
            val outputTensor = module.forward(
                IValue.from(inputTensor),
                IValue.from(pointCoordsTensor),
                IValue.from(pointLabelsTensor)
            ).toTensor()

            return outputTensor.dataAsFloatArray

        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 1024) return file.absolutePath

        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }

        return file.absolutePath
    }

    fun release() {
        module.destroy()
    }
}
```

### 5.4 Mask 可视化工具类

```kotlin
// MaskUtils.kt
package com.example.myapplication

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

object MaskUtils {
    
    /**
     * 将 mask logits 转换为 Bitmap
     */
    fun maskToBitmap(maskData: FloatArray, maskColor: Int = 0x8000FF00): Bitmap {
        val size = 512
        val pixels = IntArray(size * size)

        for (i in maskData.indices) {
            pixels[i] = if (maskData[i] > 0f) maskColor else Color.TRANSPARENT
        }

        return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
    }

    /**
     * 将 mask 叠加到原始图片上
     */
    fun overlayMaskOnBitmap(
        originalBitmap: Bitmap,
        maskBitmap: Bitmap
    ): Bitmap {
        val result = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)

        // 缩放 mask 到原始图片尺寸
        val scaledMask = Bitmap.createScaledBitmap(
            maskBitmap,
            originalBitmap.width,
            originalBitmap.height,
            true
        )

        canvas.drawBitmap(scaledMask, 0f, 0f, null)
        scaledMask.recycle()

        return result
    }

    /**
     * 在图片上绘制提示点
     */
    fun drawPointsOnBitmap(
        bitmap: Bitmap,
        points: List<Pair<Float, Float>>,
        labels: List<Int>
    ): Bitmap {
        val result = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)

        val paintPositive = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        val paintNegative = Paint().apply {
            color = Color.RED
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        val radius = 15f

        points.forEachIndexed { index, (x, y) ->
            val paint = if (labels[index] == 1) paintPositive else paintNegative
            canvas.drawCircle(x, y, radius, paint)
        }

        return result
    }
}
```

---

## 6. 完整示例代码

### 6.1 布局文件

```xml
<!-- res/layout/activity_main.xml -->
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <!-- 图片显示 -->
    <ImageView
        android:id="@+id/imageView"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:scaleType="fitCenter"
        android:adjustViewBounds="true"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintBottom_toTopOf="@id/controlsLayout" />

    <!-- 控制面板 -->
    <LinearLayout
        android:id="@+id/controlsLayout"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical"
        android:padding="16dp"
        app:layout_constraintBottom_toBottomOf="parent">

        <Button
            android:id="@+id/btnToggleMode"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="切换模式: 前景点"
            android:layout_marginBottom="8dp" />

        <Button
            android:id="@+id/btnClearPoints"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="清除所有点" />

    </LinearLayout>

</androidx.constraintlayout.widget.ConstraintLayout>
```

### 6.2 MainActivity

```kotlin
// MainActivity.kt
package com.example.myapplication

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var btnToggleMode: Button
    private lateinit var btnClearPoints: Button

    private lateinit var efficientTAM: EfficientTAMPointPrompt
    private var originalBitmap: Bitmap? = null

    // 状态
    private val clickedPoints = mutableListOf<Pair<Float, Float>>()
    private val pointLabels = mutableListOf<Int>()
    private var isPositiveMode = true  // true=前景点, false=背景点

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        initModel()
        loadImage()
        setupListeners()
    }

    private fun initViews() {
        imageView = findViewById(R.id.imageView)
        btnToggleMode = findViewById(R.id.btnToggleMode)
        btnClearPoints = findViewById(R.id.btnClearPoints)
    }

    private fun initModel() {
        efficientTAM = EfficientTAMPointPrompt(this)
    }

    private fun loadImage() {
        // 从 assets 加载图片
        assets.open("sample_image.jpg").use { input ->
            originalBitmap = BitmapFactory.decodeStream(input)
            imageView.setImageBitmap(originalBitmap)
        }
    }

    private fun setupListeners() {
        // 模式切换
        btnToggleMode.setOnClickListener {
            isPositiveMode = !isPositiveMode
            btnToggleMode.text = if (isPositiveMode) "切换模式: 前景点" else "切换模式: 背景点"
        }

        // 清除所有点
        btnClearPoints.setOnClickListener {
            clickedPoints.clear()
            pointLabels.clear()
            imageView.setImageBitmap(originalBitmap)
        }

        // 触摸事件
        imageView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN && originalBitmap != null) {
                val imageCoords = CoordinateUtils.viewToImageCoordinates(
                    event.x, event.y, imageView, originalBitmap!!
                )

                if (imageCoords != null) {
                    clickedPoints.add(imageCoords)
                    pointLabels.add(if (isPositiveMode) 1 else 0)
                    performSegmentation()
                }
            }
            true
        }
    }

    private fun performSegmentation() {
        lifecycleScope.launch {
            try {
                val mask = withContext(Dispatchers.IO) {
                    efficientTAM.predictWithPoints(
                        originalBitmap!!,
                        clickedPoints,
                        pointLabels
                    )
                }

                if (mask != null) {
                    updateDisplayWithMask(mask)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun updateDisplayWithMask(maskData: FloatArray) {
        // 1. 生成 mask bitmap
        val maskBitmap = MaskUtils.maskToBitmap(maskData)

        // 2. 叠加到原图
        var resultBitmap = MaskUtils.overlayMaskOnBitmap(originalBitmap!!, maskBitmap)

        // 3. 绘制提示点
        resultBitmap = MaskUtils.drawPointsOnBitmap(resultBitmap, clickedPoints, pointLabels)

        // 4. 更新显示
        imageView.setImageBitmap(resultBitmap)

        maskBitmap.recycle()
    }

    override fun onDestroy() {
        super.onDestroy()
        efficientTAM.release()
    }
}
```

---

## 7. 性能优化建议

### 7.1 模型优化

| 优化方式 | 说明 |
|----------|------|
| 使用 Lite 版本 | `pytorch_android_lite` 体积更小 |
| 模型量化 | INT8 量化可减少 50% 体积 |
| 单 mask 输出 | `multimask_output=False` 减少计算 |

### 7.2 内存优化

```kotlin
// 及时回收 Bitmap
if (tempBitmap != originalBitmap) {
    tempBitmap.recycle()
}

// 复用 Tensor 对象
// 避免频繁创建大数组
```

### 7.3 UI 优化

```kotlin
// 使用协程避免 UI 卡顿
lifecycleScope.launch {
    val result = withContext(Dispatchers.IO) {
        // 耗时操作
    }
    // 更新 UI
}
```

---

## 8. 常见问题

### 8.1 模型加载失败

**问题**: `UnsatisfiedLinkError` 或 `bytecode.pkl not found`

**解决**: 
- 确保使用 `pytorch_android_lite:2.1.0`
- 使用 `Module.load()` 而非 `LiteModuleLoader.load()`

### 8.2 分割结果全黑

**问题**: mask 全为 0 或负值

**解决**:
- 检查图像预处理是否正确（ImageNet 标准化）
- 检查坐标转换是否正确
- 确认点标签值正确（1=前景, 0=背景）

### 8.3 框选无效果

**问题**: 使用框选时没有分割结果

**解决**:
- 确保框转换为两个点（左上角标签2，右下角标签3）
- 检查框坐标是否正确转换到 512 空间
- 确保框尺寸足够大（>1 像素）

### 8.4 坐标偏移

**问题**: 分割区域与点击位置不匹配

**解决**:
- 检查 `ImageView.scaleType` 是否为 `fitCenter`
- 使用 `imageMatrix` 正确转换坐标
- 确保两级坐标转换都正确执行
