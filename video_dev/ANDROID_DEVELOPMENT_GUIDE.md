# EfficientTAM Android 开发指南

本文档详细介绍如何在 Android 应用中集成 EfficientTAM 视频分割功能。用户点击视频画面选择目标后，模型会自动跟踪并分割该目标。

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

- 加载 assets 中的 MP4 视频文件
- 用户点击视频画面选择要跟踪的目标
- 模型自动分割并跟踪目标物体
- 实时显示分割 mask 叠加效果（自动对齐 fitCenter 视频显示区域）
- 实时 FPS 显示

### 1.2 模型架构

EfficientTAM 使用两个模型：

| 模型 | 文件 | 用途 | 输入 | 输出 |
|---|---|---|---|---|
| 首帧编码器 | `efficienttam_video_first_v4.ptl` | 处理用户点击的首帧 | 图像 + 点坐标 | mask + memory |
| 跟踪器 | `efficienttam_video_track_v4.ptl` | 跟踪后续帧 | 图像 + memory | mask + new_memory |

### 1.3 处理流程

```
用户点击 → 首帧编码器(图像, 点) → mask₀, memory₀
                                      ↓
帧1 → 跟踪器(图像, memory₀) → mask₁, memory₁
                                      ↓
帧2 → 跟踪器(图像, memory₁) → mask₂, memory₂
                                      ↓
                                     ...
```

---

## 2. 环境配置

### 2.1 Gradle 依赖

在 `app/build.gradle` 中添加：

```gradle
android {
    defaultConfig {
        // 最低 API 21
        minSdkVersion 21
        
        // 启用 NDK
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a'
        }
    }
    
    // 防止压缩 .ptl 文件
    aaptOptions {
        noCompress "ptl"
    }
}

dependencies {
    // PyTorch Android Lite
    implementation 'org.pytorch:pytorch_android_lite:1.13.1'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.13.1'
    
    // 视频解码
    implementation 'com.google.android.exoplayer:exoplayer-core:2.18.7'
    
    // 其他
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
}
```

### 2.2 权限配置

在 `AndroidManifest.xml` 中添加：

```xml
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

### 2.3 Assets 文件

将以下文件放入 `app/src/main/assets/` 目录：

```
assets/
├── efficienttam_video_first_v4.ptl    (46.7 MB)
├── efficienttam_video_track_v4.ptl    (70.4 MB)
└── sample_video.mp4                    (测试视频)
```

---

## 3. 项目结构

```
app/src/main/
├── java/com/example/efficienttam/
│   ├── MainActivity.java              # 主界面
│   ├── EfficientTAMSegmenter.java     # 模型封装类
│   ├── VideoFrameExtractor.java       # 视频帧提取
│   └── MaskOverlayView.java           # Mask 叠加显示
├── res/
│   ├── layout/
│   │   └── activity_main.xml          # 主界面布局
│   └── values/
│       └── strings.xml
└── assets/
    ├── efficienttam_video_first_v4.ptl
    ├── efficienttam_video_track_v4.ptl
    └── sample_video.mp4
```

---

## 4. 模型文件说明

### 4.1 首帧编码器 (efficienttam_video_first_v4.ptl)

**输入参数：**

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `img` | `[1, 3, 512, 512]` | Float | RGB图像，值范围 0-255 |
| `point_coords` | `[1, N, 2]` | Float | 点击坐标 (x, y)，512空间 |
| `point_labels` | `[1, N]` | Int | 点标签：1=前景, 0=背景 |

**输出参数：**

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `mask` | `[1, 1, 512, 512]` | Float | 分割 mask logits |
| `maskmem_features` | `[1, 64, 32, 32]` | Float | Memory 特征 |
| `maskmem_pos_enc` | `[1, 64, 32, 32]` | Float | Memory 位置编码 |
| `obj_ptr` | `[1, 256]` | Float | Object Pointer |

### 4.2 跟踪器 (efficienttam_video_track_v4.ptl)

**输入参数：**

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `img` | `[1, 3, 512, 512]` | Float | RGB图像，值范围 0-255 |
| `maskmem_features` | `[1, 64, 32, 32]` | Float | 上一帧的 Memory 特征 |
| `maskmem_pos_enc` | `[1, 64, 32, 32]` | Float | Memory 位置编码 |
| `obj_ptr` | `[1, 256]` | Float | 上一帧的 Object Pointer |

**输出参数：**

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `mask` | `[1, 1, 512, 512]` | Float | 分割 mask logits |
| `new_maskmem_features` | `[1, 64, 32, 32]` | Float | 新的 Memory 特征 |
| `new_maskmem_pos_enc` | `[1, 64, 32, 32]` | Float | 新的 Memory 位置编码 |
| `new_obj_ptr` | `[1, 256]` | Float | 新的 Object Pointer |

### 4.3 坐标转换

用户点击的屏幕坐标需要转换到 512x512 模型空间：

```java
// 原始视频尺寸: videoWidth x videoHeight
// 点击位置: clickX, clickY (屏幕坐标，已转换到视频坐标系)

float modelX = clickX / videoWidth * 512.0f;
float modelY = clickY / videoHeight * 512.0f;
```

### 4.4 Mask 后处理

模型输出的是 logits，需要转换为二值 mask：

```java
// mask > 0 表示前景
boolean[][] binaryMask = new boolean[512][512];
for (int y = 0; y < 512; y++) {
    for (int x = 0; x < 512; x++) {
        binaryMask[y][x] = maskData[y * 512 + x] > 0;
    }
}

// 如果需要 resize 到原始视频尺寸，使用双线性插值
```

---

## 5. 核心代码实现

### 5.1 模型加载工具类

```java
// AssetUtils.java
package com.example.efficienttam;

import android.content.Context;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class AssetUtils {
    
    /**
     * 从 assets 复制文件到内部存储（PyTorch 需要文件路径）
     */
    public static String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        
        try (InputStream is = context.getAssets().open(assetName);
             FileOutputStream os = new FileOutputStream(file)) {
            
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
            
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        
        return file.getAbsolutePath();
    }
}
```

### 5.2 EfficientTAM 分割器封装

```java
// EfficientTAMSegmenter.java
package com.example.efficienttam;

import android.content.Context;
import android.graphics.Bitmap;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

public class EfficientTAMSegmenter {
    
    private Module firstEncoder;
    private Module tracker;
    
    // Memory 状态（跟踪时需要保持）
    private Tensor maskmemFeatures;
    private Tensor maskmemPosEnc;
    private Tensor objPtr;
    
    private boolean isInitialized = false;
    private int videoWidth;
    private int videoHeight;
    
    /**
     * 初始化模型
     */
    public void initialize(Context context) {
        String firstEncoderPath = AssetUtils.assetFilePath(context, "efficienttam_video_first_v4.ptl");
        String trackerPath = AssetUtils.assetFilePath(context, "efficienttam_video_track_v4.ptl");
        
        firstEncoder = LiteModuleLoader.load(firstEncoderPath);
        tracker = LiteModuleLoader.load(trackerPath);
    }
    
    /**
     * 设置视频尺寸（用于坐标转换）
     */
    public void setVideoSize(int width, int height) {
        this.videoWidth = width;
        this.videoHeight = height;
    }
    
    /**
     * 处理首帧（用户点击后调用）
     * 
     * @param bitmap 首帧图像
     * @param clickX 点击 X 坐标（视频坐标系）
     * @param clickY 点击 Y 坐标（视频坐标系）
     * @return 分割 mask (512x512)
     */
    public float[] processFirstFrame(Bitmap bitmap, float clickX, float clickY) {
        // 1. 图像预处理
        Tensor imgTensor = bitmapToTensor(bitmap);
        
        // 2. 坐标转换到 512 空间
        float modelX = clickX / videoWidth * 512.0f;
        float modelY = clickY / videoHeight * 512.0f;
        
        // 3. 创建点坐标 tensor [1, 1, 2]
        float[] pointData = new float[] { modelX, modelY };
        Tensor pointCoords = Tensor.fromBlob(pointData, new long[] { 1, 1, 2 });
        
        // 4. 创建点标签 tensor [1, 1]（1 = 前景点）
        int[] labelData = new int[] { 1 };
        Tensor pointLabels = Tensor.fromBlob(labelData, new long[] { 1, 1 });
        
        // 5. 运行首帧编码器
        IValue[] outputs = firstEncoder.forward(
            IValue.from(imgTensor),
            IValue.from(pointCoords),
            IValue.from(pointLabels)
        ).toTuple();
        
        // 6. 解析输出
        Tensor maskTensor = outputs[0].toTensor();
        maskmemFeatures = outputs[1].toTensor();
        maskmemPosEnc = outputs[2].toTensor();
        objPtr = outputs[3].toTensor();
        
        isInitialized = true;
        
        // 7. 返回 mask 数据
        return maskTensor.getDataAsFloatArray();
    }
    
    /**
     * 处理后续帧（跟踪）
     * 
     * @param bitmap 当前帧图像
     * @return 分割 mask (512x512)，如果未初始化返回 null
     */
    public float[] processFrame(Bitmap bitmap) {
        if (!isInitialized) {
            return null;
        }
        
        // 1. 图像预处理
        Tensor imgTensor = bitmapToTensor(bitmap);
        
        // 2. 运行跟踪器
        IValue[] outputs = tracker.forward(
            IValue.from(imgTensor),
            IValue.from(maskmemFeatures),
            IValue.from(maskmemPosEnc),
            IValue.from(objPtr)
        ).toTuple();
        
        // 3. 更新 memory 状态
        Tensor maskTensor = outputs[0].toTensor();
        maskmemFeatures = outputs[1].toTensor();
        maskmemPosEnc = outputs[2].toTensor();
        objPtr = outputs[3].toTensor();
        
        // 4. 返回 mask 数据
        return maskTensor.getDataAsFloatArray();
    }
    
    /**
     * 重置跟踪状态
     */
    public void reset() {
        isInitialized = false;
        maskmemFeatures = null;
        maskmemPosEnc = null;
        objPtr = null;
    }
    
    /**
     * 将 Bitmap 转换为模型输入 Tensor
     * 输出: [1, 3, 512, 512]，值范围 0-255
     */
    private Tensor bitmapToTensor(Bitmap bitmap) {
        // Resize 到 512x512
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 512, 512, true);
        
        // 提取像素
        int[] pixels = new int[512 * 512];
        resized.getPixels(pixels, 0, 512, 0, 0, 512, 512);
        
        // 转换为 float 数组 [C, H, W] = [3, 512, 512]
        float[] floatData = new float[3 * 512 * 512];
        
        for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
                int pixel = pixels[y * 512 + x];
                
                // 提取 RGB（Android 是 ARGB 格式）
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;
                
                // CHW 格式，值范围 0-255
                floatData[0 * 512 * 512 + y * 512 + x] = r;  // R channel
                floatData[1 * 512 * 512 + y * 512 + x] = g;  // G channel
                floatData[2 * 512 * 512 + y * 512 + x] = b;  // B channel
            }
        }
        
        // 释放临时 bitmap
        if (resized != bitmap) {
            resized.recycle();
        }
        
        return Tensor.fromBlob(floatData, new long[] { 1, 3, 512, 512 });
    }
    
    /**
     * 将 mask logits 转换为二值 Bitmap (512x512)
     * 注意：返回原始 512x512 尺寸，由 MaskOverlayView 负责缩放到正确位置
     *
     * @param maskData mask 数据 (512*512)
     * @param maskColor mask 颜色 (ARGB)
     * @return 带透明度的 mask Bitmap (512x512)
     */
    public static Bitmap maskToBitmap(float[] maskData, int maskColor) {
        // 创建 512x512 的 mask bitmap
        Bitmap maskBitmap = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888);

        int[] pixels = new int[512 * 512];

        for (int i = 0; i < 512 * 512; i++) {
            if (maskData[i] > 0) {
                pixels[i] = maskColor;
            } else {
                pixels[i] = 0x00000000;  // 透明
            }
        }

        maskBitmap.setPixels(pixels, 0, 512, 0, 0, 512, 512);

        return maskBitmap;
    }
    
    /**
     * 释放资源
     */
    public void release() {
        if (firstEncoder != null) {
            firstEncoder.destroy();
            firstEncoder = null;
        }
        if (tracker != null) {
            tracker.destroy();
            tracker = null;
        }
        reset();
    }
}
```

### 5.3 视频帧提取器

```java
// VideoFrameExtractor.java
package com.example.efficienttam;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import java.io.IOException;

public class VideoFrameExtractor {
    
    private MediaMetadataRetriever retriever;
    private int videoWidth;
    private int videoHeight;
    private long durationMs;
    private int frameCount;
    private long frameDurationUs;  // 每帧时长（微秒）
    
    /**
     * 从 assets 打开视频
     */
    public void openFromAssets(Context context, String assetName) throws IOException {
        retriever = new MediaMetadataRetriever();
        
        AssetFileDescriptor afd = context.getAssets().openFd(assetName);
        retriever.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), afd.getLength());
        afd.close();
        
        // 获取视频信息
        String width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH);
        String height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT);
        String duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
        String frameRate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE);
        
        videoWidth = Integer.parseInt(width);
        videoHeight = Integer.parseInt(height);
        durationMs = Long.parseLong(duration);
        
        // 估算帧率（如果获取不到，默认 30fps）
        float fps = 30.0f;
        if (frameRate != null) {
            fps = Float.parseFloat(frameRate);
        }
        
        frameDurationUs = (long) (1000000.0f / fps);
        frameCount = (int) (durationMs * fps / 1000);
    }
    
    /**
     * 获取指定帧
     * 
     * @param frameIndex 帧索引
     * @return Bitmap，如果失败返回 null
     */
    public Bitmap getFrame(int frameIndex) {
        if (retriever == null || frameIndex < 0) {
            return null;
        }
        
        long timeUs = frameIndex * frameDurationUs;
        return retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST);
    }
    
    /**
     * 获取指定时间的帧
     * 
     * @param timeMs 时间（毫秒）
     * @return Bitmap
     */
    public Bitmap getFrameAtTime(long timeMs) {
        if (retriever == null) {
            return null;
        }
        return retriever.getFrameAtTime(timeMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST);
    }
    
    public int getVideoWidth() { return videoWidth; }
    public int getVideoHeight() { return videoHeight; }
    public long getDurationMs() { return durationMs; }
    public int getFrameCount() { return frameCount; }
    
    /**
     * 释放资源
     */
    public void release() {
        if (retriever != null) {
            retriever.release();
            retriever = null;
        }
    }
}
```

### 5.4 Mask 叠加 View

```java
// MaskOverlayView.java
package com.example.efficienttam;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

public class MaskOverlayView extends View {

    private Bitmap maskBitmap;
    private Paint paint;
    private float clickX = -1, clickY = -1;
    private Paint clickPaint;
    private Paint fpsPaint;

    // 视频尺寸（用于计算 fitCenter 显示区域）
    private int videoWidth = 0;
    private int videoHeight = 0;

    // FPS 显示
    private float fps = 0;
    private boolean showFps = true;

    public MaskOverlayView(Context context) {
        super(context);
        init();
    }

    public MaskOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        paint = new Paint();
        paint.setFilterBitmap(true);

        clickPaint = new Paint();
        clickPaint.setColor(0xFFFF0000);  // 红色
        clickPaint.setStyle(Paint.Style.FILL);
        clickPaint.setAntiAlias(true);

        fpsPaint = new Paint();
        fpsPaint.setColor(0xFFFFFF00);  // 黄色
        fpsPaint.setTextSize(48);
        fpsPaint.setAntiAlias(true);
        fpsPaint.setShadowLayer(2, 1, 1, 0xFF000000);
    }

    /**
     * 设置视频尺寸（必须调用，用于正确计算 mask 显示位置）
     */
    public void setVideoSize(int width, int height) {
        this.videoWidth = width;
        this.videoHeight = height;
    }

    /**
     * 设置 mask bitmap（512x512 的原始 mask）
     */
    public void setMask(Bitmap mask) {
        if (maskBitmap != null && maskBitmap != mask) {
            maskBitmap.recycle();
        }
        maskBitmap = mask;
        invalidate();
    }

    /**
     * 设置 FPS
     */
    public void setFps(float fps) {
        this.fps = fps;
        invalidate();
    }

    /**
     * 设置点击位置（用于显示点击点）
     */
    public void setClickPoint(float x, float y) {
        this.clickX = x;
        this.clickY = y;
        invalidate();
    }

    /**
     * 清除点击点
     */
    public void clearClickPoint() {
        this.clickX = -1;
        this.clickY = -1;
        invalidate();
    }

    /**
     * 计算视频在 View 中的显示区域（fitCenter 模式）
     */
    private RectF getVideoDisplayRect() {
        if (videoWidth <= 0 || videoHeight <= 0) {
            return new RectF(0, 0, getWidth(), getHeight());
        }

        int viewWidth = getWidth();
        int viewHeight = getHeight();

        float scaleX = (float) viewWidth / videoWidth;
        float scaleY = (float) viewHeight / videoHeight;
        float scale = Math.min(scaleX, scaleY);

        float displayWidth = videoWidth * scale;
        float displayHeight = videoHeight * scale;

        float offsetX = (viewWidth - displayWidth) / 2;
        float offsetY = (viewHeight - displayHeight) / 2;

        return new RectF(offsetX, offsetY, offsetX + displayWidth, offsetY + displayHeight);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // 获取视频显示区域
        RectF videoRect = getVideoDisplayRect();

        // 绘制 mask（缩放到视频显示区域）
        if (maskBitmap != null) {
            canvas.drawBitmap(maskBitmap, null, videoRect, paint);
        }

        // 绘制点击点
        if (clickX >= 0 && clickY >= 0) {
            canvas.drawCircle(clickX, clickY, 20, clickPaint);
        }

        // 绘制 FPS
        if (showFps && fps > 0) {
            String fpsText = String.format("FPS: %.1f", fps);
            canvas.drawText(fpsText, 20, 60, fpsPaint);
        }
    }

    /**
     * 清除 mask
     */
    public void clear() {
        if (maskBitmap != null) {
            maskBitmap.recycle();
            maskBitmap = null;
        }
        clearClickPoint();
        fps = 0;
        invalidate();
    }
}
```

---

## 6. 完整示例代码

### 6.1 布局文件

```xml
<!-- res/layout/activity_main.xml -->
<?xml version="1.0" encoding="utf-8"?>
<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <!-- 视频显示 -->
    <ImageView
        android:id="@+id/videoFrame"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:scaleType="fitCenter" />

    <!-- Mask 叠加层 -->
    <com.example.efficienttam.MaskOverlayView
        android:id="@+id/maskOverlay"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />

    <!-- 控制面板 -->
    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_gravity="bottom"
        android:background="#80000000"
        android:orientation="vertical"
        android:padding="16dp">

        <!-- 状态文本 -->
        <TextView
            android:id="@+id/statusText"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="点击画面选择要跟踪的目标"
            android:textColor="#FFFFFF"
            android:textSize="16sp" />

        <!-- 进度条 -->
        <SeekBar
            android:id="@+id/seekBar"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="8dp" />

        <!-- 按钮 -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="8dp"
            android:gravity="center"
            android:orientation="horizontal">

            <Button
                android:id="@+id/btnPlay"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="播放" />

            <Button
                android:id="@+id/btnPause"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:layout_marginStart="16dp"
                android:text="暂停" />

            <Button
                android:id="@+id/btnReset"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:layout_marginStart="16dp"
                android:text="重置" />

        </LinearLayout>

    </LinearLayout>

    <!-- 加载指示器 -->
    <ProgressBar
        android:id="@+id/loadingIndicator"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_gravity="center"
        android:visibility="gone" />

</FrameLayout>
```

### 6.2 MainActivity

```java
// MainActivity.java
package com.example.efficienttam;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    
    private static final String VIDEO_FILE = "sample_video.mp4";
    private static final int MASK_COLOR = 0x8000FF00;  // 半透明绿色
    
    // UI 组件
    private ImageView videoFrame;
    private MaskOverlayView maskOverlay;
    private TextView statusText;
    private SeekBar seekBar;
    private Button btnPlay, btnPause, btnReset;
    private ProgressBar loadingIndicator;
    
    // 核心组件
    private EfficientTAMSegmenter segmenter;
    private VideoFrameExtractor videoExtractor;
    
    // 状态
    private boolean isModelLoaded = false;
    private boolean isTracking = false;
    private boolean isPlaying = false;
    private int currentFrameIndex = 0;
    private float clickX = -1, clickY = -1;
    
    // 线程
    private ExecutorService executor;
    private Handler mainHandler;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initViews();
        initComponents();
        loadModelsAsync();
    }
    
    private void initViews() {
        videoFrame = findViewById(R.id.videoFrame);
        maskOverlay = findViewById(R.id.maskOverlay);
        statusText = findViewById(R.id.statusText);
        seekBar = findViewById(R.id.seekBar);
        btnPlay = findViewById(R.id.btnPlay);
        btnPause = findViewById(R.id.btnPause);
        btnReset = findViewById(R.id.btnReset);
        loadingIndicator = findViewById(R.id.loadingIndicator);
        
        // 点击事件 - 选择跟踪目标
        videoFrame.setOnTouchListener(this::onVideoTouch);
        
        // 按钮事件
        btnPlay.setOnClickListener(v -> startPlayback());
        btnPause.setOnClickListener(v -> pausePlayback());
        btnReset.setOnClickListener(v -> resetTracking());
        
        // 进度条
        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    seekToFrame(progress);
                }
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                pausePlayback();
            }
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });
    }
    
    private void initComponents() {
        executor = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());
        
        segmenter = new EfficientTAMSegmenter();
        videoExtractor = new VideoFrameExtractor();
    }
    
    /**
     * 异步加载模型
     */
    private void loadModelsAsync() {
        showLoading(true);
        statusText.setText("正在加载模型...");
        
        executor.execute(() -> {
            try {
                // 加载模型
                segmenter.initialize(this);
                
                // 打开视频
                videoExtractor.openFromAssets(this, VIDEO_FILE);
                segmenter.setVideoSize(
                    videoExtractor.getVideoWidth(),
                    videoExtractor.getVideoHeight()
                );
                
                // 显示首帧
                Bitmap firstFrame = videoExtractor.getFrame(0);
                
                mainHandler.post(() -> {
                    isModelLoaded = true;
                    showLoading(false);
                    statusText.setText("点击画面选择要跟踪的目标");
                    
                    seekBar.setMax(videoExtractor.getFrameCount() - 1);
                    
                    // 设置视频尺寸给 MaskOverlayView（用于 fitCenter 对齐）
                    maskOverlay.setVideoSize(
                            videoExtractor.getVideoWidth(),
                            videoExtractor.getVideoHeight()
                    );
                    
                    if (firstFrame != null) {
                        videoFrame.setImageBitmap(firstFrame);
                    }
                });
                
            } catch (Exception e) {
                e.printStackTrace();
                mainHandler.post(() -> {
                    showLoading(false);
                    statusText.setText("加载失败: " + e.getMessage());
                    Toast.makeText(this, "模型加载失败", Toast.LENGTH_LONG).show();
                });
            }
        });
    }
    
    /**
     * 视频画面点击事件
     */
    private boolean onVideoTouch(View v, MotionEvent event) {
        if (!isModelLoaded || event.getAction() != MotionEvent.ACTION_DOWN) {
            return false;
        }
        
        // 计算点击在视频中的坐标
        float viewX = event.getX();
        float viewY = event.getY();
        
        // 转换到视频坐标系
        float[] videoCoords = viewToVideoCoords(viewX, viewY);
        if (videoCoords == null) {
            return false;
        }
        
        clickX = videoCoords[0];
        clickY = videoCoords[1];
        
        // 显示点击位置
        float[] screenCoords = videoToViewCoords(clickX, clickY);
        maskOverlay.setClickPoint(screenCoords[0], screenCoords[1]);
        
        // 开始分割
        startSegmentation();
        
        return true;
    }
    
    /**
     * 开始分割（处理首帧）
     */
    private void startSegmentation() {
        if (!isModelLoaded || clickX < 0) {
            return;
        }
        
        pausePlayback();
        showLoading(true);
        statusText.setText("正在分割...");
        
        executor.execute(() -> {
            try {
                // 获取当前帧
                Bitmap frame = videoExtractor.getFrame(currentFrameIndex);
                if (frame == null) {
                    throw new Exception("无法获取视频帧");
                }
                
                // 处理首帧
                float[] maskData = segmenter.processFirstFrame(frame, clickX, clickY);
                
                // 生成 mask bitmap (512x512)
                Bitmap maskBitmap = EfficientTAMSegmenter.maskToBitmap(maskData, MASK_COLOR);
                
                mainHandler.post(() -> {
                    isTracking = true;
                    showLoading(false);
                    statusText.setText("分割完成，点击播放继续跟踪");
                    
                    videoFrame.setImageBitmap(frame);
                    maskOverlay.setMask(maskBitmap);
                });
                
            } catch (Exception e) {
                e.printStackTrace();
                mainHandler.post(() -> {
                    showLoading(false);
                    statusText.setText("分割失败: " + e.getMessage());
                });
            }
        });
    }
    
    /**
     * 开始播放（跟踪）
     */
    private void startPlayback() {
        if (!isTracking) {
            Toast.makeText(this, "请先点击选择目标", Toast.LENGTH_SHORT).show();
            return;
        }
        
        isPlaying = true;
        statusText.setText("跟踪中...");
        processNextFrame();
    }
    
    /**
     * 处理下一帧
     */
    private void processNextFrame() {
        if (!isPlaying || !isTracking) {
            return;
        }

        currentFrameIndex++;
        if (currentFrameIndex >= videoExtractor.getFrameCount()) {
            // 视频结束，循环播放
            currentFrameIndex = 0;
            segmenter.reset();
            isTracking = false;
            statusText.setText("播放结束，点击重新选择目标");
            return;
        }

        long frameStartTime = System.currentTimeMillis();

        executor.execute(() -> {
            try {
                Bitmap frame = videoExtractor.getFrame(currentFrameIndex);
                if (frame == null) {
                    return;
                }

                // 跟踪
                float[] maskData = segmenter.processFrame(frame);

                Bitmap maskBitmap = EfficientTAMSegmenter.maskToBitmap(maskData, MASK_COLOR);

                long endTime = System.currentTimeMillis();
                float frameTime = endTime - frameStartTime;
                float currentFps = frameTime > 0 ? 1000.0f / frameTime : 0;

                mainHandler.post(() -> {
                    videoFrame.setImageBitmap(frame);
                    maskOverlay.setMask(maskBitmap);
                    maskOverlay.setFps(currentFps);
                    seekBar.setProgress(currentFrameIndex);

                    // 继续下一帧
                    processNextFrame();
                });

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }
    
    /**
     * 暂停播放
     */
    private void pausePlayback() {
        isPlaying = false;
        if (isTracking) {
            statusText.setText("已暂停");
        }
    }
    
    /**
     * 重置跟踪
     */
    private void resetTracking() {
        pausePlayback();
        
        isTracking = false;
        currentFrameIndex = 0;
        clickX = -1;
        clickY = -1;
        
        segmenter.reset();
        maskOverlay.clear();
        seekBar.setProgress(0);
        
        // 显示首帧
        executor.execute(() -> {
            Bitmap firstFrame = videoExtractor.getFrame(0);
            mainHandler.post(() -> {
                if (firstFrame != null) {
                    videoFrame.setImageBitmap(firstFrame);
                }
                statusText.setText("点击画面选择要跟踪的目标");
            });
        });
    }
    
    /**
     * 跳转到指定帧
     */
    private void seekToFrame(int frameIndex) {
        currentFrameIndex = frameIndex;
        
        executor.execute(() -> {
            Bitmap frame = videoExtractor.getFrame(frameIndex);
            mainHandler.post(() -> {
                if (frame != null) {
                    videoFrame.setImageBitmap(frame);
                }
                
                // 如果正在跟踪，需要重新从首帧开始
                if (isTracking && frameIndex != 0) {
                    Toast.makeText(this, "跳转后需要重新选择目标", Toast.LENGTH_SHORT).show();
                    resetTracking();
                }
            });
        });
    }
    
    /**
     * View 坐标转视频坐标
     */
    private float[] viewToVideoCoords(float viewX, float viewY) {
        // 获取 ImageView 的实际显示区域
        int viewWidth = videoFrame.getWidth();
        int viewHeight = videoFrame.getHeight();
        int videoWidth = videoExtractor.getVideoWidth();
        int videoHeight = videoExtractor.getVideoHeight();
        
        // 计算缩放比例（fitCenter 模式）
        float scaleX = (float) viewWidth / videoWidth;
        float scaleY = (float) viewHeight / videoHeight;
        float scale = Math.min(scaleX, scaleY);
        
        // 计算偏移
        float offsetX = (viewWidth - videoWidth * scale) / 2;
        float offsetY = (viewHeight - videoHeight * scale) / 2;
        
        // 转换坐标
        float videoX = (viewX - offsetX) / scale;
        float videoY = (viewY - offsetY) / scale;
        
        // 检查是否在视频范围内
        if (videoX < 0 || videoX >= videoWidth || videoY < 0 || videoY >= videoHeight) {
            return null;
        }
        
        return new float[] { videoX, videoY };
    }
    
    /**
     * 视频坐标转 View 坐标
     */
    private float[] videoToViewCoords(float videoX, float videoY) {
        int viewWidth = videoFrame.getWidth();
        int viewHeight = videoFrame.getHeight();
        int videoWidth = videoExtractor.getVideoWidth();
        int videoHeight = videoExtractor.getVideoHeight();
        
        float scaleX = (float) viewWidth / videoWidth;
        float scaleY = (float) viewHeight / videoHeight;
        float scale = Math.min(scaleX, scaleY);
        
        float offsetX = (viewWidth - videoWidth * scale) / 2;
        float offsetY = (viewHeight - videoHeight * scale) / 2;
        
        float viewX = videoX * scale + offsetX;
        float viewY = videoY * scale + offsetY;
        
        return new float[] { viewX, viewY };
    }
    
    private void showLoading(boolean show) {
        loadingIndicator.setVisibility(show ? View.VISIBLE : View.GONE);
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        pausePlayback();
        
        if (executor != null) {
            executor.shutdown();
        }
        if (segmenter != null) {
            segmenter.release();
        }
        if (videoExtractor != null) {
            videoExtractor.release();
        }
    }
}
```

---

## 7. 性能优化建议

### 7.1 模型优化

```gradle
// 使用量化版本（如果可用）
implementation 'org.pytorch:pytorch_android_lite:1.13.1'
```

### 7.2 内存优化

```java
// 及时释放 Bitmap
if (oldBitmap != null && oldBitmap != newBitmap) {
    oldBitmap.recycle();
}

// 使用 inBitmap 复用
BitmapFactory.Options options = new BitmapFactory.Options();
options.inBitmap = reusableBitmap;
```

### 7.3 推理优化

```java
// 使用 GPU（如果支持）
// 注意：PyTorch Mobile 对 GPU 支持有限

// 降低分辨率（如果可接受）
// 可以将输入从 512x512 降到 256x256

// 跳帧处理
int frameSkip = 2;  // 每隔2帧处理一次
```

### 7.4 线程优化

```java
// 使用线程池
ExecutorService executor = Executors.newFixedThreadPool(2);

// 预加载下一帧
executor.execute(() -> {
    nextFrame = videoExtractor.getFrame(currentIndex + 1);
});
```

---

## 8. 常见问题

### Q1: 模型加载失败

**原因**: .ptl 文件被压缩或损坏

**解决**: 
```gradle
android {
    aaptOptions {
        noCompress "ptl"
    }
}
```

### Q2: 内存不足 (OOM)

**原因**: 模型较大（~120MB）

**解决**:
- 使用 `android:largeHeap="true"`
- 及时释放 Bitmap
- 考虑使用更小的模型

### Q3: 推理速度慢

**原因**: CPU 推理较慢

**解决**:
- 使用 arm64-v8a 架构
- 降低输入分辨率
- 跳帧处理

### Q4: Mask 位置不准

**原因**: 坐标转换错误

**解决**:
- 检查 ImageView 的 scaleType
- 确保视频尺寸正确
- 验证坐标转换逻辑

### Q5: 跟踪漂移

**原因**: 单帧 memory 限制

**解决**:
- 这是当前简化版本的限制
- 完整版本需要多帧 memory 累积

---

## 附录：文件清单

```
Android 项目需要的文件：
├── efficienttam_video_first_v4.ptl   # 首帧编码器（从 Python 导出）
├── efficienttam_video_track_v4.ptl   # 跟踪器（从 Python 导出）
└── sample_video.mp4                   # 测试视频

Java 源文件：
├── MainActivity.java                  # 主界面
├── EfficientTAMSegmenter.java        # 模型封装
├── VideoFrameExtractor.java          # 视频帧提取
├── MaskOverlayView.java              # Mask 显示
└── AssetUtils.java                   # 工具类
```

---

**文档版本**: 1.0  
**最后更新**: 2024年12月  
**模型版本**: EfficientTAM V4
