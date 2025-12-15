"""
EfficientTAM 视频跟踪演示脚本

生成 GIF 动图用于 README 展示

使用方法:
    python demo_video_tracking.py

输出:
    assets/demo_video_tracking.gif - 视频跟踪效果 GIF
"""

import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# 配置 (相对于脚本所在目录)
FIRST_ENCODER_PATH = 'efficienttam_video_first_v4.pt'  # 需要先导出模型到此目录
TRACKER_PATH = 'efficienttam_video_track_v4.pt'
VIDEO_PATH = 'examples/videos/cat.mp4'  # 示例视频
OUTPUT_DIR = Path('assets')
MAX_FRAMES = 60
GIF_FPS = 10


def load_video_frames_from_mp4(video_path, max_frames=60):
    """从 MP4 视频文件加载帧"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        frame_count += 1
    
    cap.release()
    return frames


def preprocess_image(img, target_size=512):
    """预处理图像为模型输入（不使用 ImageNet 标准化，V4 模型内部处理）"""
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, img_resized


def create_overlay(img, mask, color=(0, 255, 0), alpha=0.5):
    """创建掩码叠加图像"""
    img_np = np.array(img)
    mask_bool = mask > 0
    
    overlay = img_np.copy()
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    return Image.fromarray(overlay)


def main():
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 检查模型文件
    if not os.path.exists(FIRST_ENCODER_PATH):
        print(f"错误: 首帧编码器不存在: {FIRST_ENCODER_PATH}")
        print("请先运行 export_video_tracking_v4.py 导出模型")
        return
    
    if not os.path.exists(TRACKER_PATH):
        print(f"错误: 跟踪器不存在: {TRACKER_PATH}")
        print("请先运行 export_video_tracking_v4.py 导出模型")
        return
    
    # 加载模型
    print("加载模型...")
    device = 'cpu'
    first_encoder = torch.jit.load(FIRST_ENCODER_PATH, map_location=device).eval()
    tracker = torch.jit.load(TRACKER_PATH, map_location=device).eval()
    print("模型加载完成")
    
    # 加载视频帧
    if not os.path.exists(VIDEO_PATH):
        print(f"错误: 视频文件不存在: {VIDEO_PATH}")
        return
    
    print(f"加载视频: {VIDEO_PATH}")
    frames = load_video_frames_from_mp4(VIDEO_PATH, max_frames=MAX_FRAMES)
    print(f"加载了 {len(frames)} 帧")
    
    # 设置点提示 - 猫的身体中心位置
    # cat.mp4 是竖屏视频 (720x1280)，resize 到 512x512 后猫在中心
    point_coords = torch.tensor([[[256.0, 200.0]]], dtype=torch.float32)
    point_labels = torch.tensor([[1]], dtype=torch.int32)
    
    print(f"点提示: ({point_coords[0,0,0].item()}, {point_coords[0,0,1].item()})")
    
    # 跟踪并生成帧
    print("\n开始视频跟踪...")
    result_frames = []
    
    with torch.no_grad():
        for i, frame in enumerate(frames):
            # 预处理图像
            img_tensor, img_resized = preprocess_image(frame)
            
            if i == 0:
                # 首帧：使用点提示
                print(f"帧 {i:3d}: 首帧处理 (带点提示)")
                mask, maskmem_features, maskmem_pos_enc, obj_ptr = first_encoder(
                    img_tensor, point_coords, point_labels
                )
            else:
                # 后续帧：使用 memory 跟踪
                if i % 10 == 0:
                    print(f"帧 {i:3d}: 跟踪中...")
                mask, maskmem_features, maskmem_pos_enc, obj_ptr = tracker(
                    img_tensor, maskmem_features, maskmem_pos_enc, obj_ptr
                )
            
            # 处理掩码
            mask_np = (mask[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
            mask_resized = np.array(Image.fromarray(mask_np).resize(img_resized.size, Image.NEAREST))
            
            # 创建叠加图像
            overlay = create_overlay(img_resized, mask_resized > 0)
            
            # 在首帧添加点标记（红色星形，更醒目）
            if i == 0:
                overlay_np = np.array(overlay)
                px, py = int(point_coords[0, 0, 0].item()), int(point_coords[0, 0, 1].item())
                # 画一个红色星形标记
                star_color = [255, 0, 0]  # 红色
                # 外圈白色边框
                for dx in range(-12, 13):
                    for dy in range(-12, 13):
                        dist = (dx*dx + dy*dy) ** 0.5
                        if 8 <= dist <= 12:
                            x, y = px + dx, py + dy
                            if 0 <= x < 512 and 0 <= y < 512:
                                overlay_np[y, x] = [255, 255, 255]
                # 内圈红色实心
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
                        if dx*dx + dy*dy <= 64:
                            x, y = px + dx, py + dy
                            if 0 <= x < 512 and 0 <= y < 512:
                                overlay_np[y, x] = star_color
                # 十字线
                for d in range(-15, 16):
                    if 0 <= px + d < 512:
                        overlay_np[py, px + d] = star_color
                    if 0 <= py + d < 512:
                        overlay_np[py + d, px] = star_color
                overlay = Image.fromarray(overlay_np)
            
            result_frames.append(overlay)
    
    print(f"\n处理完成，共 {len(result_frames)} 帧")
    
    # 保存为 GIF
    print("生成 GIF...")
    gif_path = OUTPUT_DIR / 'demo_video_tracking.gif'
    
    result_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=result_frames[1:],
        duration=int(1000 / GIF_FPS),
        loop=0
    )
    
    print(f"\nGIF 已保存: {gif_path}")
    print(f"帧数: {len(result_frames)}, FPS: {GIF_FPS}")
    
    # 同时保存首帧和最后一帧作为静态图
    first_frame_path = OUTPUT_DIR / 'demo_video_first_frame.png'
    last_frame_path = OUTPUT_DIR / 'demo_video_last_frame.png'
    
    result_frames[0].save(str(first_frame_path))
    result_frames[-1].save(str(last_frame_path))
    
    print(f"首帧: {first_frame_path}")
    print(f"末帧: {last_frame_path}")


if __name__ == "__main__":
    main()
