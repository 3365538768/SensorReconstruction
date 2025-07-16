import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import time

# 配置设备：尽量使用GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载SAM模型（使用ViT-H模型以获得最高精度）
checkpoint_path = "utils_checkpoint/sam_vit_h_4b8939.pth"  # TODO: 替换为您的SAM权重文件路径
model_type = "vit_h"  # 可选 "vit_l" 或 "vit_b"
print(f"Loading SAM model {model_type} from {checkpoint_path} ...")
start_load = time.time()
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
print(f"Model loaded in {time.time() - start_load:.1f}s")

# 初始化自动掩码生成器
mask_generator = SamAutomaticMaskGenerator(sam)

# 打开视频文件
video_path = "data/my_video/tube.mov"  # TODO: 替换为您的视频文件路径
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video file: {video_path}")
print(f"Opened video {video_path}")

# 设置输出帧率
fps_target = 5  # 目标提取帧率
orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps_target
frame_interval = max(1, int(round(orig_fps / fps_target)))
print(f"Original FPS: {orig_fps:.1f}, extracting every {frame_interval} frame(s) -> ~{orig_fps/frame_interval:.1f} FPS")

frame_idx = 0
saved_frame_idx = 0
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取结束
    
    frame_idx += 1
    if frame_idx % frame_interval != 0:
        continue

    print(f"\nProcessing frame #{frame_idx} ...")
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 生成所有候选掩码
    t_mask_start = time.time()
    masks = mask_generator.generate(image_rgb)
    t_mask = time.time() - t_mask_start
    print(f"  → Generated {len(masks)} masks in {t_mask:.2f}s")
    
    # 筛选位于画面中心且合理大小的主要掩码
    img_h, img_w, _ = frame.shape
    center_xy = np.array([img_w/2, img_h/2])
    main_mask = None
    best_score = float("inf")
    
    # 参数：最小掩码面积 & 最大相对面积
    min_area = 5000
    max_rel_area = 0.9

    for i, mask in enumerate(masks):
        area = mask["area"]
        # 过滤过小或过大的掩码
        if area < min_area or area > img_h * img_w * max_rel_area:
            continue

        x, y, w, h = mask["bbox"]
        # 计算 bbox 中心到图像中心的距离
        bbox_center = np.array([x + w/2, y + h/2])
        dist = np.linalg.norm(bbox_center - center_xy)

        if dist < best_score:
            best_score = dist
            main_mask = mask["segmentation"]

    if main_mask is None:
        print("  ! 未找到合适掩码，跳过此帧")
        continue
    print(f"  → Selected mask with center-distance {best_score:.1f}")

    # 构造透明背景图 (BGRA)
    mask_array = (main_mask.astype(np.uint8) * 255)
    bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask_array

    # 保存结果
    output_path = f"data/your-ns-data/frame_{saved_frame_idx:04d}.png"
    cv2.imwrite(output_path, bgra)
    print(f"  ✓ Saved to {output_path}")
    saved_frame_idx += 1

cap.release()
t_total = time.time() - t0
print(f"\nAll done! Processed ~{saved_frame_idx} frames in {t_total:.1f}s (~{saved_frame_idx/t_total:.2f} FPS)")
