import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

def make_heatmap_video(
    csv_path: str,
    out_path: str = "sensor_heatmap.mp4",
    sensor_h: int = 10,
    sensor_w: int = 10,
    duration_s: float = 5.0,
    fps: float = None,        # ← 新增 fps 参数
    cmap: str = 'viridis',       # 可选配色
    figsize: tuple = (6, 6),
    dpi: int = 200
):
    # 1) 读取 CSV
    df = pd.read_csv(csv_path)
    frames = df.iloc[:, 0].astype(int).values
    data   = df.iloc[:, 1:].values.astype(float)
    assert data.shape[1] == sensor_h * sensor_w, \
        f"数据列数 {data.shape[1]} ≠ {sensor_h}×{sensor_w}"

    num_frames = len(frames)

    # 2) 计算或使用传入的 fps
    if fps is None:
        fps = num_frames / duration_s
    else:
        # 如果用户指定了 fps，则覆盖 duration_s
        duration_s = num_frames / fps

    # 3) 统一色标范围
    vmin, vmax = np.nanmin(data), np.nanmax(data)

    # 4) 创建视频 writer
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec='libx264',
        ffmpeg_params=['-pix_fmt', 'yuv420p']
    )

    # 5) 绘制每一帧热力图
    for idx, frame_id in enumerate(frames):
        arr = data[idx].reshape(sensor_h, sensor_w)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(
            arr,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin='lower'
        )
        ax.set_title(f"Frame {frame_id}", fontsize=14)
        ax.axis('off')
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        h, w = fig.canvas.get_width_height()
        img = img.reshape((h, w, 3))
        plt.close(fig)

        writer.append_data(img)

    writer.close()
    print(f"✅ Heatmap video saved to {out_path}")
    print(f"   → {num_frames} frames @ {fps:.2f} FPS (~{duration_s:.2f}s)")

if __name__ == "__main__":
    # 示例：直接指定 fps 为 15
    make_heatmap_video(
        csv_path="bend_interpolate.csv",
        out_path="sensor_heatmap.mp4",
        sensor_h=10,
        sensor_w=10,
        duration_s=5.0,   # 如果设置了 fps，这里会被覆盖
        fps=15.0,         # ← 手动调节帧率
        cmap='YlGn',
        figsize=(6,6),
        dpi=200
    )
