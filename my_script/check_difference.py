import os
import numpy as np
import pandas as pd
from plyfile import PlyData

def load_points(filepath):
    """ 从 .ply 文件中读取顶点坐标，返回 (N,3) 的 numpy 数组 """
    ply = PlyData.read(filepath)
    data = ply['vertex'].data
    # 假设字段名为 x, y, z
    pts = np.vstack((data['x'], data['y'], data['z'])).T
    return pts

def compute_displacement_histogram(frame_dir, num_bins=10):
    # 列出并排序所有 .ply 文件
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith('.ply'))
    
    disp_list = []
    for i in range(len(files) - 1):
        p0 = load_points(os.path.join(frame_dir, files[i]))
        p1 = load_points(os.path.join(frame_dir, files[i+1]))
        if p0.shape != p1.shape:
            raise ValueError(f"顶点数量不一致：{files[i]} vs {files[i+1]}")
        # 计算同序号点的欧氏位移
        disp = np.linalg.norm(p1 - p0, axis=1)
        disp_list.append(disp)
    
    # 拼成一维数组，并做直方图
    all_disp = np.concatenate(disp_list)
    counts, bin_edges = np.histogram(all_disp, bins=num_bins)
    
    # 构造 DataFrame
    ranges = [f"{bin_edges[i]:.6f}–{bin_edges[i+1]:.6f}" for i in range(len(bin_edges)-1)]
    df = pd.DataFrame({
        'displacement_range': ranges,
        'count': counts
    })
    return df

if __name__ == "__main__":
    frame_dir = "data/scene2/frames"
    # 你可以调整 num_bins，比如 20、50 以获得更细的区间划分
    df = compute_displacement_histogram(frame_dir, num_bins=10)
    print(df.to_string(index=False))
    # 保存到 CSV 文件，便于后续分析、绘图
    df.to_csv("displacement_histogram.csv", index=False)