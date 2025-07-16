#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_plot_dynamic_trajectories.py

自动计算全序列 AABB、加权主方向，选取主要动态点并绘制它们的轨迹。
"""

import os
import glob
import json
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============ 配置 ============
ply_dir        = "/users/lshou/4DGaussians/filter_3dgs/filter_ply"
output_json    = "/users/lshou/4DGaussians/filter_3dgs/box_and_normal.json"
output_png     = "/users/lshou/4DGaussians/filter_3dgs/box_normal_dynamic_plot.png"
dynamic_count  = 200   # 选择多少个主要动态点来绘制轨迹

# ============ 1. 读取并排序所有 PLY ============
ply_paths = sorted(glob.glob(os.path.join(ply_dir, "frame_*.ply")))
if len(ply_paths) < 2:
    raise RuntimeError(f"需要至少 2 帧 ply，当前找到 {len(ply_paths)} 个文件")

# ============ 2. 加载所有帧点云 ============
coords_list = []
for p in ply_paths:
    ply = PlyData.read(p)
    v   = ply['vertex'].data
    pts = np.vstack([v['x'], v['y'], v['z']]).T  # (N,3)
    coords_list.append(pts)
coords = np.stack(coords_list, axis=0)         # (F, N, 3)
coords0 = coords[0]                            # 第一帧
F,  N, _ = coords.shape

# ============ 3. 计算全序列 AABB ============
# 对整个 (F*N,3) 数据做极值
all_pts = coords.reshape(-1, 3)   # (F*N, 3)
min_all = all_pts.min(axis=0)     # [xmin, ymin, zmin]
max_all = all_pts.max(axis=0)     # [xmax, ymax, zmax]
bbox0   = [min_all.tolist(), max_all.tolist()]

# ============ 4. 计算加权主方向 ============
deltas = coords[1:] - coords[:-1]               # (F-1, N, 3)
deltas_sel = deltas.reshape(-1, 3)              # ((F-1)*N, 3)
weights    = np.linalg.norm(deltas_sel, axis=1)
valid      = weights > 1e-6

if valid.sum() < 3:
    avg    = deltas_sel.mean(axis=0)
    normal = (avg / np.linalg.norm(avg)).tolist()
else:
    Xw      = deltas_sel[valid] * np.sqrt(weights[valid])[:, None]
    Xc      = Xw - Xw.mean(axis=0)
    _, _, VT = np.linalg.svd(Xc, full_matrices=False)
    normal  = VT[0].tolist()

# ============ 5. 保存 JSON ============
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w") as f:
    json.dump({"bbox": bbox0, "normal": normal}, f, indent=2)
print(f"✅ 已保存参数: {output_json}")
print(json.dumps({"bbox": bbox0, "normal": normal}, indent=2))

# ============ 6. 选择主要动态点 ============
disp_all = np.linalg.norm(coords - coords0[None,:,:], axis=2)  # (F, N)
max_disp = disp_all.max(axis=0)                               # (N,)
dyn_idx  = np.argsort(-max_disp)[:dynamic_count]

# ============ 7. 绘图并保存 ============
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

# 7.1 所有帧点云，半透明渐变色
all_pts   = coords.reshape(-1, 3)
frame_idx = np.repeat(np.arange(F), N)
sc = ax.scatter(
    all_pts[:,0], all_pts[:,1], all_pts[:,2],
    c=frame_idx, cmap='viridis', s=1,
    alpha=0.3, marker='.', linewidths=0
)
cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Frame Index")

# 7.2 绘制主要动态点的轨迹
for idx in dyn_idx:
    traj = coords[:, idx, :]  # (F,3)
    ax.plot(
        traj[:,0], traj[:,1], traj[:,2],
        linewidth=1,
        alpha=0.8,
        color='blue'
    )

# 7.3 绘制全局 AABB
# 构造 8 个角点
corners = np.array([[i,j,k] for i in [min_all[0], max_all[0]]
                           for j in [min_all[1], max_all[1]]
                           for k in [min_all[2], max_all[2]]])
# 12 条边索引
edges = [
    (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),
    (2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
]
for i, j in edges:
    p1, p2 = corners[i], corners[j]
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        c='red', linewidth=0.5, alpha=0.8
    )

# 7.4 绘制法向箭头
center_pt = (min_all + max_all) / 2
diag_len  = np.linalg.norm(max_all - min_all)
ax.quiver(
    center_pt[0], center_pt[1], center_pt[2],
    normal[0], normal[1], normal[2],
    length=0.7*diag_len,
    linewidth=4,
    color='magenta',
    arrow_length_ratio=0.2,
    alpha=1.0
)

# 7.5 坐标与标题
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Dynamic Trajectories + Global AABB + Weighted Normal")

plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.show()
print(f"✅ 已保存图像: {output_png}")
