import os
import glob
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ==================== Configuration ====================
ply_dir             = "/users/lshou/4DGaussians/output/dnerf/SPLITS/gaussian_pertimestamp"
output_ply_dir      = "/users/lshou/4DGaussians/filter_3dgs/filter_ply"
angles_dir          = "/users/lshou/4DGaussians/filter_3dgs/angles"

# Clustering parameters for main object extraction
obj_cluster_eps     = 0.5      # DBSCAN 半径
obj_min_samples     = 5000       # DBSCAN 最小簇大小

# Dynamic filtering parameters
top_k               = 15000     # 初步动态点候选数量
cluster_eps         = 0.5       # DBSCAN 半径
cluster_min_s       = 5000      # DBSCAN 最小簇大小
plot_M              = 1000      # 最终绘制轨迹的点数
neighbor_radius     = 0.05      # KDTree 邻域半径（扩展点云输出）
n_angles            = 2         # 多视角图像数量

# ================ Prepare Directories ================
os.makedirs(output_ply_dir, exist_ok=True)
os.makedirs(angles_dir, exist_ok=True)

# ================ 1. Load & Stack Coordinates ================
ply_paths   = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
coords_list = []
for p in ply_paths:
    ply   = PlyData.read(p)
    v     = ply['vertex'].data
    pts   = np.vstack([v['x'], v['y'], v['z']]).T
    coords_list.append(pts)
coords = np.stack(coords_list, axis=0)  # (F, N, 3)
F, N, _ = coords.shape

# ================ 2. Extract Main Object by Clustering ================
# Use DBSCAN on the first frame to separate the main object from background
db_obj      = DBSCAN(eps=obj_cluster_eps, min_samples=obj_min_samples)
labels0     = db_obj.fit_predict(coords[0])
valid0      = labels0 >= 0
unique0, counts0 = np.unique(labels0[valid0], return_counts=True)
main_cluster    = unique0[np.argmax(counts0)]
main_idx        = np.where(labels0 == main_cluster)[0]

# Restrict coords to main object points only
coords_main = coords[:, main_idx, :]
_, N_main, _ = coords_main.shape

# ================ 3. Compute Max Displacement within Main Object ================
disp        = np.linalg.norm(coords_main - coords_main[0:1], axis=2)
max_disp    = disp.max(axis=0)
candidates_local = np.argsort(-max_disp)[:top_k]

# ================ 4. DBSCAN Clustering on Dynamic Candidates ================
first_pts_local = coords_main[0, candidates_local, :]
db_dyn          = DBSCAN(eps=cluster_eps, min_samples=cluster_min_s)
labels_dyn      = db_dyn.fit_predict(first_pts_local)
valid_dyn       = labels_dyn >= 0
unique_dyn, counts_dyn = np.unique(labels_dyn[valid_dyn], return_counts=True)
largest_dyn      = unique_dyn[np.argmax(counts_dyn)]
cluster_loc_idx  = candidates_local[labels_dyn == largest_dyn]

# ================ 5. Pick Top-M Dynamic in Cluster ================
cluster_disp = max_disp[cluster_loc_idx]
order        = np.argsort(-cluster_disp)[:plot_M]
plot_loc_idx = cluster_loc_idx[order]
# Convert to global indices
plot_idx     = main_idx[plot_loc_idx]

# ================ 6. Save Filtered PLYs (Cluster + Neighbors) ================
tree       = KDTree(coords[0], leaf_size=40)
nbrs       = tree.query_radius(coords[0, plot_idx], r=neighbor_radius)
union_idx  = np.unique(np.concatenate(nbrs))

for i, p in enumerate(ply_paths):
    ply   = PlyData.read(p)
    verts = ply['vertex'].data[union_idx]
    elem  = PlyElement.describe(verts, 'vertex')
    new   = PlyData([elem], text=ply.text)
    new.comments = list(ply.comments)
    new.obj_info = list(ply.obj_info)
    new.write(os.path.join(output_ply_dir, f"frame_{i:03d}.ply"))
print("Filtered PLYs saved to:", output_ply_dir)

# ================ 7. Prepare Line Segments for Trajectories ================
segments = []
times    = []
for idx in plot_loc_idx:
    traj = coords_main[:, idx, :]
    segs = np.stack([traj[:-1], traj[1:]], axis=1)
    segments.append(segs)
    times.append(np.arange(F-1))
segments = np.concatenate(segments, axis=0)
times    = np.concatenate(times, axis=0)

all_pts = coords_main[:, plot_loc_idx, :].reshape(-1, 3)
p5, p95 = np.percentile(all_pts, [5, 95], axis=0)

# ================ 8. Generate Multi-Angle Trajectory Images ================
for i in range(n_angles):
    azim = 360 * i / n_angles
    fig  = plt.figure(figsize=(6, 6))
    ax   = fig.add_subplot(111, projection='3d')
    lc   = Line3DCollection(
        segments,
        cmap='plasma',
        norm=plt.Normalize(0, F-1),
        linewidth=0.5,
        alpha=0.8
    )
    lc.set_array(times)
    ax.add_collection(lc)
    ax.set_xlim(p5[0], p95[0])
    ax.set_ylim(p5[1], p95[1])
    ax.set_zlim(p5[2], p95[2])
    ax.view_init(elev=30, azim=azim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Frame index")
    ax.set_title(f"Azimuth {int(azim)}°")
    plt.tight_layout()
    out_png = os.path.join(angles_dir, f"traj_{i:02d}.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

print(f"Generated {n_angles} trajectory images in:", angles_dir)
