import os
import glob
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import networkx as nx

# ================ Configuration ================
ply_dir    = "/users/lshou/4DGaussians/output/dnerf/SPLITS/gaussian_pertimestamp"
sensor_csv = "/users/lshou/4DGaussians/sensor/sensor_data_half_68.csv"
out_dir    = "/users/lshou/4DGaussians/output/auto_roi_fixed"

# Dynamic detection params
percent_dynamic    = 0.05    # top 5% points by max displacement
eps_coarse_ratio   = 0.1     # coarse DBSCAN eps = 10% of scene diagonal
min_coarse_pts     = 30      # coarse DBSCAN min samples
neighbor_radius    = 0.02    # neighbor filter radius = 2% of diag
min_neighbors      = 300     # require at least 300 dynamic neighbors
connect_radius     = 0.05    # graph connect if within 5% of diag
pad_ratio          = 0.05    # pad bounding box by 5%

# ================ Setup ================
os.makedirs(out_dir, exist_ok=True)
ply_paths = sorted(glob.glob(os.path.join(ply_dir, "time_*.ply")))
if not ply_paths:
    raise RuntimeError(f"No PLY files found in {ply_dir}")

# ================ Load Sensor Data (frame count check) ================
sensor_df = pd.read_csv(sensor_csv)
if sensor_df.shape[0] != len(ply_paths):
    raise RuntimeError("Sensor frame count does not match PLY frame count")

# ================ Stack point clouds to compute dynamic region ================
coords_list = []
for p in ply_paths:
    ply   = PlyData.read(p)
    verts = ply['vertex'].data
    pts   = np.vstack([verts['x'], verts['y'], verts['z']]).T
    coords_list.append(pts)
coords = np.stack(coords_list, axis=0)  # (F, N, 3)
F, N, _ = coords.shape

# ================ Compute scene diagonal ================
all_pts = coords.reshape(-1, 3)
diag    = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))

# ================ Compute max displacement per point ================
disp     = np.linalg.norm(coords - coords[0:1], axis=2)  # (F, N)
max_disp = disp.max(axis=0)                              # (N,)

# ================ Select top dynamic candidates ================
K        = int(N * percent_dynamic)
cand_idx = np.argsort(-max_disp)[:K]

# ================ Neighborhood filtering ================
tree   = KDTree(coords[0], leaf_size=40)
r_nb   = diag * neighbor_radius
nbrs   = tree.query_radius(coords[0, cand_idx], r=r_nb)
mask_nb = np.array([len(nb) >= min_neighbors for nb in nbrs])
filtered = cand_idx[mask_nb]
if len(filtered) == 0:
    raise RuntimeError("No dynamic candidates survive neighbor filtering")

# ================ Coarse DBSCAN to remove noise ================
pts0       = coords[0, filtered]
eps_coarse = diag * eps_coarse_ratio
db         = DBSCAN(eps=eps_coarse, min_samples=min_coarse_pts).fit(pts0)
mask_core  = db.labels_ >= 0
core_pts   = filtered[mask_core]
if len(core_pts) == 0:
    raise RuntimeError("No core dynamic points after coarse DBSCAN")

# ================ Graph-based largest connected component ================
tree2  = KDTree(coords[0], leaf_size=40)
r_conn = diag * connect_radius
nbrs2  = tree2.query_radius(coords[0, core_pts], r=r_conn)

G = nx.Graph()
G.add_nodes_from(range(len(core_pts)))
for i, nbr in enumerate(nbrs2):
    for j in nbr:
        idx_j = np.where(core_pts == j)[0]
        if idx_j.size:
            G.add_edge(i, idx_j[0])

ccs     = list(nx.connected_components(G))
main_cc = max(ccs, key=len)
main_idx= core_pts[list(main_cc)]

# ================ Compute ROI bounding box (with padding) ================
pts_main = coords[0, main_idx]
xmin, ymin, zmin = pts_main.min(axis=0)
xmax, ymax, zmax = pts_main.max(axis=0)
pad = diag * pad_ratio
xmin, xmax = xmin - pad, xmax + pad
ymin, ymax = ymin - pad, ymax + pad
zmin, zmax = zmin - pad, zmax + pad

print("ROI bounds:")
print(f"  x: [{xmin:.4f}, {xmax:.4f}]")
print(f"  y: [{ymin:.4f}, {ymax:.4f}]")
print(f"  z: [{zmin:.4f}, {zmax:.4f}]")

# ================ Compute fixed indices on frame 0 ================
ply0 = PlyData.read(ply_paths[0])
v0   = ply0['vertex'].data
xyz0 = np.vstack([v0['x'], v0['y'], v0['z']]).T
mask0 = (
    (xyz0[:,0] >= xmin) & (xyz0[:,0] <= xmax) &
    (xyz0[:,1] >= ymin) & (xyz0[:,1] <= ymax) &
    (xyz0[:,2] >= zmin) & (xyz0[:,2] <= zmax)
)
indices = np.nonzero(mask0)[0]
print(f"Number of Gaussians inside ROI (frame 0): {len(indices)}")

# ================ Batch-crop all PLYs using fixed indices ================
for path in ply_paths:
    ply   = PlyData.read(path)
    verts = ply['vertex'].data
    sel   = verts[indices]  # fixed-index cropping
    elem  = PlyElement.describe(sel, 'vertex')
    new_ply = PlyData([elem], text=False)
    out_p = os.path.join(out_dir, os.path.basename(path))
    new_ply.write(out_p)

print("Batch cropping complete. Cropped PLYs in:", out_dir)
