
# utils.py

import os
import glob
import json
import numpy as np
import pandas as pd
from plyfile import PlyData
from tqdm import tqdm
import torch

# ... [load_region_transform 和 compute_least_squares 函数保持不变] ...
def load_region_transform(box_json):
    """Loads transformation info from region.json to normalize the point cloud."""
    with open(box_json, 'r') as f:
        cfg = json.load(f)
    bbox = np.array(cfg['bbox'], dtype=np.float32)
    center = (bbox[0] + bbox[1]) / 2.0
    n = np.array(cfg['normal'], dtype=np.float32)
    n /= (np.linalg.norm(n) + 1e-8)
    v = np.cross(n, np.array([0, 0, 1], dtype=np.float32))
    c = n.dot([0, 0, 1])
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + vx + vx.dot(vx) / (1.0 + c)
    corners = (bbox - center) @ R.T
    bb_min = corners.min(axis=0)
    bb_rng = corners.max(axis=0) - bb_min
    bb_rng[bb_rng < 1e-7] = 1.0
    return center, R, bb_min, bb_rng

def compute_least_squares(grid, b, ridge=1e-6):
    """Solves for grid motion using least squares."""
    P = b.shape[0]
    rows = torch.arange(P, device=grid.device).unsqueeze(1).expand(-1, 8).flatten()
    cols = grid.point_indices.flatten()
    vals = grid.point_weights.flatten()
    W = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (P, grid.num_nodes), device=grid.device
    ).to_dense()  # [P, N_grid]
    
    A = W.T @ W
    A += torch.eye(grid.num_nodes, device=grid.device) * ridge
    B = W.T @ b  # [N_grid, 3]
    x = torch.linalg.solve(A, B)
    return x  # [N_grid,3]


# =========================================================================================
#   Deformation Grid (修正后)
# =========================================================================================
class DeformationGrid:
    """A helper class to manage the deformation grid and its operations."""
    def __init__(self, resolution=(8,8,8), bounds=((-0.5,0.5),(-0.5,0.5),(-0.5,0.5))):
        self.resolution = resolution
        self.bounds = torch.tensor(bounds, dtype=torch.float32)
        
        # ==================== 核心修正开始 ====================
        # 在构造函数中初始化device属性，确保它始终存在
        self.device = 'cpu'
        # ==================== 核心修正结束 ====================

        gx = torch.linspace(self.bounds[0,0], self.bounds[0,1], resolution[0])
        gy = torch.linspace(self.bounds[1,0], self.bounds[1,1], resolution[1])
        gz = torch.linspace(self.bounds[2,0], self.bounds[2,1], resolution[2])
        gy, gx, gz = torch.meshgrid(gy, gx, gz, indexing='ij')
        self.nodes = torch.stack([gx, gy, gz], dim=-1).reshape(-1,3)
        self.num_nodes = self.nodes.shape[0]
        self.cell_size = (self.bounds[:,1] - self.bounds[:,0]) / torch.tensor(resolution, dtype=torch.float32)
        self.point_indices = None
        self.point_weights = None

    def to(self, device):
        self.device = device
        self.nodes = self.nodes.to(device)
        self.bounds = self.bounds.to(device)
        self.cell_size = self.cell_size.to(device)
        # 如果权重已经计算，也一并移动
        if self.point_indices is not None:
            self.point_indices = self.point_indices.to(device)
        if self.point_weights is not None:
            self.point_weights = self.point_weights.to(device)
        return self

    def find_interpolation_weights(self, points):
        # 这一行现在可以安全地执行了
        points = points.to(self.device)
        
        scaled = (points - self.bounds[:,0]) / self.cell_size
        floor = torch.floor(scaled).long()
        rx, ry, rz = self.resolution
        floor[:,0].clamp_(0, rx-2)
        floor[:,1].clamp_(0, ry-2)
        floor[:,2].clamp_(0, rz-2)
        
        local = scaled - floor.float()
        u, v, w = local[:,0], local[:,1], local[:,2]
        w000 = (1-u)*(1-v)*(1-w); w100 = u*(1-v)*(1-w)
        w010 = (1-u)*v*(1-w);     w001 = (1-u)*(1-v)*w
        w110 = u*v*(1-w);         w101 = u*(1-v)*w
        w011 = (1-u)*v*w;         w111 = u*v*w
        self.point_weights = torch.stack([w000,w100,w010,w001,w110,w101,w011,w111], dim=-1)

        i, j, k = floor[:,0], floor[:,1], floor[:,2]
        idx000 = i + j*rx + k*rx*ry
        idx100 = (i+1) + j*rx + k*rx*ry
        idx010 = i + (j+1)*rx + k*rx*ry
        idx001 = i + j*rx + (k+1)*rx*ry
        idx110 = (i+1) + (j+1)*rx + k*rx*ry
        idx101 = (i+1) + j*rx + (k+1)*rx*ry
        idx011 = i + (j+1)*rx + (k+1)*rx*ry
        idx111 = (i+1) + (j+1)*rx + (k+1)*rx*ry
        self.point_indices = torch.stack(
            [idx000,idx100,idx010,idx001,idx110,idx101,idx011,idx111],
            dim=-1
        )

    def deform_points(self, pts, velocities):
        # 确保权重和索引在正确的设备上
        if self.point_indices.device != velocities.device:
            self.point_indices = self.point_indices.to(velocities.device)
            self.point_weights = self.point_weights.to(velocities.device)
            
        corner_vel = velocities[self.point_indices]  # [P,8,3]
        disp = (corner_vel * self.point_weights.unsqueeze(-1)).sum(dim=1)
        return pts + disp

# ... [load_and_prepare_data 函数保持不变] ...
def load_and_prepare_data(scene_path, device, grid):
    center, R, bb_min, bb_rng = load_region_transform(os.path.join(scene_path,'region.json'))
    with open(os.path.join(scene_path,'region.json'),'r') as f:
        bbox_coords = np.array(json.load(f)['bbox'], dtype=np.float32)
    bmin, bmax = bbox_coords[0], bbox_coords[1]

    # load point clouds
    ply_paths = sorted(glob.glob(f"{scene_path}/frames/time_*.ply"))
    pos_list = []
    for p in tqdm(ply_paths, desc="Preprocess PLYs"):
        ply = PlyData.read(p)
        pts = np.vstack([
            ply['vertex'].data['x'],
            ply['vertex'].data['y'],
            ply['vertex'].data['z']
        ]).T
        mask = np.all((pts>=bmin)&(pts<=bmax), axis=1)
        unit = (((pts[mask]-center)@R.T - bb_min)/(bb_rng+1e-8)) - 0.5
        pos_list.append(torch.from_numpy(unit).float())
    Pmin = min(p.shape[0] for p in pos_list)
    all_pos = [p[:Pmin].to(device) for p in pos_list]

    # load sensors
    df = pd.read_csv(f"{scene_path}/sensor.csv")
    all_sens = [
        torch.from_numpy(row.iloc[1:].values.astype(np.float32)
                         .reshape(1,1,16,16)/255.0).to(device)
        for _, row in df.iterrows()
    ]

    # setup grid
    grid.find_interpolation_weights(all_pos[0])

    # compute GT grid positions and velocities via least squares
    all_pos_gt = []
    all_vel_gt = [torch.zeros_like(grid.nodes)]
    for t in tqdm(range(len(all_pos)), desc="Compute GT grid"):
        X_t = compute_least_squares(grid, all_pos[t])  # grid positions
        all_pos_gt.append(X_t)
        if t > 0:
            vel_t = compute_least_squares(grid, all_pos[t] - all_pos[t-1])
            all_vel_gt.append(vel_t)

    # normalize velocities
    vel_tensor = torch.stack(all_vel_gt)
    vel_mean   = vel_tensor.mean(dim=[0,1])
    vel_std    = vel_tensor.std (dim=[0,1])
    vel_std[vel_std<1e-8] = 1.0
    all_vel_norm = [(v - vel_mean)/vel_std for v in all_vel_gt]

    # normalize positions
    pos_tensor = torch.stack(all_pos_gt)
    pos_mean   = pos_tensor.mean(dim=[0,1])
    pos_std    = pos_tensor.std (dim=[0,1])
    pos_std[pos_std<1e-8] = 1.0
    all_pos_norm = [(X - pos_mean)/pos_std for X in all_pos_gt]

    return (
        all_pos, all_sens,
        all_vel_norm, all_pos_norm,
        vel_mean, vel_std, pos_mean, pos_std
    )