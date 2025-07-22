import os
import glob
import json
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KDTree
from tqdm import tqdm
from torch_geometric.nn import GraphConv, GATConv


# =========================================
# 0. 公共函数: PLY 读写 (更新以支持高斯球格式)
# =========================================
def write_ply(path, pts):
    """保存 N x 3 数组为 ASCII PLY 文件 (用于笼网格等简单XYZ数据)"""
    verts = np.core.records.fromarrays(pts.T,
                                       names='x,y,z',
                                       formats='f4,f4,f4')
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=True).write(path)


def write_gaussian_ply(path, structured_pts_array):
    """
    保存结构化 numpy 数组为 PLY 文件，适用于高斯球等包含多属性的数据。
    保留所有原始属性。
    """
    el = PlyElement.describe(structured_pts_array, 'vertex')
    PlyData([el], text=False, byte_order='<').write(path)
    print(f"Saved Gaussian PLY to {path}")


# =========================================
# 1. 数据预处理 & Dataset (为推理修改)
# =========================================
class DeformDataset(Dataset):
    def __init__(self, data_dir, init_ply_path, cage_res=(12,12,12), sensor_res=(16,16)):
        self.data_dir = data_dir
        self.cage_res = cage_res
        self.sensor_res = sensor_res  # 传感器分辨率参数化
        H, W = sensor_res

        # 读取传感器 CSV
        raw = pd.read_csv(os.path.join(data_dir, 'sensor.csv'), header=None)
        col0 = pd.to_numeric(raw.iloc[:, 0], errors='coerce')
        valid = col0.notna().values
        self.frames = col0[valid].astype(int).values
        sensors = raw[valid].iloc[:, 1:].values.astype(np.float32)

        # 检查列数是否匹配 sensor_res
        expected = H * W
        assert sensors.shape[1] == expected, \
            f"sensor.csv 列数 {sensors.shape[1]} ≠ {H}×{W}"
        self.sensors = sensors

        # 读取 region.json
        cfg = json.load(open(os.path.join(data_dir, 'region.json')))
        self.bbox_min = np.array(cfg['bbox'][0], dtype=np.float32)
        self.bbox_max = np.array(cfg['bbox'][1], dtype=np.float32)
        self.normal   = np.array(cfg['normal'], dtype=np.float32)

        # 加载初始 PLY 文件并归一化
        pts0_xyz, self.init_full_ply_data = self._load_ply(init_ply_path)
        mask0 = self._in_box(pts0_xyz)
        self.idx0 = np.where(mask0)[0]
        self._compute_norm(pts0_xyz[self.idx0])

        # 构建笼节点
        cage = self._build_cage(cage_res).astype(np.float32)
        self.cage_coords = cage
        self.weights = self._compute_weights(self.norm_init, cage)
        self.norm_init_tensor = torch.from_numpy(self.norm_init)
        self.translate        = torch.from_numpy(self.translate)
        self.rot_scale        = torch.from_numpy(self.rot_scale)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        H, W = self.sensor_res
        sensor = torch.from_numpy(self.sensors[idx].reshape(1, H, W))
        t_frame = float(self.frames[idx]) / float(self.frames.max())
        t_norm = torch.tensor(t_frame, dtype=torch.float32)
        gt_def_placeholder = torch.zeros_like(self.norm_init_tensor)
        return sensor, gt_def_placeholder, t_norm

    def _load_ply(self, path):
        ply = PlyData.read(path)
        vertex_data = ply['vertex']
        pts_xyz = np.vstack([vertex_data[k] for k in ['x','y','z']]).T.astype(np.float32)
        full_structured_array = np.array(vertex_data.data, dtype=vertex_data.data.dtype)
        return pts_xyz, full_structured_array

    def _in_box(self, pts):
        return np.all((pts >= self.bbox_min) & (pts <= self.bbox_max), axis=1)

    def _compute_norm(self, pts):
        c = (self.bbox_min + self.bbox_max) / 2
        pc = pts - c
        n = self.normal / np.linalg.norm(self.normal)
        z = np.array([0,0,1], dtype=np.float32)
        v = np.cross(n, z); s = np.linalg.norm(v); c0 = n.dot(z)
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + vx + vx.dot(vx) * ((1-c0)/(s**2+1e-8))
        pr = pc.dot(R.T)
        S = 1.0/(2*np.abs(pr).max()+1e-8)
        self.translate = -c.astype(np.float32)
        self.rot_scale = (R*S).T.astype(np.float32)
        self.norm_init = pr * S

    def _build_cage(self, res):
        xs, ys, zs = [np.linspace(-.5, .5, n) for n in res]
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1)
        return grid.reshape(-1, 3)

    def _compute_weights(self, pts, cage):
        d = np.linalg.norm(pts[:, None, :] - cage[None, :, :], axis=2)
        inv = 1.0 / (d + 1e-6)
        w = inv / inv.sum(axis=1, keepdims=True)
        return w.astype(np.float32)


# =========================================
# 2. 模型定义 (与训练代码相同)
# =========================================
class TimeEncoding(nn.Module):
    def __init__(self, num_bands=6):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer('freqs', freqs)
    def forward(self, t):
        out = [t.unsqueeze(1)]
        for f in self.freqs:
            out.append(torch.sin(t.unsqueeze(1) * f))
            out.append(torch.cos(t.unsqueeze(1) * f))
        return torch.cat(out, dim=1)

class FourierEncoding(nn.Module):
    def __init__(self, num_bands=8):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer('freqs', freqs)
    def forward(self, coords):
        out = [coords]
        for f in self.freqs:
            out.append(torch.sin(coords * f))
            out.append(torch.cos(coords * f))
        return torch.cat(out, dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.layers(x) + x)

class EnhancedSensorEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 128, 7, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(7)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x)

class GNNDeformer(nn.Module):
    def __init__(self, input_dim, cage_nodes, edge_index, fourier_dim):
        super().__init__()
        self.register_buffer('edge_index', edge_index)
        h1, h2, h3 = 512, 256, 128
        self.node_init = nn.Linear(input_dim + fourier_dim, h1)
        self.conv1 = GATConv(h1, h2 // 4, heads=4)
        self.conv2 = GATConv(h2, h3 // 4, heads=4)
        self.conv3 = GraphConv(h3, h3)
        self.fc_out = nn.Sequential(
            nn.Linear(h3, h3),
            nn.ReLU(inplace=True),
            nn.Linear(h3, 3)
        )
    def forward(self, feat, fcoords):
        B, K = feat.size(0), fcoords.size(0)
        fexp = feat.unsqueeze(1).expand(B, K, -1)
        cexp = fcoords.unsqueeze(0).expand(B, K, -1)
        h = torch.cat([fexp, cexp], dim=2)
        h = self.node_init(h)
        outs = []
        for b in range(B):
            x = F.relu(self.conv1(h[b], self.edge_index))
            x = F.relu(self.conv2(x, self.edge_index))
            x = F.relu(self.conv3(x, self.edge_index))
            outs.append(self.fc_out(x))
        return torch.stack(outs, dim=0)

class DeformModelEnhanced(nn.Module):
    def __init__(self,
                 sensor_dim: int,
                 cage_coords: np.ndarray,
                 edge_index: torch.Tensor,
                 num_fourier_bands: int = 8,
                 num_time_bands: int = 6):
        super().__init__()
        self.encoder = EnhancedSensorEncoder(out_dim=sensor_dim)
        self.timeenc = TimeEncoding(num_bands=num_time_bands)
        self.fourier = FourierEncoding(num_bands=num_fourier_bands)

        cage_coords = cage_coords.astype(np.float32)
        fourier_dim = self.fourier(torch.from_numpy(cage_coords)).shape[1]
        time_dim = 1 + 2 * num_time_bands

        self.deformer = GNNDeformer(
            input_dim=sensor_dim + time_dim,
            cage_nodes=cage_coords.shape[0],
            edge_index=edge_index,
            fourier_dim=fourier_dim
        )

        self.register_buffer('cage_coords', torch.from_numpy(cage_coords))

    def forward(self, sensor: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(sensor)
        tfeat = self.timeenc(t_norm)
        x = torch.cat([feat, tfeat], dim=1)

        fcoords = self.fourier(self.cage_coords)
        return self.deformer(x, fcoords)


# =========================================
# 3. 辅助函数: 边索引 & 融合权重
# =========================================
def build_edge_index(res):
    nx, ny, nz = res
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    edges = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for di, dj, dk in [(1,0,0),(0,1,0),(0,0,1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if ni < nx and nj < ny and nk < nz:
                        u = int(idx[i,j,k]); v = int(idx[ni,nj,nk])
                        edges += [(u, v), (v, u)]
    u, v = zip(*edges)
    return torch.tensor([u, v], dtype=torch.long)


def smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def calculate_blending_weights(points_xyz, bbox_min, bbox_max, falloff_distance):
    outer_bbox_min = bbox_min - falloff_distance
    outer_bbox_max = bbox_max + falloff_distance

    dist_min = (outer_bbox_min - points_xyz) / falloff_distance
    dist_max = (points_xyz - outer_bbox_max) / falloff_distance
    dist = torch.max(torch.cat([dist_min, dist_max], dim=1), dim=1).values
    weights = 1.0 - smoothstep(0.0, 1.0, dist)
    return weights


# =========================================
# 4. 推理函数
# =========================================
def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ds = DeformDataset(
        args.data_dir,
        args.init_ply_path,
        cage_res=tuple(args.cage_res),
        sensor_res=tuple(args.sensor_res)
    )
    edge_index = build_edge_index(tuple(args.cage_res)).to(device)

    print("Initializing model...")
    model = DeformModelEnhanced(
        sensor_dim=args.sensor_dim,
        cage_coords=ds.cage_coords,
        edge_index=edge_index,
        num_fourier_bands=args.num_fourier_bands,
        num_time_bands=args.num_time_bands
    ).to(device)

    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}")
        return
    print(f"Loading weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("Pre-calculating blending data...")
    with torch.no_grad():
        all_pts_xyz_np, all_full_ply_data = ds._load_ply(args.init_ply_path)
        all_pts_xyz = torch.from_numpy(all_pts_xyz_np).to(device)

        translate = ds.translate.to(device)
        rot_scale = ds.rot_scale.to(device)
        inv_RS = torch.inverse(rot_scale)

        all_pts_norm = (all_pts_xyz + translate) @ rot_scale

        def compute_weights_torch(pts, cage):
            d = torch.cdist(pts, cage)
            inv = 1.0 / (d + 1e-8)
            return inv / inv.sum(axis=1, keepdims=True)

        cage_coords_torch = torch.from_numpy(ds.cage_coords).to(device)
        all_weights = compute_weights_torch(all_pts_norm, cage_coords_torch)

        bbox_min_torch = torch.from_numpy(ds.bbox_min).to(device)
        bbox_max_torch = torch.from_numpy(ds.bbox_max).to(device)
        blending_weights = calculate_blending_weights(
            all_pts_xyz, bbox_min_torch, bbox_max_torch, args.falloff_distance
        ).unsqueeze(1)

    cages_dir = os.path.join(args.out_dir, 'cages_pred')
    objs_dir = os.path.join(args.out_dir, 'objects_world')
    os.makedirs(cages_dir, exist_ok=True)
    os.makedirs(objs_dir, exist_ok=True)
    print(f"Saving outputs to {args.out_dir}")

    print("Starting inference...")
    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc='Inferring'):
            sensor, _, t_norm = ds[idx]
            frame_id = ds.frames[idx]

            s = sensor.unsqueeze(0).to(device)
            t = t_norm.unsqueeze(0).to(device)

            delta = model(s, t).squeeze(0)
            full_deformation_norm = all_weights @ delta
            blended_norm = full_deformation_norm * blending_weights

            deformed_pts_norm = all_pts_norm + blended_norm
            deformed_pts_world = (deformed_pts_norm @ inv_RS) - translate

            deformed_np = deformed_pts_world.cpu().numpy()
            output_ply = all_full_ply_data.copy()
            output_ply['x'] = deformed_np[:, 0].astype(output_ply['x'].dtype)
            output_ply['y'] = deformed_np[:, 1].astype(output_ply['y'].dtype)
            output_ply['z'] = deformed_np[:, 2].astype(output_ply['z'].dtype)

            write_gaussian_ply(os.path.join(objs_dir, f'object_{frame_id:05d}.ply'), output_ply)
            cage_pred = ds.cage_coords + delta.cpu().numpy()
            write_ply(os.path.join(cages_dir, f'cage_{frame_id:05d}.ply'), cage_pred)

    print('Inference complete.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Deformation Prediction Inference Script")
    parser.add_argument('--data_dir', type=str, default='experiment2', help='Directory with sensor.csv & region.json')
    parser.add_argument('--init_ply_path', type=str,
                        default='experiment2/point_cloud.ply', help='Initial PLY path')
    parser.add_argument('--model_path', type=str,
                        default='outputs/experiment1/deform_model_final.pth', help='Model weights path')
    parser.add_argument('--out_dir', type=str, default='inference_outputs/experiment2', help='Output directory')
    parser.add_argument('--sensor_dim', type=int, default=512, help='Sensor encoding dim')
    parser.add_argument('--cage_res', nargs=3, type=int, default=[15,15,15], help='Cage resolution')
    parser.add_argument('--sensor_res', nargs=2, type=int, default=[10,10], help='Sensor resolution H W')
    parser.add_argument('--num_fourier_bands', type=int, default=8, help='Fourier bands')
    parser.add_argument('--num_time_bands', type=int, default=6, help='Time encoding bands')
    parser.add_argument('--falloff_distance', type=float, default=0, help='Blending falloff distance')
    args = parser.parse_args()
    inference(args)
