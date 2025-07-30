import os
import glob
import json
import re
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch_geometric.nn import GraphConv, GATConv


# =========================================
# 0. 公共函数: PLY 读写
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

def _load_ply(path):
    """从PLY文件加载顶点数据和完整的结构化数组"""
    ply = PlyData.read(path)
    vertex_data = ply['vertex']
    pts_xyz = np.vstack([vertex_data[k] for k in ['x','y','z']]).T.astype(np.float32)
    full_structured_array = np.array(vertex_data.data, dtype=vertex_data.data.dtype)
    return pts_xyz, full_structured_array


# =========================================
# 1. 模型定义 (与原始代码相同)
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
# 2. 辅助函数: 边索引 & 融合权重
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
    dist_min = (outer_bbox_min - points_xyz) / (falloff_distance + 1e-8)
    dist_max = (points_xyz - outer_bbox_max) / (falloff_distance + 1e-8)
    dist = torch.max(torch.cat([dist_min, dist_max], dim=1), dim=1).values
    weights = 1.0 - smoothstep(0.0, 1.0, dist)
    return weights


# =========================================
# 3. 新增: 区域数据处理器
# =========================================
class RegionProcessor:
    """封装单个区域的数据加载和预处理逻辑"""
    def __init__(self, region_json_path, sensor_csv_path, all_pts_xyz_tensor, cage_res, sensor_res, falloff_distance, device):
        self.device = device
        self.sensor_res = sensor_res
        self.region_id = os.path.basename(region_json_path)

        # 加载 region.json
        cfg = json.load(open(region_json_path))
        self.bbox_min = torch.tensor(cfg['bbox'][0], dtype=torch.float32, device=device)
        self.bbox_max = torch.tensor(cfg['bbox'][1], dtype=torch.float32, device=device)
        self.normal = np.array(cfg['normal'], dtype=np.float32)

        # 加载 sensor.csv
        raw = pd.read_csv(sensor_csv_path, header=None)
        col0 = pd.to_numeric(raw.iloc[:, 0], errors='coerce')
        valid = col0.notna().values
        self.frames = col0[valid].astype(int).values
        sensors = raw[valid].iloc[:, 1:].values.astype(np.float32)
        H, W = self.sensor_res
        expected = H * W
        assert sensors.shape[1] == expected, f"sensor.csv 列数 {sensors.shape[1]} ≠ {H}×{W} for {sensor_csv_path}"
        self.sensors_np = sensors
        self.frame_to_sensor_idx = {frame: i for i, frame in enumerate(self.frames)}

        # 计算该区域的归一化变换
        self._compute_norm(all_pts_xyz_tensor)

        # 根据此区域的变换对所有点进行归一化
        all_pts_norm = (all_pts_xyz_tensor + self.translate) @ self.rot_scale

        # 构建笼
        self.cage_coords = torch.from_numpy(self._build_cage(cage_res).astype(np.float32)).to(device)

        # 预计算所有点到此区域笼的权重
        self.all_weights = self._compute_weights_torch(all_pts_norm, self.cage_coords)

        # 预计算基于此区域bbox的所有点的融合权重
        self.blending_weights = calculate_blending_weights(
            all_pts_xyz_tensor, self.bbox_min, self.bbox_max, falloff_distance
        ).unsqueeze(1)

    def _compute_norm(self, all_pts_xyz):
        """计算从世界坐标到归一化坐标的变换"""
        c = (self.bbox_min.cpu().numpy() + self.bbox_max.cpu().numpy()) / 2.0
        
        # 仅使用包围盒内的点来计算旋转和缩放，以匹配原始逻辑
        in_box_mask = torch.all((all_pts_xyz >= self.bbox_min) & (all_pts_xyz <= self.bbox_max), axis=1)
        pts_in_box = all_pts_xyz[in_box_mask].cpu().numpy()
        if len(pts_in_box) == 0: # 如果盒子为空，则使用bbox本身来确定缩放
            pts_in_box = self.bbox_min.cpu().numpy().reshape(1,3)

        pc = pts_in_box - c
        n = self.normal / np.linalg.norm(self.normal)
        z = np.array([0,0,1], dtype=np.float32)
        v = np.cross(n, z); s = np.linalg.norm(v); c0 = n.dot(z)
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + vx + vx.dot(vx) * ((1-c0)/(s**2+1e-8))
        pr = pc.dot(R.T)
        S = 1.0/(2*np.abs(pr).max()+1e-8)
        
        self.translate = torch.from_numpy(-c.astype(np.float32)).to(self.device)
        self.rot_scale = torch.from_numpy((R*S).T.astype(np.float32)).to(self.device)
        self.inv_RS = torch.inverse(self.rot_scale)

    def _build_cage(self, res):
        xs, ys, zs = [np.linspace(-.5, .5, n) for n in res]
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1)
        return grid.reshape(-1, 3)

    def _compute_weights_torch(self, pts, cage):
        d = torch.cdist(pts, cage)
        inv = 1.0 / (d + 1e-8)
        return inv / inv.sum(axis=1, keepdims=True)

    def get_sensor_data_for_frame(self, frame_id):
        if frame_id not in self.frame_to_sensor_idx:
            return None, None
        
        idx = self.frame_to_sensor_idx[frame_id]
        H, W = self.sensor_res
        sensor = torch.from_numpy(self.sensors_np[idx].reshape(1, H, W)).to(self.device)
        
        max_frame = float(self.frames.max()) if len(self.frames) > 0 else 1.0
        t_frame = float(frame_id) / max_frame
        t_norm = torch.tensor(t_frame, dtype=torch.float32).to(self.device)
        
        return sensor, t_norm


# =========================================
# 4. 更新后的多区域推理函数
# =========================================
def inference_multi_region(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. 发现区域文件 ---
    region_files = sorted(glob.glob(os.path.join(args.data_dir, 'region_*.json')))
    if not region_files:
        print(f"Error: No 'region_*.json' files found in '{args.data_dir}'.")
        return

    regions_data_paths = []
    for region_file in region_files:
        match = re.search(r'region_(\d+).json', os.path.basename(region_file))
        if not match: continue
        idx = match.group(1)
        sensor_file = os.path.join(args.data_dir, f'sensor_{idx}.csv')
        if os.path.exists(sensor_file):
            regions_data_paths.append({'region_json': region_file, 'sensor_csv': sensor_file})
        else:
            print(f"Warning: Found {os.path.basename(region_file)} but missing corresponding sensor_{idx}.csv.")
    
    print(f"Found {len(regions_data_paths)} region(s) to process.")

    # --- 2. 全局数据和模型加载 ---
    print(f"Loading initial mesh from {args.init_ply_path}...")
    all_pts_xyz_np, all_full_ply_data = _load_ply(args.init_ply_path)
    all_pts_xyz_tensor = torch.from_numpy(all_pts_xyz_np).to(device)

    print("Initializing model...")
    edge_index = build_edge_index(tuple(args.cage_res)).to(device)
    # 笼坐标对于所有区域都是相同的归一化坐标，因此我们可以即时构建一个用于模型初始化
    temp_cage_coords_np = RegionProcessor._build_cage(None, tuple(args.cage_res))

    model = DeformModelEnhanced(
        sensor_dim=args.sensor_dim,
        cage_coords=temp_cage_coords_np,
        edge_index=edge_index,
        num_fourier_bands=args.num_fourier_bands,
        num_time_bands=args.num_time_bands
    ).to(device)

    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}")
        return
    print(f"Loading weights from: {args.model_path}")
    raw_state = torch.load(args.model_path, map_location=device)
    mapped_state = {}
    for k, v in raw_state.items():
        if '.att_src' in k: nk = k.replace('.att_src', '.att_l')
        elif '.att_dst' in k: nk = k.replace('.att_dst', '.att_r')
        elif '.lin_src.weight' in k: nk = k.replace('.lin_src.weight', '.lin_l.weight')
        elif '.lin_dst.weight' in k: nk = k.replace('.lin_dst.weight', '.lin_r.weight')
        elif '.lin_rel.weight' in k: nk = k.replace('.lin_rel.weight', '.lin_l.weight')
        elif '.lin_rel.bias' in k: nk = k.replace('.lin_rel.bias', '.lin_l.bias')
        elif '.lin_root.weight' in k: nk = k.replace('.lin_root.weight', '.lin_r.weight')
        else: nk = k
        mapped_state[nk] = v
    model.load_state_dict(mapped_state, strict=False)
    model.eval()

    # --- 3. 区域特定的预处理 ---
    print("Pre-processing data for each region...")
    processors = []
    all_frames = set()
    for paths in tqdm(regions_data_paths, desc="Processing Regions"):
        proc = RegionProcessor(
            region_json_path=paths['region_json'],
            sensor_csv_path=paths['sensor_csv'],
            all_pts_xyz_tensor=all_pts_xyz_tensor,
            cage_res=tuple(args.cage_res),
            sensor_res=tuple(args.sensor_res),
            falloff_distance=args.falloff_distance,
            device=device
        )
        processors.append(proc)
        all_frames.update(proc.frames)
    
    sorted_frames = sorted(list(all_frames))
    print(f"Total unique frames to process: {len(sorted_frames)}")

    # --- 4. 主推理循环 ---
    objs_dir = os.path.join(args.out_dir, 'objects_world_combined')
    os.makedirs(objs_dir, exist_ok=True)
    print(f"Saving combined outputs to {objs_dir}")

    with torch.no_grad():
        for frame_id in tqdm(sorted_frames, desc='Inferring Frames'):
            total_world_displacement = torch.zeros_like(all_pts_xyz_tensor)

            # 为当前帧累加来自所有区域的位移
            for proc in processors:
                sensor, t_norm = proc.get_sensor_data_for_frame(frame_id)
                if sensor is None:
                    continue

                # 预测笼的偏移
                delta = model(sensor.unsqueeze(0), t_norm.unsqueeze(0)).squeeze(0)
                
                # 计算归一化空间中的形变
                deformation_norm = proc.all_weights @ delta
                
                # 应用融合权重
                blended_norm = deformation_norm * proc.blending_weights
                
                # 将位移向量转换到世界坐标系
                world_displacement = blended_norm @ proc.inv_RS
                
                # 累加世界坐标系下的位移
                total_world_displacement += world_displacement

            # 应用最终的累加位移
            deformed_pts_world = all_pts_xyz_tensor + total_world_displacement

            # 保存结果PLY文件
            deformed_np = deformed_pts_world.cpu().numpy()
            output_ply = all_full_ply_data.copy()
            output_ply['x'] = deformed_np[:, 0].astype(output_ply['x'].dtype)
            output_ply['y'] = deformed_np[:, 1].astype(output_ply['y'].dtype)
            output_ply['z'] = deformed_np[:, 2].astype(output_ply['z'].dtype)
            
            output_path = os.path.join(objs_dir, f'object_{frame_id:05d}.ply')
            write_gaussian_ply(output_path, output_ply)

    print('Inference complete.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Region Deformation Prediction Inference Script")
    parser.add_argument('--data_dir', type=str, default='test', help="Directory with sensor_*.csv & region_*.json files")
    parser.add_argument('--init_ply_path', type=str,
                        default='test/time_00000.ply', help='Path to the initial, undeformed PLY mesh')
    parser.add_argument('--model_path', type=str,
                        default=r'outputs/bend/deform_model_final.pth', help='Path to the trained model weights')
    parser.add_argument('--out_dir', type=str, default='inference_outputs/test_combined', help='Parent directory for outputs')
    parser.add_argument('--sensor_dim', type=int, default=512, help='Dimension of the sensor encoding')
    parser.add_argument('--cage_res', nargs=3, type=int, default=[15,15,15], help='Resolution of the deformation cage (e.g., 15 15 15)')
    parser.add_argument('--sensor_res', nargs=2, type=int, default=[10,10], help='Resolution of the sensor grid H W (e.g., 10 10)')
    parser.add_argument('--num_fourier_bands', type=int, default=8, help='Number of Fourier frequency bands for position encoding')
    parser.add_argument('--num_time_bands', type=int, default=6, help='Number of frequency bands for time encoding')
    parser.add_argument('--falloff_distance', type=float, default=0, help='Blending falloff distance from the bounding box')
    args = parser.parse_args()
    
    # 将 falloff_distance 设为大于 0 的值以实现平滑融合
    if args.falloff_distance <= 0:
        print("Warning: falloff_distance is 0. Deformations may have sharp edges. Consider a small positive value.")

    inference_multi_region(args)