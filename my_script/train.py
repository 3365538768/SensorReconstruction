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
# 0. 公共函数: PLY 读写
# =========================================
def write_ply(path, pts):
    """保存 N x 3 数组为 ASCII PLY 文件"""
    verts = np.core.records.fromarrays(pts.T,
                                       names='x,y,z',
                                       formats='f4,f4,f4')
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=True).write(path)

# =========================================
# 1. 数据预处理 & Dataset
# =========================================
class DeformDataset(Dataset):
    def __init__(self, data_dir, cage_res=(12,12,12), sensor_res=(16,16)):
        """
        data_dir: 数据路径
        cage_res: 笼网格分辨率 (nx,ny,nz)
        sensor_res: 传感器分辨率 (H, W)
        """
        self.data_dir = data_dir
        self.cage_res = cage_res
        self.sensor_res = sensor_res  # (H, W)

        # 读取传感器 CSV
        raw = pd.read_csv(os.path.join(data_dir,'sensor.csv'), header=None)
        # 检查 CSV 列数是否和指定分辨率匹配
        expected_cols = sensor_res[0] * sensor_res[1]
        assert raw.shape[1] - 1 == expected_cols, \
            f"sensor.csv 列数 {raw.shape[1]-1} ≠ {sensor_res[0]}×{sensor_res[1]}"

        col0 = pd.to_numeric(raw.iloc[:,0], errors='coerce')
        valid = col0.notna().values
        self.frames = col0[valid].astype(int).values
        self.sensors = raw[valid].iloc[:,1:].values.astype(np.float32)

        # 读取 region.json
        cfg = json.load(open(os.path.join(data_dir,'region.json')))
        self.bbox_min = np.array(cfg['bbox'][0],dtype=np.float32)
        self.bbox_max = np.array(cfg['bbox'][1],dtype=np.float32)
        self.normal   = np.array(cfg['normal'],dtype=np.float32)

        # 收集 PLY 文件
        files = sorted(glob.glob(os.path.join(data_dir,'frames','*.ply')))
        self.ply = {int(os.path.basename(f).split('_')[1].split('.')[0]): f for f in files}

        # 初始帧归一化
        pts0 = self._load_ply(self.ply[self.frames[0]])
        mask0 = self._in_box(pts0)
        self.idx0 = np.where(mask0)[0]
        self._compute_norm(pts0[self.idx0])

        # 构建笼节点
        cage = self._build_cage(cage_res).astype(np.float32)
        self.cage_coords = cage

        # 规范化基准点云 & 计算权重
        self.norm_init = self.norm_init  # numpy
        self.weights = self._compute_weights(self.norm_init, cage)

        # 转为张量
        self.norm_init_tensor = torch.from_numpy(self.norm_init)
        self.translate        = torch.from_numpy(self.translate)
        self.rot_scale        = torch.from_numpy(self.rot_scale)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # 传感器数据 reshape
        H, W = self.sensor_res
        sensor = torch.from_numpy(
            self.sensors[idx].reshape(1, H, W)
        )

        # 时间归一化
        t_frame = float(self.frames[idx]) / float(self.frames.max())
        t_norm = torch.tensor(t_frame, dtype=torch.float32)

        # 真实偏移
        pts = self._load_ply(self.ply[self.frames[idx]])[self.idx0]
        pts = (torch.from_numpy(pts) + self.translate) @ self.rot_scale
        gt_def = pts - self.norm_init_tensor

        return sensor, gt_def, t_norm

    def _load_ply(self, path):
        ply = PlyData.read(path)
        pts = np.vstack([ply['vertex'][k] for k in ['x','y','z']]).T
        return pts.astype(np.float32)

    def _in_box(self, pts):
        return np.all((pts>=self.bbox_min)&(pts<=self.bbox_max),axis=1)

    def _compute_norm(self, pts):
        c = (self.bbox_min + self.bbox_max)/2
        pc = pts - c
        n = self.normal/np.linalg.norm(self.normal)
        z = np.array([0,0,1],dtype=np.float32)
        v = np.cross(n,z); s = np.linalg.norm(v); c0 = n.dot(z)
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]],dtype=np.float32)
        R = np.eye(3,dtype=np.float32) + vx + vx.dot(vx)*((1-c0)/(s**2+1e-8))
        pr = pc.dot(R.T)
        S = 1.0/(2*np.abs(pr).max()+1e-8)
        self.translate = -c.astype(np.float32)
        self.rot_scale = (R*S).T.astype(np.float32)
        self.norm_init = pr * S

    def _build_cage(self, res):
        xs,ys,zs = [np.linspace(-.5,.5,n) for n in res]
        grid = np.stack(np.meshgrid(xs,ys,zs,indexing='ij'),-1)
        return grid.reshape(-1,3)

    def _compute_weights(self, pts, cage):
        d = np.linalg.norm(pts[:,None,:] - cage[None,:,:], axis=2)
        inv = 1.0/(d+1e-6)
        w = inv / inv.sum(axis=1,keepdims=True)
        return w.astype(np.float32)


# =========================================
# 2. 模型定义
# =========================================
class TimeEncoding(nn.Module):
    def __init__(self, num_bands=6):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer('freqs', freqs)
    def forward(self, t):
        out = [t.unsqueeze(1)]
        for f in self.freqs:
            out.append(torch.sin(t.unsqueeze(1)*f))
            out.append(torch.cos(t.unsqueeze(1)*f))
        return torch.cat(out, dim=1)

class FourierEncoding(nn.Module):
    def __init__(self, num_bands=8):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer('freqs', freqs)
    def forward(self, coords):
        out = [coords]
        for f in self.freqs:
            out.append(torch.sin(coords*f))
            out.append(torch.cos(coords*f))
        return torch.cat(out, dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels,channels,3,padding=1,bias=False),
            nn.BatchNorm2d(channels),nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,3,padding=1,bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.layers(x)+x)

class EnhancedSensorEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1,128,7,padding=3,bias=False),
            nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.MaxPool2d(2)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(7)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,out_dim),nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x)

class GNNDeformer(nn.Module):
    def __init__(self, input_dim, cage_nodes, edge_index, fourier_dim):
        super().__init__()
        self.register_buffer('edge_index', edge_index)
        h1,h2,h3 = 512,256,128
        self.node_init = nn.Linear(input_dim+fourier_dim, h1)
        self.conv1 = GATConv(h1, h2//4, heads=4)
        self.conv2 = GATConv(h2, h3//4, heads=4)
        self.conv3 = GraphConv(h3, h3)
        self.fc_out = nn.Sequential(nn.Linear(h3,h3),nn.ReLU(inplace=True),nn.Linear(h3,3))
    def forward(self, feat, fcoords):
        B,K = feat.size(0), fcoords.size(0)
        fexp = feat.unsqueeze(1).expand(B,K,-1)
        cexp = fcoords.unsqueeze(0).expand(B,K,-1)
        h = torch.cat([fexp,cexp],dim=2)
        h = self.node_init(h)
        outs=[]
        for b in range(B):
            x = F.relu(self.conv1(h[b],self.edge_index))
            x = F.relu(self.conv2(x,self.edge_index))
            x = F.relu(self.conv3(x,self.edge_index))
            outs.append(self.fc_out(x))
        return torch.stack(outs,dim=0)

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
        feat = self.encoder(sensor)            # (B, sensor_dim)
        tfeat = self.timeenc(t_norm)          # (B, time_dim)
        x = torch.cat([feat, tfeat], dim=1)   # (B, sensor_dim+time_dim)
        fcoords = self.fourier(self.cage_coords)  
        return self.deformer(x, fcoords)      # (B, K, 3)

# =========================================
# 3. 训练 & 推理
# =========================================
def train_and_infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- dataset & adjacency ---
    ds = DeformDataset(
        args.data_dir,
        cage_res=tuple(args.cage_res),
        sensor_res=tuple(args.sensor_res)
    )

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
                            u = int(idx[i,j,k])
                            v = int(idx[ni,nj,nk])
                            edges += [(u, v), (v, u)]
        u, v = zip(*edges)
        return torch.tensor([u, v], dtype=torch.long)

    edge_index = build_edge_index(tuple(args.cage_res)).to(device)
    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers)

    # --- model, optimizer, loss ---
    model = DeformModelEnhanced(
        sensor_dim=args.sensor_dim,
        cage_coords=ds.cage_coords,
        edge_index=edge_index,
        num_fourier_bands=args.num_fourier_bands,
        num_time_bands=args.num_time_bands
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # --- output dirs ---
    bbox_dir  = os.path.join(args.out_dir, 'cropped_bbox')
    cages_dir = os.path.join(args.out_dir, 'cages_pred')
    objs_dir  = os.path.join(args.out_dir, 'objects_world')
    os.makedirs(bbox_dir,  exist_ok=True)
    os.makedirs(cages_dir, exist_ok=True)
    os.makedirs(objs_dir,  exist_ok=True)

    # --- training loop ---
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for sensor, gt_def, t_norm in tqdm(dl, desc=f'Epoch {epoch+1}'):
            sensor = sensor.to(device)
            gt_def  = gt_def.to(device)
            t_norm  = t_norm.to(device)

            delta = model(sensor, t_norm)  # (B, K, 3)

            # reconstruct point-wise def: (B, N_pts, 3)
            W = torch.from_numpy(ds.weights).to(device)    # (N_pts, K)
            pred_def = torch.einsum('nk,bkd->bnd', W, delta)

            loss = mse(pred_def, gt_def)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg = total_loss / len(dl)
        print(f'Epoch {epoch+1} avg_loss: {avg:.6f}')

    save_path = os.path.join(args.out_dir, 'deform_model_final.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    # --- inference & save PLYs ---
    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            sensor, gt_def, t_norm = ds[idx]
            frame = ds.frames[idx]

            # crop bbox
            raw = ds._load_ply(ds.ply[frame])
            mask = ds._in_box(raw)
            write_ply(os.path.join(bbox_dir, f'crop_{idx:05d}.ply'), raw[mask])

            # predict
            s = sensor.unsqueeze(0).to(device)
            t = t_norm.unsqueeze(0).to(device)
            d = model(s, t).squeeze(0).cpu().numpy()  # (K,3)

            # save deformed cage
            cage_p = ds.cage_coords + d
            write_ply(os.path.join(cages_dir, f'cage_{idx:05d}.ply'), cage_p)

            # reconstruct object in world coords
            pred_norm = ds.norm_init + (ds.weights @ d)  # (N_pts,3)
            inv_RS    = np.linalg.inv(ds.rot_scale.numpy())
            world     = pred_norm.dot(inv_RS) - ds.translate.numpy()
            write_ply(os.path.join(objs_dir, f'object_{idx:05d}.ply'), world)

    print('Done')

if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/experiment1')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--sensor_dim', type=int, default=512)
    p.add_argument('--cage_res', nargs=3, type=int, default=[15,15,15])
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--out_dir', type=str, default='outputs/experiment1')
    p.add_argument('--num_fourier_bands', type=int, default=8)
    p.add_argument('--num_time_bands', type=int, default=6)
    p.add_argument('--sensor_res', nargs=2, type=int, default=[10,10],
                   help='传感器网格的高和宽，例如 16 16')
    args = p.parse_args()
    train_and_infer(args)