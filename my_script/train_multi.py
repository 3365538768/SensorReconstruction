import os
import glob
import json
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
import logging
import datetime
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch_geometric.nn import GraphConv, GATConv

# =========================================
# 0. 公共函数: PLY 读写 和 日志
# =========================================
def write_ply(path, pts):
    """保存 N x 3 数组为 ASCII PLY 文件"""
    verts = np.core.records.fromarrays(pts.T,
                                       names='x,y,z',
                                       formats='f4,f4,f4')
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=True).write(path)

class TrainingLogger:
    """一个用于记录训练过程的简单日志记录器。"""
    def __init__(self, model_name, experiment_name):
        self.log_dir = os.path.join("logs", model_name, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_file = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = os.path.join(self.log_dir, log_file)

        self.logger = logging.getLogger(f"{model_name}_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # 使用 utf-8 编码以获得更好的兼容性
        fh = logging.FileHandler(self.log_path, encoding='utf-8')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            
        self.metrics = {"epochs": []}
        self.config = {}
        self.summary = {}

    def log_config(self, config_dict):
        self.config = config_dict
        self.logger.info("="*30 + "\nTRAINING CONFIGURATION\n" + "="*30)
        for key, value in config_dict.items(): self.logger.info(f"{key}: {value}")
        self.logger.info("="*30)

    def log_training_start(self, dataset_size, batch_size, total_epochs, model_parameters):
        self.summary['start_time'] = datetime.datetime.now().isoformat()
        # 移除了表情符号以避免 UnicodeEncodeError
        self.logger.info(f"Training started! Dataset size: {dataset_size}, Batch size: {batch_size}, Epochs: {total_epochs}, Model params: {model_parameters}")

    def log_epoch_stats(self, epoch, avg_loss, min_loss, max_loss):
        epoch_data = {"epoch": epoch, "avg_loss": avg_loss, "min_loss": min_loss, "max_loss": max_loss}
        self.metrics["epochs"].append(epoch_data)
        self.logger.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f} | Min Loss: {min_loss:.6f} | Max Loss: {max_loss:.6f}")

    def log_training_complete(self, model_save_path, output_directories, inference_stats):
        self.summary.update({
            'end_time': datetime.datetime.now().isoformat(),
            'model_saved_at': model_save_path,
            'output_directories': output_directories,
            'inference_stats': inference_stats
        })
        self.logger.info(f"Training complete! Model saved to: {model_save_path}")
        for key, value in inference_stats.items(): self.logger.info(f"  - Generated {value} {key} files")

    def save_metrics(self):
        metrics_path = os.path.join(self.log_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f: json.dump(self.metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_path}")

    def get_log_summary(self):
        final_summary = self.config.copy()
        final_summary.update(self.summary)
        if self.metrics["epochs"]: final_summary['final_avg_loss'] = self.metrics["epochs"][-1]['avg_loss']
        return final_summary

def create_training_logger(model_name, experiment_name):
    return TrainingLogger(model_name, experiment_name)


# =========================================
# 1. 数据预处理 & Dataset (Multi-Region)
# =========================================
class DeformDataset(Dataset):
    def __init__(self, data_dir, cage_res=(12,12,12), sensor_res=(16,16)):
        self.data_dir = data_dir
        self.cage_res = cage_res
        self.sensor_res = sensor_res
        
        region_files = sorted(glob.glob(os.path.join(data_dir, 'region*.json')))
        sensor_files = sorted(glob.glob(os.path.join(data_dir, 'sensor*.csv')))
        
        def get_key(filepath):
            match = re.search(r'(region|sensor)(\d*)\.(json|csv)', os.path.basename(filepath))
            return int(match.group(2)) if match.group(2) else 0
        
        region_map = {get_key(f): f for f in region_files}
        sensor_map = {get_key(f): f for f in sensor_files}
        
        assert region_map.keys() == sensor_map.keys(), "Mismatch between region and sensor files."
        
        self.regions = {}
        self.samples = []

        ply_files = sorted(glob.glob(os.path.join(data_dir, 'frames', '*.ply')))
        self.ply_map = {int(os.path.basename(f).split('_')[1].split('.')[0]): f for f in ply_files}
        max_frame = max(self.ply_map.keys()) if self.ply_map else 0

        for region_id in sorted(region_map.keys()):
            region_data = self._load_region_data(region_id, region_map[region_id], sensor_map[region_id], max_frame)
            self.regions[region_id] = region_data
            
            for i in range(len(region_data['frames'])):
                self.samples.append({'region_id': region_id, 'sample_idx': i})

    def _load_region_data(self, region_id, region_path, sensor_path, max_frame):
        cfg = json.load(open(region_path))
        bbox_min = np.array(cfg['bbox'][0], dtype=np.float32)
        bbox_max = np.array(cfg['bbox'][1], dtype=np.float32)
        normal = np.array(cfg['normal'], dtype=np.float32)
        
        raw = pd.read_csv(sensor_path, header=None)
        expected_cols = self.sensor_res[0] * self.sensor_res[1]
        assert raw.shape[1] - 1 == expected_cols, f"Sensor file {sensor_path} has wrong column count."
        
        col0 = pd.to_numeric(raw.iloc[:,0], errors='coerce')
        valid = col0.notna().values
        frames = col0[valid].astype(int).values
        sensors = raw[valid].iloc[:,1:].values.astype(np.float32)
        
        if not frames.size:
             raise ValueError(f"No valid frames found in {sensor_path} for region {region_id}")

        pts0 = self._load_ply(self.ply_map[frames[0]])
        mask0 = self._in_box(pts0, bbox_min, bbox_max)
        idx0 = np.where(mask0)[0]
        
        if not idx0.size > 0:
            raise ValueError(f"No points found inside bbox for initial frame of region {region_id}")

        translate, rot_scale, norm_init = self._compute_norm(pts0[idx0], bbox_min, bbox_max, normal)

        cage_coords = self._build_cage(self.cage_res).astype(np.float32)
        weights = self._compute_weights(norm_init, cage_coords)

        return {
            'frames': frames,
            'sensors': sensors,
            'idx0': idx0,
            'translate': torch.from_numpy(translate),
            'rot_scale': torch.from_numpy(rot_scale),
            'norm_init': torch.from_numpy(norm_init),
            'weights': torch.from_numpy(weights),
            'max_frame': max_frame
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        region_id = sample_info['region_id']
        sample_idx = sample_info['sample_idx']
        
        region = self.regions[region_id]
        
        H, W = self.sensor_res
        sensor = torch.from_numpy(region['sensors'][sample_idx].reshape(1, H, W))
        
        t_frame = float(region['frames'][sample_idx]) / float(region['max_frame']) if region['max_frame'] > 0 else 0.0
        t_norm = torch.tensor(t_frame, dtype=torch.float32)
        
        frame_num = region['frames'][sample_idx]
        pts = self._load_ply(self.ply_map[frame_num])[region['idx0']]
        pts = (torch.from_numpy(pts) + region['translate']) @ region['rot_scale']
        gt_def = pts - region['norm_init']
        
        return region_id, sensor, gt_def, t_norm

    def _load_ply(self, path):
        ply = PlyData.read(path)
        pts = np.vstack([ply['vertex'][k] for k in ['x','y','z']]).T
        return pts.astype(np.float32)

    def _in_box(self, pts, bbox_min, bbox_max):
        return np.all((pts >= bbox_min) & (pts <= bbox_max), axis=1)

    def _compute_norm(self, pts, bbox_min, bbox_max, normal):
        c = (bbox_min + bbox_max) / 2
        pc = pts - c
        n = normal / np.linalg.norm(normal)
        z = np.array([0,0,1], dtype=np.float32)
        v = np.cross(n, z); s = np.linalg.norm(v); c0 = n.dot(z)
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + vx + vx.dot(vx) * ((1-c0) / (s**2 + 1e-8))
        pr = pc.dot(R.T)
        S = 1.0 / (2 * np.abs(pr).max() + 1e-8)
        translate = -c.astype(np.float32)
        rot_scale = (R * S).T.astype(np.float32)
        norm_init = pr * S
        return translate, rot_scale, norm_init

    def _build_cage(self, res):
        xs,ys,zs = [np.linspace(-.5,.5,n) for n in res]
        grid = np.stack(np.meshgrid(xs,ys,zs,indexing='ij'),-1)
        return grid.reshape(-1,3)

    def _compute_weights(self, pts, cage):
        d = np.linalg.norm(pts[:,None,:] - cage[None,:,:], axis=2)
        inv = 1.0 / (d + 1e-6)
        w = inv / inv.sum(axis=1, keepdims=True)
        return w.astype(np.float32)

# =========================================
# 2. 模型定义 (无变化)
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

        fourier_dim = self.fourier(torch.from_numpy(cage_coords.astype(np.float32))).shape[1]
        time_dim = 1 + 2 * num_time_bands

        self.deformer = GNNDeformer(
            input_dim=sensor_dim + time_dim,
            cage_nodes=cage_coords.shape[0],
            edge_index=edge_index,
            fourier_dim=fourier_dim
        )
        self.register_buffer('cage_coords', torch.from_numpy(cage_coords.astype(np.float32)))

    def forward(self, sensor: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(sensor)
        tfeat = self.timeenc(t_norm)
        x = torch.cat([feat, tfeat], dim=1)
        fcoords = self.fourier(self.cage_coords)  
        return self.deformer(x, fcoords)


# =========================================
# 3. 辅助函数 (移至顶层)
# =========================================
def build_edge_index(res):
    """为 GNN 构建边索引。"""
    nx, ny, nz = res
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    edges = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for di, dj, dk in [(1,0,0),(0,1,0),(0,0,1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if ni < nx and nj < ny and nk < nz:
                        u, v = int(idx[i,j,k]), int(idx[ni,nj,nk])
                        edges += [(u, v), (v, u)]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(list(zip(*edges)), dtype=torch.long)

def collate_fn(batch):
    """自定义整理函数，按 region_id 分组。"""
    # 使用标准字典代替 defaultdict(lambda) 以解决 pickle 问题
    grouped_batch = {}
    for region_id, sensor, gt_def, t_norm in batch:
        if region_id not in grouped_batch:
            grouped_batch[region_id] = {'sensors': [], 'gt_defs': [], 't_norms': []}
        
        grouped_batch[region_id]['sensors'].append(sensor)
        grouped_batch[region_id]['gt_defs'].append(gt_def)
        grouped_batch[region_id]['t_norms'].append(t_norm)
    
    for region_id, data in grouped_batch.items():
        data['sensors'] = torch.stack(data['sensors'])
        data['gt_defs'] = torch.stack(data['gt_defs'])
        data['t_norms'] = torch.stack(data['t_norms'])
    return grouped_batch

# =========================================
# 4. 训练 & 推理 (Multi-Region)
# =========================================
def train_and_infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_name = os.path.basename(os.path.normpath(args.data_dir))
    training_logger = create_training_logger("cage_model", experiment_name)
    
    training_config = vars(args).copy()
    training_config["device"] = str(device)
    training_logger.log_config(training_config)

    # --- dataset & adjacency ---
    ds = DeformDataset(
        args.data_dir,
        cage_res=tuple(args.cage_res),
        sensor_res=tuple(args.sensor_res)
    )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    # --- model, optimizer, loss ---
    edge_index = build_edge_index(tuple(args.cage_res)).to(device)
    shared_cage_coords = ds._build_cage(tuple(args.cage_res))
    model = DeformModelEnhanced(
        sensor_dim=args.sensor_dim,
        cage_coords=shared_cage_coords,
        edge_index=edge_index,
        num_fourier_bands=args.num_fourier_bands,
        num_time_bands=args.num_time_bands
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # --- output dirs ---
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- training loop ---
    training_logger.log_training_start(len(ds), args.batch_size, args.epochs, sum(p.numel() for p in model.parameters()))
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(dl, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch in progress_bar:
            opt.zero_grad()
            total_loss = 0
            
            for region_id, data in batch.items():
                sensor, gt_def, t_norm = data['sensors'].to(device), data['gt_defs'].to(device), data['t_norms'].to(device)
                
                delta = model(sensor, t_norm)
                
                W = ds.regions[region_id]['weights'].to(device)
                pred_def = torch.einsum('nk,bkd->bnd', W, delta)
                
                loss = mse(pred_def, gt_def)
                total_loss += loss

            if isinstance(total_loss, torch.Tensor) and total_loss != 0:
                total_loss.backward()
                opt.step()
                epoch_losses.append(total_loss.item() / len(batch))
                progress_bar.set_postfix(avg_loss=epoch_losses[-1])

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        training_logger.log_epoch_stats(epoch + 1, avg_loss, np.min(epoch_losses) if epoch_losses else 0, np.max(epoch_losses) if epoch_losses else 0)

    save_path = os.path.join(out_dir, 'deform_model_final.pth')
    torch.save(model.state_dict(), save_path)

    # --- inference & save PLYs ---
    training_logger.logger.info("Starting inference phase...")
    model.eval()
    inference_stats = defaultdict(int)
    
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="Inference"):
            region_id, sensor, _, t_norm = ds[i]
            region = ds.regions[region_id]
            frame_num = region['frames'][ds.samples[i]['sample_idx']]
            
            region_out_dir = os.path.join(out_dir, f'region_{region_id}')
            cages_dir = os.path.join(region_out_dir, 'cages_pred')
            objs_dir = os.path.join(region_out_dir, 'objects_world')
            os.makedirs(cages_dir, exist_ok=True)
            os.makedirs(objs_dir, exist_ok=True)

            s, t = sensor.unsqueeze(0).to(device), t_norm.unsqueeze(0).to(device)
            d = model(s, t).squeeze(0).cpu()
            
            cage_p = shared_cage_coords + d.numpy()
            write_ply(os.path.join(cages_dir, f'cage_{frame_num:05d}.ply'), cage_p)
            inference_stats["cage_files"] += 1

            pred_norm = region['norm_init'] + (region['weights'] @ d)
            inv_RS = np.linalg.inv(region['rot_scale'].numpy())
            world = pred_norm.numpy().dot(inv_RS) - region['translate'].numpy()
            write_ply(os.path.join(objs_dir, f'object_{frame_num:05d}.ply'), world)
            inference_stats["object_files"] += 1

    training_logger.log_training_complete(save_path, {"output_dir": out_dir}, inference_stats)
    training_logger.save_metrics()
    
    log_summary = training_logger.get_log_summary()
    print("\nTraining Log Summary:")
    for key, value in log_summary.items(): print(f"  {key}: {value}")
    print('\nDone')


if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser(description="Train a GNN-based deformation model on multiple regions.")
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
    p.add_argument('--sensor_res', nargs=2, type=int, default=[10,10], help='传感器网格的高和宽，例如 10 10')
    
    args = p.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Default data directory '{args.data_dir}' not found. Creating dummy data.")
        frame_dir = os.path.join(args.data_dir, 'frames')
        os.makedirs(frame_dir, exist_ok=True)
        
        total_frames = 90 
        num_regions = 3
        frames_per_region = total_frames // num_regions

        print(f"Creating {total_frames} continuous dummy PLY files...")
        for i in range(total_frames):
            points = np.random.rand(100, 3) * 5 - 2.5
            write_ply(os.path.join(frame_dir, f'frame_{i:05d}.ply'), points)

        print(f"Creating data for {num_regions} regions...")
        for region_idx in range(num_regions):
            suffix = str(region_idx) if region_idx > 0 else ""
            
            region_data = {
                "bbox": [[-1.5 + region_idx, -1.5, -1.5], [1.5 + region_idx, 1.5, 1.5]],
                "normal": [np.sin(region_idx), np.cos(region_idx), 1.0]
            }
            with open(os.path.join(args.data_dir, f'region{suffix}.json'), 'w') as f:
                json.dump(region_data, f)
            
            sensor_res_val = args.sensor_res
            sensor_data = []
            
            start_frame = region_idx * frames_per_region
            end_frame = start_frame + frames_per_region
            
            for frame_id in range(start_frame, end_frame):
                readings = np.random.rand(sensor_res_val[0] * sensor_res_val[1])
                sensor_data.append([frame_id] + readings.tolist())
            
            pd.DataFrame(sensor_data).to_csv(os.path.join(args.data_dir, f'sensor{suffix}.csv'), header=False, index=False)
        print("Dummy data creation complete.")

    train_and_infer(args)
