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
    # PlyElement.describe 会根据 structured_pts_array 的 dtype 自动创建属性
    el = PlyElement.describe(structured_pts_array, 'vertex')
    # 强制使用 binary_little_endian 格式，与高斯球的常见格式一致
    PlyData([el], text=False, byte_order='<').write(path)
    print(f"Saved Gaussian PLY to {path}")


# =========================================
# 1. 数据预处理 & Dataset (为推理修改)
# =========================================
class DeformDataset(Dataset):
    def __init__(self, data_dir, init_ply_path, cage_res=(12,12,12)):
        self.data_dir = data_dir
        self.cage_res = cage_res
        # 读取传感器 CSV
        raw = pd.read_csv(os.path.join(data_dir,'sensor.csv'), header=None)
        col0 = pd.to_numeric(raw.iloc[:,0], errors='coerce')
        valid = col0.notna().values
        self.frames = col0[valid].astype(int).values # 传感器数据对应的帧ID
        self.sensors = raw[valid].iloc[:,1:].values.astype(np.float32)
        # 读取 region.json
        cfg = json.load(open(os.path.join(data_dir,'region.json')))
        self.bbox_min = np.array(cfg['bbox'][0],dtype=np.float32)
        self.bbox_max = np.array(cfg['bbox'][1],dtype=np.float32)
        self.normal   = np.array(cfg['normal'],dtype=np.float32)

        # 加载初始 PLY 文件并进行归一化
        print(f"Loading initial PLY from: {init_ply_path}")
        # _load_ply 现在返回 (xyz_coords, full_structured_array)
        pts0_xyz, self.init_full_ply_data = self._load_ply(init_ply_path)
        
        # 仅使用XYZ坐标进行边界框检查和归一化计算
        mask0 = self._in_box(pts0_xyz) 
        self.idx0 = np.where(mask0)[0] # 初始 PLY 中在边界框内的点索引
        self._compute_norm(pts0_xyz[self.idx0]) # 计算 self.norm_init, self.translate, self.rot_scale

        # 构建笼节点
        cage = self._build_cage(cage_res).astype(np.float32)
        self.cage_coords = cage
        # 规范化基准点云 (self.norm_init 已经从 init_ply_path 计算得到)
        self.weights = self._compute_weights(self.norm_init, cage)
        # 转为张量
        self.norm_init_tensor = torch.from_numpy(self.norm_init)
        self.translate        = torch.from_numpy(self.translate)
        self.rot_scale        = torch.from_numpy(self.rot_scale)

    def __len__(self):
        return len(self.frames) # 传感器读数的数量

    def __getitem__(self, idx):
        # 传感器数据
        sensor = torch.from_numpy(self.sensors[idx].reshape(1,16,16))
        # 时间归一化
        t_frame = float(self.frames[idx]) / float(self.frames.max())
        t_norm = torch.tensor(t_frame, dtype=torch.float32)
        # 真实偏移 (在推理时不需要，返回一个占位符以匹配DataLoader的输出格式)
        gt_def_placeholder = torch.zeros_like(self.norm_init_tensor)
        return sensor, gt_def_placeholder, t_norm

    def _load_ply(self, path):
        """
        加载PLY文件。返回XYZ坐标和包含所有属性的结构化NumPy数组。
        """
        ply = PlyData.read(path)
        vertex_data = ply['vertex']
        
        # 提取XYZ坐标
        pts_xyz = np.vstack([vertex_data[k] for k in ['x','y','z']]).T.astype(np.float32)
        
        # 获取所有属性的名称和数据类型，并创建完整的结构化数组副本
        # 直接使用 vertex_data.data 可以获取包含所有属性的原始结构化数组
        full_structured_array = np.array(vertex_data.data, dtype=vertex_data.data.dtype)

        return pts_xyz, full_structured_array

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
        # --- sensor & time encoders ---
        self.encoder = EnhancedSensorEncoder(out_dim=sensor_dim)
        self.timeenc = TimeEncoding(num_bands=num_time_bands)
        self.fourier = FourierEncoding(num_bands=num_fourier_bands)

        # ensure cage_coords is float32 numpy
        cage_coords = cage_coords.astype(np.float32)
        # compute size of fourier embedding
        fourier_dim = self.fourier(torch.from_numpy(cage_coords)).shape[1]
        time_dim = 1 + 2 * num_time_bands

        # graph deformer takes (feat + fourier) per node
        self.deformer = GNNDeformer(
            input_dim=sensor_dim + time_dim,
            cage_nodes=cage_coords.shape[0],
            edge_index=edge_index,
            fourier_dim=fourier_dim
        )

        # register the real cage coordinates as a buffer
        self.register_buffer('cage_coords', torch.from_numpy(cage_coords))

    def forward(self, sensor: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        sensor: (B, 1, H, W)
        t_norm: (B,)
        returns: (B, K, 3)  node-wise displacement
        """
        feat = self.encoder(sensor)             # (B, sensor_dim)
        tfeat = self.timeenc(t_norm)            # (B, time_dim)
        x = torch.cat([feat, tfeat], dim=1)     # (B, sensor_dim+time_dim)

        # compute fourier features of cage coords: (K, fourier_dim)
        fcoords = self.fourier(self.cage_coords)
        # run GNN on each batch
        return self.deformer(x, fcoords)        # (B, K, 3)

# =========================================
# 3. 推理函数
# =========================================
def build_edge_index(res):
    """根据笼网格分辨率构建图的边索引"""
    nx, ny, nz = res
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    edges = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 连接到相邻节点 (只考虑正方向，避免重复)
                for di, dj, dk in [(1,0,0),(0,1,0),(0,0,1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if ni < nx and nj < ny and nk < nz:
                        u = int(idx[i,j,k])
                        v = int(idx[ni,nj,nk])
                        edges += [(u, v), (v, u)] # 添加双向边
    u, v = zip(*edges)
    return torch.tensor([u, v], dtype=torch.long)

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- dataset & adjacency ---
    # 使用新的数据目录和初始PLY文件路径初始化数据集
    print(f"Loading data from: {args.data_dir} with initial PLY: {args.init_ply_path}")
    ds = DeformDataset(args.data_dir, args.init_ply_path, cage_res=tuple(args.cage_res))

    # 构建边索引，确保与训练时使用的分辨率一致
    edge_index = build_edge_index(tuple(args.cage_res)).to(device)

    # --- model initialization ---
    # 初始化模型，参数需与训练模型一致
    print("Initializing model architecture...")
    model = DeformModelEnhanced(
        sensor_dim=args.sensor_dim,
        cage_coords=ds.cage_coords,
        edge_index=edge_index,
        num_fourier_bands=args.num_fourier_bands,
        num_time_bands=args.num_time_bands
    ).to(device)

    # 加载训练好的模型权重
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}")
        return
    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() # 设置为评估模式

    # --- output directories ---
    bbox_dir    = os.path.join(args.out_dir, 'cropped_bbox')
    cages_dir   = os.path.join(args.out_dir, 'cages_pred')
    objs_dir    = os.path.join(args.out_dir, 'objects_world')
    os.makedirs(bbox_dir,    exist_ok=True)
    os.makedirs(cages_dir,   exist_ok=True)
    os.makedirs(objs_dir,    exist_ok=True)
    print(f"Output will be saved to: {args.out_dir}")

    # --- inference & save PLYs ---
    print("Starting inference...")
    with torch.no_grad():
        # 保存一次初始裁剪的边界框点云 (高斯球格式)
        # ds.init_full_ply_data 包含了所有属性
        # ds.idx0 包含了在边界框内的点的索引
        initial_cropped_gaussian_data = ds.init_full_ply_data[ds.idx0]
        write_gaussian_ply(os.path.join(bbox_dir, f'initial_cropped_bbox.ply'), initial_cropped_gaussian_data)

        for idx in tqdm(range(len(ds)), desc='Inferring frames'):
            sensor, _, t_norm = ds[idx] # gt_def 在推理中不使用
            frame_id = ds.frames[idx] # 使用 sensor.csv 中的实际帧ID作为输出文件名

            # 预测形变
            s = sensor.unsqueeze(0).to(device)
            t = t_norm.unsqueeze(0).to(device)
            delta = model(s, t).squeeze(0).cpu().numpy()  # (K,3) - 笼节点位移

            # 保存形变后的笼节点 (简单XYZ格式)
            cage_pred_world = ds.cage_coords + delta
            write_ply(os.path.join(cages_dir, f'cage_{frame_id:05d}.ply'), cage_pred_world)

            # 重构物体在世界坐标系中的点云 (高斯球格式，保留其他属性)
            # 1. 复制原始的完整结构化数据
            output_full_ply_data = ds.init_full_ply_data.copy()

            # 2. 计算归一化空间中点云的预测位置
            pred_norm_space = ds.norm_init + (ds.weights @ delta)  # (N_pts,3)

            # 3. 将预测位置逆变换回世界坐标系
            inv_RS = np.linalg.inv(ds.rot_scale.numpy())
            world_reconstructed_xyz = pred_norm_space.dot(inv_RS) - ds.translate.numpy()

            # 4. 更新结构化数组中的 'x', 'y', 'z' 字段
            # 注意：world_reconstructed_xyz 对应的是 ds.idx0 中的点
            # 确保数据类型匹配原始PLy文件的属性类型
            output_full_ply_data['x'][ds.idx0] = world_reconstructed_xyz[:, 0].astype(output_full_ply_data['x'].dtype)
            output_full_ply_data['y'][ds.idx0] = world_reconstructed_xyz[:, 1].astype(output_full_ply_data['y'].dtype)
            output_full_ply_data['z'][ds.idx0] = world_reconstructed_xyz[:, 2].astype(output_full_ply_data['z'].dtype)
            
            # 其他属性 (nx, ny, nz, f_dc_*, f_rest_*, opacity, scale_*, rot_*) 保持不变

            # 5. 保存更新后的完整结构化数据为高斯球PLY文件
            write_gaussian_ply(os.path.join(objs_dir, f'object_{frame_id:05d}.ply'), output_full_ply_data)

    print('Inference complete. Deformed cages and objects saved.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description="Deformation Prediction Inference Script")
    p.add_argument('--data_dir',type=str,default='test2',
                   help='Path to the directory containing sensor.csv and region.json for inference.')
    p.add_argument('--init_ply_path',type=str,default='test2/init.ply',
                   help='Path to the initial PLY file (e.g., data/init.ply) that will be deformed.')
    p.add_argument('--model_path',type=str,default='outputs/deform_model_final.pth',
                   help='Path to the trained model weights file (e.g., outputs/deform_model_final.pth).')
    p.add_argument('--out_dir',type=str,default='inference_outputs',
                   help='Directory to save inference results (deformed cages and objects).')
    p.add_argument('--sensor_dim',type=int,default=512,
                   help='Dimension of the sensor encoding. Must match training config.')
    p.add_argument('--cage_res',nargs=3,type=int,default=[15,15,15],
                   help='Resolution of the deformation cage (e.g., 15 15 15). Must match training config.')
    p.add_argument('--num_fourier_bands',type=int,default=8,
                   help='Number of Fourier bands for position encoding. Must match training config.')
    p.add_argument('--num_time_bands',type=int,default=6,
                   help='Number of Fourier bands for time encoding. Must match training config.')
    # batch_size and num_workers are not strictly necessary for inference loop over individual items,
    # but kept for consistency if one were to use a DataLoader for batched inference.
    p.add_argument('--batch_size',type=int,default=1,
                   help='Batch size for inference (usually 1 for sequential processing).')
    p.add_argument('--num_workers',type=int,default=4,
                   help='Number of workers for data loading (set to 0 for single-process inference).')
    args=p.parse_args()
    inference(args)

