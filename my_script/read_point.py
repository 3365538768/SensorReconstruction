import os
import re
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
from plyfile import PlyData

class GaussianPlyWithSensorDataset(Dataset):
    """
    以 time_00000.ply... 和一个对应的传感器 txt 文件(每行 10000 个浮点数)为输入，
    每个样本返回：
      {
        'xyz':      Tensor[N,3],
        'normals':  Tensor[N,3],
        'f_dc':     Tensor[N,3],
        'f_rest':   Tensor[N,45],
        'opacity':  Tensor[N,1],
        'scale':    Tensor[N,3],
        'rotation': Tensor[N,4],
        'timestamp': int,
        'sensor':   Tensor[10000]
      }
    """
    FILENAME_RE = re.compile(r'time_(\d+)\.ply$')

    def __init__(self,
                 ply_dir: str,
                 sensor_txt: str,
                 device: str = 'cpu'):
        # 找到并排序所有 ply
        self.files = sorted(glob(os.path.join(ply_dir, '*.ply')))
        # 读取整个 sensor txt：每行 10000 个数
        self.sensor_data = np.loadtxt(sensor_txt)  # shape (T,10000)
        if self.sensor_data.ndim == 1:
            # 只有一行时也要变成 (1,10000)
            self.sensor_data = self.sensor_data[np.newaxis, :]
        self.device = device

        # 检查 ply 数量与 sensor 行数
        max_ts = max(int(self.FILENAME_RE.match(os.path.basename(f)).group(1))
                     for f in self.files)
        if max_ts >= len(self.sensor_data):
            raise ValueError(
                f"传感器数据行数({len(self.sensor_data)})不足以覆盖最大时间步 {max_ts}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = self.files[idx]
        fname = os.path.basename(ply_path)
        m = self.FILENAME_RE.match(fname)
        if not m:
            raise ValueError(f"文件名格式不匹配: {fname}")
        timestamp = int(m.group(1))

        # 1. 读 ply
        ply = PlyData.read(ply_path)
        v   = ply['vertex'].data
        names = list(v.dtype.names)
        arr = np.vstack([v[f] for f in names]).T
        idmap = {n:i for i,n in enumerate(names)}

        sample = {
            'xyz':      arr[:, [idmap['x'], idmap['y'], idmap['z']]],
            'normals':  arr[:, [idmap['nx'], idmap['ny'], idmap['nz']]],
            'f_dc':     arr[:, [idmap[f"f_dc_{i}"] for i in range(3)]],
            'f_rest':   arr[:, [idmap[f"f_rest_{i}"] for i in range(45)]],
            'opacity':  arr[:, [idmap['opacity']]],
            'scale':    arr[:, [idmap[f"scale_{i}"] for i in range(3)]],
            'rotation': arr[:, [idmap[f"rot_{i}"]   for i in range(4)]],
            'timestamp': timestamp,
            'sensor':   self.sensor_data[timestamp]  # shape (10000,)
        }

        # 转为 torch.Tensor
        for k, v in sample.items():
            if k in ('timestamp',):
                # timestamp 保留 int
                continue
            t = torch.from_numpy(v)
            if k == 'sensor':
                # sensor 一维 -> shape (10000,)
                sample[k] = t.to(self.device).float()
            else:
                # ply 中各 field -> shape (N,dim)
                sample[k] = t.to(self.device).float()
        sample['timestamp'] = torch.tensor(timestamp, device=self.device)
        return sample

# === 使用示例 ===
if __name__ == "__main__":
    ply_dir    = "/users/lshou/4DGaussians/output/dnerf/jumpingjacks/gaussian_interpolated"
    sensor_txt = "/users/lshou/4DGaussians/sensor/sensor_data.txt"
    dataset    = GaussianPlyWithSensorDataset(ply_dir, sensor_txt, device='cpu')
    loader     = DataLoader(dataset, batch_size=1, shuffle=False)

    first = next(iter(loader))
    print("timestamp:", first['timestamp'].item())
    print("xyz     shape:", first['xyz'].shape)
    print("sensor  shape:", first['sensor'].shape)
    print("sensor  data:", first['sensor'][:10])  # 打印前10个值
