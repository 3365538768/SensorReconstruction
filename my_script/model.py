import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================================
#   Model Architecture (supports batch)
# =========================================================================================
class SensorEncoder(nn.Module):
    """Encodes the 16x16 sensor data into a context vector."""
    def __init__(self, d_out=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(32*4*4,128)
        self.fc2   = nn.Linear(128,d_out)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GridCNNUpdater(nn.Module):
    """The core state updater combining spatial CNNs and temporal GRUs."""
    def __init__(
        self, context_dim=128, model_dim=256, grid_res=(8,8,8),
        encoder_layers=2, cnn_layers=2, decoder_layers=3
    ):
        super().__init__()
        self.grid_res = grid_res

        # Encoder MLP
        enc = [nn.Linear(3+3+context_dim, model_dim), nn.ReLU(inplace=True)]
        for _ in range(encoder_layers-1):
            enc += [nn.Linear(model_dim,model_dim), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*enc)

        # 3D-CNN Block
        cnn = []
        for i in range(cnn_layers):
            cnn += [nn.Conv3d(model_dim,model_dim,3,padding=1),
                    nn.BatchNorm3d(model_dim)]
            if i<cnn_layers-1:
                cnn += [nn.ReLU(inplace=True)]
        self.cnn_block = nn.Sequential(*cnn)
        self.ln1 = nn.LayerNorm(model_dim)

        # Temporal GRUCell
        self.gru = nn.GRUCell(model_dim,model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        # Decoder MLP
        dec = [nn.Linear(model_dim,model_dim//2), nn.ReLU(inplace=True)]
        for _ in range(decoder_layers-2):
            dec += [nn.Linear(model_dim//2,model_dim//2), nn.ReLU(inplace=True)]
        dec += [nn.Linear(model_dim//2,3)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, pos, vel_norm, ctx_exp, hidden):
        # pos, vel_norm, hidden: [B, N, C]   ctx_exp: [B, N, ctx_dim]
        B, N, _ = pos.shape
        D, H, W = self.grid_res

        # 1) Per-node feature fusion
        inp = torch.cat([pos, vel_norm, ctx_exp], dim=2)  # [B,N,feat]
        flat = inp.view(B*N, -1)                          # [B*N,feat]
        fused = self.encoder(flat).view(B, N, -1)         # [B,N,C]

        # 2) 3D convolution for spatial context
        vol = fused.view(B, D, H, W, -1).permute(0,4,1,2,3).contiguous()  # [B,C,D,H,W]
        out = self.cnn_block(vol)                                         # [B,C,D,H,W]
        feat = out.permute(0,2,3,4,1).reshape(B, N, -1)                    # [B,N,C]
        sp = self.ln1(fused + feat)                                       # [B,N,C]

        # 3) Temporal update with GRU
        sp_flat = sp.view(B*N, -1)
        h_flat  = hidden.view(B*N, -1)
        nh_flat = self.gru(sp_flat, h_flat)
        nh = nh_flat.view(B, N, -1)
        tm = self.ln2(nh)                                                 # [B,N,C]

        # 4) Decode final velocity
        tm_flat = tm.view(B*N, -1)
        v_norm_flat = self.decoder(tm_flat)                               # [B*N,3]
        v_norm = v_norm_flat.view(B, N, 3)
        
        return v_norm, nh

class GridCNNAutoRegressiveDeformer(nn.Module):
    """The main model class that orchestrates the sub-modules."""
    def __init__(self, context_dim=128, model_dim=256, grid_res=(8,8,8), **kwargs):
        super().__init__()
        self.sensor_enc = SensorEncoder(d_out=context_dim)
        self.updater    = GridCNNUpdater(context_dim, model_dim, grid_res, **kwargs)

    def forward(self, pos, vel_norm, sensor, hidden):
        # pos, vel_norm, hidden: [B,N,C]; sensor: [B,1,16,16]
        ctx = self.sensor_enc(sensor)                                 # [B,ctx]
        ctx_exp = ctx.unsqueeze(1).expand(-1, pos.shape[1], -1)       # [B,N,ctx]
        return self.updater(pos, vel_norm, ctx_exp, hidden)
