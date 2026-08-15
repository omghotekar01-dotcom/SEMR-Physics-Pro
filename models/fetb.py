import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet import HaarDWT, HaarIDWT

class FETB(nn.Module):
    def __init__(self, dim, num_heads=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # depthwise
        self.conv2 = nn.Conv2d(dim, dim, 1)                         # pointwise
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.freq_conv = nn.Conv2d(dim * 4, dim, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, sigma_f=None):
        B, C, H, W = x.shape

        # Spatial branch: depthwise conv + pointwise conv with residual
        x_identity = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.conv1(x_norm)
        x_spatial = self.conv2(x_spatial)
        x_spatial = x_identity + x_spatial

        # Feed-forward network
        x_norm2 = self.norm2(x_spatial.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_ffn = self.ffn(x_norm2)
        x_spatial = x_spatial + x_ffn

        # Frequency branch (wavelet)
        yl, yh = self.dwt(x)
        yh_cat = torch.cat([yh[0], yh[1], yh[2]], dim=1)
        freq_feat = torch.cat([yl, yh_cat], dim=1)
        freq_out = self.freq_conv(freq_feat)
        freq_out = self.idwt(freq_out, yh)

        # Adaptive fusion
        gate_in = torch.cat([x_spatial, freq_out], dim=1)
        gate = self.gate(gate_in)
        out = gate[:, 0:1] * x_spatial + gate[:, 1:2] * freq_out
        return out