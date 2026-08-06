import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet import HaarDWT, HaarIDWT

class FETB(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
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
        # Spatial branch
        x_flat = x.flatten(2).permute(0, 2, 1)  # B, HW, C
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_spatial = x_flat + attn_out
        x_spatial = x_spatial + self.ffn(self.norm2(x_spatial))
        x_spatial = x_spatial.permute(0, 2, 1).view(B, C, H, W)

        # Frequency branch
        yl, yh = self.dwt(x)
        yh_cat = torch.cat([yh[0], yh[1], yh[2]], dim=1)
        freq_feat = torch.cat([yl, yh_cat], dim=1)
        freq_out = self.freq_conv(freq_feat)
        freq_out = self.idwt(freq_out, yh)        # <-- no tuple

        # Adaptive fusion
        gate_in = torch.cat([x_spatial, freq_out], dim=1)
        gate = self.gate(gate_in)
        out = gate[:, 0:1] * x_spatial + gate[:, 1:2] * freq_out
        return out