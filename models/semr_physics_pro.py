import torch
import torch.nn as nn
import torch.nn.functional as F
from .speckle_token import SpeckleAwareTokenization
from .self_calibration import SelfCalibrationHead
from .fetb import FETB
from .uncertainty_head import UncertaintyHead

class SEMRPhysicsPro(nn.Module):
    def __init__(self, inp_channels=1, dim=48, num_blocks=[4,6,6,8], heads=8, scale=4):
        super().__init__()
        self.scale = scale
        self.sat = SpeckleAwareTokenization()
        self.scn = SelfCalibrationHead()
        self.shallow = nn.Conv2d(inp_channels, dim, 3, 1, 1)

        # ---- Encoder ----
        self.enc_blks = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_dim = dim
        for i, n in enumerate(num_blocks[:-1]):
            blks = nn.Sequential(*[FETB(in_dim, heads) for _ in range(n)])
            self.enc_blks.append(blks)
            self.downs.append(nn.Conv2d(in_dim, in_dim*2, 2, 2))
            in_dim *= 2

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(*[FETB(in_dim, heads) for _ in range(num_blocks[-1])])

        # ---- Decoder ----
        self.ups = nn.ModuleList()
        self.reduces = nn.ModuleList()
        self.dec_blks = nn.ModuleList()

        for i in range(len(num_blocks)-1):
            out_dim = in_dim // 2
            self.ups.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim * 4, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            # after concat: 2*out_dim, reduce to out_dim
            self.reduces.append(nn.Conv2d(out_dim * 2, out_dim, 1, bias=False))
            n_blocks = num_blocks[-2 - i]
            self.dec_blks.append(nn.Sequential(*[FETB(out_dim, heads) for _ in range(n_blocks)]))
            in_dim = out_dim

        # ---- Super-resolution head ----
        if scale == 2:
            self.sr_head = nn.Sequential(
                nn.Conv2d(in_dim, dim*4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(dim, 1, 3, 1, 1)
            )
        elif scale == 4:
            self.sr_head = nn.Sequential(
                nn.Conv2d(in_dim, dim*4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim*4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(dim, 1, 3, 1, 1)
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale}. Only 2 or 4 are supported.")

        self.uncertainty_head = UncertaintyHead(in_dim)

    def forward(self, noisy_lr):
        sigma_f, sigma_b, eta, k, readout = self.scn(noisy_lr)
        x = self.sat(noisy_lr)
        feat = self.shallow(x)          # (B, dim, H_lr, W_lr)

        # Encoder
        skips = []
        for blk, down in zip(self.enc_blks, self.downs):
            for b in blk:
                feat = b(feat, sigma_f)
            skips.append(feat)
            feat = down(feat)

        # Bottleneck
        for b in self.bottleneck:
            feat = b(feat, sigma_f)

        # Decoder
        for up, reduce, blk, skip in zip(self.ups, self.reduces, self.dec_blks, reversed(skips)):
            feat = up(feat)
            feat = torch.cat([feat, skip], dim=1)
            feat = reduce(feat)
            for b in blk:
                feat = b(feat, sigma_f)

        # SR head and uncertainty
        hr = self.sr_head(feat)                     # (B,1, H, W)
        uncert_lr = self.uncertainty_head(feat)     # (B,1, H_lr, W_lr)
        uncertainty = F.interpolate(uncert_lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        return hr, uncertainty