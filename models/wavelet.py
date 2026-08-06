import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('ll', torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]))
        self.register_buffer('lh', torch.tensor([[[[0.5, 0.5], [-0.5, -0.5]]]]))
        self.register_buffer('hl', torch.tensor([[[[0.5, -0.5], [0.5, -0.5]]]]))
        self.register_buffer('hh', torch.tensor([[[[0.5, -0.5], [-0.5, 0.5]]]]))

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.reshape(-1, 1, H, W)
        if H % 2 != 0: x_reshaped = F.pad(x_reshaped, (0, 0, 0, 1))
        if W % 2 != 0: x_reshaped = F.pad(x_reshaped, (0, 1, 0, 0))
        ll = F.conv2d(x_reshaped, self.ll, stride=2)
        lh = F.conv2d(x_reshaped, self.lh, stride=2)
        hl = F.conv2d(x_reshaped, self.hl, stride=2)
        hh = F.conv2d(x_reshaped, self.hh, stride=2)
        yl = ll.reshape(B, C, ll.size(2), ll.size(3))
        yh = [
            lh.reshape(B, C, lh.size(2), lh.size(3)),
            hl.reshape(B, C, hl.size(2), hl.size(3)),
            hh.reshape(B, C, hh.size(2), hh.size(3))
        ]
        return yl, yh

class HaarIDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('ll_rec', torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]))
        self.register_buffer('lh_rec', torch.tensor([[[[0.5, 0.5], [-0.5, -0.5]]]]))
        self.register_buffer('hl_rec', torch.tensor([[[[0.5, -0.5], [0.5, -0.5]]]]))
        self.register_buffer('hh_rec', torch.tensor([[[[-0.5, 0.5], [0.5, -0.5]]]]))

    def forward(self, yl, yh):
        B, C, H_half, W_half = yl.shape
        yl_up = F.conv_transpose2d(yl.reshape(-1, 1, H_half, W_half), self.ll_rec, stride=2, output_padding=0)
        lh_up = F.conv_transpose2d(yh[0].reshape(-1, 1, H_half, W_half), self.lh_rec, stride=2, output_padding=0)
        hl_up = F.conv_transpose2d(yh[1].reshape(-1, 1, H_half, W_half), self.hl_rec, stride=2, output_padding=0)
        hh_up = F.conv_transpose2d(yh[2].reshape(-1, 1, H_half, W_half), self.hh_rec, stride=2, output_padding=0)
        out = yl_up + lh_up + hl_up + hh_up
        out = out.reshape(B, C, out.size(2), out.size(3))
        return out