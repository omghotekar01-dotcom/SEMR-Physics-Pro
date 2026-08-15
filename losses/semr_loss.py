import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

def charbonnier_loss(pred, target, eps=1e-6):
    return torch.sqrt((pred - target)**2 + eps).mean()

def gradient_loss(pred, target):
    dx = pred[:,:,1:,:] - pred[:,:,:-1,:]
    dy = pred[:,:,:,1:] - pred[:,:,:,:-1]
    tx = target[:,:,1:,:] - target[:,:,:-1,:]
    ty = target[:,:,:,1:] - target[:,:,:,:-1]
    return F.l1_loss(dx, tx) + F.l1_loss(dy, ty)

def physics_loss(restored, input_lr, scale=2):
    # Downsample restored image by the same factor as SR scale
    # For scale=2, restored is 256x256 -> downsampled to 128x128 (matches input_lr)
    down = F.avg_pool2d(restored, scale)
    return F.l1_loss(down, input_lr)

class SEMRLoss(nn.Module):
    def __init__(self, w_pix=1.0, w_ssim=0.0, w_grad=0.3, w_phys=0.1, w_unc=0.0, scale=2):
        super().__init__()
        self.w_pix = w_pix
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_phys = w_phys
        self.w_unc = w_unc
        self.scale = scale

    def forward(self, pred, target, uncertainty, input_lr):
        loss = 0.0

        # 1. Charbonnier L1 (robust pixel loss)
        loss += self.w_pix * charbonnier_loss(pred, target)

        # 2. MS-SSIM (structural)
        if self.w_ssim > 0:
            loss += self.w_ssim * (1 - ms_ssim(pred, target, data_range=1.0, size_average=True))

        # 3. Gradient loss (edge preservation)
        if self.w_grad > 0:
            loss += self.w_grad * gradient_loss(pred, target)

        # 4. Physics consistency (downsample restored == input LR)
        if self.w_phys > 0:
            loss += self.w_phys * physics_loss(pred, input_lr, scale=self.scale)

        # 5. Uncertainty-weighted NLL (only if w_unc > 0)
        if self.w_unc > 0:
            precision = 1.0 / (uncertainty + 1e-6)
            loss += self.w_unc * (precision * torch.abs(pred - target) + torch.log(uncertainty + 1e-6)).mean()

        return loss