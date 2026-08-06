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

def physics_loss(restored, input_lr, scale=4):
    down = F.avg_pool2d(restored, scale)
    return F.l1_loss(down, input_lr)

class SEMRLoss(nn.Module):
    def __init__(self, w_pix=1.0, w_ssim=0.5, w_grad=0.3, w_phys=0.2, w_unc=0.1):
        super().__init__()
        self.w_pix = w_pix
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_phys = w_phys
        self.w_unc = w_unc

    def forward(self, pred, target, uncertainty, input_lr):
        l_pix = charbonnier_loss(pred, target)
        l_ssim = 1 - ms_ssim(pred, target, data_range=1.0, size_average=True)
        l_grad = gradient_loss(pred, target)
        l_phys = physics_loss(pred, input_lr)
        l_unc = (torch.abs(pred - target) / (uncertainty + 1e-6) + torch.log(uncertainty + 1e-6)).mean()
        return self.w_pix*l_pix + self.w_ssim*l_ssim + self.w_grad*l_grad + self.w_phys*l_phys + self.w_unc*l_unc