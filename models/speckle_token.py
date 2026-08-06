import torch
import torch.nn as nn

class SpeckleAwareTokenization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x_log = torch.log(x + self.alpha + self.eps)
        mean = x_log.mean(dim=[2,3], keepdim=True)
        std  = x_log.std(dim=[2,3], keepdim=True)
        return (x_log - mean) / (std + self.eps)