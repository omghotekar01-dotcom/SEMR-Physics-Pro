import torch
import torch.nn as nn

class UncertaintyHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim//2, 1, 3, padding=1),
            nn.Softplus()
        )
    def forward(self, feat):
        return self.head(feat)