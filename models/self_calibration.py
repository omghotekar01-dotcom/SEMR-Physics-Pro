import torch
import torch.nn as nn

class SelfCalibrationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

    def forward(self, x):
        params = self.net(x)
        sigma_f   = torch.nn.functional.softplus(params[:,0])
        sigma_b   = torch.nn.functional.softplus(params[:,1])
        eta       = torch.sigmoid(params[:,2]) * 0.3
        speckle_k = torch.nn.functional.softplus(params[:,3]) + 1.0
        readout   = torch.nn.functional.softplus(params[:,4])
        return sigma_f, sigma_b, eta, speckle_k, readout