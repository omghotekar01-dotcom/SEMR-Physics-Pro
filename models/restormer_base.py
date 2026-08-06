"""Restormer base placeholder.

NOTE: The real Restormer implementation should be copied here.
"""
import torch.nn as nn


class RestormerBase(nn.Module):
    """Minimal placeholder for Restormer base."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Identity()

    def forward(self, x):
        return self.layer(x)
