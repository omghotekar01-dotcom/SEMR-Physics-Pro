import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.semr_physics_pro import SEMRPhysicsPro

model = SEMRPhysicsPro(scale=2, dim=12, num_blocks=[2,2,2,2], heads=4)
model.eval()

dummy_lr = torch.randn(1, 1, 128, 128)
dummy_hr = torch.randn(1, 1, 256, 256)

with torch.no_grad():
    pred_hr, uncertainty = model(dummy_lr)
print(f"Output HR: {pred_hr.shape}, Uncertainty: {uncertainty.shape}")

from losses.semr_loss import SEMRLoss
loss_fn = SEMRLoss()
loss = loss_fn(pred_hr, dummy_hr, uncertainty, dummy_lr)
print(f"Loss: {loss.item():.4f}")