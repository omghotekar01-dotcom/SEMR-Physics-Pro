import torch
from models.semr_physics_pro import SEMRPhysicsPro
from losses.semr_loss import SEMRLoss

print("Creating model...")
model = SEMRPhysicsPro(scale=4, dim=8, num_blocks=[2,2,2,2], heads=2)  # tiny
model.eval()

dummy_lr = torch.randn(1, 1, 16, 16)   # tiny image
dummy_hr = torch.randn(1, 1, 64, 64)

print("Running forward pass...")
with torch.no_grad():
    pred_hr, uncertainty = model(dummy_lr)
print(f"Output shape: {pred_hr.shape}, Uncertainty shape: {uncertainty.shape}")

loss_fn = SEMRLoss()
loss = loss_fn(pred_hr, dummy_hr, uncertainty, dummy_lr)
print(f"Loss: {loss.item()}")
print("Test passed!")   