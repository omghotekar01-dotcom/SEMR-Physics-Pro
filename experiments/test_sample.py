import torch, numpy as np, cv2
from models.semr_physics_pro import SEMRPhysicsPro
from utils.degradation import generate_sem_pair

# Load the tiny trained model
device = torch.device('cpu')
model = SEMRPhysicsPro(scale=4, dim=8, num_blocks=[2,2,2,2], heads=2).to(device)
model.load_state_dict(torch.load('pretrained/best_light.pth', map_location=device))
model.eval()

# Generate a synthetic test image (or draw a simple shape)
clean = np.zeros((256, 256), dtype=np.float32)
# Add some patterns – a few rectangles to look like a wafer layout
cv2.rectangle(clean, (40, 40), (100, 120), 1.0, -1)
cv2.rectangle(clean, (140, 80), (200, 200), 1.0, -1)
cv2.circle(clean, (80, 180), 20, 1.0, -1)

# Degrade it
lr, _ = generate_sem_pair(clean, scale=4)   # lr is 64x64

# Restore
lr_tensor = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0)  # [1,1,64,64]
with torch.no_grad():
    hr_pred, uncertainty = model(lr_tensor)

hr_pred = hr_pred.squeeze().numpy()
hr_pred = np.clip(hr_pred, 0, 1)
uncertainty = uncertainty.squeeze().numpy()

# Save images
cv2.imwrite('clean_original.png', (clean*255).astype(np.uint8))
cv2.imwrite('degraded_input.png', (lr*255).astype(np.uint8))
cv2.imwrite('restored.png', (hr_pred*255).astype(np.uint8))

# Create an uncertainty heatmap
unc_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-6)
heatmap = cv2.applyColorMap((unc_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite('uncertainty.png', heatmap)

print("Saved: clean_original.png, degraded_input.png, restored.png, uncertainty.png")
print("Check the folder!")