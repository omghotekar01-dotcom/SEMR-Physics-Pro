import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, numpy as np, cv2
from models.semr_physics_pro import SEMRPhysicsPro
from utils.degradation import generate_sem_pair

device = torch.device('cpu')
model = SEMRPhysicsPro(scale=2, dim=12, num_blocks=[2,2,2,2], heads=4).to(device)
model.load_state_dict(torch.load('pretrained/best_scale2.pth', map_location=device))
model.eval()

clean = np.zeros((256, 256), dtype=np.float32)
cv2.rectangle(clean, (40, 40), (100, 120), 1.0, -1)
cv2.rectangle(clean, (140, 80), (200, 200), 1.0, -1)
cv2.circle(clean, (80, 180), 20, 1.0, -1)

lr, _ = generate_sem_pair(clean, scale=2)   # LR = 128x128
lr_t = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    hr_pred, uncertainty = model(lr_t)

hr_pred = hr_pred.squeeze().numpy()
hr_pred = np.clip(hr_pred, 0, 1)
uncertainty = uncertainty.squeeze().numpy()

cv2.imwrite('results/clean_original.png', (clean*255).astype(np.uint8))
cv2.imwrite('results/degraded_input.png', (lr*255).astype(np.uint8))
cv2.imwrite('results/restored.png', (hr_pred*255).astype(np.uint8))
unc_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-6)
heatmap = cv2.applyColorMap((unc_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite('results/uncertainty.png', heatmap)
print("Saved images in results/")