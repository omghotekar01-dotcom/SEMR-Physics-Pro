import gradio as gr
import torch
import numpy as np
import cv2
from models.semr_physics_pro import SEMRPhysicsPro

device = torch.device('cpu')
model = SEMRPhysicsPro(scale=2, dim=12, num_blocks=[2,2,2,2], heads=4).to(device)
model.load_state_dict(torch.load('pretrained/best_scale2.pth', map_location=device))
model.eval()

def restore(image):
    try:
        # Convert to grayscale float32
        if image.ndim == 3:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img = image
        img = img.astype(np.float32) / 255.0

        # Resize to 64x64 (model input)
        lr = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        lr_t = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0)  # (1,1,64,64)

        with torch.no_grad():
            hr, unc = model(lr_t)

        hr = hr.squeeze().numpy()
        hr = np.clip(hr, 0, 1)
        unc = unc.squeeze().numpy()

        # Resize back to original image dimensions
        orig_h, orig_w = img.shape
        hr_display = cv2.resize(hr, (orig_w, orig_h))
        unc_display = cv2.resize(unc, (orig_w, orig_h))

        # Normalize uncertainty (avoid division by zero)
        unc_range = unc_display.max() - unc_display.min()
        if unc_range < 1e-6:
            # If all uncertainty values are the same, create a uniform heatmap
            heatmap = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        else:
            unc_norm = (unc_display - unc_display.min()) / unc_range
            heatmap = cv2.applyColorMap((unc_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Convert grayscale restored image to 3-channel for blending
        hr_color = cv2.cvtColor((hr_display * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(hr_color, 0.6, heatmap, 0.4, 0)

        return (hr_display * 255).astype(np.uint8), overlay

    except Exception as e:
        print("Error:", e)
        return None, None

iface = gr.Interface(
    fn=restore,
    inputs=gr.Image(label="Upload degraded SEM image"),
    outputs=[gr.Image(label="Restored Image"), gr.Image(label="Uncertainty Overlay")],
    title="SEMR-Physics Pro – Semiconductor Image Restoration",
    description="AI restoration with speckle‑aware tokenization, frequency‑enhanced transformers, and uncertainty quantification."
)
iface.launch(share=False)