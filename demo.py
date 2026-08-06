import gradio as gr, torch, numpy as np, cv2
from models.semr_physics_pro import SEMRPhysicsPro

model = SEMRPhysicsPro()
model.load_state_dict(torch.load('pretrained/best.pth', map_location='cpu'))
model.eval()

def restore(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    h,w = img.shape
    # Ensure divisible by 4
    new_h, new_w = (h//4)*4, (w//4)*4
    img = img[:new_h, :new_w]
    lr_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        hr, uncert = model(lr_t)
    hr = hr.squeeze().numpy()
    uncert = uncert.squeeze().numpy()
    # Overlay uncertainty as heatmap
    uncert_norm = (uncert - uncert.min()) / (uncert.max() - uncert.min() + 1e-6)
    heatmap = cv2.applyColorMap((uncert_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (0.6*hr[..., None] + 0.4*heatmap).astype(np.uint8)
    return (hr*255).astype(np.uint8), overlay

gr.Interface(
    fn=restore,
    inputs=gr.Image(label="Degraded SEM Image"),
    outputs=[gr.Image(label="Restored"), gr.Image(label="Uncertainty Overlay")],
    title="SEMR-Physics Pro"
).launch(share=True)