import io, os, uuid, time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch, numpy as np, cv2
from PIL import Image

# Add parent to path so we can import our model
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.semr_physics_pro import SEMRPhysicsPro

app = FastAPI(title="SEMR-Physics Pro API")

# Allow requests from React dev server
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model once at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SEMRPhysicsPro(scale=2, dim=12, num_blocks=[2,2,2,2], heads=4).to(device)
model.load_state_dict(torch.load('../pretrained/best_light.pth', map_location=device))
model.eval()

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.post("/restore")
async def restore_image(file: UploadFile = File(...)):
    # Save uploaded file
    file_ext = file.filename.split('.')[-1]
    unique_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Load and preprocess
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    # Resize to model input size (64x64)
    lr = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        hr, uncertainty = model(lr_t)
    t1 = time.time()

    hr = hr.squeeze().cpu().numpy()
    hr = np.clip(hr, 0, 1)
    uncertainty = uncertainty.squeeze().cpu().numpy()

    # Resize back to original dimensions
    orig_h, orig_w = img.shape
    hr_large = cv2.resize(hr, (orig_w, orig_h))
    unc_large = cv2.resize(uncertainty, (orig_w, orig_h))

    # Save results
    hr_path = os.path.join(RESULT_DIR, f"restored_{unique_name}")
    unc_path = os.path.join(RESULT_DIR, f"uncertainty_{unique_name}")
    cv2.imwrite(hr_path, (hr_large * 255).astype(np.uint8))
    cv2.imwrite(unc_path, (unc_large * 255).astype(np.uint8))

    return JSONResponse({
        "restored_url": f"/download/restored_{unique_name}",
        "uncertainty_url": f"/download/uncertainty_{unique_name}",
        "inference_time_ms": (t1 - t0) * 1000,
        "original_shape": [orig_h, orig_w]
    })

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "File not found"})