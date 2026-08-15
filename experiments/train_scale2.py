import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.semr_physics_pro import SEMRPhysicsPro
from losses.semr_loss import SEMRLoss
from utils.dataset import SEMDataset, create_clean_shapes

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU.")

    print("Building 2x super-resolution model...")
    model = SEMRPhysicsPro(scale=2, dim=12, num_blocks=[2,2,2,2], heads=4).to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    clean_imgs = create_clean_shapes(num=50, size=256)
    dataset = SEMDataset(clean_imgs, scale=2)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)  # 0 for Windows safety

    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = SEMRLoss(w_pix=1.0, w_ssim=0.0, w_grad=0.3, w_phys=0.1, w_unc=0.0).to(device)

    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = 10
    print(f"Starting {epochs}-epoch training...\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        t0 = time.time()

        # tqdm progress bar over batches
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for i, (lr, hr) in pbar:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred_hr, uncertainty = model(lr)
                    loss = loss_fn(pred_hr, hr, uncertainty, lr)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_hr, uncertainty = model(lr)
                loss = loss_fn(pred_hr, hr, uncertainty, lr)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(loader)
        t1 = time.time()
        print(f"→ Epoch {epoch+1} Avg Loss: {avg_loss:.4f} | Time: {t1-t0:.1f}s\n")

    os.makedirs('pretrained', exist_ok=True)
    torch.save(model.state_dict(), 'pretrained/best_scale2.pth')
    print("Training finished. Model saved to pretrained/best_scale2.pth")

if __name__ == '__main__':
    train()