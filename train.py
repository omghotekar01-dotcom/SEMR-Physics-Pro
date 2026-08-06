import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from models.semr_physics_pro import SEMRPhysicsPro
from losses.semr_loss import SEMRLoss
from utils.dataset import SEMDataset, create_clean_shapes
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SEMRPhysicsPro(scale=4).to(device)
    model.train()

    clean_imgs = create_clean_shapes(num=500, size=256)
    dataset = SEMDataset(clean_imgs, scale=4)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    curriculum = [
        (1.0, 0.5, 0.1, 0.1, 0.0),
        (0.8, 0.6, 0.3, 0.2, 0.05),
        (0.5, 0.7, 0.4, 0.3, 0.1)
    ]

    for epoch in range(200):
        phase = 0 if epoch < 50 else (1 if epoch < 100 else 2)
        wp, ws, wg, wphys, wunc = curriculum[phase]
        loss_fn = SEMRLoss(wp, ws, wg, wphys, wunc).to(device)

        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                with autocast('cuda'):
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

        scheduler.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.6f}")

        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), f'checkpoints/epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), 'pretrained/best.pth')
    print("Training done.")

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('pretrained', exist_ok=True)
    train()