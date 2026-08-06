import torch, torch.optim as optim, os, time
from torch.utils.data import DataLoader
from models.semr_physics_pro import SEMRPhysicsPro
from losses.semr_loss import SEMRLoss
from utils.dataset import SEMDataset, create_clean_shapes

def train():
    device = torch.device('cpu')
    print("Building tiny model...")
    model = SEMRPhysicsPro(scale=4, dim=8, num_blocks=[2,2,2,2], heads=2).to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    print("Generating 20 training images (size 256)...")
    clean_imgs = create_clean_shapes(num=20, size=256)   # clean HR = 256, LR = 64
    dataset = SEMDataset(clean_imgs, scale=4)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = SEMRLoss(1.0, 0.5, 0.1, 0.1, 0.0).to(device)

    print("🚀 Starting 5-epoch training...\n")
    for epoch in range(5):
        epoch_loss = 0.0
        t0 = time.time()
        for i, (lr, hr) in enumerate(loader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            pred_hr, uncertainty = model(lr)
            loss = loss_fn(pred_hr, hr, uncertainty, lr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"  Epoch {epoch+1}, Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")
        t1 = time.time()
        avg = epoch_loss / len(loader)
        print(f"  → Epoch {epoch+1} Avg Loss: {avg:.4f} | Time: {t1-t0:.1f}s\n")

    os.makedirs('pretrained', exist_ok=True)
    torch.save(model.state_dict(), 'pretrained/best_light.pth')
    print("✅ Training finished. Model saved.")

if __name__ == '__main__':
    train()