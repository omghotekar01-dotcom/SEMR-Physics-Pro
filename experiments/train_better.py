import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.optim as optim, time
from torch.utils.data import DataLoader
from models.semr_physics_pro import SEMRPhysicsPro
from losses.semr_loss import SEMRLoss
from utils.dataset import SEMDataset, create_clean_shapes

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Bigger model: dim=32, 3 encoder levels
    model = SEMRPhysicsPro(scale=4, dim=32, num_blocks=[4,4,4,4], heads=4).to(device)
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # More diverse training data: 200 images of size 256
    print("Generating 200 training images...")
    clean_imgs = create_clean_shapes(num=200, size=256)
    dataset = SEMDataset(clean_imgs, scale=4)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2 if device.type=='cuda' else 0)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    # Start with all loss components on (no curriculum)
    loss_fn = SEMRLoss(1.0, 0.3, 0.2, 0.1, 0.05).to(device)

    epochs = 20
    print(f"Starting {epochs}-epoch training...\n")
    for epoch in range(epochs):
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
            if i % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")
        avg_loss = epoch_loss / len(loader)
        t1 = time.time()
        print(f"→ Epoch {epoch+1} Avg Loss: {avg_loss:.4f} | Time: {t1-t0:.1f}s\n")

    os.makedirs('pretrained', exist_ok=True)
    torch.save(model.state_dict(), 'pretrained/best_improved.pth')
    print("✅ Improved model saved.")

if __name__ == '__main__':
    train()