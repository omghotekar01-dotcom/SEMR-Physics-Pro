import torch, cv2, numpy as np, os, argparse
from models.semr_physics_pro import SEMRPhysicsPro
from metrics.semiconductor_metrics import compute_all_metrics  # you implement this
from utils.degradation import generate_sem_pair

def evaluate(model_path, test_clean_dir, output_dir):
    device = torch.device('cuda')
    model = SEMRPhysicsPro().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    metrics_sum = []

    for fname in os.listdir(test_clean_dir):
        if not fname.endswith('.png'): continue
        clean = cv2.imread(os.path.join(test_clean_dir, fname), 0).astype(np.float32)/255.0
        lr, _ = generate_sem_pair(clean, scale=4)
        lr_t = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            hr_pred, _ = model(lr_t)
        hr_pred = hr_pred.squeeze().cpu().numpy()
        hr_pred = np.clip(hr_pred, 0, 1)

        # Compute metrics
        metrics = compute_all_metrics(hr_pred, clean)
        metrics_sum.append(metrics)
        print(f"{fname}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")

        # Save output
        cv2.imwrite(os.path.join(output_dir, f'restored_{fname}'), (hr_pred*255).astype(np.uint8))

    print("Average Metrics:", np.mean([m['psnr'] for m in metrics_sum]), np.mean([m['ssim'] for m in metrics_sum]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pretrained/best.pth')
    parser.add_argument('--clean_dir', required=True)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    evaluate(args.model, args.clean_dir, args.output)