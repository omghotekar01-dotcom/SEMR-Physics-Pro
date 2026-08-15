import os, argparse, time
import torch, numpy as np, cv2
from models.semr_physics_pro import SEMRPhysicsPro

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True, help='Path to folder containing degraded test images')
    parser.add_argument('--output_dir', required=True, help='Path to folder where restored images will be saved')
    parser.add_argument('--model_path', default='pretrained/best_scale2.pth', help='Path to model weights')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SEMRPhysicsPro(scale=2, dim=12, num_blocks=[2,2,2,2], heads=4).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    total_time = 0.0
    num_imgs = 0
    for fname in sorted(os.listdir(args.test_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.bmp', '.tif', '.tiff')):
            continue
        img = cv2.imread(os.path.join(args.test_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        h2, w2 = (h // 2) * 2, (w // 2) * 2
        img = img[:h2, :w2]
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            restored, _ = model(img_t)
        t1 = time.time()
        total_time += (t1 - t0)
        num_imgs += 1

        restored = restored.squeeze().cpu().numpy()
        restored = np.clip(restored, 0, 1)
        out_path = os.path.join(args.output_dir, f"restored_{fname}")
        cv2.imwrite(out_path, (restored * 255).astype(np.uint8))
        print(f"Processed {fname} -> {out_path}")

    if num_imgs > 0:
        print(f"Average inference time per image: {total_time/num_imgs*1000:.2f} ms")

if __name__ == '__main__':
    main()