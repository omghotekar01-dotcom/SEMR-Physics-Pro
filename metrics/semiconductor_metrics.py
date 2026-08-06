import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_all_metrics(pred, target):
    psnr = peak_signal_noise_ratio(target, pred, data_range=1)
    ssim = structural_similarity(target, pred, data_range=1)
    # Edge preservation score (simplified)
    pred_grad = np.gradient(pred)
    target_grad = np.gradient(target)
    eps = np.corrcoef(pred_grad[0].flat, target_grad[0].flat)[0,1] if pred_grad[0].std()>0 else 0
    return {'psnr': psnr, 'ssim': ssim, 'eps': eps}