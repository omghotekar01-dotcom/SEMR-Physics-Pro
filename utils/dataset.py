import torch
from torch.utils.data import Dataset
import numpy as np, cv2, os
from .degradation import generate_sem_pair

def create_clean_shapes(num=1000, size=512):
    # Generate simple shapes (lines, rectangles, circles) as clean images
    imgs = []
    for _ in range(num):
        img = np.zeros((size, size), dtype=np.float32)
        # random rectangles
        for _ in range(np.random.randint(3,10)):
            x,y,w,h = np.random.randint(0,size-50,4)
            img[y:y+h, x:x+w] = 1.0
        imgs.append(img)
    return imgs

class SEMDataset(Dataset):
    def __init__(self, clean_list, scale=4):
        self.clean = clean_list
        self.scale = scale
    def __len__(self):
        return len(self.clean)
    def __getitem__(self, idx):
        hr = self.clean[idx].copy()
        lr, hr_ref = generate_sem_pair(hr, self.scale)
        lr = torch.from_numpy(lr).float().unsqueeze(0)
        hr_ref = torch.from_numpy(hr_ref).float().unsqueeze(0)
        return lr, hr_ref