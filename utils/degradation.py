import numpy as np, cv2

def generate_sem_pair(clean, scale=4):
    # clean: uint8 or float [0,1]
    clean = clean.astype(np.float32)
    k = np.random.uniform(10, 50)
    speckle = np.random.gamma(k, 1/k, clean.shape)
    noisy = clean * speckle
    noisy += np.random.normal(0, np.random.uniform(0.01, 0.05), clean.shape)
    # PSF
    sf = np.random.uniform(0.5, 2.0)
    sb = np.random.uniform(3.0, 8.0)
    eta = np.random.uniform(0.1, 0.3)
    ksize = int(2*sb*3 + 1) | 1
    gauss_f = cv2.getGaussianKernel(ksize, sf)
    gauss_b = cv2.getGaussianKernel(ksize, sb)
    kernel = (1-eta)*(gauss_f*gauss_f.T) + eta*(gauss_b*gauss_b.T)
    blurred = cv2.filter2D(noisy, -1, kernel)
    h,w = clean.shape
    lr = cv2.resize(blurred, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
    return lr, clean