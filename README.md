# SEMR-Physics Pro

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**

SEMR-Physics Pro is a deep learning solution designed for the SEMICON India 2026 Hackathon (KLA Challenge) to restore degraded semiconductor inspection images using advanced neural network architectures. The model employs physics-informed learning with uncertainty quantification to achieve high-quality 2× super-resolution restoration.

## Key Innovations

### 1. Speckle-Aware Tokenization (SAT)
Addresses semiconductor-specific noise patterns by applying logarithmic normalization to raw pixel intensities, making the model robust to multiplicative speckle noise inherent in SEM imaging.

### 2. Frequency-Enhanced Transformer Block (FETB)
A memory-efficient fusion of spatial and frequency domains:
- **Spatial Branch**: Depthwise convolution with residual connections
- **Frequency Branch**: Haar wavelet decomposition for multi-scale frequency analysis
- **Adaptive Gating**: Learned fusion mechanism to balance spatial and frequency features

### 3. Self-Calibrating HyperNetwork
A lightweight network that estimates physics-aware hyperparameters (noise std dev, blur kernel width, speckle shape) from input images, enabling adaptive model behavior per image.

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup
``ash
# Clone or download the repository
cd SEMR-Physics-Pro

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
``

## How to Run

### 1. Quick Evaluation on Test Images
Run inference on a folder of degraded semiconductor images:

``ash
python evaluate.py --test_dir <path_to_test_images> --output_dir <output_folder>
``

**Example:**
``ash
python evaluate.py --test_dir ./sample_images --output_dir ./restored_images
``

**Output:**
- Restored images saved with prefix estored_ to the output folder
- Average inference time per image printed to console

### 2. Interactive Demo with Gradio
Launch a web-based interface for single image restoration and uncertainty visualization:

``ash
python demo_app.py
``

Then open your browser and navigate to the URL displayed in the terminal (typically http://localhost:7860).

### 3. Retrain the Model
To retrain the model with synthetic training data:

``ash
python experiments/train_scale2.py
``

**Training Details:**
- **Epochs**: 10
- **Batch Size**: 1 (Windows compatible)
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW
- **Model Configuration**: 2× super-resolution, dim=12, [2,2,2,2] blocks, 4 heads
- **Output**: Saves model to pretrained/best_scale2.pth

**Note**: Training uses synthetically generated data (geometric shapes with SEM-realistic degradation). For production-grade results, provide real degraded/clean image pairs.

## Model Architecture

### Main Model: models/semr_physics_pro.py
- **Input**: Grayscale LR image (1 channel)
- **Output**: Restored HR image + uncertainty map (2×2 scale)
- **Encoder-Decoder**: U-Net-like architecture with skip connections
- **Bottleneck**: Multiple FETB blocks for deep feature processing

### Pretrained Weights
- **File**: pretrained/best_scale2.pth
- **Configuration**: scale=2, dim=12, num_blocks=[2,2,2,2], heads=4
- **Size**: ~8 MB

## Project Structure

``
SEMR-Physics-Pro/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── evaluate.py                     # Standalone inference script
├── train.py                        # Main training entry point
├── demo_app.py                     # Gradio interactive demo
├── demo.py                         # Alternative demo script
├── export_onnx.py                  # Export model to ONNX format
├── models/
│   ├── semr_physics_pro.py        # Main model architecture
│   ├── fetb.py                     # Frequency-Enhanced Transformer Block
│   ├── speckle_token.py            # Speckle-Aware Tokenization
│   ├── self_calibration.py         # Self-Calibrating HyperNetwork
│   ├── uncertainty_head.py         # Uncertainty estimation head
│   ├── wavelet.py                  # Haar DWT/IDWT implementations
│   └── __init__.py
├── losses/
│   ├── semr_loss.py                # Combined loss function
│   └── __init__.py
├── utils/
│   ├── dataset.py                  # Dataset and data generation utilities
│   ├── degradation.py              # SEM image degradation simulation
│   └── __init__.py
├── metrics/
│   ├── semiconductor_metrics.py    # Custom evaluation metrics
│   └── __init__.py
├── experiments/
│   ├── train_scale2.py             # 2× SR training script
│   ├── train_light.py              # Lightweight variant
│   ├── train_better.py             # Improved variant
│   ├── test_model.py               # Unit tests
│   ├── test_sample.py              # Quick test on synthetic data
│   └── __init__.py
├── pretrained/
│   ├── best_scale2.pth             # Pretrained weights (2× SR)
│   └── best_light.pth              # Lightweight variant weights
├── backend/                        # FastAPI backend for web service
│   ├── main.py
│   ├── uploads/
│   ├── results/
│   └── frontend/                   # React frontend
└── results/                        # Sample output images
``

## Loss Function

The model optimizes a multi-component loss function:

``
Loss = w_pix * L_charb + w_ssim * L_ssim + w_grad * L_grad + w_phys * L_phys + w_unc * L_unc
``

Where:
- **L_charb**: Charbonnier loss (robust pixel-wise error)
- **L_ssim**: Structural similarity loss (perceptual quality)
- **L_grad**: Gradient matching loss (edge preservation)
- **L_phys**: Physics consistency loss (downsampling correspondence)
- **L_unc**: Uncertainty-aware loss (epistemic uncertainty)

Default weights: (1.0, 0.0, 0.3, 0.1, 0.0)

## Performance

- **Inference Speed**: ~50-100 ms per 256×256 image (CPU), ~10-20 ms (GPU)
- **VRAM Usage**: ~200 MB (inference), ~1-2 GB (training)
- **Model Size**: ~8 MB

## Submission

This project was developed for the **SEMICON India 2026 Hackathon** - **KLA Challenge: AI-Based Restoration of Degraded Images for Semiconductor Inspection**.

### Demo
[Video Demo Placeholder - To be added by contributor]

## Requirements

See equirements.txt for the complete list. Key dependencies:
- 	orch>=2.0.0 - Deep learning framework
- opencv-python>=4.5.0 - Image processing
- 
umpy>=1.20.0 - Numerical computing
- scikit-image>=0.18.0 - Image metrics
- pytorch_msssim>=0.2.1 - MS-SSIM loss
- gradio>=6.0.0 - Interactive demo interface
- astapi>=0.100.0 - REST API backend
- 	qdm>=4.60.0 - Progress bars

## License

This project is provided as-is for the SEMICON India 2026 Hackathon.

## Contact

For questions or issues, please refer to the GitHub repository.

---

**Last Updated**: February 2026
