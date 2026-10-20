# SEMR-Physics Pro

Image restoration experiments for semiconductor inspection, developed for the SEMICON India 2026 KLA challenge.

The project trains a PyTorch model to restore low-resolution, degraded grayscale images at 2× scale. It combines image-processing ideas with learned restoration and produces an uncertainty output alongside the restored image. Training examples in the supplied workflow are synthetic; results on real SEM images need separate evaluation.

## Approach

The model in `models/semr_physics_pro.py` combines:

- **Speckle-aware tokenization** to handle multiplicative noise.
- **Frequency-enhanced transformer blocks** for spatial and wavelet-domain processing.
- **A self-calibration head** for image-dependent degradation estimates.
- **An uncertainty head** that produces a map alongside the restored image.

The 2× training configuration uses a compact encoder/decoder with skip connections. Implementation details live in `models/`, `losses/`, and `utils/`.

## Setup

Use Python with the dependencies in `requirements.txt`. A CUDA-capable GPU can accelerate training, but the scripts also support CPU execution.

```powershell
git clone https://github.com/omghotekar01-dotcom/SEMR-Physics-Pro.git
cd SEMR-Physics-Pro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Run a restoration

The evaluator accepts a directory of input images and writes restored images to the selected output directory. By default it reads `pretrained/best_scale2.pth`.

```bash
python evaluate.py --test_dir ./input_images --output_dir ./restored_images
```

You can supply a different checkpoint using `--model_path`. Input images may be PNG, JPEG, BMP, or TIFF. Output filenames begin with `restored_`.

For the interactive Gradio demo:

```bash
python demo_app.py
```

The demo loads `pretrained/best_scale2.pth` and opens a local interface for viewing the restored image and an uncertainty overlay. It requires the checkpoint file to be present.

## Training

To train the 2× configuration on the synthetic shape dataset:

```bash
python experiments/train_scale2.py
```

The script generates clean shapes, applies the dataset's degradation pipeline, trains for 10 epochs, and writes its checkpoint to `pretrained/best_scale2.pth`.

Important files:

| Path | Description |
| --- | --- |
| `evaluate.py` | Batch inference and output-image export |
| `demo_app.py` | Gradio comparison interface |
| `experiments/train_scale2.py` | Synthetic-data training run |
| `models/` | Restoration model and its component blocks |
| `losses/` | Combined training loss |
| `utils/` | Dataset and degradation utilities |
| `metrics/` | Image-quality measurement helpers |
| `backend/` | Separate web-service experiment |

## Interpretation and limitations

Synthetic degradations are useful for controlled experiments, but they do not establish performance on real inspection data. Quality and timing depend on the input set, checkpoint, and hardware. Validate restored defects against real reference images before considering any inspection-related use.

The uncertainty overlay is a model output, not a calibrated probability that a visual feature is correct. Keep source images and evaluation results when comparing checkpoints.

## Background

This repository is a hackathon research prototype for semiconductor image restoration. It is provided as-is for experimentation and review.
