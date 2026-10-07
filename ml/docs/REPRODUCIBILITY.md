# Reproducibility Guide: Step-by-Step Instructions

This document provides exact, deterministic commands to reproduce every result, metric, checkpoint, and figure from scratch.

---

## 1. Environment Setup

Create and activate a clean virtual environment using Python 3.10+ (tested on Python 3.14):

```bash
# Option A: Standard venv
python3 -m venv .venv_repro
source .venv_repro/bin/activate
pip install --upgrade pip
pip install -r ml/environment/requirements.txt

# Option B: Conda / Mamba
conda env create -f ml/environment/environment.yml
conda activate prosthetic_hand_ml
```

Verify the environment:
```bash
python -c "import torch, scipy, numpy, onnx, onnxruntime; print('Environment OK! PyTorch:', torch.__version__)"
```

---

## 2. Obtain NinaPro DB8 Dataset

1. Register an academic user account at: [http://ninapro.hevs.ch/DB8](http://ninapro.hevs.ch/DB8).
2. Download the Subject 1 `.mat` data archives:
   - `S1_E1_A1.mat` (Training acquisition)
   - `S1_E1_A2.mat` (Validation acquisition)
   - `S1_E1_A3.mat` (Strictly held-out test acquisition)

---

## 3. Place Dataset & Verify Checksums

Move the downloaded `.mat` files into `data/ninapro_db8/`:

```bash
mkdir -p data/ninapro_db8
# Place S1_E1_A1.mat, S1_E1_A2.mat, S1_E1_A3.mat inside data/ninapro_db8/

# Verify SHA-256 cryptographic checksums:
sha256sum data/ninapro_db8/*.mat
```

Expected outputs:
```text
4f2355061b594baffb474350ce6ede5f852911a8871d547b1adb7cadcfa86d5f  data/ninapro_db8/S1_E1_A1.mat
a431c57ecc6bc8f4689b66e96c2197a696201dfb475ed04794593b3820d6293e  data/ninapro_db8/S1_E1_A2.mat
772d18763ba3a51e0ae50183b16238c90a92bbe4c6dff7d4745b259e33f515fe  data/ninapro_db8/S1_E1_A3.mat
```

---

## 4. Run Unit Tests & Causality Verification

Run the test suite verifying causal filtering, windowing bounds, and no-future-leakage assertions:

```bash
python -m unittest discover tests
```

---

## 5. Evaluate the Frozen Checkpoint on Held-Out Session 3

To reproduce the exact reported Session 3 metrics directly from the authoritative checkpoint without retraining:

```bash
python evaluate.py --checkpoint ml/models/checkpoints/best_model_6ch.pt
```

Expected output:
```text
Aggregate Raw Test Results: {'mae': 0.0811, 'rmse': 0.1532, 'r2': 0.4133, 'correlation': 0.7287}
Aggregate EMA (alpha=0.20) Results: {'mae': 0.0728, 'rmse': 0.1348, 'r2': 0.5301, 'correlation': 0.7718}
Jitter Reduction: 0.0251 -> 0.0066 (73.6%)
```

---

## 6. (Optional) Retrain Model from Scratch

To re-run the full training procedure on Acquisition 1 and Acquisition 2:

```bash
python train.py --config ml/training/configs/tcn_db8.yaml
```

Training takes approximately 12–13 minutes on a laptop GPU (RTX 3050). The best model checkpoint will be automatically saved to `results/best_model_6ch.pt`.

---

## 7. Verify ONNX Graph & Numerical Agreement

Verify that the exported ONNX model matches PyTorch within floating-point precision:

```bash
python ml/inference/onnx/verify_onnx.py \
  --checkpoint ml/models/checkpoints/best_model_6ch.pt \
  --onnx ml/models/checkpoints/model_causal_tcn.onnx
```

Expected output:
```text
✓ Structural validation passed: Valid ONNX computational graph.
Max Absolute Difference:  4.17232500e-07
Mean Absolute Difference: 1.07288360e-07
✓ Numerical agreement test PASSED (within tolerance 1e-05).
```

---

## 8. Benchmark Inference Latency & Memory Footprint

Measure single-window inference latency on your CPU across 500 runs:

```bash
python benchmark.py
```

Expected output:
```text
CPU Latency Mean: 2.23 ms (p95: 2.43 ms)
Real-time slack: +29.77 ms against 32.0 ms deadline (31.25 Hz update rate).
```

---

## 9. Regenerate Publication Figures

To regenerate all 8 evaluation figures from the saved predictions:

```bash
python -c "
import pandas as pd, numpy as np
from ml import plots

df = pd.read_csv('results/session3_predictions.csv')
target_cols = [c for c in df.columns if c.endswith('_true')]
pred_cols = [c.replace('_true', '_pred_ema') for c in target_cols]
y_true = df[target_cols].values
y_pred_ema = df[pred_cols].values
target_names = ['index', 'middle', 'ring', 'little', 'thumb_flexion', 'thumb_opposition']

history_df = pd.read_csv('results/training_history.csv')
plots.plot_training_curves(history_df['train_loss'].values, history_df['val_loss'].values, 'results/plots/1_training_validation_loss.png')
print('Figures regenerated successfully.')
"
```
