# Training Pipeline & Hyperparameter Documentation

## Overview

The training module trains the 6-output Causal Dilated Temporal Convolutional Network (TCN) on NinaPro DB8 Subject 1 data under strict data isolation protocols.

---

## Experimental Split Protocol

- **Training Partition:** `S1_E1_A1.mat` (Acquisition 1) — 35,817 sliding windows.
- **Validation Partition:** `S1_E1_A2.mat` (Acquisition 2) — 36,144 sliding windows.
- **Strictly Held-Out Test Partition:** `S1_E1_A3.mat` (Acquisition 3) — 8,264 sliding windows.
  - Never touched during gradient updates, learning rate scheduling, or early stopping decisions.

---

## Frozen Training Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| **Optimizer** | `Adam` | Stable convergence with adaptive per-parameter moments ($\beta_1=0.9, \beta_2=0.999$) |
| **Initial Learning Rate** | $1.0 \times 10^{-4}$ | Prevents gradient divergence in deep dilated stacks |
| **Weight Decay** | $1.0 \times 10^{-5}$ | $L_2$ regularization against overfitting |
| **Loss Function** | Mean Absolute Error (`L1Loss`) | Robust to non-Gaussian sEMG amplitude spikes compared to MSE |
| **Batch Size** | 64 | Balances GPU memory throughput with stochastic gradient noise |
| **Maximum Epochs** | 35 | Sufficient for asymptotic convergence |
| **Early Stopping** | Patience $= 7$ epochs | Evaluated on Session 2 validation loss |
| **LR Scheduler** | `ReduceLROnPlateau` | `factor=0.5`, `patience=3`, `min_lr=1e-6` on validation MAE |
| **Random Seed** | 42 | Full deterministic reproducibility |

---

## Convergence & Checkpoint History

- **Best Validation Epoch:** **Epoch 33**
- **Best Validation MAE:** `0.0863104` (on Session 2)
- **Total Training Duration:** $773.27\text{ seconds}$ (~12.9 minutes on NVIDIA GeForce RTX 3050 Laptop GPU)
- **Saved Checkpoint:** `ml/models/checkpoints/best_model_6ch.pt`

---

## Execution Command

To run full end-to-end training and evaluation:

```bash
# Using the root entrypoint:
python train.py --config configs/tcn_db8.yaml

# Or using the modular package entrypoint:
python ml/training/train.py --config ml/training/configs/tcn_db8.yaml
```
