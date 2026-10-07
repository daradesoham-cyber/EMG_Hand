# NinaPro DB8 Causal TCN Machine Learning Results Report

**Date:** October 7, 2026
**Pipeline:** Causal Dilated Temporal Convolutional Network (TCN) for Continuous sEMG-to-Joint Kinematics
**Dataset:** NinaPro Database 8 (DB8) — Subject 1 Empirical Benchmark
**Evaluation Standard:** Strictly Held-Out Acquisition 3 (Session 3) — No hyperparameter tuning, model selection, or normalization fitting on test data.

---

## 1. Complete Pipeline Architecture Overview

$$\text{16-Ch Raw sEMG (2 kHz)} \longrightarrow \text{Causal IIR Filter (20-450 Hz + 50 Hz Notch)} \longrightarrow \text{Z-score Normalization (Fitted on A1)} \longrightarrow \text{256-Sample Window (128 ms)}$$
$$\longrightarrow \text{6-Block Causal Dilated TCN (16}\to\text{32}\to\text{64}\to\text{64}\to\text{128}\to\text{128}\to\text{128)} \longrightarrow \text{Linear Head} \longrightarrow \text{EMA Filter ($\alpha=0.20$)} \longrightarrow \text{6 HASEL Channels}$$

---

## 2. Dataset & Sample Counts

| Stage | Acquisition File | Raw Duration | Total Samples | Causal Windows ($W=256, S=64$) | Role in Protocol |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | `S1_E1_A1.mat` | 1,146.3 s (19.1 min) | 2,292,526 | **35,817 windows** | Model optimization & normalizer fitting |
| **Validation Set** | `S1_E1_A2.mat` | 1,161.9 s (19.4 min) | 2,313,470 | **36,144 windows** | Architecture selection & EMA grid search |
| **Strict Test Set** | `S1_E1_A3.mat` | 264.5 s (4.4 min) | 529,099 | **8,264 windows** | **STRICT UNSEEN HELD-OUT EVALUATION** |
| **Total Protocol** | All 3 Sessions | 2,572.7 s (42.9 min) | 5,135,095 | **80,225 windows** | Full Subject 1 Protocol |

---

## 3. Signal Conditioning & Preprocessing Parameters

- **Hardware Sampling Rate:** Native 1,111 Hz, sinc-interpolated to $2,000\text{ Hz}$ in benchmark files.
- **DC Handling:** Offline mean removal verified; running IIR Direct Form II Transposed filter.
- **Bandpass Filter:** 4th-order causal Butterworth bandpass, $20.0\text{ Hz}$ to $450.0\text{ Hz}$ ($f_s = 2,000\text{ Hz}$). Causal `scipy.signal.lfilter` forward pass only; zero-phase `filtfilt` strictly prohibited.
- **Powerline Notch:** Causal IIR notch filter at $50.0\text{ Hz}$ ($Q = 30.0$).
- **Normalization:** Channel-wise Z-score normalization $(\mathbf{x} - \mathbf{\mu}_{\text{A1}}) / \mathbf{\sigma}_{\text{A1}}$ fitted **exclusively on Acquisition 1**.
- **Window Parameters:**
  - Window length: $128\text{ ms} = 256\text{ samples}$
  - Update stride: $32\text{ ms} = 64\text{ samples}$
  - Inference frequency: $2000 / 64 = 31.25\text{ Hz}$

---

## 4. Receptive Field Optimization (Model A vs B vs C)

Evaluated strictly on Session 2 (Validation Set):

| Architecture | Dilations | Residual Blocks | Receptive Field | Buffer Coverage | Val MAE | Val $R^2$ | Val Correlation | Output Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Baseline)** | `[1, 2, 4, 8]` | 4 | 61 samples (30.5 ms) | 23.8% | 0.1073 | 0.3486 | 0.6197 | 0.0556 |
| **Model B** | `[1, 2, 4, 8, 16]` | 5 | 125 samples (62.5 ms) | 48.8% | 0.1019 | 0.3985 | 0.6663 | 0.0435 |
| **Model C (Selected)** | `[1, 2, 4, 8, 16, 32]` | 6 | **253 samples (126.5 ms)** | **98.8%** | **0.0941** | **0.5019** | **0.7304** | **0.0295** |

Model C captures the entire 128 ms muscular pre-activation window, yielding a 44.0% relative improvement in explained kinematic variance.

---

## 5. Model Specifications & Memory Footprint

| Metric | Measured Empirical Value |
| :--- | :--- |
| **Model Class** | `CausalTCN` (Model C) |
| **Trainable Parameters** | **332,966 parameters** |
| **Weight Memory (FP32)** | **1,300.65 kB** (1.27 MB) |
| **Weight Memory (FP16)** | **650.32 kB** |
| **Weight Memory (INT8 Target)** | **325.16 kB** |
| **ONNX Graph File (`.onnx`)** | **65.50 kB** (Graph topology only) |
| **ONNX Data File (`.onnx.data`)** | **1,300.65 kB** (Tensor weights) |
| **INT8 Checkpoint File** | **1.4 MB** |
| **Single-Window CPU Latency** | **2.23 ms** (p95: 2.43 ms) |
| **Single-Window GPU Latency** | **1.50 ms** (p95: 1.62 ms) |
| **Real-Time Headroom** | **92.4%** slack relative to 32.0 ms deadline |

---

## 6. Baselines Comparison on Strictly Held-Out Session 3

| Model | Session 3 MAE | Session 3 RMSE | Session 3 $R^2$ | Pearson $r$ | Step Jitter | CPU Latency (mean) | Causal sEMG-Only |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Persistence Reference** | 0.0023 | 0.0080 | 0.9985 | 0.9992 | 0.0023 | N/A | NO (Oracle Feedback) |
| **Linear Regression (TD Features)** | 0.1065 | 0.1779 | 0.2164 | 0.6086 | 0.0252 | < 0.1 ms | YES |
| **Simple MLP Baseline** | 0.1199 | 0.2067 | 0.0219 | 0.3954 | 0.0730 | 0.42 ms | YES |
| **Causal GRU Baseline (Raw)** | **0.0726** | 0.1361 | **0.5345** | **0.7794** | 0.0264 | 57.35 ms (Exceeds budget) | YES |
| **Causal GRU Baseline + EMA ($\alpha=0.20$)** | **0.0653** | **0.1162** | **0.6483** | **0.8258** | 0.0069 | 57.35 ms (Exceeds budget) | YES |
| **Causal TCN Model C (Raw)** | 0.0811 | 0.1532 | 0.4133 | 0.7287 | 0.0251 | **2.23 ms (Feasible)** | YES |
| **Causal TCN Model C + EMA ($\alpha=0.20$)** | 0.0728 | **0.1348** | 0.5301 | 0.7718 | **0.0066** | **2.23 ms (Feasible)** | YES |

---

## 7. Per-Actuator Kinematic Performance (Session 3)

| Actuator Channel | Raw MAE | Raw RMSE | Raw $R^2$ | Raw $r$ | EMA MAE | EMA RMSE | EMA $R^2$ | EMA $r$ | Empirical Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1: Index Flexion** | 0.0746 | 0.1605 | 0.4283 | 0.7187 | 0.0669 | 0.1435 | 0.5427 | 0.7728 | -96.0 ms |
| **H2: Middle Flexion** | 0.0823 | 0.1588 | 0.5164 | 0.7599 | 0.0742 | 0.1415 | 0.6165 | 0.8098 | -64.0 ms |
| **H3: Ring Flexion** | 0.0675 | 0.1320 | 0.6475 | 0.8391 | 0.0576 | 0.1091 | 0.7592 | 0.8860 | -96.0 ms |
| **H4: Little Flexion** | 0.0689 | 0.1344 | 0.7030 | 0.8534 | 0.0594 | 0.1078 | 0.8089 | 0.9024 | -96.0 ms |
| **H5: Thumb Flexion** | 0.1166 | 0.1881 | -0.1623 | 0.4410 | 0.1095 | 0.1753 | -0.0094 | 0.4713 | +160.0 ms |
| **H6: Thumb Opposition** | 0.0764 | 0.1453 | 0.3468 | 0.7603 | 0.0693 | 0.1318 | 0.4625 | 0.7884 | +96.0 ms |

---

## 8. Publication Figures Generated in `results/plots/`

1. `1_training_validation_loss.png` — Convergence curves over 35 epochs.
2. `2_predicted_vs_ground_truth.png` — Scatter plots and ideal lines for all 6 actuators.
3. `3_time_series_overlay.png` — Multi-channel continuous trajectory tracking.
4. `4_per_finger_error.png` — MAE and RMSE bar chart by digit.
5. `5_residual_distribution.png` — Error probability density histograms.
6. `6_correlation_scatter.png` — Empirical correlation and best-fit slopes.
7. `7_prediction_error_over_time.png` — Temporal tracking error envelope.
8. `8_jitter_and_ema_comparison.png` — 73.6% sample-to-sample derivative jitter reduction.
