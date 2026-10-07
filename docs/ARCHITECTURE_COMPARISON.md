# TCN Architecture Comparison & Receptive Field Optimization

**Date:** October 7, 2026
**Evaluation Standard:** Strictly Session 2 (Validation Acquisition `S1_E1_A2.mat`)
**Strict Protocol Compliance:** Session 3 (Held-Out Test `S1_E1_A3.mat`) was **NOT accessed** for architecture selection or EMA tuning.

---

## 1. Candidate Architectures

To address the receptive field deficiency identified in Problem 1 (where Model A covered only 23.8% of the 128 ms input buffer), three causal dilated convolutional architectures were trained on Acquisition 1 (A1) and evaluated on Acquisition 2 (A2):

| Parameter | Model A (Baseline) | Model B (Intermediate) | Model C (Full Window RF) |
| :--- | :---: | :---: | :---: |
| **Dilations** | `[1, 2, 4, 8]` | `[1, 2, 4, 8, 16]` | `[1, 2, 4, 8, 16, 32]` |
| **Residual Blocks** | 4 | 5 | 6 |
| **Hidden Channels** | `[32, 64, 64, 128]` | `[32, 64, 64, 128, 128]` | `[32, 64, 64, 128, 128, 128]` |
| **Convolutions / Block** | 2 | 2 | 2 |
| **Kernel Size ($k$)** | 3 | 3 | 3 |
| **Receptive Field ($RF$)** | **61 samples (30.5 ms)** | **125 samples (62.5 ms)** | **253 samples (126.5 ms)** |
| **Buffer Coverage Ratio** | **23.8%** of 128 ms | **48.8%** of 128 ms | **98.8%** of 128 ms |
| **Trainable Parameters** | 134,822 | 233,894 | 332,966 |
| **FP32 Memory Footprint** | 526.6 kB | 913.6 kB | 1,300.6 kB |
| **Single-Window CPU Latency** | 2.04 ms (p95: 3.49 ms) | 3.21 ms (p95: 3.49 ms) | 4.79 ms (p95: 5.02 ms) |

---

## 2. Empirical Validation Results on Session 2 (A2)

All models were evaluated on the 36,144 sliding windows of Session 2 using 6 continuous output channels ($y \in [0, 1]^6$):

| Model Candidate | Receptive Field | Val MAE $\downarrow$ | Val RMSE $\downarrow$ | Val $R^2$ $\uparrow$ | Pearson $r$ $\uparrow$ | Output Jitter $\downarrow$ | CPU Latency (p95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Current)** | 30.50 ms | 0.1073 | 0.1787 | 0.3486 | 0.6197 | 0.0556 | 3.49 ms |
| **Model B** | 62.50 ms | 0.1019 | 0.1695 | 0.3985 | 0.6663 | 0.0435 | 3.49 ms |
| **Model C (Full Window)** | **126.50 ms** | **0.0941** | **0.1533** | **0.5019** | **0.7304** | **0.0295** | **5.02 ms** |

### Quantitative Comparison:
1. **Kinematic Variance Explained ($R^2$):** Model C achieves $R^2 = 0.5019$ vs $0.3486$ for Model A (+44.0% relative improvement).
2. **Correlation ($r$):** Improves from $0.6197$ (Model A) to $0.7304$ (Model C).
3. **Tracking Error (MAE):** Decreases by $12.3\%$ (0.1073 $\to$ 0.0941).
4. **Trajectory Smoothness (Jitter):** Raw output jitter decreases by $46.9\%$ (0.0556 $\to$ 0.0295), indicating that having full temporal context over the entire 128 ms window prevents high-frequency chattering.
5. **Computational Feasibility:** Model C executes in **4.79 ms on CPU** (p95: 5.02 ms). Given a stride of 32 ms ($31.25\text{ Hz}$ update rate), Model C consumes only $15.0\%$ of the available computation budget, leaving $27\text{ ms}$ of processing headroom for sensing, filtering, and CAN-bus communication.

---

## 3. EMA Alpha ($\alpha$) Grid Search on Session 2

To tune the post-processing filter without touching Session 3, an empirical grid search over $\alpha \in [0.10, 0.15, 0.20, 0.30, 0.40]$ was performed on Session 2 predictions:

| Model Architecture | Smoothing ($\alpha$) | MAE $\downarrow$ | RMSE $\downarrow$ | $R^2$ $\uparrow$ | Pearson $r$ $\uparrow$ | Jitter | Jitter Reduction | Zero-Lag $r$ | Best-Lag $r$ | Empirical Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model C** | **RAW (None)** | 0.0941 | 0.1533 | 0.5019 | 0.7304 | 0.0295 | 0.0% | 0.7304 | 0.7387 | -128.0 ms |
| Model C | $\alpha = 0.10$ | 0.0827 | 0.1293 | 0.6285 | 0.7947 | 0.0044 | 85.2% | 0.7947 | 0.7996 | +101.3 ms |
| Model C | $\alpha = 0.15$ | 0.0840 | 0.1316 | 0.6185 | 0.7894 | 0.0058 | 80.2% | 0.7894 | 0.7917 | +21.3 ms |
| **Model C (Selected)** | **$\alpha = 0.20$** | **0.0852** | **0.1341** | **0.6068** | **0.7831** | **0.0073** | **75.4%** | **0.7831** | **0.7854** | **-16.0 ms** |
| Model C | $\alpha = 0.30$ | 0.0872 | 0.1383 | 0.5864 | 0.7721 | 0.0101 | 65.9% | 0.7721 | 0.7758 | -64.0 ms |
| Model C | $\alpha = 0.40$ | 0.0887 | 0.1415 | 0.5699 | 0.7634 | 0.0128 | 56.7% | 0.7634 | 0.7684 | -85.3 ms |

### Engineering Trade-Off Analysis:
- $\alpha = 0.10$ achieves the highest mathematical $R^2$ (0.6285), but introduces an empirical lag of $+101.3\text{ ms}$ (> 3 frames of delay), which exceeds the acceptable prosthetic control latency threshold ($< 100\text{ ms}$ total).
- $\alpha = 0.20$ achieves $75.4\%$ jitter reduction with near-zero phase distortion (mean delay $-16.0\text{ ms}$, zero-lag $r = 0.7831$ within $0.002$ of best-lag $r = 0.7854$).

---

## 4. Final Architecture Decision

Based strictly on Session 2 validation performance:
1. **Primary Model Architecture:** **Model C** (`dilations=[1, 2, 4, 8, 16, 32]`, `hidden_channels=[32, 64, 64, 128, 128, 128]`, receptive field = 253 samples / 126.5 ms).
2. **Post-Processing Filter:** Causal EMA with **$\alpha = 0.20$**.
3. **Freezing Policy:** The architecture, receptive field parameters, normalizer statistics, and EMA parameter $\alpha$ are now **FROZEN**. Session 3 will be evaluated strictly once against this frozen model and baselines.
