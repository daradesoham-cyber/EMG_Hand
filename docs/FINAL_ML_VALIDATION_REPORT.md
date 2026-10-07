# Final Machine Learning Validation Report: Six-Output sEMG Regression Validation for the Six-Actuator Prosthetic-Hand Control Architecture

**Date:** October 7, 2026
**Pipeline:** Causal Dilated Temporal Convolutional Network (TCN) for Continuous sEMG-to-Joint Kinematics
**Scope:** Six-output sEMG regression validation for the six-actuator prosthetic-hand control architecture
**Dataset:** NinaPro Database 8 (DB8) — Subject 1 Empirical Benchmark
**Evaluation Standard:** Strictly Held-Out Acquisition 3 (Session 3) — No hyperparameter tuning, model selection, or normalization fitting on test data.
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU / Intel Core CPU Baseline

---

## 1. Objective

The objective of this pipeline is to provide a **six-output sEMG regression validation for the six-actuator prosthetic-hand control architecture**.

Importantly, this machine-learning experiment validates:
$$\text{Forearm sEMG} \longrightarrow \text{Six continuous control / kinematic outputs } \mathbf{y}(t) \in [0, 1]^6$$
It does **NOT** claim physical experimental validation of the physical HASEL actuators themselves, which are modeled deterministically within the separate Phase 1 digital-twin plant. Rather, it validates that multi-channel surface EMG can reliably decode continuous proportional reference commands for the six prosthetic channels:
- **H1:** Index finger flexion command
- **H2:** Middle finger flexion command
- **H3:** Ring finger flexion command
- **H4:** Little finger flexion command
- **H5:** Thumb flexion command
- **H6:** Thumb opposition / CMC circumduction command

---

## 2. Dataset

The benchmark utilizes the **NinaPro Database 8 (DB8)** dataset:
- **Subject:** Subject 1 (intact limb, right hand).
- **Sensors:** 16 surface EMG channels placed circumferentially around the forearm; synchronized 18-sensor CyberGlove II recording contralateral hand joint kinematics.
- **Sampling Rate:** Hardware sampled at 1,111 Hz, sinc-interpolated to exactly $2,000\text{ Hz}$ in distributed `.mat` benchmark files.
- **Protocol:** 9 functional hand movements (individual finger flexions, thumb abduction, power grasp, precision pinches) plus rest.
- **Data Volume:** Total Subject 1 protocol duration is $2,572.7\text{ s}$ ($42.9\text{ min}$), comprising $5,135,095$ raw samples.

---

## 3. Experimental Split

To evaluate realistic prosthetic deployment and prevent data leakage:
- **Acquisition 1 (`S1_E1_A1.mat`):** Training Set. $1,146.3\text{ s}$ ($2,292,526$ frames, **35,817 sliding windows**).
- **Acquisition 2 (`S1_E1_A2.mat`):** Validation Set. $1,161.9\text{ s}$ ($2,313,470$ frames, **36,144 sliding windows**). Used exclusively for early stopping, architecture comparison (Model A vs B vs C), and EMA hyperparameter grid search.
- **Acquisition 3 (`S1_E1_A3.mat`):** Strictly Held-Out Test Set. $264.5\text{ s}$ ($529,099$ frames, **8,264 sliding windows**). **Untouched throughout architecture selection and hyperparameter tuning**; evaluated strictly once after model parameters were frozen.

---

## 4. Preprocessing

The signal conditioning pipeline is causal and streamable:
1. **DC Offset Removal:** Running mean subtraction.
2. **Bandpass Filtering:** 4th-order causal Butterworth IIR bandpass ($20.0\text{ Hz} \to 450.0\text{ Hz}$) implemented via Direct Form II Transposed (`scipy.signal.lfilter`).
3. **Mains Interference Notch:** 2nd-order causal IIR notch filter at $50.0\text{ Hz}$ ($Q = 30.0$).
4. **Z-Score Normalization:** Channel-wise zero-mean, unit-variance normalization fitted **strictly on Acquisition 1 (A1)**. Normalization statistics ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) are frozen and applied unchanged to A2 and A3.
5. **Causal Sliding Window:** Window length $W = 256\text{ samples}$ ($128.0\text{ ms}$ buffer), stride $S = 64\text{ samples}$ ($32.0\text{ ms}$ interval), generating $31.25\text{ predictions/s}$. The prediction at step $t$ corresponds strictly to the rightmost (newest) sample.

---

## 5. Causality Audit

As documented in `docs/CAUSALITY_AUDIT.md`, the pipeline enforces strict temporal causality across all stages:
- **Zero Phase Filtering (`filtfilt`) Prohibited:** No bidirectional or forward-backward filters are utilized.
- **Causal Convolutional Padding:** All 1D dilated convolutions use asymmetric left-side zero padding: $\text{pad}_{\text{left}} = (k - 1) \times d$, $\text{pad}_{\text{right}} = 0$.
- **No Future Data Leakage:** Empirical verification via gradient backpropagation demonstrates that gradients with respect to input indices $t' > t$ are identically zero.
- **Causal Post-Processing:** Exponential moving average uses strictly past outputs: $y_t = \alpha x_t + (1 - \alpha) y_{t-1}$.

---

## 6. TCN Architecture

The frozen primary architecture (**Model C**) consists of 6 stacked causal residual blocks:
- **Input Channels:** 16 bipolar sEMG channels.
- **Hidden Channels:** $[32, 64, 64, 128, 128, 128]$.
- **Kernel Size ($k$):** 3.
- **Dilations ($d_l$):** $[1, 2, 4, 8, 16, 32]$.
- **Convolutions per Block:** 2 (following standard Bai et al. 2018 design).
- **Regularization & Normalization:** BatchNorm1d, LeakyReLU ($\alpha = 0.01$), Dropout ($p = 0.20$).
- **Residual Connections:** $1 \times 1$ pointwise Conv1d shortcut projections when channel dimensions change.
- **Regression Head:** Linear layer ($128 \to 6$) with $[0, 1]$ clamping.

---

## 7. Receptive Field

The receptive field of a multi-layer dilated causal convolutional network with $M$ convolutions per residual block is given by:
$$RF = 1 + M (k - 1) \sum_{l=1}^{L} d_l$$
For Model C ($M=2, k=3, d \in [1, 2, 4, 8, 16, 32]$):
$$\sum d = 1 + 2 + 4 + 8 + 16 + 32 = 63$$
$$RF = 1 + 2 \times (3 - 1) \times 63 = 1 + 252 = \mathbf{253\text{ samples}}\;(\mathbf{126.50\text{ ms}})$$
Model C covers **98.8%** of the 256-sample ($128.0\text{ ms}$) input window, completely resolving the receptive field limitation of Model A ($RF = 61\text{ samples} / 30.5\text{ ms}$, which discarded 76.2% of the buffer).

---

## 8. Six-Actuator Mapping

As audited in `docs/DB8_ACTUATOR_MAPPING.md`, NinaPro DB8 CyberGlove II channels map 1-to-1 to the physical prosthetic hand:

| Output | Prosthetic Actuator | CyberGlove Columns | Anatomical Joint | Calibration Range ($q_{\min} \to q_{\max}$) |
| :---: | :--- | :--- | :--- | :---: |
| **$y_0$ (H1)** | Index Flexion | Col 4 (MCP) + Col 5 (PIP) | Index MCP & PIP flexion | $-1.2^\circ \to +94.8^\circ$ |
| **$y_1$ (H2)** | Middle Flexion | Col 6 (MCP) + Col 7 (PIP) | Middle MCP & PIP flexion | $-2.1^\circ \to +98.6^\circ$ |
| **$y_2$ (H3)** | Ring Flexion | Col 9 (MCP) + Col 10 (PIP) | Ring MCP & PIP flexion | $-1.8^\circ \to +89.2^\circ$ |
| **$y_3$ (H4)** | Little Flexion | Col 12 (MCP) + Col 13 (PIP) | Little MCP & PIP flexion | $-1.5^\circ \to +88.4^\circ$ |
| **$y_4$ (H5)** | Thumb Flexion | Col 1 (MCP) + Col 15 (IP) | Thumb MCP & IP flexion | $-55.2^\circ \to +68.4^\circ$ |
| **$y_5$ (H6)** | Thumb Opposition | Col 0 (CMC Rotation) | Trapeziometacarpal Saddle CMC | $+5.8^\circ \to +138.6^\circ$ |

Normalizer extrema were fitted strictly on Acquisition 1.

---

## 9. Training Procedure

- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-5}$).
- **Learning Rate:** Initial $10^{-4}$, scheduled via `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`.
- **Loss Function:** Mean Absolute Error (L1 Loss).
- **Batch Size:** 64 sliding windows.
- **Early Stopping:** Patience $= 7$ epochs evaluated on Session 2 (A2).
- **Convergence:** Best validation epoch occurred at **Epoch 33** (Val MAE $= 0.0863$). Total training duration was $773.3\text{ seconds}$ on GPU.

---

## 10. Baselines

To benchmark the causal TCN against non-temporal and recurrent alternatives on identical data splits:
1. **Oracle Persistence Reference:** $y_t = y_{t-1}^{\text{true}}$. Smoothness reference; requires real-time glove measurement, not an sEMG-only predictor.
2. **Linear Regression Baseline:** Linear model mapping 64 causal time-domain features (MAV, RMS, Variance, Waveform Length across 16 channels) directly to 6 outputs.
3. **Simple MLP Baseline:** 2-layer feedforward network on flattened $16 \times 256$ inputs ($128 \to 64 \to 6$).
4. **Causal GRU Baseline:** 2-layer causal Gated Recurrent Unit ($16 \to 64 \to 6$).
5. **Causal TCN Model C (Raw):** Frozen 6-block causal dilated TCN.
6. **Causal TCN Model C + EMA:** Frozen Model C with causal EMA ($\alpha = 0.20$).

---

## 11. Session-3 Results

Empirical results evaluated on strictly held-out Session 3 (8,264 windows, unseen during training/tuning):

| Model | MAE $\downarrow$ | RMSE $\downarrow$ | $R^2$ $\uparrow$ | Pearson $r$ $\uparrow$ | Jitter $\downarrow$ | Real-Time Latency (tested CPU) | Causal sEMG-Only |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Persistence Reference** | 0.0023 | 0.0080 | 0.9985 | 0.9992 | 0.0023 | N/A | NO (Oracle Feedback) |
| **Linear Regression (TD Features)** | 0.1065 | 0.1779 | 0.2164 | 0.6086 | 0.0252 | < 0.1 ms | YES |
| **Simple MLP Baseline** | 0.1199 | 0.2067 | 0.0219 | 0.3954 | 0.0730 | 0.42 ms | YES |
| **Causal GRU Baseline (Raw)** | **0.0726** | 0.1361 | **0.5345** | **0.7794** | 0.0264 | 57.35 ms (Exceeds budget) | YES |
| **Causal GRU Baseline + EMA ($\alpha=0.20$)** | **0.0653** | **0.1162** | **0.6483** | **0.8258** | 0.0069 | 57.35 ms (Exceeds budget) | YES |
| **Causal TCN Model C (Raw)** | 0.0811 | 0.1532 | 0.4133 | 0.7287 | 0.0251 | **2.23 ms (Feasible)** | YES |
| **Causal TCN Model C + EMA ($\alpha=0.20$)** | 0.0728 | 0.1348 | 0.5301 | 0.7718 | **0.0066** | **2.23 ms (Feasible)** | YES |

---

## 12. Per-Finger Results

Breakdown across all 6 actuator channels on held-out Session 3:

| Actuator Channel | Raw MAE | Raw RMSE | Raw $R^2$ | Raw $r$ | EMA MAE | EMA RMSE | EMA $R^2$ | EMA $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1: Index Flexion** | 0.0746 | 0.1605 | 0.4283 | 0.7187 | 0.0669 | 0.1435 | 0.5427 | 0.7728 |
| **H2: Middle Flexion** | 0.0823 | 0.1588 | 0.5164 | 0.7599 | 0.0742 | 0.1415 | 0.6165 | 0.8098 |
| **H3: Ring Flexion** | 0.0675 | 0.1320 | 0.6475 | 0.8391 | 0.0576 | 0.1091 | 0.7592 | 0.8860 |
| **H4: Little Flexion** | 0.0689 | 0.1344 | 0.7030 | 0.8534 | 0.0594 | 0.1078 | 0.8089 | 0.9024 |
| **H5: Thumb Flexion** | 0.1166 | 0.1881 | -0.1623 | 0.4410 | 0.1095 | 0.1753 | -0.0094 | 0.4713 |
| **H6: Thumb Opposition** | 0.0764 | 0.1453 | 0.3468 | 0.7603 | 0.0693 | 0.1318 | 0.4625 | 0.7884 |

**Scientific Observation on H5:**
The audit identified anatomical, target-construction, sensor-polarity, and low-variance factors that plausibly explain the reduced H5 performance. No implementation error was identified in the audited pipeline; however, the present experiment does not establish that these factors are the sole causes. Channel H5 is retained in the architecture to preserve the six-actuator command interface without synthetic manipulation.

---

## 13. Temporal Lag Analysis

Empirical cross-correlation delay analysis ($\Delta t = 32.0\text{ ms}$ per step):

| Configuration | Zero-Lag Correlation ($r_0$) | Best-Lag Correlation ($r^*$) | Mean Empirical Delay | Median Delay |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Causal TCN** | 0.7287 | 0.7383 | -133.3 ms | -192.0 ms |
| **Causal TCN + EMA ($\alpha=0.20$)** | 0.7718 | 0.7749 | -16.0 ms | -64.0 ms |
| **Causal GRU Baseline (Raw)** | 0.7794 | 0.7832 | -112.0 ms | -128.0 ms |
| **Causal GRU Baseline + EMA ($\alpha=0.20$)** | 0.8258 | 0.8281 | -5.3 ms | 0.0 ms |

**Scientific Interpretation of Temporal Lead:**
Cross-correlation identified an ensemble prediction lead of approximately $-133.3\text{ ms}$ relative to the measured CyberGlove displacement. This lead is physiologically plausible given the known temporal separation between neural/muscle activation and observable joint movement; however, **the present experiment does not independently isolate physiological electromechanical delay from model/windowing effects**. When post-filtered with EMA ($\alpha = 0.20$), the filter delay counteracts the observed lead, bringing mean alignment to $-16.0\text{ ms}$ relative to measured kinematics.

---

## 14. EMA Analysis

- **Formula:** $y_t = \alpha x_t + (1 - \alpha) y_{t-1}$.
- **Selected Parameter:** $\alpha = 0.20$ (selected on Session 2 validation grid).
- **Jitter Reduction:** Step-to-step sample derivative $|dy/dt|$ reduced from $0.0251$ to $0.0066$ (**73.6% reduction**).
- **Accuracy Effect:** Session 3 MAE improved from $0.0811$ to $0.0728$ ($\Delta = -0.0082$), while $R^2$ improved from $0.4133$ to $0.5301$.
- **Latency Trade-Off:** Theoretical time constant $\tau = -32.0 / \ln(1 - 0.20) = 143.4\text{ ms}$. Under dynamic motion, cross-correlation reveals effective response delay remains well within real-time tolerances.

---

## 15. Latency Benchmark

Measured across 500 single-window inferences ($B=1, C=16, L=256$):

| Processing Platform | Mean Latency | Median (p50) | 95th Percentile (p95) | 99th Percentile (p99) | Real-Time Slack (vs 32 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intel CPU Baseline** | **2.23 ms** | 2.19 ms | 2.43 ms | 2.71 ms | **+29.77 ms (93.0% headroom)** |
| **RTX 3050 Laptop GPU** | **1.50 ms** | 1.50 ms | 1.62 ms | 1.74 ms | **+30.50 ms (95.3% headroom)** |

Real-time inference feasibility was demonstrated on the tested CPU platform, with a measured mean TCN inference latency of 2.23 ms against the 32.0 ms inference deadline (31.25 Hz). In contrast, the evaluated GRU implementation required 57.35 ms (p95: 91.04 ms), exceeding the 32 ms real-time execution budget.

---

## 16. Model Size Audit

Detailed breakdown of memory structures:
- **Total Trainable Parameters:** 332,966.
- **Weight Memory (FP32):** $1,300.65\text{ kB}$ ($1.27\text{ MB}$, calculated as $332,966 \times 4\text{ bytes}$).
- **Weight Memory (FP16):** $650.32\text{ kB}$ ($332,966 \times 2\text{ bytes}$).
- **Weight Memory (INT8 Target):** $325.16\text{ kB}$ ($332,966 \times 1\text{ byte}$).
- **ONNX Graph File (`.onnx`):** $65.50\text{ kB}$ (contains computational graph topology, operator nodes, and tensor shapes only).
- **ONNX Data File (`.onnx.data`):** $1,300.65\text{ kB}$ (external binary container storing uncompressed FP32 tensor weights).
- **Quantized Checkpoint (`model_quantized_int8.pt`):** $1.4\text{ MB}$ (PyTorch state_dict container with quantized tensor parameters).
- **Peak Activation Memory:** Approximately $262\text{ kB}$ (estimated for double-buffered intermediate feature maps at batch size 1).
- **Total Estimated Inference SRAM:** Approximately $647\text{ kB}$ ($325\text{ kB}$ INT8 weights + $262\text{ kB}$ activations + ~$60\text{ kB}$ runtime overhead).

---

## 17. Limitations

1. **Thumb Flexion SNR & Anatomy:** Thumb flexion (H5) exhibits modest correlation ($r = 0.4713, R^2 \approx 0$) due to deep muscle placement of the flexor pollicis longus and superficial cross-talk.
2. **Subject Generalization Scope:** Evaluations were conducted strictly within Subject 1 across sessions. Inter-subject cross-validation was not evaluated here.
3. **Physical Loop Scope:** The experiment evaluated offline dataset streams and local host CPU inference latency. Physical HASEL actuator dynamics, sensor feedback, and clinical prosthetic control loops were not evaluated in this phase.

---

## 18. Deployment Status

- **Status:** **Prospective target deployment platform** identified as embedded microprocessor / ARM Cortex-M or Espressif ESP32-S3.
- **Hardware Testing Clarification:** Physical flashing and profiling on physical ESP32-S3 hardware was **NOT performed** in this phase. Latency numbers reflect local host CPU and GPU execution.
- **Memory Constraint:** An ESP32-S3 possesses $512\text{ kB}$ internal SRAM, with approximately $380\text{ kB}$ typically available for user applications. Because total estimated inference memory is $\approx 647\text{ kB}$, Model C cannot execute entirely within internal SRAM without external octal PSRAM or streaming buffer optimizations. The platform remains a prospective target.

---

## 19. Reproducibility

Full experimental parameters are recorded in `results/experiment_manifest.json`:
- **Random Seed:** 42.
- **Git Commit:** Recorded in manifest.
- **Python:** 3.14.0.
- **PyTorch:** 2.14.0+cu130.
- **Artifacts:**
  - Checkpoint: `results/best_model_6ch.pt`
  - Manifest: `results/experiment_manifest.json`
  - Per-output metrics: `results/metrics_per_output.csv`
  - Baselines: `results/baselines_comparison.csv`
  - Visual plots: `results/plots/1` through `8`

---

## 20. Final Conclusion & Baseline Freezing Declaration

> **ML BASELINE READY TO FREEZE**
>
> The six-output causal TCN baseline is ready to freeze as the computational ML control model for the prosthetic-hand architecture. The model was evaluated on a strictly held-out acquisition, satisfies the measured inference-time requirement on the tested CPU platform, and has been audited for temporal indexing, architecture comparison, memory accounting, and target construction.
>
> The result should not be interpreted as physical HASEL validation, end-to-end prosthetic control validation, physiological EMD measurement, ESP32 deployment validation, or clinical validation.
