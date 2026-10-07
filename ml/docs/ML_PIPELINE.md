# Machine Learning Pipeline Architecture

## 1. End-to-End Pipeline Data Flow

```text
Raw Forearm sEMG (16 bipolar channels @ 2,000 Hz)
   │
   ▼
[1] DC Offset Removal & Causal IIR Bandpass (4th-order Butterworth, 20.0 - 450.0 Hz)
   │
   ▼
[2] Causal IIR Notch Filter (2nd-order, 50.0 Hz, Q = 30.0)
   │
   ▼
[3] Channel-Wise Z-Score Normalization (μ, σ strictly fitted on Session 1 Training Set)
   │
   ▼
[4] Causal Sliding Windowing (Buffer W = 256 samples / 128.0 ms, Stride S = 64 samples / 32.0 ms)
   │   Target aligned strictly at window endpoint: y(t) = target[end_idx - 1]
   ▼
[5] Causal Dilated Temporal Convolutional Network (Model C)
   │   6 Residual Blocks, Dilations [1, 2, 4, 8, 16, 32], Kernel k = 3, Convs/Block = 2
   │   Receptive Field: 253 samples (126.5 ms), Parameters: 332,966
   ▼
[6] Linear Regression Head + Clamp: 6 continuous outputs in [0.0, 1.0]
   │
   ▼
[7] Causal Exponential Moving Average (EMA) Post-Filter (α = 0.20, τ = 143.4 ms)
   │   Jitter reduced by 73.6%, Empirical delay = -16.0 ms relative to kinematics
   ▼
Six Continuous HASEL Actuator Reference Commands [H1, H2, H3, H4, H5, H6] ∈ [0.0, 1.0]^6
```

---

## 2. Signal Conditioning & Preprocessing

- **Direct Form II Transposed IIR Filtering:** Executed using `scipy.signal.lfilter`.
- **Zero Future Leakage:** Strictly forward-only filtering. Zero-phase bidirectional filtering (`filtfilt`) is strictly forbidden because it accesses future samples that do not exist during real-time online inference.
- **Normalization Isolation:** Mean and standard deviation vectors ($\boldsymbol{\mu}, \boldsymbol{\sigma} \in \mathbb{R}^{16}$) are computed exclusively on Acquisition 1 (`S1_E1_A1.mat`). The frozen normalization parameters are applied to Acquisition 2 (`A2`) and Acquisition 3 (`A3`) without re-estimation.

---

## 3. Causal Sliding Windowing & Temporal Indexing

- **Window Duration:** $128.0\text{ ms}$ ($W = 256\text{ samples}$ at $2\text{ kHz}$).
- **Inference Stride:** $32.0\text{ ms}$ ($S = 64\text{ samples}$ at $2\text{ kHz}$).
- **Prediction Frequency:** $f_{\text{update}} = \frac{2000}{64} = 31.25\text{ Hz}$ (one control vector generated every $32.0\text{ ms}$).
- **Endpoint Alignment:** For window interval $[t - W + 1, \dots, t]$, the target is evaluated at time $t$. No lookahead or centering is applied.

---

## 4. Model Architecture: Causal TCN (Model C)

### Residual Block Structure
Each of the 6 residual blocks contains:
1. `CausalConv1d(in_ch, out_ch, kernel_size=3, dilation=d)` with asymmetric left-padding $(k-1) \times d$.
2. `BatchNorm1d(out_ch)`
3. `LeakyReLU(negative_slope=0.01)`
4. `Dropout(p=0.20)`
5. `CausalConv1d(out_ch, out_ch, kernel_size=3, dilation=d)` with asymmetric left-padding $(k-1) \times d$.
6. `BatchNorm1d(out_ch)`
7. `LeakyReLU(negative_slope=0.01)`
8. `Dropout(p=0.20)`
9. Residual addition: $1 \times 1$ pointwise convolution if channel count changes, else identity.

### Receptive Field Derivation
$$RF = 1 + M (k - 1) \sum_{l=1}^{L} d_l = 1 + 2 \times (3 - 1) \times (1 + 2 + 4 + 8 + 16 + 32) = 1 + 4 \times 63 = 253\text{ samples}\;(126.5\text{ ms})$$
Coverage: $\frac{253}{256} = 98.8\%$ of the temporal buffer.

---

## 5. Actuator Command Mapping (H1–H6)

The model regression head outputs a continuous vector $\mathbf{y} \in [0.0, 1.0]^6$:
- **H1:** Index finger flexor reference command
- **H2:** Middle finger flexor reference command
- **H3:** Ring finger flexor reference command
- **H4:** Little finger flexor reference command
- **H5:** Thumb flexion reference command (MCP + IP)
- **H6:** Thumb opposition reference command (CMC saddle rotation / circumduction)

These continuous outputs provide proportional normalized reference setpoints for the downstream electrohydraulic actuator controller.
