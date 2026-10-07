# Current Authoritative ML Baseline

## Declaration of Authoritative Model

This document establishes the exact, authoritative machine-learning baseline for the parameterized six-HASEL prosthetic hand control pipeline.

---

### Authoritative Checkpoint File

- **File Name:** `best_model_6ch.pt`
- **Primary Paths:**
  - `ml/models/checkpoints/best_model_6ch.pt`
  - `results/best_model_6ch.pt`
- **Compatibility Copy:**
  - `results/best_model.pt` (Contains identical model weights and configuration; retained for legacy scripts)
- **SHA-256 Checksum:** `744fc0a602547c0bd167a2d7f478e1b15f04f2d40ce6e48e457239aa9380cd49`

---

### Architectural Identity

- **Architecture:** Causal Dilated Temporal Convolutional Network (TCN) Model C
- **Input Channels:** 16 bipolar sEMG channels (forearm circumferential array)
- **Input Buffer Duration:** 128.0 ms (256 samples at 2,000 Hz)
- **Inference Stride:** 32.0 ms (64 samples at 2,000 Hz $\to$ 31.25 Hz update rate)
- **Residual Blocks:** 6 stacked causal residual blocks
- **Convolutions per Block:** 2 (kernel size = 3)
- **Dilations:** $[1, 2, 4, 8, 16, 32]$
- **Hidden Channels:** $[32, 64, 64, 128, 128, 128]$
- **Receptive Field:** 253 samples ($126.5\text{ ms}$, covering 98.8% of the 256-sample buffer)
- **Parameter Count:** **332,966 trainable parameters**
- **Outputs:** 6 continuous regression commands in $[0.0, 1.0]$:
  1. H1 Index Flexion
  2. H2 Middle Flexion
  3. H3 Ring Flexion
  4. H4 Little Flexion
  5. H5 Thumb Flexion
  6. H6 Thumb Opposition

---

### Verified Benchmark Performance (Strictly Held-Out Session 3 / A3)

| Metric | Raw Causal TCN | Causal TCN + EMA ($\alpha=0.20$) | Note |
| :--- | :---: | :---: | :--- |
| **MAE** | 0.0811 | **0.0728** | Mean Absolute Error |
| **RMSE** | 0.1532 | **0.1348** | Root Mean Squared Error |
| **$R^2$** | 0.4133 | **0.5301** | Kinematic variance explained |
| **Pearson $r$** | 0.7287 | **0.7718** | Linear correlation |
| **Step-to-Step Jitter** | 0.0251 | **0.0066** | **73.6% reduction** |
| **Mean Empirical Lag** | -133.3 ms | **-16.0 ms** | Counteracts biological neural lead |
| **Mean CPU Latency** | 2.23 ms | **2.23 ms** | Intel host baseline (32 ms deadline) |
| **p95 CPU Latency** | 2.43 ms | **2.43 ms** | Real-time slack $> 29.5\text{ ms}$ |

---

### Duplicate Check & Clarifications

1. **`best_model_6ch.pt` vs `best_model.pt`**:
   - Both checkpoints store the identical 6-channel state dictionary (`regression_head.weight` shape `[6, 128]`) trained to Epoch 33 with validation loss `0.0863104`.
   - `best_model_6ch.pt` is the authoritative reference name. `best_model.pt` is kept strictly for backwards compatibility.
2. **`model_causal_tcn.onnx`**:
   - The authoritative exported ONNX graph matching `best_model_6ch.pt`.
   - Verified numerically against PyTorch with maximum absolute difference $< 4.18 \times 10^{-7}$.
3. **`model_quantized_int8.pt`**:
   - Dynamic INT8 quantized PyTorch model.
