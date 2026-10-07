# Checkpoint Metadata & Specification

## Authoritative Model Checkpoint

- **Primary Checkpoint File:** `best_model_6ch.pt`
- **Location:** `ml/models/checkpoints/best_model_6ch.pt` (and `results/best_model_6ch.pt`)
- **Compatibility Symlink / Copy:** `results/best_model.pt`
- **File Size:** 4,088,253 bytes (~3.90 MB)
- **SHA-256 Checksum:** `744fc0a602547c0bd167a2d7f478e1b15f04f2d40ce6e48e457239aa9380cd49`
- **Training Epoch:** 33 (converged out of 35 max epochs, early stopping patience 7)
- **Validation Metric:** Validation MAE = `0.0863104` on Session 2 (A2)
- **Random Seed:** 42

---

## Architectural Specifications

- **Model Name:** Causal Dilated Temporal Convolutional Network (Model C)
- **Input Dimensions:** `(Batch, 16, 256)` — 16 bipolar sEMG channels, 128.0 ms window at 2,000 Hz
- **Residual Blocks:** 6 stacked causal dilated blocks
- **Convolutions per Block:** 2 (standard Bai et al. 2018 design)
- **Kernel Size ($k$):** 3
- **Dilation Schedule ($d_l$):** $[1, 2, 4, 8, 16, 32]$
- **Hidden Channel Progression:** $[32, 64, 64, 128, 128, 128]$
- **Mathematical Receptive Field:**
  $$RF = 1 + 2 \times (3 - 1) \times (1 + 2 + 4 + 8 + 16 + 32) = 1 + 4 \times 63 = 253\text{ samples}\;(126.5\text{ ms})$$
  Covers **98.8%** of the 256-sample ($128.0\text{ ms}$) input buffer.
- **Total Trainable Parameters:** **332,966**
- **Linear Regression Head:** `Linear(128, 6)` mapping hidden representations directly to six continuous actuator outputs clamped to $[0.0, 1.0]$.
- **Normalization & Regularization:** `BatchNorm1d`, `LeakyReLU(negative_slope=0.01)`, `Dropout(p=0.20)`, $1 \times 1$ Conv shortcut projections.

---

## Six Continuous Actuator Outputs

1. **H1 ($y_0$):** Index Finger Flexion (CyberGlove MCP col 4 + PIP col 5)
2. **H2 ($y_1$):** Middle Finger Flexion (CyberGlove MCP col 6 + PIP col 7)
3. **H3 ($y_2$):** Ring Finger Flexion (CyberGlove MCP col 9 + PIP col 10)
4. **H4 ($y_3$):** Little Finger Flexion (CyberGlove MCP col 12 + PIP col 13)
5. **H5 ($y_4$):** Thumb Flexion (CyberGlove MCP col 1 + IP col 15)
6. **H6 ($y_5$):** Thumb Opposition / CMC Circumduction (CyberGlove CMC saddle col 0)

---

## ONNX Export & INT8 Quantization

- **ONNX Graph File:** `model_causal_tcn.onnx` (67,069 bytes / ~65.50 kB)
  - SHA-256: `200c69c163cd6a251d12749c773accc3de933a93d4db932e8512842189eabeb2`
  - Contains computational graph topology, nodes, dynamic axes, and metadata.
- **ONNX External Tensor Data:** `model_causal_tcn.onnx.data` (1,321,472 bytes / ~1,290.5 kB)
  - SHA-256: `33034d798640e1bece6a3850b69a842a0b939516edeaf3b4f2f748f735791397`
  - Stores uncompressed FP32 model weights.
  - Verification: Evaluated on test inputs; PyTorch and ONNX Runtime predictions match within `4.17e-7` maximum absolute error.
- **Dynamic INT8 Quantized Checkpoint:** `model_quantized_int8.pt` (1,371,101 bytes / ~1.37 MB)
  - SHA-256: `48ba7124809d267a8ee46388ce3eee5fe75ca1f10a5ccbc63b4fff6a3aaff77e`

### Critical Note on File Size vs Embedded Runtime RAM
> **Do NOT equate ONNX file size or weight file size with runtime RAM.**
> - **Weight Memory (INT8 Target):** $332,966 \times 1\text{ byte} = 325.16\text{ kB}$.
> - **Activation Buffer Memory:** $\approx 262\text{ kB}$ (intermediate feature maps for batch size 1).
> - **Runtime Overhead:** $\approx 60\text{ kB}$.
> - **Total Estimated Inference SRAM:** $\approx 647\text{ kB}$.
> An ESP32-S3 microcontroller possesses $512\text{ kB}$ internal SRAM with approximately $380\text{ kB}$ available to user applications; execution requires external octal PSRAM or streaming ring buffer optimizations.
