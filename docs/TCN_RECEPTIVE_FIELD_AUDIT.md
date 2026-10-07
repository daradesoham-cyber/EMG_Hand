# TCN Receptive Field Audit & Mathematical Derivation

**Date:** October 7, 2026
**Module Audited:** `ml/model.py` (`CausalTCN`, `CausalResidualBlock`, `CausalConv1d`)
**Audit Purpose:** Resolve discrepancy between the 128 ms input window and the effective temporal receptive field of the causal TCN.

---

## 1. Computational Graph Architecture Inspection

Direct inspection of `ml/model.py` reveals the exact computational implementation:

- **Input Dimension:** $\mathbf{X} \in \mathbb{R}^{B \times 16 \times 256}$ (Batch, 16 sEMG Channels, 256 time steps at $2\text{ kHz} \implies 128.0\text{ ms}$).
- **Residual Blocks:** 4 sequential blocks with dilations $\mathbf{d} = [1, 2, 4, 8]$.
- **Kernel Size:** $k = 3$ throughout all convolutional layers.
- **Convolutions per Block:** `convs_per_block = 2`. Each block contains:
  1. Causal Conv1D ($k=3, d$) with left-padding of $(k-1)d$ zeros
  2. `BatchNorm1d`
  3. `LeakyReLU` ($\alpha=0.01$)
  4. `Dropout(0.2)`
  5. Causal Conv1D ($k=3, d$) with left-padding of $(k-1)d$ zeros
  6. `BatchNorm1d`
  7. `LeakyReLU` ($\alpha=0.01$)
  8. `Dropout(0.2)`
- **Residual Shortcut:** $1 \times 1$ Conv1D (when channel dimension changes) or `Identity` (when channels match).
- **Pooling / Downsampling:** None. Stride is strictly $1$ throughout the entire temporal dimension. Sequence length remains $L = 256$ across all layers.
- **Prediction Head:** Linear layer applied strictly to the final time step:
  $$\hat{\mathbf{y}}[t] = \mathbf{W}_{\text{head}} \mathbf{h}_{L}[t, :] + \mathbf{b}_{\text{head}}$$
  where $\mathbf{h}_{L}[t, :]$ is the feature vector at the causal boundary (sample index 255).

---

## 2. Mathematical Receptive Field Derivation

### Layer-by-Layer Propagation Formula:
For a 1D convolution with kernel size $k$ and dilation $d$, each layer expands the receptive field according to:
$$RF_{\text{out}} = RF_{\text{in}} + (k - 1) \cdot d$$

For a residual block containing two convolutions with dilation $d_l$:
$$\Delta RF_l = 2 \cdot (k - 1) \cdot d_l$$

With kernel size $k = 3$:
$$\Delta RF_l = 2 \cdot (3 - 1) \cdot d_l = 4 \cdot d_l$$

Because skip connections are identity or $1 \times 1$ convolutions ($k=1 \implies \Delta RF = 0$), the maximum temporal path is governed by the two dilated convolutions.

### Expansion Across the 4 Residual Blocks:
1. **Block 1 ($d_1 = 1$):** $\Delta RF_1 = 4 \times 1 = 4$ samples. Cumulative $RF_1 = 1 + 4 = 5$ samples.
2. **Block 2 ($d_2 = 2$):** $\Delta RF_2 = 4 \times 2 = 8$ samples. Cumulative $RF_2 = 5 + 8 = 13$ samples.
3. **Block 3 ($d_3 = 4$):** $\Delta RF_3 = 4 \times 4 = 16$ samples. Cumulative $RF_3 = 13 + 16 = 29$ samples.
4. **Block 4 ($d_4 = 8$):** $\Delta RF_4 = 4 \times 8 = 32$ samples. Cumulative $RF_4 = 29 + 32 = \mathbf{61\text{ samples}}$.

### General Formula:
$$RF = 1 + 2 \cdot (k - 1) \cdot \sum_{l=1}^M d_l = 1 + 4 \cdot (1 + 2 + 4 + 8) = 1 + 4 \cdot 15 = \mathbf{61\text{ samples}}$$

### Conversion to Physical Time at $f_s = 2,000\text{ Hz}$:
$$T_{\text{RF}} = \frac{61\text{ samples}}{2,000\text{ Hz}} = \mathbf{0.0305\text{ s}} = \mathbf{30.50\text{ ms}}$$

---

## 3. Empirical Verification via Gradient Backpropagation

To verify the derivation without relying on analytical assumptions, we computed the gradient of the final output with respect to all 256 input samples:
$$g[\tau] = \sum_{c=1}^{16} \sum_{k=1}^5 \left| \frac{\partial \hat{y}_k[t]}{\partial x_c[t - \tau]} \right|$$

### Empirical Measurement Results:
- **Input Window Size:** 256 samples ($t \in [0, 255]$)
- **Current Prediction Sample:** $t = 255$
- **Earliest Non-Zero Gradient Sample:** $t_{\min} = 195$
- **Measured Receptive Field:** $255 - 195 + 1 = \mathbf{61\text{ samples}}$
- **Physical Context:** $\mathbf{30.50\text{ ms}}$
- **Zero-Influence Samples:** Input indices $0$ to $194$ (195 samples = $97.50\text{ ms}$) have strictly $g[\tau] = 0.000000$.

---

## 4. Receptive Field Comparison Table

| Architecture Variant | Dilations | Convs / Block | Total Layers | $RF$ [samples] | $RF$ [ms at 2 kHz] | Coverage of 256-sample Buffer | Trainable Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Current Model A** | $1, 2, 4, 8$ | 2 | 8 | **61** | **30.50 ms** | **23.8%** | 134,693 (5-out) / 134,822 (6-out) |
| **Model A (Single Conv)** | $1, 2, 4, 8$ | 1 | 4 | **31** | **15.50 ms** | **12.1%** | 57,029 (5-out) / 57,158 (6-out) |
| **Model B (Extended 16)** | $1, 2, 4, 8, 16$ | 2 | 10 | **125** | **62.50 ms** | **48.8%** | 233,638 (6-out) |
| **Model C (Extended 32)** | $1, 2, 4, 8, 16, 32$ | 2 | 12 | **253** | **126.50 ms** | **98.8%** | 332,454 (6-out) |

---

## 5. Technical Conclusion & Justification of the 128 ms Window

1. **The Previous Claim Was Imprecise:** Describing Model A as having *"128 ms temporal context"* was technically inaccurate. The model operates on a **128 ms input buffer**, but its causal receptive field is **30.50 ms (61 samples)**.
2. **Role of Past Samples in Model A:** Samples $0 \to 194$ in Model A do not influence the prediction at time $t=255$. However, in a convolutional network computing full sequence outputs, those earlier samples provide causal context for earlier time steps $[t - 195, \dots, t - 61]$ in the intermediate feature maps.
3. **Engineering Opportunity:** Extending dilations to include $d=16$ (Model B, 62.5 ms) or $d=16, 32$ (Model C, 126.5 ms) allows the model to utilize the full 128 ms buffer. This will be evaluated empirically on Session 2 (Validation) in Problem 2.
