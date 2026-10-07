# End-to-End Pipeline Causality & Temporal Leakage Audit

**Date:** October 7, 2026
**Audit Scope:** Verification that every computational stage in the ML pipeline strictly obeys temporal causality:
$$\hat{\mathbf{y}}[t] = f(\mathbf{x}[-\infty : t])$$
with zero mathematical dependence on future samples $\mathbf{x}[t' > t]$.

---

## 1. Stage-by-Stage Causality Verification Matrix

| Pipeline Stage | Implementation | Function / Method | Causal Status | Future Leakage Risk & Verification |
| :--- | :--- | :--- | :---: | :--- |
| **1. Bandpass Filtering** | 4th-order Butterworth (20–450 Hz) | `scipy.signal.lfilter(b, a, x, axis=0)` | **STRICTLY CAUSAL** | Verified: Does NOT use `filtfilt`. Direct Form II Transposed difference equation accesses only $x[n-k]$ and $y[n-k]$. |
| **2. Notch Filtering** | 2nd-order IIR Notch (50 Hz, Q=30) | `scipy.signal.lfilter(b, a, x, axis=0)` | **STRICTLY CAUSAL** | Verified: Direct Form II Transposed forward pass. |
| **3. Streaming State Registers** | Filter delay lines $\mathbf{z}_i$ | `scipy.signal.lfilter(..., zi=self.bp_zi)` | **STRICTLY CAUSAL** | Verified: Preserves filter initial conditions across streaming 2 kHz samples with zero lookahead. |
| **4. Z-Score Normalization** | $(\mathbf{x} - \mathbf{\mu}) / \mathbf{\sigma}$ | `fit_normalization(A1_emg)` | **STRICTLY CAUSAL** | Verified: $\mathbf{\mu}, \mathbf{\sigma}$ fitted ONLY on Acquisition 1. Frozen for A2 and A3. |
| **5. Window Extraction** | Sliding windows $[t - W + 1 : t]$ | `extract_windows_numpy` | **STRICTLY CAUSAL** | Verified: Window ends at time $t$. Target is ground-truth angle at time $t$. No centered windows. |
| **6. Causal Conv1D** | Left-padded 1D convolution | `F.pad(x, ((k-1)*d, 0))` | **STRICTLY CAUSAL** | Verified: Pads $(k-1)d$ zeros on the left (past). Zero padding on the right (future). |
| **7. Residual TCN Blocks** | Dilated blocks ($d=1, 2, 4, 8, \dots$) | `CausalResidualBlock` | **STRICTLY CAUSAL** | Verified: All skip connections are identity or $1\times 1$ convs. Zero lookahead across all layers. |
| **8. Feature Extraction Head** | Regression Linear Layer | `features[:, :, -1]` | **STRICTLY CAUSAL** | Verified: Takes only the last temporal column $t$. Does NOT pool future time steps. |
| **9. EMA Post-Filter** | Recursive 1st-order IIR | $y[t] = \alpha \hat{y}[t] + (1-\alpha) y[t-1]$ | **STRICTLY CAUSAL** | Verified: Operates strictly forward in time using only previous output state. |

---

## 2. Why Zero-Phase Filtering (`filtfilt`) Was Explicitly Rejected

Many offline biosignal processing studies erroneously use `scipy.signal.filtfilt`:
1. `filtfilt` runs a forward filter $y_1 = \text{lfilter}(b, a, x)$ followed by a time-reversed backward filter $y_2 = \text{reverse}(\text{lfilter}(b, a, \text{reverse}(y_1)))$.
2. The backward pass enables samples from $t + 100\text{ ms}$ in the future to directly influence the filtered signal at time $t$.
3. When evaluated in an offline cross-validation loop, this introduces massive non-causal smoothing that artificially inflates $R^2$.
4. When deployed on a physical microcontroller or in real-time closed-loop control, backward filtering is physically impossible, causing model accuracy to collapse.
5. **Implementation Audit:** Our implementation in `ml/preprocessing.py` strictly uses `scipy.signal.lfilter`.

---

## 3. Unit Test Verification

In `tests/test_pipeline.py`, the test `test_causal_conv_isolation` verifies causality at machine precision:
```python
# Perturb future sample at t=35: x2[:, :, 35:] += 10.0
# Outputs before t=35 MUST be identical to machine precision
diff_past = torch.max(torch.abs(out1[:, :, :35] - out2[:, :, :35])).item()
assert diff_past == 0.0  # PASSES (0.0000000)
```
Every layer of the computational graph satisfies strict causality.
