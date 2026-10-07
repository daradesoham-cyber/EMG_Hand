# Temporal Alignment & Timestamp Convention Audit

**Document:** Temporal Synchronization, Indexing Convention, and Delay Characterization
**Date:** October 7, 2026
**Project:** Six-Output sEMG Regression Validation for the Six-Actuator Prosthetic-Hand Control Architecture
**Dataset:** NinaPro Database 8 (DB8) — Subject 1 Empirical Benchmark

---

## 1. Executive Summary

This audit traces and mathematically verifies the end-to-end timestamping and synchronization chain across the NinaPro DB8 dataset, causal sliding window extraction, Causal TCN receptive field integration, and post-processing filters.

### Key Finding & Scientific Interpretation:
Cross-correlation analysis identified an ensemble prediction lead of approximately **$-133.3\text{ ms}$** relative to the measured CyberGlove displacement ($-192\text{ ms}$ on index, middle, ring; $-224\text{ ms}$ on little; $+32\text{ ms}$ on thumb flexion; $-32\text{ ms}$ on thumb opposition).

- **Indexing Verification:** A rigorous audit of the code confirms that this lead is **NOT an indexing bug or sample alignment error**. The ground-truth target index and the rightmost sample of the sEMG window share identical sample timestamps ($0.0\text{ ms}$ index offset).
- **Scientific Interpretation:** This observed prediction lead is **physiologically plausible** given the known temporal separation between neural/muscle activation and observable joint movement (electromechanical delay, EMD).
- **Critical Scientific Qualification:** However, **the present experiment does not independently isolate physiological electromechanical delay from model receptive-field dynamics, windowing effects, temporal prediction effects, or dataset acquisition filtering**. The $-133.3\text{ ms}$ value must be understood as an *empirically observed cross-correlation lead*, not a directly measured physiological constant.

---

## 2. Complete End-to-End Timestamp Chain

```
Raw Data Clock (fs = 2000 Hz, dt = 0.5 ms per sample)
---------------------------------------------------------------------------------------------------------
Sample Index:      0       1       2    ...     192    ...     254     255     256    ...     319
Timestamp (ms):   0.0     0.5     1.0   ...     96.0   ...    127.0   127.5   128.0   ...    159.5
---------------------------------------------------------------------------------------------------------

Window i = 0 (Duration W = 256 samples = 128.0 ms):
sEMG Window:     [sample 0 ------------------------------------> sample 255]
                 (t = 0.0 ms)                                    (t = 127.5 ms)
                                                                       |
Model C RF (253 samples = 126.5 ms):                                   |
Causal Span:               [sample 3 --------------------------> sample 255]
                                                                       |
                                                                       v
                                                           TCN Prediction Generated:
                                                           y_hat[0] evaluated at t = 127.5 ms
                                                                       |
                                                                       v
                                                           Kinematic Target Sampled:
                                                           y_true[0] = glove[255] (t = 127.5 ms)

Window i = 1 (Stride S = 64 samples = 32.0 ms):
sEMG Window:     [sample 64 -----------------------------------> sample 319]
                 (t = 32.0 ms)                                   (t = 159.5 ms)
                                                                       v
                                                           TCN Prediction: y_hat[1] at t = 159.5 ms
                                                           Kinematic Target: y_true[1] = glove[319]
```

### Mathematical Indexing Proof:
In `ml/windowing.py` (lines 88–95):
```python
for i in range(num_windows):
    start_idx = i * S
    end_idx = start_idx + W
    X[i] = emg[start_idx:end_idx].T
    Y[i] = targets[end_idx - 1]
```
For window index $i$:
1. The sEMG buffer slices samples $k \in [i \cdot S, \; i \cdot S + W - 1]$.
2. The final sample entering the causal convolution is sample index $t_{\text{end}} = i \cdot S + W - 1$.
3. The ground-truth target is extracted at `targets[end_idx - 1]` = `targets[i * S + W - 1]`.
4. Therefore:
   $$\text{Timestamp}(X_{\text{last}}) = \frac{i \cdot S + W - 1}{f_s} \equiv \text{Timestamp}(Y) = \frac{i \cdot S + W - 1}{f_s}$$
   There is **strictly 0.0 ms index offset** between the final sEMG sample and the target sample.

---

## 3. Dataset Synchronization Audit

In NinaPro DB8, 16 sEMG channels (Delsys Trigno) and 18 CyberGlove II joint angles were synchronously acquired:
- Hardware acquisition was synchronized via hardware trigger pulses.
- Signals were sinc-interpolated to a common, uniform $2,000\text{ Hz}$ clock ($\Delta t = 0.5\text{ ms}$).
- Matrix row $m$ in `emg` corresponds strictly to matrix row $m$ in `glove`.

Verification check on `S1_E1_A3.mat`:
- `emg.shape = (529099, 16)`
- `glove.shape = (529099, 18)`
- `stimulus.shape = (529099,)`
All three streams possess identical row lengths with sample-level clock parity.

---

## 4. Cross-Correlation Lag Convention & Empirical Observations

### Mathematical Definition:
In `ml/evaluate.py`, the cross-correlation lag shift $k \in [-20, +20]$ ($\pm 640\text{ ms}$ at $\Delta t = 32.0\text{ ms}$) aligns $y_{\text{true}}$ and $y_{\text{pred}}$:
- **Positive lag ($k > 0$):** $y_{\text{true}}[t]$ correlates best with $y_{\text{pred}}[t + k]$. The prediction responds **after** the physical movement (computational or filter delay).
- **Negative lag ($k < 0$):** $y_{\text{true}}[t + |k|]$ correlates best with $y_{\text{pred}}[t]$. The prediction pattern occurs **before** the physical glove movement.

### Measured Values on Held-Out Session 3:
| Actuator Channel | Raw TCN Lag ($k^* \cdot \Delta t$) | TCN + EMA ($\alpha=0.20$) Lag | Causal GRU Raw Lag | Causal GRU + EMA Lag |
| :--- | :---: | :---: | :---: | :---: |
| **H1: Index** | $-192.0\text{ ms}$ | $-96.0\text{ ms}$ | $-160.0\text{ ms}$ | $-32.0\text{ ms}$ |
| **H2: Middle** | $-192.0\text{ ms}$ | $-64.0\text{ ms}$ | $-160.0\text{ ms}$ | $-32.0\text{ ms}$ |
| **H3: Ring** | $-192.0\text{ ms}$ | $-96.0\text{ ms}$ | $-128.0\text{ ms}$ | $0.0\text{ ms}$ |
| **H4: Little** | $-224.0\text{ ms}$ | $-96.0\text{ ms}$ | $-160.0\text{ ms}$ | $-32.0\text{ ms}$ |
| **H5: Thumb Flexion** | $+32.0\text{ ms}$ | $+160.0\text{ ms}$ | $+32.0\text{ ms}$ | $+128.0\text{ ms}$ |
| **H6: Thumb Opposition** | $-32.0\text{ ms}$ | $+96.0\text{ ms}$ | $-96.0\text{ ms}$ | $+32.0\text{ ms}$ |
| **Ensemble Mean** | **$-133.3\text{ ms}$** | **$-16.0\text{ ms}$** | **$-112.0\text{ ms}$** | **$-5.3\text{ ms}$** |

### Scientific Distinction: Observed Lead vs Physiological Interpretation
- **Observed Fact:** The raw TCN and raw GRU predictions systematically lead measured joint angles by approximately $110\text{--}135\text{ ms}$ on average across the 6 channels.
- **Physiological Interpretation:** In human biomechanics, neuromuscular motor unit action potentials precede observable joint torque and kinematic movement due to calcium dynamics, series elasticity, and limb inertia (commonly termed electromechanical delay, typically $50\text{--}150\text{ ms}$). The observed lead is consistent with this physiological delay.
- **Scientific Limitation:** We do not claim this $-133.3\text{ ms}$ metric is an isolated physiological measurement of EMD. The lead is a composite result of:
  1. Biological neuromuscular activation timing,
  2. The network's effective temporal receptive field and weight weighting within the 128 ms window,
  3. Preprocessing filter phase response,
  4. Visual stimulus anticipation by the subject during the NinaPro protocol.
- **EMA Phase Shift:** Applying the causal EMA filter ($\alpha = 0.20$) introduces an empirical smoothing delay ($\tau = 143.4\text{ ms}$). This delay counteracts the anticipatory lead, bringing the ensemble net lag from **$-133.3\text{ ms}$ down to $-16.0\text{ ms}$** relative to the measured CyberGlove trajectory.

---

## 5. Conclusion

- Temporal indexing implementation has **zero sample offset**.
- The observed $-133.3\text{ ms}$ lead is real in the data and physiologically plausible, but should not be characterized as a direct measurement of biological EMD.
- With EMA ($\alpha = 0.20$), the composite system achieves close temporal synchronization ($-16.0\text{ ms}$ mean lag) with measured kinematics.
