# Final Pre-Freeze Machine Learning Audit Report

**Document:** Final Scientific Pre-Freeze ML Baseline Audit
**Date:** October 7, 2026
**Project:** Six-Output sEMG Regression Validation for the Six-Actuator Prosthetic-Hand Control Architecture
**Dataset:** NinaPro Database 8 (DB8) — Subject 1 Empirical Benchmark
**Evaluation Standard:** Strictly Held-Out Acquisition 3 (Session 3)

---

## 1. H5 Thumb Flexion: Root Cause Diagnosis

On strictly held-out Session 3, Channel H5 (Thumb Flexion) reported:
- Raw TCN: $\text{MAE} = 0.1166$, $\text{RMSE} = 0.1881$, $R^2 = -0.1623$, Pearson $r = +0.4410$
- TCN + EMA ($\alpha=0.20$): $\text{MAE} = 0.1095$, $\text{RMSE} = 0.1753$, $R^2 = -0.0094$, Pearson $r = +0.4713$

A diagnostic audit of DB8 raw data, glove sensor channels, anatomy, and sEMG activations identified several contributing factors:

### 1.1 Uncorrelated Dual-Sensor Target Construction
For the 4 primary fingers, the MCP and PIP joints are strongly coupled mechanically and physiologically:
- Index (MCP Col 4 vs PIP Col 5): Pearson $r = \mathbf{+0.8451}$
- Middle (MCP Col 6 vs PIP Col 7): Pearson $r = \mathbf{+0.7726}$
- Ring (MCP Col 9 vs PIP Col 10): Pearson $r = \mathbf{+0.8471}$
- Little (MCP Col 12 vs PIP Col 13): Pearson $r = \mathbf{+0.8724}$
Because finger MCP and PIP joints flex together in synchrony, their arithmetic mean $\frac{q_{\text{MCP}} + q_{\text{PIP}}}{2}$ reinforces the dominant flexion signal.

In contrast, for the thumb:
- **Thumb MCP (Col 1) vs Thumb IP (Col 15):** Pearson $r = \mathbf{+0.1081}$ (Session A1), $\mathbf{+0.0204}$ (Session A2), $\mathbf{+0.0726}$ (Session A3).
The two thumb flexion sensors exhibit minimal linear correlation. Anatomically, Thumb MCP is actuated by both thenar intrinsic muscles in the palm (flexor pollicis brevis) and extrinsic tendons, whereas Thumb IP is driven by the extrinsic flexor pollicis longus (FPL) tendon. During precision pinch (Movement 8) and tripod grasp (Movement 9), the MCP flexes while the distal IP joint stabilizes or hyperextends. Averaging Col 1 and Col 15 creates destructive interference between differing joint dynamics.

### 1.2 Inverted Sensor Polarity in Hardware Calibration
Inspection of movement-specific angle deltas relative to rest reveals:
- Finger MCP/PIP angles **increase** during flexion: Index MCP $+46.13^\circ$, PIP $+32.11^\circ$.
- Thumb MCP (Col 1) **decreases** during Movement 1: rest $-27.59^\circ \to$ active $-41.09^\circ$ ($\Delta = -13.50^\circ$).
- Thumb IP (Col 15) **decreases** during Movement 1: rest $+42.11^\circ \to$ active $+21.59^\circ$ ($\Delta = -20.52^\circ$).
In the CyberGlove II calibration used in DB8, thumb flexion angles become more negative. Consequently, the linear normalization formula $\frac{q - q_{\min}}{q_{\max} - q_{\min}}$ drives the target toward $0.0$ during active flexion (M1 true mean $= 0.303$) while rest hovers at $0.456$. Although the TCN learned this inverse mapping (yielding positive Pearson $r = +0.4713$), conflicting grasp synergies (e.g. M7 Fist where thumb true $= 0.537$) penalized variance prediction.

### 1.3 Deep Muscle Anatomy vs Superficial Surface sEMG
The sole muscle driving thumb IP flexion is the **Flexor Pollicis Longus (FPL)**. The FPL originates deep on the anterior surface of the radius and lies beneath superficial muscle layers (pronator teres, flexor carpi radialis, and flexor digitorum superficialis). Non-invasive surface electrodes placed circumferentially on the skin primarily record cross-talk from superficial digit flexors, providing limited isolated signal power for deep FPL motor units.

### 1.4 Target Variance Deficiency
Session 3 variance across channels:
- Little: $\text{var} = 0.0608$
- Middle: $\text{var} = 0.0522$
- Ring: $\text{var} = 0.0494$
- Index: $\text{var} = 0.0450$
- Thumb Opposition (H6): $\text{var} = 0.0323$
- **Thumb Flexion (H5):** $\text{var} = \mathbf{0.0304}$ (lowest in dataset)
Because ground-truth variance is low ($\text{var} = 0.0304$), a modest mean squared error ($\text{MSE} = 0.0354$) results in negative $R^2 = 1 - \frac{0.0354}{0.0304} = -0.1623$, despite a positive Pearson correlation of $r = +0.4713$.

### Scientific Conclusion on H5:
The audit identified anatomical, target-construction, sensor-polarity, and low-variance factors that plausibly explain the reduced H5 performance. No implementation error was identified in the audited pipeline; however, the present experiment does not establish that these factors are the sole causes. Channel H5 is retained in the six-output architecture without artificial data manipulation, representing an honest technical limitation of non-invasive surface sEMG thumb flexion decoding.

---

## 2. Temporal Alignment Audit Conclusion

A systematic audit of `ml/windowing.py`, `ml/model.py`, and `ml/evaluate.py` confirmed:
1. **Zero Indexing Offset:** In sliding window extraction, window $i$ spans sample indices $[i \cdot S, \; i \cdot S + W - 1]$. The ground-truth target is sampled at index $i \cdot S + W - 1$. The target timestamp is **strictly identical** to the timestamp of the last sample in the sEMG window ($0.0\text{ ms}$ index error).
2. **Dataset Clock Parity:** In NinaPro DB8, sEMG and CyberGlove II streams share a uniform hardware-synchronized $2,000\text{ Hz}$ clock ($\Delta t = 0.5\text{ ms}$). Row lengths across all acquisitions match to the exact sample.
3. **Observed Prediction Lead:** Cross-correlation identified an ensemble prediction lead of approximately **$-133.3\text{ ms}$** relative to the measured CyberGlove displacement ($-192\text{ ms}$ on index, middle, ring; $-224\text{ ms}$ on little; $+32\text{ ms}$ on thumb flexion; $-32\text{ ms}$ on thumb opposition).
4. **Physiological Plausibility vs Direct Proof:** This lead is physiologically plausible given the known temporal separation between neural/muscle activation and observable joint movement; however, **the present experiment does not independently isolate physiological electromechanical delay from model/windowing effects**. The $-133.3\text{ ms}$ metric should be cited as an observed cross-correlation lead, not a directly measured biological constant.
5. **EMA Low-Pass Phase Compensation:** Post-filtering with causal EMA ($\alpha = 0.20$) introduces an empirical smoothing delay that counteracts the observed lead, bringing net ensemble delay to **$-16.0\text{ ms}$** relative to the measured CyberGlove trajectory.

The complete timestamp derivations and diagrams are archived in [`docs/TEMPORAL_ALIGNMENT_AUDIT.md`](file:///home/soham-darade/prosthetic_hand_phase1/docs/TEMPORAL_ALIGNMENT_AUDIT.md).

---

## 3. GRU vs GRU + EMA Symmetric Benchmark

To ensure scientific parity, the 2-layer Causal GRU baseline was evaluated both raw and with EMA post-filtering. The EMA smoothing parameter $\alpha$ was tuned exclusively on **Session 2 (Validation)** ($\alpha = 0.20$ selected) and evaluated on strictly held-out Session 3:

### Session 2 (Validation) GRU EMA Grid Search:
| Candidate $\alpha$ | Val MAE $\downarrow$ | Val RMSE $\downarrow$ | Val $R^2$ $\uparrow$ | Pearson $r$ $\uparrow$ | Jitter Reduction | Mean Delay |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\alpha = 0.10$ | 0.0698 | 0.1046 | 0.7570 | 0.8730 | 84.7% | $+128.0\text{ ms}$ |
| $\alpha = 0.15$ | 0.0701 | 0.1060 | 0.7506 | 0.8695 | 79.4% | $+48.0\text{ ms}$ |
| **$\alpha = 0.20$ (Frozen)** | **0.0709** | **0.1083** | **0.7404** | **0.8641** | **74.4%** | **$+5.3\text{ ms}$** |
| $\alpha = 0.30$ | 0.0724 | 0.1127 | 0.7206 | 0.8541 | 64.6% | $-42.7\text{ ms}$ |
| $\alpha = 0.40$ | 0.0736 | 0.1163 | 0.7037 | 0.8458 | 55.3% | $-58.7\text{ ms}$ |

$\alpha = 0.20$ was selected on Session 2 (achieving 74.4% jitter reduction with $+5.3\text{ ms}$ net phase shift).

### Session 3 Held-Out Benchmark (Symmetric Comparison):
| Architecture | Post-Filter | Parameters | CPU Latency (mean) | Real-Time Slack (vs 32 ms) | Test MAE $\downarrow$ | Test RMSE $\downarrow$ | Test $R^2$ $\uparrow$ | Pearson $r$ $\uparrow$ | Jitter $\downarrow$ | Net Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Causal TCN Model C** | **Raw** | 332,966 | **2.23 ms** | **+29.77 ms (93.0% headroom)** | 0.0811 | 0.1532 | 0.4133 | 0.7287 | 0.0251 | -133.3 ms |
| **Causal TCN Model C** | **EMA ($\alpha=0.20$)** | 332,966 | **2.23 ms** | **+29.77 ms (93.0% headroom)** | 0.0728 | 0.1348 | 0.5301 | 0.7718 | **0.0066** | -16.0 ms |
| **Causal GRU Baseline** | **Raw** | 41,094 | 57.35 ms | **Exceeds budget (-25.35 ms)** | 0.0726 | 0.1361 | 0.5345 | 0.7794 | 0.0264 | -112.0 ms |
| **Causal GRU Baseline** | **EMA ($\alpha=0.20$)** | 41,094 | 57.35 ms | **Exceeds budget (-25.35 ms)** | **0.0653** | **0.1162** | **0.6483** | **0.8258** | 0.0069 | -5.3 ms |

---

## 4. Final Architecture Comparison: TCN vs GRU Rationale

Although the evaluated GRU achieved higher offline predictive accuracy ($R^2 = 0.6483$ vs $0.5301$), **the evaluated GRU implementation exceeded the 32 ms real-time execution deadline on the tested CPU platform**:

1. **Inference Latency Constraint:**
   - The system stride is $64\text{ samples}$ at $2\text{ kHz}$, defining a $32.0\text{ ms}$ ($31.25\text{ Hz}$) execution deadline.
   - **Evaluated GRU Implementation:** Measured mean CPU latency was **$57.35\text{ ms}$ (p95: $91.04\text{ ms}$)**, exceeding the $32\text{ ms}$ execution budget by $+79.2\%$.
   - **Causal TCN Implementation:** Measured mean CPU latency was **$2.23\text{ ms}$ (p95: $2.43\text{ ms}$)**, consuming only $7.0\%$ of the budget ($29.77\text{ ms}$ headroom).
2. **Computational Structure:**
   - The GRU executes 256 sequential recurrent cell transitions per window.
   - The 1D dilated TCN evaluates temporal convolutions in a single feedforward pass using hardware-accelerated vectorization.
3. **Finite Receptive Field:**
   - Model C possesses a mathematically guaranteed finite memory horizon of exactly **253 samples (126.5 ms)**, ensuring transient input artifacts clear the network predictably.

**Engineering Decision:** The Causal TCN is retained as the primary architecture because it provides substantially lower inference latency on the tested platform while maintaining competitive predictive performance.

---

## 5. Model Memory & Quantization Audit

Detailed accounting of Model C parameters and memory allocations:

| Component | Mathematical / File Measurement | Physical Meaning & Hardware Relevance |
| :--- | :---: | :--- |
| **Trainable Parameters** | **332,966** | Exact count across 6 residual blocks + linear head. |
| **FP32 Weight Footprint** | **1,300.65 kB** (1.27 MB) | $332,966 \times 4\text{ bytes} = 1,331,864\text{ bytes} \approx 1,300.65\text{ KiB}$. |
| **FP16 Weight Footprint** | **650.32 kB** | $332,966 \times 2\text{ bytes} = 665,932\text{ bytes} \approx 650.32\text{ KiB}$. |
| **INT8 Target Footprint** | **325.16 kB** | $332,966 \times 1\text{ byte} = 332,966\text{ bytes} \approx 325.16\text{ KiB}$. |
| **ONNX Graph (`.onnx`)** | **65.50 kB** | Protobuf file containing **graph topology, node operators, dilated padding attributes, and tensor shapes only**. |
| **ONNX Data (`.onnx.data`)** | **1,300.65 kB** | External binary container storing uncompressed FP32 weights. |
| **Peak Activation Memory** | **~262 kB** (Estimated) | Double-buffered feature maps ($(1, 128, 256) \times 4 \times 2$). |
| **Total Inference SRAM** | **~647 kB** (Estimated) | INT8 weights ($325\text{ kB}$) + activations ($262\text{ kB}$) + runtime overhead (~$60\text{ kB}$). |

### Embedded Deployment Note:
An Espressif ESP32-S3 possesses $512\text{ kB}$ internal SRAM, with approximately $380\text{ kB}$ typically available for user applications. Because total estimated inference memory is $\approx 647\text{ kB}$, Model C cannot execute entirely within internal SRAM without external octal PSRAM or streaming buffer optimizations. The ESP32-S3 is accurately characterized as a **prospective target deployment platform**, not a physically demonstrated deployment.

---

## 6. Final Architecture Specification

- **Network:** 6-Block Causal Dilated Residual Temporal Convolutional Network (Model C)
- **Input Dimensions:** $(B, 16, 256)$ representing 16 sEMG channels over 128.0 ms at 2 kHz
- **Convolutions per Block:** 2 (Bai et al. residual design)
- **Kernel Size:** $k = 3$
- **Dilation Schedule:** $[1, 2, 4, 8, 16, 32]$
- **Hidden Channel Schedule:** $[32, 64, 64, 128, 128, 128]$
- **Receptive Field:** $RF = 1 + 2 \times (3 - 1) \times (1 + 2 + 4 + 8 + 16 + 32) = \mathbf{253\text{ samples}}\;(\mathbf{126.50\text{ ms}})$
- **Buffer Coverage:** **98.8%** of the 128 ms input buffer
- **Regression Head:** Linear projection $(128 \to 6)$ with $[0.0, 1.0]$ output clamping
- **Post-Filter:** Causal Exponential Moving Average ($y_t = 0.20 x_t + 0.80 y_{t-1}$)

---

## 7. Final Six-Output Actuator Mapping Table

This mapping provides continuous control targets for the six-actuator prosthetic hand control architecture:

| Output | Actuator Target | CyberGlove Columns | Anatomical Joint | Calibration Range | Test $r$ | Test $R^2$ |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **$y_0$** | **H1: Index Flexion** | Col 4 (MCP) + Col 5 (PIP) | Index MCP & PIP flexion | $-1.2^\circ \to +94.8^\circ$ | 0.7728 | 0.5427 |
| **$y_1$** | **H2: Middle Flexion** | Col 6 (MCP) + Col 7 (PIP) | Middle MCP & PIP flexion | $-2.1^\circ \to +98.6^\circ$ | 0.8098 | 0.6165 |
| **$y_2$** | **H3: Ring Flexion** | Col 9 (MCP) + Col 10 (PIP) | Ring MCP & PIP flexion | $-1.8^\circ \to +89.2^\circ$ | 0.8860 | 0.7592 |
| **$y_3$** | **H4: Little Flexion** | Col 12 (MCP) + Col 13 (PIP) | Little MCP & PIP flexion | $-1.5^\circ \to +88.4^\circ$ | 0.9024 | 0.8089 |
| **$y_4$** | **H5: Thumb Flexion** | Col 1 (MCP) + Col 15 (IP) | Thumb MCP & IP flexion | $-55.2^\circ \to +68.4^\circ$ | 0.4713 | -0.0094 |
| **$y_5$** | **H6: Thumb Opposition** | Col 0 (CMC Rotation) | Trapeziometacarpal Saddle CMC | $+5.8^\circ \to +138.6^\circ$ | 0.7884 | 0.4625 |

Normalizer extrema were fitted strictly on Acquisition 1.

---

## 8. Remaining Limitations

1. **Thumb Flexion Accuracy:** While thumb opposition (H6) achieves strong correlation ($r = 0.7884, R^2 = 0.4625$), isolated thumb flexion (H5) exhibits modest correlation ($r = 0.4713, R^2 \approx 0$) due to deep FPL anatomy and superficial cross-talk.
2. **Subject Generalization Scope:** Results represent cross-session generalization within Subject 1. Inter-subject generalization across multiple subjects was not evaluated in this protocol.
3. **Physical Loop Scope:** The experiment evaluated offline dataset streams and local CPU inference latency. Physical HASEL actuator dynamics, sensor feedback, and clinical prosthetic control loops were not evaluated in this phase.

---

## 9. Baseline Freezing Declaration

Real-time inference feasibility was demonstrated on the tested CPU platform, with a measured mean TCN inference latency of 2.23 ms against the 32 ms inference deadline. All audits, symmetric benchmarks, and timestamp verifications have been completed:

> **ML BASELINE READY TO FREEZE**
>
> The six-output causal TCN baseline is ready to freeze as the computational ML control model for the prosthetic-hand architecture. The model was evaluated on a strictly held-out acquisition, satisfies the measured inference-time requirement on the tested CPU platform, and has been audited for temporal indexing, architecture comparison, memory accounting, and target construction.
>
> The result should not be interpreted as physical HASEL validation, end-to-end prosthetic control validation, physiological EMD measurement, ESP32 deployment validation, or clinical validation.
