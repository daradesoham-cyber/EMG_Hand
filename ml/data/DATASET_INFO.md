# NinaPro DB8 Dataset Technical Specification

## 1. Overview & Provenance

- **Dataset Identifier:** NinaPro Database 8 (DB8)
- **Primary Source:** [http://ninapro.hevs.ch/DB8](http://ninapro.hevs.ch/DB8)
- **Subject Evaluated:** Subject 1 (intact limb, right hand dominant)
- **Sampling Frequency:** Sinc-interpolated to exactly $2,000\text{ Hz}$ in distributed `.mat` format (native hardware sampled at 1,111 Hz).
- **Sensors:**
  - **sEMG:** 16 double-differential channels (Trigno Wireless, Delsys Inc.) placed circumferentially around the proximal forearm musculature.
  - **Kinematics:** 18-sensor CyberGlove II strain-gauge data glove recorded synchronously on the contralateral/ipsilateral hand.
- **Protocol:** 9 functional hand movements (individual finger flexions, thumb abduction, power grasp, precision pinches) alternating with rest phases.
- **Total Duration (Subject 1):** $2,572.7\text{ s}$ ($42.88\text{ minutes}$), comprising $5,135,095$ continuous time frames.

---

## 2. Partitioning & Leakage Prevention

To evaluate realistic prosthetic deployment and prevent window-overlap data leakage:

| Session | File | Duration | Raw Frames | Sliding Windows ($W=256, S=64$) | Role in Pipeline |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Session 1 (A1)** | `S1_E1_A1.mat` | $1,146.3\text{ s}$ | $2,292,526$ | 35,817 | **Training Partition.** Fits filter stats and normalizers. |
| **Session 2 (A2)** | `S1_E1_A2.mat` | $1,161.9\text{ s}$ | $2,313,470$ | 36,144 | **Validation Partition.** Early stopping, EMA grid search ($\alpha=0.20$). |
| **Session 3 (A3)** | `S1_E1_A3.mat` | $264.5\text{ s}$ | $529,099$ | 8,264 | **Strictly Held-Out Test Partition.** Evaluated strictly once. |

> **Strict Rule:** Session 3 was never touched during training, normalization fitting, architecture search, or hyperparameter tuning.

---

## 3. Target Kinematics & CyberGlove Channel Mapping

The six continuous output commands map to NinaPro DB8 CyberGlove columns:

| Output | Actuator Command | CyberGlove Columns | Anatomical Joint Description | Calibration Range ($q_{1\%} \to q_{99\%}$) |
| :---: | :--- | :--- | :--- | :---: |
| **H1** | Index Flexion | Columns 4, 5 | Index MCP & PIP flexion angle | $-7.68^\circ \to +92.95^\circ$ |
| **H2** | Middle Flexion | Columns 6, 7 | Middle MCP & PIP flexion angle | $-14.82^\circ \to +96.59^\circ$ |
| **H3** | Ring Flexion | Columns 9, 10 | Ring MCP & PIP flexion angle | $-14.87^\circ \to +92.16^\circ$ |
| **H4** | Little Flexion | Columns 12, 13 | Little MCP & PIP flexion angle | $-6.30^\circ \to +85.83^\circ$ |
| **H5** | Thumb Flexion | Columns 1, 15 | Thumb MCP & IP flexion angle | $-35.91^\circ \to +61.90^\circ$ |
| **H6** | Thumb Opposition | Column 0 | Trapeziometacarpal Saddle CMC rotation | $+16.77^\circ \to +145.35^\circ$ |

Continuous targets are scaled into $[0.0, 1.0]$:
$$y_k(t) = \text{clip}\left(\frac{q_k(t) - q_{k, 1\%}}{q_{k, 99\%} - q_{k, 1\%}}, 0.0, 1.0\right)$$

---

## 4. Signal Conditioning & Windowing Parameters

1. **DC Baseline Removal:** Running mean subtraction.
2. **Causal Bandpass:** 4th-order causal Butterworth IIR ($20.0\text{ Hz} \to 450.0\text{ Hz}$) via `scipy.signal.lfilter`.
3. **Mains Notch Filter:** 2nd-order causal IIR notch at $50.0\text{ Hz}$ ($Q = 30.0$).
4. **Z-Score Normalization:** Channel-wise zero-mean, unit-variance fitted strictly on Session 1 (A1).
5. **Causal Window Length ($W$):** 256 samples ($128.0\text{ ms}$ at 2 kHz).
6. **Inference Stride ($S$):** 64 samples ($32.0\text{ ms}$ at 2 kHz $\to 31.25\text{ predictions/s}$).
7. **Causal Endpoint Alignment:** The target vector $\mathbf{y}_i$ is sampled strictly at sample $t = \text{end\_idx} - 1$ (the newest sample in the window).
