# Machine Learning Benchmark Results & Empirical Audit

## 1. Executive Summary

The primary ML model is a six-output causal dilated Temporal Convolutional Network (Model C) trained on Acquisition 1 (A1), validated on Acquisition 2 (A2), and evaluated strictly once on held-out Acquisition 3 (Session 3 / A3) from NinaPro Database 8, Subject 1.

### Primary Benchmark Metrics (Held-Out Session 3)

| Model Configuration | MAE $\downarrow$ | RMSE $\downarrow$ | $R^2$ $\uparrow$ | Pearson $r$ $\uparrow$ | Jitter $\downarrow$ | Host CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Causal TCN Model C (Raw)** | 0.0811 | 0.1532 | 0.4133 | 0.7287 | 0.0251 | **2.23 ms** |
| **Causal TCN Model C + EMA ($\alpha=0.20$)** | **0.0728** | **0.1348** | **0.5301** | **0.7718** | **0.0066** | **2.23 ms** |
| **Causal GRU Reference Baseline (Raw)** | 0.0726 | 0.1361 | 0.5345 | 0.7794 | 0.0264 | 57.35 ms |
| **Causal GRU Reference + EMA ($\alpha=0.20$)** | **0.0653** | **0.1162** | **0.6483** | **0.8258** | 0.0069 | 57.35 ms |

> [!IMPORTANT]
> **Metric Interpretation**: $R^2 = 0.5301$ represents that the model explains **53.0% of the total continuous kinematic variance** across all six held-out joints. This is continuous regression, **NOT** "53% classification accuracy". Accuracy is an inappropriate metric for continuous proportional joint control.

---

## 2. Per-Actuator Channel Breakdown (Session 3)

| Actuator Channel | Raw MAE | Raw RMSE | Raw $R^2$ | Raw Pearson $r$ | EMA MAE | EMA RMSE | EMA $R^2$ | EMA Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1: Index Flexion** | 0.0746 | 0.1605 | 0.4283 | 0.7187 | **0.0669** | 0.1435 | **0.5427** | **0.7728** |
| **H2: Middle Flexion** | 0.0823 | 0.1588 | 0.5164 | 0.7599 | **0.0742** | 0.1415 | **0.6165** | **0.8098** |
| **H3: Ring Flexion** | 0.0675 | 0.1320 | 0.6475 | 0.8391 | **0.0576** | 0.1091 | **0.7592** | **0.8860** |
| **H4: Little Flexion** | 0.0689 | 0.1344 | 0.7030 | 0.8534 | **0.0594** | 0.1078 | **0.8089** | **0.9024** |
| **H5: Thumb Flexion** | 0.1166 | 0.1881 | -0.1623 | 0.4410 | **0.1095** | 0.1753 | **-0.0094** | **0.4713** |
| **H6: Thumb Opposition** | 0.0764 | 0.1453 | 0.3468 | 0.7603 | **0.0693** | 0.1318 | **0.4625** | **0.7884** |

### Detailed Scientific Analysis of H5 Performance
- **Explained Variance:** $R^2 = -0.0094$ indicates that a simple running mean would predict as well as the model for continuous thumb flexion angle in Session 3.
- **Correlation:** Despite low $R^2$, Pearson $r = 0.4713$ confirms a positive linear relationship between predicted and true activations.
- **Root Causes:**
  1. *Anatomy:* The primary thumb interphalangeal flexor (*flexor pollicis longus*) lies deep beneath superficial flexor digitorum superficialis and pronator teres, heavily attenuating surface EMG signal-to-noise ratio.
  2. *Sensor Polarity & Range:* CyberGlove thumb flexion calibration range exhibited lower variance (0.0304) and occasional sensor polarity shifts during rapid movements.
- **Handoff Policy:** Channel H5 is retained in the six-channel interface without synthetic heuristic adjustments to preserve scientific integrity.

---

## 3. Why TCN Was Selected Over GRU

While the Causal GRU baseline achieved higher offline predictive metrics ($R^2 = 0.6483$ vs $0.5301$), it failed the real-time execution constraint:
- **Measured CPU Latency:** GRU required **$57.35\text{ ms}$** (mean) and **$91.04\text{ ms}$** (p95), heavily exceeding the **$32.0\text{ ms}$** execution budget.
- **TCN Execution:** The Causal TCN evaluates all receptive field convolutions via parallel 1D dilated kernels, executing in **$2.23\text{ ms}$** on CPU and **$1.50\text{ ms}$** on GPU.
- **Headroom:** TCN leaves **$> 93\%$ idle processing headroom** per 32 ms cycle for digital-twin kinematics, telemetry, and safety supervisory tasks.

---

## 4. Receptive Field Evolution (Model A vs Model C)

- **Model A (4 blocks, dilations $[1, 2, 4, 8]$):** Receptive field = 61 samples ($30.5\text{ ms}$), discarding 76.2% of the 128 ms buffer. Session 2 $R^2 = 0.3486$.
- **Model C (6 blocks, dilations $[1, 2, 4, 8, 16, 32]$):** Receptive field = 253 samples ($126.5\text{ ms}$), covering 98.8% of the buffer. Session 2 $R^2 = 0.5019$ (**+44.0% relative improvement**).

---

## 5. Post-Processing EMA Filter ($\alpha = 0.20$)

- **Jitter Reduction:** Derivative jitter $|dy/dt|$ reduced from $0.0251$ to $0.0066$ (**73.6% reduction**), preventing mechanical chatter in soft HASEL actuators.
- **Temporal Lead Compensation:** Cross-correlation analysis reveals raw TCN predictions lead joint motion by $-133.3\text{ ms}$; the EMA filter introduces a stabilizing delay that brings empirical ensemble lag to **$-16.0\text{ ms}$**.
