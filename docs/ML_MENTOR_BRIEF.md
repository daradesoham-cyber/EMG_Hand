# Executive Mentor Brief: Continuous sEMG-to-Kinematic Regression Pipeline

**Project:** Biomechatronic Six-HASEL Prosthetic Hand — Machine Learning Architecture
**Target:** 2-Minute Technical Thesis Defense / Faculty Presentation

---

### The 2-Minute Spoken Summary

> *"Instead of classifying discrete hand gestures into brittle state machines, our machine-learning model performs continuous causal regression from multi-channel surface electromyography (sEMG) to six continuous actuator commands.*
>
> *Our frozen architecture is a 6-block causal dilated Temporal Convolutional Network (Model C) with an exact 126.5 ms receptive field that covers 98.8% of the input buffer. It decodes all six prosthetic actuators—including independent thumb opposition—achieving a held-out Pearson correlation of 0.7718 and an MAE of 0.0728 on strictly held-out Session 3 data, while executing in just 2.23 ms on a standard CPU."*

---

### Key Technical Talking Points

#### 1. Why a Temporal Convolutional Network (TCN) instead of LSTM/GRU?
- **Real-Time Execution Feasibility:** While recurrent networks like LSTMs/GRUs must execute 256 sequential time-step updates (our evaluated GRU baseline required **57.35 ms on CPU**, exceeding the 32.0 ms deadline), a 1D dilated TCN evaluates the entire temporal buffer in parallel, achieving **2.23 ms on CPU** and **1.50 ms on GPU** (leaving $> 92\%$ idle processing headroom per 32 ms cycle).
- **Direct Temporal Feature Extraction:** Raw bipolar sEMG is an oscillatory interference signal ($20\text{--}450\text{ Hz}$). Dilated convolutions act as learnable, multi-band temporal finite impulse response (FIR) filter banks that extract motor-unit firing rate envelopes directly without manual feature engineering.

#### 2. Receptive Field Engineering (Model A vs Model C)
- Our initial 4-block network (Model A, dilations $[1, 2, 4, 8]$) had a mathematical receptive field of only **61 samples (30.5 ms)**, discarding 76.2% of the 128 ms buffer.
- By extending dilations to $[1, 2, 4, 8, 16, 32]$ in Model C, the receptive field expanded to **253 samples (126.5 ms)**, capturing 98.8% of the buffer.
- On Session 2 validation, Model C increased explained kinematic variance ($R^2$) by **44.0% relative** ($0.3486 \to 0.5019$) and reduced tracking jitter by 46.9%, demonstrating that capturing the full muscular pre-activation window is critical.

#### 3. Why Strict Temporal Causality?
- A prosthetic hand operates in the real world; it cannot access future muscular action potentials.
- By using strictly **left-padded causal convolutions** and causal IIR filters (`scipy.signal.lfilter`), we mathematically guarantee that an input at time $t$ depends only on samples from $t' \le t$. Zero-phase non-causal filters (`filtfilt`) and bidirectional networks were rejected because they induce artificial offline accuracy that collapses during physical deployment.

#### 4. The Six-Actuator Ground-Truth Resolution
- The prosthetic hand has **six physical HASEL channels**: H1 Index, H2 Middle, H3 Ring, H4 Little, H5 Thumb Flexion, and H6 Thumb Opposition (CMC circumduction).
- Empirical audit of NinaPro DB8 verified that CyberGlove Column 0 records continuous **Thumb CMC saddle joint rotation** ($+0.9^\circ \to +150.9^\circ$), exercised during isolated thumb abduction and pinch grasps.
- We mapped all 6 channels directly to ground-truth labels without synthetic heuristics or fabricated targets.

#### 5. Why Session 3 (A3) Was Strictly Held Out?
- Overlapping sliding windows within the same recording session introduce massive temporal data leakage, creating artificially inflated $R^2 > 0.95$ scores that fail in reality.
- We enforced an uncompromising split: **Acquisition 1 for training**, **Acquisition 2 for validation**, and **Acquisition 3 strictly held out**. All preprocessors, normalizers, and feature scalers were fitted exclusively on A1, ensuring our reported held-out metrics (**MAE = 0.0728, Pearson $r = 0.7718$**) reflect true predictive generalization.

#### 6. Baselines & The TCN vs GRU Decision
- Naive persistence achieves MAE $\approx 0.0023$ because joint trajectories are continuous. However, we proved this is an **Oracle Smoothness Reference** using ground-truth glove feedback $y_{t-1}^{\text{true}}$, not an sEMG-only predictor.
- Against causal sEMG baselines on Session 3, Causal TCN + EMA ($R^2 = 0.5301$, MAE = $0.0728$) dramatically outperforms Linear Regression ($R^2 = 0.2164$, MAE = $0.1065$) and Simple MLP ($R^2 = 0.0219$, MAE = $0.1199$).
- Although the evaluated GRU achieved higher offline predictive accuracy ($R^2 = 0.6483$ with EMA), its evaluated CPU latency of **$57.35\text{ ms}$** failed the 32 ms real-time deadline. The Causal TCN was therefore selected because it satisfies the real-time execution requirement ($2.23\text{ ms}$) while maintaining competitive predictive accuracy.

#### 7. Post-Processing EMA Filter ($\alpha = 0.20$)
- An EMA filter with $\alpha = 0.20$ reduces sample-to-sample derivative jitter by **73.6%**, preventing mechanical actuator chattering.
- Cross-correlation temporal analysis reveals an empirical delay of **$-16.0\text{ ms}$**, which counteracts the observed physiological neural lead without introducing perceptible lag.

#### 8. Deployment Target Clarification
- The model contains **332,966 parameters** ($1.30\text{ MB}$ FP32, $325.2\text{ kB}$ INT8 target).
- Embedded microcontrollers such as the Espressif ESP32-S3 are designated as **prospective target deployment platforms**. Because total estimated inference SRAM is $\approx 647\text{ kB}$ (exceeding the ~380 kB user-available internal SRAM), deployment would require external PSRAM or buffer streaming. Physical MCU on-chip profiling remains prospective future work, while host CPU latency ($2.23\text{ ms}$) demonstrates inference feasibility.
