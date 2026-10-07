# Machine Learning Project Handoff Document

**To:** Incoming ML / sEMG Pipeline Owner
**From:** Actuator / Digital-Twin Plant Lead
**Date:** October 7, 2026
**Subject:** Complete Handoff of the Parameterized Six-HASEL Hand ML Control Pipeline

---

## 1. Division of Responsibilities

To accelerate overall progress, the prosthetic hand project is partitioned into two independent subsystems:

```text
┌─────────────────────────────────────────────────────────┐
│                     ML OWNER (You)                      │
│ - Forearm sEMG signal processing & filtering           │
│ - NinaPro DB8 & multi-modal dataset management          │
│ - Neural network architecture (TCN, Transformers, etc.) │
│ - Multi-modal sensor fusion (sEMG + FMG)               │
│ - Quantization & embedded inference (ESP32 / ARM)       │
│ - Cross-session & cross-subject generalization         │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼  [Interface Contract: 6 continuous setpoints H1..H6 ∈ [0, 1]]
┌─────────────────────────────────────────────────────────┐
│                  ACTUATOR OWNER (Me)                    │
│ - Electrohydraulic HASEL actuator physics & modeling    │
│ - Pouch geometry, stack count & dielectric breakdown    │
│ - High-voltage power circuitry (0 to 8 kV)             │
│ - Tendon routing, pulley ratios & non-linear friction   │
│ - MuJoCo rigid-body mechanics & flexure compliance      │
│ - Physical prototype fabrication & force testing        │
└─────────────────────────────────────────────────────────┘
```

The two subsystems meet cleanly at the **six continuous control commands H1–H6**.

---

## 2. What Is Finished

1. **Complete causal preprocessing pipeline:** 4th-order causal Butterworth bandpass (20–450 Hz) and 50 Hz notch filter via Direct Form II Transposed (`scipy.signal.lfilter`). No future data leakage.
2. **Train-isolated normalization:** Channel-wise Z-score statistics ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) and target angle percentiles (1st and 99th percentiles) are strictly fitted on Acquisition 1 only.
3. **Causal TCN Model C:** 6 stacked causal dilated blocks, dilations $[1, 2, 4, 8, 16, 32]$, 2 convs/block, receptive field of 253 samples ($126.5\text{ ms}$, covering 98.8% of the buffer).
4. **Authoritative Checkpoint:** Trained to Epoch 33, evaluated on strictly held-out Session 3 (A3).
5. **EMA Smoothing Filter:** Parameterized at $\alpha = 0.20$, reducing step-to-step derivative jitter by 73.6% and stabilizing the ensemble temporal lead to $-16.0\text{ ms}$.
6. **ONNX Export & Verification:** Verified numerically with maximum absolute error $< 4.18 \times 10^{-7}$ against PyTorch.
7. **Comprehensive Benchmarking:** Host CPU latency is $2.23\text{ ms}$ (p95: $2.43\text{ ms}$), leaving $> 93\%$ idle processing headroom against the $32.0\text{ ms}$ inference deadline.
8. **Unit Tests:** 7 comprehensive unit tests verifying causality, window shapes, and normalization isolation.

---

## 3. What Is Frozen (Do NOT Touch in V1)

- **Do NOT retrain or fit anything on Session 3 (A3).** Session 3 must remain strictly held out as the final test benchmark.
- **Do NOT reintroduce bidirectional filtering (`filtfilt`).** All filters must remain strictly causal.
- **Do NOT modify the existing frozen checkpoint (`ml/models/checkpoints/best_model_6ch.pt`).** Save new models under distinct filenames.
- **Do NOT modify physical HASEL plant files:** `hand.xml`, `hasel_model.py`, `parameters.py`, `simulation.py`, `grasp_tests.py`, `optimization.py`.

---

## 4. Where the Code & Artifacts Reside

- **Root Entrypoints:**
  - `python train.py --config configs/tcn_db8.yaml`
  - `python evaluate.py --checkpoint ml/models/checkpoints/best_model_6ch.pt`
  - `python benchmark.py`
- **Modular Package Structure (`ml/`):**
  - `ml/preprocessing/`: Signal filters (`filters.py`), windowing (`windowing.py`), normalization (`normalization.py`), temporal alignment (`alignment.py`).
  - `ml/models/`: Causal TCN (`tcn/`), GRU reference (`gru_reference/`), checkpoints (`checkpoints/`).
  - `ml/training/`: Training runner (`train.py`), configs (`configs/tcn_db8.yaml`).
  - `ml/evaluation/`: Test runner (`evaluate.py`), metrics (`metrics.py`), bootstrap CIs (`bootstrap.py`).
  - `ml/inference/`: Real-time controller (`inference.py`), latency profiler (`benchmark.py`), ONNX exporter & verifier (`onnx/`).
  - `ml/results/`: Full result tables (`RESULTS_TABLE.csv`), Session 3 JSON/CSV, per-channel metrics, publication figures.
  - `ml/docs/`: Full documentation suite (`ML_PIPELINE.md`, `RESULTS.md`, `LIMITATIONS.md`, `REPRODUCIBILITY.md`).

---

## 5. Where the Dataset Goes

Place NinaPro DB8 Subject 1 `.mat` files into `data/ninapro_db8/`:
- `data/ninapro_db8/S1_E1_A1.mat` (Training)
- `data/ninapro_db8/S1_E1_A2.mat` (Validation)
- `data/ninapro_db8/S1_E1_A3.mat` (Strictly held-out test)

*(See `ml/data/README.md` for download links and SHA-256 checksums).*

---

## 6. Current Authoritative Checkpoint & Metrics

- **Checkpoint File:** `ml/models/checkpoints/best_model_6ch.pt`
- **SHA-256:** `744fc0a602547c0bd167a2d7f478e1b15f04f2d40ce6e48e457239aa9380cd49`
- **Session 3 Performance (TCN + EMA $\alpha=0.20$):**
  - **MAE:** $0.0728$
  - **RMSE:** $0.1348$
  - **$R^2$:** $0.5301$ (Kinematic variance explained)
  - **Pearson $r$:** $0.7718$
  - **Step Jitter:** $0.0066$
  - **Host CPU Latency:** $2.23\text{ ms}$ (mean), $2.43\text{ ms}$ (p95)

---

## 7. Output Interface Contract (H1–H6)

The model emits 6 continuous control commands:
$$\mathbf{y}(t) = [H_1, H_2, H_3, H_4, H_5, H_6]^T \in [0.0, 1.0]^6$$

| Channel | Actuator Role | Physical Motion | Controlled Finger |
| :---: | :--- | :--- | :--- |
| **H1** | Index Flexor | MCP + PIP flexion | Index |
| **H2** | Middle Flexor | MCP + PIP flexion | Middle |
| **H3** | Ring Flexor | MCP + PIP flexion | Ring |
| **H4** | Little Flexor | MCP + PIP flexion | Little |
| **H5** | Thumb Flexor | MCP + IP flexion | Thumb |
| **H6** | Thumb Opposer | CMC trapeziometacarpal rotation | Thumb saddle |

---

## 8. Known Issues & Scientific Limitations

1. **Thumb Flexion (H5) is weak:** $R^2 = -0.0094$ (Pearson $r = 0.4713$). Caused by anatomical depth of the *flexor pollicis longus* muscle and low ground-truth variance. Do NOT fabricate or delete H5; address it scientifically via sensor fusion or feature engineering.
2. **Single subject evaluated:** Trained exclusively on Subject 1. Cross-subject adaptation remains to be investigated.
3. **No physical hardware loop:** The pipeline is feedforward ($sEMG \to TCN \to y$). Physical HASEL actuators and grip force sensor feedback are future actuator-layer integration work.
4. **Embedded SRAM constraint:** Estimated inference SRAM is ~647 kB, which exceeds the ~380 kB internal user SRAM on the ESP32-S3. Deployment requires external PSRAM or streaming buffer optimizations.

---

## 9. Suggested Next Experiments for You (The ML Owner)

1. **Multi-Subject Generalization:** Evaluate Subject 2 through Subject 12 from NinaPro DB8.
2. **Multi-Modal Sensor Fusion:** Fuse sEMG with Force Myography (FMG) or ultrasound to capture deep thumb muscle activations.
3. **TinyML Optimization:** Implement streaming buffer convolutions to fit the 128-channel feature maps into $< 40\text{ kB}$ internal SRAM for ESP32-S3 microcontrollers without PSRAM latency penalties.
4. **Attention / Light Conformer:** Benchmark whether lightweight linear attention can improve multi-finger coordination without exceeding the 32 ms latency deadline.
