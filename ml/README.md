# Machine Learning Control Pipeline: Parameterized Six-HASEL Prosthetic Hand

This directory contains the frozen, audited, and self-contained machine-learning subsystem for continuous surface electromyography (sEMG) decoding into six proportional actuator commands.

---

## 1. Division of Responsibilities

The project is structured with clear ownership boundaries:

### ML OWNER (Incoming Teammate)
Responsible for:
- Forearm sEMG signal processing & artifact rejection
- NinaPro DB8 & multimodal dataset ingestion/curation
- Deep neural network architectures (TCN, Attention, Transformers)
- Multimodal sEMG + Force Myography (FMG) sensor fusion
- Model compression, INT8 quantization & TinyML embedded optimization
- Cross-session, cross-subject, and real-time clinical validation

### ACTUATOR OWNER (Project Lead)
Responsible for:
- Electrohydraulic HASEL actuator physics, charge-voltage modeling & dielectric pouches
- Pouch stacking, kinematics, mechanical transmission & tendon routing
- MuJoCo mechanical digital twin plant & flexure joint compliances
- High-voltage power circuitry (0 to 8 kV) & driver safety isolation
- Grasp force, stroke margins & physical prototype validation

**Interface Meeting Point:**
The two independent systems meet strictly at the **six continuous control commands H1–H6**.

---

## 2. Interface Contract with the Actuator System

The ML subsystem acts as a continuous feedforward proportional command generator:

$$\mathbf{y}(t) = [H_1, H_2, H_3, H_4, H_5, H_6]^T \in [0.0, 1.0]^6$$

| Actuator Output | Downstream Interpretation | Physical Mechanism | Range |
| :---: | :--- | :--- | :---: |
| **H1** | Index Finger Flexor | Tension on index flexor tendon | $[0.0, 1.0]$ |
| **H2** | Middle Finger Flexor | Tension on middle flexor tendon | $[0.0, 1.0]$ |
| **H3** | Ring Finger Flexor | Tension on ring flexor tendon | $[0.0, 1.0]$ |
| **H4** | Little Finger Flexor | Tension on little flexor tendon | $[0.0, 1.0]$ |
| **H5** | Thumb Flexor | Tension on thumb MCP/IP flexor tendon | $[0.0, 1.0]$ |
| **H6** | Thumb Opposer | Tension on oblique CMC circumduction tendon | $[0.0, 1.0]$ |

### Temporal & Electrical Interface Specifications:
- **Input Sampling Rate:** $2,000\text{ Hz}$ across 16 bipolar forearm channels.
- **Inference Window Buffer:** $128.0\text{ ms}$ (256 samples).
- **Control Update Rate:** $31.25\text{ Hz}$ (one 6-command vector emitted every $32.0\text{ ms}$).
- **Inference Latency:** $2.23\text{ ms}$ on standard x86 CPU (leaves $> 93\%$ idle headroom per cycle).
- **Post-Filter Smoothing:** Causal Exponential Moving Average ($\alpha = 0.20$, effective time constant $\tau = 143.4\text{ ms}$) prevents mechanical actuator chatter.

> **Future Actuator Layer Role:** The downstream actuator layer converts $\mathbf{y}(t) \to \text{target voltage } V_k(t) \to \text{high-voltage amplifier} \to \text{physical HASEL pouch expansion}$.

---

## 3. Current Scientific Status

> **The current ML pipeline is a six-output sEMG regression baseline evaluated on strictly held-out Session 3 data from Subject 1.**

### Primary Performance Metrics (Strictly Held-Out Session 3):
- **Coefficient of Determination ($R^2$):** **$0.5301$**
- **Pearson Linear Correlation ($r$):** **$0.7718$**
- **Mean Absolute Error (MAE):** **$0.0728$**
- **Root Mean Squared Error (RMSE):** **$0.1348$**
- **Step-to-Step Jitter:** **$0.0066$** (73.6% reduction via EMA)
- **Mean CPU Latency:** **$2.23\text{ ms}$** (p95: $2.43\text{ ms}$)

> [!IMPORTANT]
> **$R^2 = 0.53$ is NOT "53% accuracy."**
> Because this task is continuous regression rather than discrete gesture classification, accuracy is not a valid primary metric. $R^2 = 0.5301$ indicates that the model explains 53.0% of the continuous variance of multi-finger joint motion across unseen recording sessions.

---

## 4. Explicit Scientific & Technical Boundaries

To maintain scientific integrity:
1. **NOT Closed-Loop Control:** The pipeline does not yet demonstrate closed-loop control. It is feedforward ($sEMG \to TCN \to \mathbf{y}$). Grip force sensor feedback and user co-adaptation are future work.
2. **NOT Flashed on Hardware:** Microcontroller deployment (ESP32-S3) is **prospective**. Although weight storage fits (325 kB INT8), total estimated inference SRAM (~647 kB) exceeds internal SRAM, requiring external PSRAM or buffer streaming.
3. **NOT Physiological EMD Measurement:** While ensemble predictions lead kinematics by $-133.3\text{ ms}$ (reduced to $-16.0\text{ ms}$ by EMA), this experiment does not isolate physiological electromechanical delay from windowing/filtering dynamics.
4. **H5 Thumb Flexion Limitation:** Thumb flexion achieves low explained variance ($R^2 \approx -0.009$, $r = 0.4713$) due to deep muscle anatomy (*flexor pollicis longus*) and low sensor calibration variance. It is reported transparently and retained without synthetic manipulation.

---

## 5. Directory Structure

```text
ml/
├── README.md                      # This document (division of labor & system overview)
├── MANIFEST.json                  # Reproducibility anchor with hashes and metadata
├── CURRENT_BASELINE.md            # Declaration of authoritative model checkpoint
│
├── environment/                   # Pinned dependency specifications
│   ├── requirements.txt
│   ├── environment.yml
│   └── VERSION_INFO.md
│
├── data/                          # Dataset handling & legal documentation
│   ├── README.md                  # Licensing, download links, and checksum verification
│   ├── DATASET_INFO.md            # Detailed sensor and session protocols
│   └── manifests/
│       └── ninapro_db8_manifest.json
│
├── preprocessing/                 # Causal signal conditioning modules
│   ├── filters.py                 # 4th-order causal Butterworth & 50 Hz notch
│   ├── windowing.py               # 256-sample windowing & ring buffer
│   ├── normalization.py           # Train-isolated Z-score & quantile scaling
│   └── alignment.py               # Endpoint causal alignment & cross-correlation lag
│
├── models/                        # Neural network architectures
│   ├── tcn/                       # Frozen Causal TCN Model C (332k parameters)
│   ├── gru_reference/             # Fair GRU benchmark baseline (41k parameters)
│   └── checkpoints/               # Authoritative PyTorch checkpoints & ONNX graph
│
├── training/                      # Training procedures
│   ├── train.py                   # Full training pipeline runner
│   ├── configs/                   # Experiment YAML configuration
│   └── README.md
│
├── evaluation/                    # Validation & testing
│   ├── evaluate.py                # Standalone Session 3 evaluator
│   ├── metrics.py                 # MAE, RMSE, R², correlation, jitter, EMA
│   ├── bootstrap.py               # 95% bootstrap confidence intervals
│   ├── compare_architectures.py   # Validation-set architecture comparison
│   └── configs/                   # Evaluation configuration
│
├── inference/                     # Real-time deployment & benchmarking
│   ├── inference.py               # Streaming controller class & H1-H6 mapping
│   ├── benchmark.py               # CPU/GPU latency & throughput profiler
│   ├── onnx/                      # ONNX export and parity verification
│   └── deployment/                # Embedded C header & SRAM feasibility report
│
├── results/                       # Verified experimental outputs
│   ├── RESULTS_TABLE.csv          # Comprehensive multi-model benchmark table
│   ├── session3/                  # Predictions and test metric JSON
│   ├── per_channel/               # H1-H6 per-output breakdown
│   ├── latency/                   # Latency profiles and TCN vs GRU table
│   └── figures/                   # 12 Publication-grade visualization plots
│
└── docs/                          # Detailed audit and handoff reports
    ├── ML_PIPELINE.md             # In-depth architectural dataflow
    ├── ML_HANDOFF.md              # Detailed onboarding guide for the incoming ML owner
    ├── RESULTS.md                 # Full empirical audit of all models
    ├── LIMITATIONS.md             # 10 Explicit scientific limitations
    └── REPRODUCIBILITY.md         # Deterministic reproduction commands
```

---

## 6. Quickstart Commands

```bash
# 1. Run unit tests verifying causality and shapes
python -m unittest discover tests

# 2. Evaluate frozen model on Session 3
python evaluate.py --checkpoint ml/models/checkpoints/best_model_6ch.pt

# 3. Verify ONNX numerical parity against PyTorch
python ml/inference/onnx/verify_onnx.py

# 4. Benchmark real-time CPU latency
python benchmark.py
```
