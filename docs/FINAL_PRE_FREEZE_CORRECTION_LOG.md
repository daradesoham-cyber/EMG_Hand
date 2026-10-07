# Final ML Pre-Freeze Scientific & Terminology Correction Log

**Document:** Audit Change Log & Pre-Freeze Integrity Verification
**Date:** October 7, 2026
**Project:** Six-Output sEMG Regression Validation for the Six-Actuator Prosthetic-Hand Control Architecture
**Dataset:** NinaPro Database 8 (DB8) — Subject 1 Empirical Benchmark

---

## 1. Summary of Scientific & Wording Corrections

### 1.1 Temporal Lead / EMD Claim Softened
- **Previous Overclaim:** Stated that the observed $-133.3\text{ ms}$ lead was "the direct manifestation of biological Electromechanical Delay (EMD)".
- **Correction:** Replaced with scientifically defensible wording:
  > *"Cross-correlation identified an ensemble prediction lead of approximately $-133.3\text{ ms}$ relative to the measured CyberGlove displacement. This lead is physiologically plausible given the known temporal separation between neural/muscle activation and observable joint movement; however, the present experiment does not independently isolate physiological electromechanical delay from model/windowing effects."*
- **Distinction Enforced:** Clearly differentiated observed fact (prediction leads glove movement by X ms) from physiological hypothesis (consistent with muscle activation preceding joint rotation).

### 1.2 H5 Thumb Flexion Interpretation
- **Previous Overclaim:** Asserted that reduced H5 performance was proven to be exclusively caused by anatomical and sensor-polarization factors.
- **Correction:** Replaced with evidence-backed formulation:
  > *"The audit identified anatomical, target-construction, sensor-polarity, and low-variance factors that plausibly explain the reduced H5 performance. No implementation error was identified in the audited pipeline; however, the present experiment does not establish that these factors are the sole causes."*
- H5 is preserved in the six-output architecture without artificial smoothing or target synthesis.

### 1.3 GRU Real-Time Constraint Formulation
- **Previous Overclaim:** Claimed the GRU was "physically incapable of running at 31.25 Hz on CPU".
- **Correction:** Narrowed to tested empirical reality:
  > *"The evaluated GRU implementation fails the 32 ms real-time deadline on the tested CPU platform. The measured mean latency was 57.35 ms, with p95 latency of 91.04 ms, exceeding the 32 ms execution budget."*

### 1.4 Real-Time Scope Clarification
- **Previous Overclaim:** Claimed "real-time execution confirmed (2.23 ms on CPU, 93% headroom)".
- **Correction:** Clarified scope to inference feasibility:
  > *"Real-time inference feasibility was demonstrated on the tested CPU platform, with a measured mean TCN inference latency of 2.23 ms against the 32.0 ms inference deadline."*
- Explicitly stated that physical end-to-end prosthetic control loop (sensor acquisition $\to$ communication $\to$ HASEL dynamics $\to$ physical grasp) was not evaluated in this ML phase.

### 1.5 TCN vs. GRU Engineering Decision Justification
- Preserved Causal TCN Model C as the selected baseline.
- Clarified that while the GRU achieves higher offline predictive accuracy ($R^2 = 0.6483$ vs $0.5301$), it exceeds the allowable 32 ms budget on the tested CPU platform ($57.35\text{ ms}$ vs $2.23\text{ ms}$). The TCN provides viable inference speed, deterministic feedforward vectorization, and a mathematically bounded finite receptive field (126.5 ms).

### 1.6 Memory & File Size Accounting
- Maintained exact accounting: 332,966 parameters; $1,300.65\text{ kB}$ FP32 weights; $325.16\text{ kB}$ INT8 target; $65.50\text{ kB}$ ONNX graph topology; $1,300.65\text{ kB}$ ONNX external data blob; $\approx 262\text{ kB}$ activation SRAM; $\approx 647\text{ kB}$ total estimated runtime SRAM.

### 1.7 Prospective Embedded Target Platform Clarification
- ESP32-S3 and ARM Cortex-M microcontrollers remain prospective deployment targets.
- Documented that $647\text{ kB}$ estimated inference memory exceeds the $\approx 380\text{ kB}$ user-available internal SRAM of an ESP32-S3, requiring external PSRAM or temporal buffer streaming. Physical microcontroller deployment is not claimed.

### 1.8 Scope & Terminology Alignment
- Enforced required standard title and scope across all documentation:
  *"Six-output sEMG regression validation for the six-actuator prosthetic-hand control architecture"*.

---

## 2. Numerical Results Changed?

**NO — numerical model results were not changed.**

All reported values:
- Session 3 metrics (Raw TCN MAE 0.0811, $R^2$ 0.4133; TCN+EMA MAE 0.0728, $R^2$ 0.5301; Raw GRU MAE 0.0726, $R^2$ 0.5345; GRU+EMA MAE 0.0653, $R^2$ 0.6483);
- Parameter counts (332,966 TCN, 41,094 GRU);
- CPU latencies (2.23 ms TCN, 57.35 ms GRU);
- Receptive fields (253 samples / 126.5 ms);
- Temporal cross-correlation leads ($-133.3\text{ ms}$ raw, $-16.0\text{ ms}$ EMA);
remain identical to the actual measured logs.

---

## 3. Final Pre-Freeze Audit Table

| Audit Item | Status | Verification Detail |
| :--- | :---: | :--- |
| **Six-output target mapping** | **PASS** | Mapped 1-to-1 to H1–H6 using verified CyberGlove channels. |
| **Session 3 strict holdout** | **PASS** | Untouched during training, architecture comparison, and EMA grid tuning. |
| **Causal preprocessing** | **PASS** | Causal IIR forward filter (`lfilter`), left-padded convolutions, A1-only normalization. |
| **Receptive field calculation** | **PASS** | Mathematically proven and gradient-verified ($RF = 253\text{ samples} / 126.5\text{ ms}$). |
| **Temporal indexing** | **PASS** | Zero indexing offset between sEMG window end and target sample. |
| **EMD interpretation** | **CORRECTED** | Softened to observed cross-correlation lead + physiological plausibility. |
| **H5 diagnosis** | **CORRECTED** | Softened to plausible contributing factors without claiming sole causation. |
| **GRU symmetric benchmark** | **PASS** | Evaluated both Raw and EMA ($\alpha = 0.20$ selected on Session 2 only). |
| **TCN real-time inference** | **PASS** | Measured mean CPU latency of $2.23\text{ ms}$ satisfies 32 ms deadline (93% headroom). |
| **Full physical real-time loop** | **NOT VALIDATED** | Actuator dynamics, sensor feedback, and clinical loop remain prospective. |
| **Model parameter count** | **PASS** | Verified at exactly 332,966 trainable parameters. |
| **FP32/INT8 memory accounting** | **PASS** | $1,300.65\text{ kB}$ FP32, $325.16\text{ kB}$ INT8, ONNX graph vs data blob distinguished. |
| **ESP32 deployment** | **NOT VALIDATED** | Accurately framed as prospective hardware target. |
| **HASEL physical validation** | **OUT OF SCOPE** | Deterministic Phase 1 digital-twin physics unchanged. |
| **Clinical validation** | **OUT OF SCOPE** | Single intact subject cross-session dataset; patient trials not performed. |

---

## 4. Final Baseline Freezing Declaration

> **ML BASELINE READY TO FREEZE**
>
> The six-output causal TCN baseline is ready to freeze as the computational ML control model for the prosthetic-hand architecture. The model was evaluated on a strictly held-out acquisition, satisfies the measured inference-time requirement on the tested CPU platform, and has been audited for temporal indexing, architecture comparison, memory accounting, and target construction.
>
> The result should not be interpreted as physical HASEL validation, end-to-end prosthetic control validation, physiological EMD measurement, ESP32 deployment validation, or clinical validation.
