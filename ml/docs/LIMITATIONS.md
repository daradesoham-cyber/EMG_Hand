# Scientific & Technical Limitations of the Current ML Baseline

This document explicitly enumerates the ten primary limitations of the current machine learning pipeline to ensure rigorous scientific standards during the handoff.

---

## 1. Single-Subject Scope (Subject 1 Only)
The current pipeline was fitted and evaluated exclusively on data from NinaPro Database 8, **Subject 1** (right-handed able-bodied male). Performance across subjects with different forearm circumferences, muscle volumes, or skin impedance remains unvalidated.

## 2. No Inter-Subject Cross-Validation
While Session 3 was strictly held out across time, **inter-subject cross-validation** was not evaluated. The generalization of the learned convolutional filter weights to unseen users is expected to exhibit performance degradation without transfer learning or subject-specific calibration.

## 3. Poor H5 Thumb Flexion Performance ($R^2 \approx -0.009$)
Thumb flexion (H5) achieves positive Pearson correlation ($r = 0.4713$), but negative explained variance ($R^2 = -0.0094$). The model output should not be relied upon for high-precision autonomous thumb tip pinching without sensor fusion or mechanical compliance safeguards.

## 4. Deep Muscle Attenuation & Modality Mismatch
The flexor pollicis longus (FPL) muscle responsible for thumb IP flexion resides deep in the forearm anterior compartment beneath superficial flexor muscles. Standard surface EMG electrodes measure low signal-to-noise ratio and significant volume-conductor crosstalk from adjacent finger flexors.

## 5. Physical sEMG Acquisition Hardware Not Tested
The pipeline operates on pre-recorded, sinc-interpolated NinaPro benchmark files. Physical front-end amplifier saturation, electrode motion artifacts, sweating, impedance drift over hours of wearing, and 50/60 Hz power-line interference in clinical environments have not been evaluated in this setup.

## 6. Physical HASEL Actuator System Not Connected
The ML control commands ($y_0 \dots y_5 \in [0, 1]^6$) represent proportional setpoints. Physical electrohydraulic HASEL actuators, high-voltage amplification circuits (0–8 kV), and fluid dielectric pouches were not connected to this ML pipeline.

## 7. No Closed-Loop Sensorimotor Feedback
The current pipeline is an **open-loop feedforward estimator** ($sEMG \to TCN \to y$). It does not incorporate grasp force sensor feedback, tactile slips, visual closed-loop correction, or user co-adaptation.

## 8. ESP32-S3 Microcontroller Deployment Unvalidated on Hardware
Inference on microcontrollers (e.g., Espressif ESP32-S3) is **prospective**. Although weight memory (325 kB INT8) and execution time (2.23 ms CPU) demonstrate theoretical feasibility, total estimated inference SRAM (~647 kB) exceeds internal SRAM, requiring external PSRAM or buffer streaming optimizations that have not been physically flashed or profiled.

## 9. CyberGlove Target Sensor Non-Linearities
Kinematic ground-truth angles were captured via strain-gauge CyberGlove II sensors. Calibration drift, nonlinear glove fabric stretching, and joint cross-axis coupling introduce target noise that limits theoretical model upper bounds.

## 10. Generalization to Free-Form Prosthetic Use Unverified
The protocol evaluated 9 structured, repetitive movements under controlled laboratory seated conditions. Arm posture changes, shoulder abduction, limb orientation shifts relative to gravity, and co-contraction during real-life activities were not evaluated.
