# Target Mapping & Six-Actuator Architecture Analysis

**Date:** October 7, 2026
**Project:** Biomechatronic Prosthetic Hand ML Pipeline (Continuous Kinematic Regression)
**Scope:** CyberGlove II 18-channel decomposition, target normalization, and 5-output vs 6-actuator architecture resolution.

---

## 1. CyberGlove II 18-Channel Kinematic Decomposition

The NinaPro DB8 dataset records continuous contralateral hand kinematics using an 18-sensor CyberGlove II. In raw `.mat` files, `glove` is stored as an $(N, 18)$ matrix of calibrated joint angles in degrees ($^\circ$).

### Empirical Channel Inventory:
| Column Index | Anatomical Joint | Motion Measured | Physical Range ($S_1$) | Mean Baseline ($S_1$) |
| :---: | :--- | :--- | :---: | :---: |
| **0** | **Thumb CMC (Trapeziometacarpal)** | Abduction / Opposition Rotation | $+14.9^\circ \to +167.7^\circ$ | $+58.4^\circ$ |
| **1** | **Thumb MCP** | Flexion / Extension | $-67.8^\circ \to +71.7^\circ$ | $-25.2^\circ$ |
| **2** | **Thumb-Index Web Space** | Inter-digital Abduction | $-6.5^\circ \to +69.8^\circ$ | $+23.7^\circ$ |
| **3** | Uncalibrated / Palm Arch | Palmar curvature | $+14.9^\circ \to +48.7^\circ$ | $+36.0^\circ$ |
| **4** | **Index MCP** | Flexion / Extension | $-22.1^\circ \to +85.4^\circ$ | $+19.0^\circ$ |
| **5** | **Index PIP** | Flexion / Extension | $-4.3^\circ \to +112.0^\circ$ | $+36.9^\circ$ |
| **6** | **Middle MCP** | Flexion / Extension | $-18.5^\circ \to +58.7^\circ$ | $+20.2^\circ$ |
| **7** | **Middle PIP** | Flexion / Extension | $-19.3^\circ \to +137.3^\circ$ | $+29.1^\circ$ |
| **8** | Index-Middle Abduction | Inter-digital Abduction | $-10.7^\circ \to +25.9^\circ$ | $+10.3^\circ$ |
| **9** | **Ring MCP** | Flexion / Extension | $-12.9^\circ \to +68.4^\circ$ | $+17.2^\circ$ |
| **10** | **Ring PIP** | Flexion / Extension | $-23.3^\circ \to +121.8^\circ$ | $+25.6^\circ$ |
| **11** | Middle-Ring Abduction | Inter-digital Abduction | $-43.4^\circ \to +21.9^\circ$ | $+5.0^\circ$ |
| **12** | **Little MCP** | Flexion / Extension | $-8.9^\circ \to +48.9^\circ$ | $+6.9^\circ$ |
| **13** | **Little PIP** | Flexion / Extension | $-7.5^\circ \to +141.6^\circ$ | $+26.6^\circ$ |
| **14** | Ring-Little Abduction | Inter-digital Abduction | $+2.5^\circ \to +15.7^\circ$ | $+6.7^\circ$ |
| **15** | **Thumb IP** | Flexion / Extension | $-34.0^\circ \to +70.6^\circ$ | $+35.2^\circ$ |
| **16** | Wrist Flexion / Extension | Wrist Pitch | $-2.3^\circ \to +14.5^\circ$ | $+2.6^\circ$ |
| **17** | Wrist Radial / Ulnar Deviation | Wrist Yaw | $-1.4^\circ \to +6.9^\circ$ | $+4.6^\circ$ |

---

## 2. Five-Output Continuous Digit Flexion Mapping

To formulate continuous finger flexion states $\theta \in [0, 1]^5$ where $0 = \text{fully extended / open}$ and $1 = \text{fully flexed}$, we map the dual-sensor flexion pairs (MCP + PIP for fingers, MCP + IP for thumb) while preserving all 18 raw channels in preprocessed files:

### Target Formulations:
1. **Thumb Flexion ($\theta_{\text{thumb}}$):**
   $$\theta_{\text{thumb}} = \frac{1}{2} \left[ \frac{q_1 - q_{1,\min}}{q_{1,\max} - q_{1,\min}} + \frac{q_{15} - q_{15,\min}}{q_{15,\max} - q_{15,\min}} \right]$$
   - Combines Thumb MCP (Col 1) and Thumb IP (Col 15).
2. **Index Flexion ($\theta_{\text{index}}$):**
   $$\theta_{\text{index}} = \frac{1}{2} \left[ \frac{q_4 - q_{4,\min}}{q_{4,\max} - q_{4,\min}} + \frac{q_5 - q_{5,\min}}{q_{5,\max} - q_{5,\min}} \right]$$
   - Combines Index MCP (Col 4) and Index PIP (Col 5).
3. **Middle Flexion ($\theta_{\text{middle}}$):**
   $$\theta_{\text{middle}} = \frac{1}{2} \left[ \frac{q_6 - q_{6,\min}}{q_{6,\max} - q_{6,\min}} + \frac{q_7 - q_{7,\min}}{q_{7,\max} - q_{7,\min}} \right]$$
   - Combines Middle MCP (Col 6) and Middle PIP (Col 7).
4. **Ring Flexion ($\theta_{\text{ring}}$):**
   $$\theta_{\text{ring}} = \frac{1}{2} \left[ \frac{q_9 - q_{9,\min}}{q_{9,\max} - q_{9,\min}} + \frac{q_{10} - q_{10,\min}}{q_{10,\max} - q_{10,\min}} \right]$$
   - Combines Ring MCP (Col 9) and Ring PIP (Col 10).
5. **Pinky Flexion ($\theta_{\text{pinky}}$):**
   $$\theta_{\text{pinky}} = \frac{1}{2} \left[ \frac{q_{12} - q_{12,\min}}{q_{12,\max} - q_{12,\min}} + \frac{q_{13} - q_{13,\min}}{q_{13,\max} - q_{13,\min}} \right]$$
   - Combines Little MCP (Col 12) and Little PIP (Col 13).

> **Crucial Data Integrity Rule:** The normalization extrema $[q_{\min}, q_{\max}]$ are fitted **strictly on the Training Set (Acquisition 1)** and applied frozen to Validation (A2) and Held-Out Test (A3). This prevents test data leakage.

---

## 3. The Six-Actuator Architectural Issue & Scientific Resolution

### The Physical Problem:
The Phase 1 physical plant contains **six independent HASEL actuator channels**:
- **H1:** Index finger flexion
- **H2:** Middle finger flexion
- **H3:** Ring finger flexion
- **H4:** Little finger flexion
- **H5:** Thumb flexion
- **H6:** Thumb opposition / CMC circumduction

A 5-output model $\theta = [\text{thumb}, \text{index}, \text{middle}, \text{ring}, \text{pinky}]$ only controls 5 degrees of actuation, leaving H6 unaddressed.

### Evaluation of Candidate Solutions:

| Option | Architecture | Dataset Feasibility | Biomechanical Rationale | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: 5-Output TCN + Separate Thumb-Opposition Estimator** | Two independent models: one 5-output TCN for flexion + one separate regression head for opposition. | Feasible, but adds compute overhead and parameter duplication. | Biomechanically uncoupled; ignores common flexor-abductor synergies. | Sub-optimal |
| **Option B: 6-Output Unified Causal TCN** | Single causal TCN with 6 continuous outputs: $[\theta_{\text{idx}}, \theta_{\text{mid}}, \theta_{\text{ring}}, \theta_{\text{pinky}}, \theta_{\text{th\_flex}}, \theta_{\text{th\_opp}}]$. | **Directly Supported by NinaPro DB8.** Movement 2 explicitly exercises thumb abduction, and Glove Col 0 directly measures CMC opposition rotation. | Biomechanically optimal: captures multi-digit synergies and shared motor unit activations across the entire 16-channel array. | **RECOMMENDED PRIMARY** |
| **Option C: 5-Output TCN + Deterministic Kinematic Synergy Mapper** | 5 outputs drive H1–H5; H6 is computed via grasp synergy: $\theta_{\text{H6}} = f(\theta_{\text{thumb}}, \theta_{\text{index}}, \bar{\theta})$. | Requires no extra labels, uses known prosthetic synergies. | Standard fallback in commercial prosthetics (e.g. Azzurra hand), but cannot independently command open-palm opposition. | **SUPPORTED FALLBACK** |

### Implementation Decision:
1. **Configurable Output Head:** The pipeline supports both `num_outputs: 5` and `num_outputs: 6`.
2. In the 6-output configuration, target channel 6 is mapped to normalized **CyberGlove Column 0 (Thumb CMC Abduction/Rotation)**:
   $$\theta_{\text{th\_opp}} = \frac{q_0 - q_{0,\min}}{q_{0,\max} - q_{0,\min}} \in [0, 1]$$
3. In the 5-output configuration, an explicit deterministic synergy adapter maps $\theta \to \mathbf{u}_{\text{HASEL}} \in [0, 1]^6$ for seamless integration with `simulation.py`.
