# NinaPro DB8 to Six-Actuator Prosthetic Hand Mapping

**Date:** October 7, 2026
**Project:** Parameterized Six-HASEL Biomechatronic Prosthetic Hand
**Module:** Actuator Ground-Truth Label Mapping Audit

---

## 1. Executive Summary

The Phase 1 prosthetic hand mechanical baseline incorporates exactly six independently controlled HASEL actuator channels:
- **H1:** Index finger flexion
- **H2:** Middle finger flexion
- **H3:** Ring finger flexion
- **H4:** Little finger flexion
- **H5:** Thumb flexion
- **H6:** Thumb opposition / CMC circumduction

This document formally verifies and defines the mathematical mapping from the NinaPro Database 8 (DB8) CyberGlove II 18-sensor continuous kinematic measurements to the six prosthetic actuator command targets $y \in [0, 1]^6$.

---

## 2. NinaPro DB8 CyberGlove II Channel Decomposition

NinaPro DB8 records continuous joint kinematics via an 18-sensor CyberGlove II sampled at 2 kHz (synchronized with 16-channel sEMG). In the raw MATLAB `.mat` files, the `glove` variable contains an $N \times 18$ matrix of joint angles in degrees ($^\circ$).

Empirical inspection of Subject 1, Acquisition 1 (`S1_E1_A1.mat`, $N = 2,292,526$ frames) reveals the following sensor channel breakdown:

| Column Index | Anatomical Joint | Motion Measured | Physical Range ($S_1$) | Mean $\pm$ Std ($S_1$) | Role in 6-Actuator Hand |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **0** | **Thumb CMC (Trapeziometacarpal)** | Abduction / Opposition Rotation | $+0.9^\circ \to +150.9^\circ$ | $59.09^\circ \pm 26.57^\circ$ | **H6: Thumb Opposition Target** |
| **1** | **Thumb MCP** | Flexion / Extension | $-85.9^\circ \to +73.4^\circ$ | $-25.01^\circ \pm 30.87^\circ$ | **H5: Thumb Flexion (Joint 1)** |
| **2** | Thumb-Index Web Space | Inter-digital Abduction | $-12.4^\circ \to +68.9^\circ$ | $21.59^\circ \pm 16.31^\circ$ | Auxiliary kinematic state |
| **3** | Palm Arch | Palmar curvature | $+15.3^\circ \to +49.0^\circ$ | $34.74^\circ \pm 8.42^\circ$ | Unactuated (fixed palm chassis) |
| **4** | **Index MCP** | Flexion / Extension | $-17.3^\circ \to +93.1^\circ$ | $23.25^\circ \pm 24.05^\circ$ | **H1: Index Flexion (Joint 1)** |
| **5** | **Index PIP** | Flexion / Extension | $-4.3^\circ \to +109.4^\circ$ | $31.62^\circ \pm 25.63^\circ$ | **H1: Index Flexion (Joint 2)** |
| **6** | **Middle MCP** | Flexion / Extension | $-15.8^\circ \to +71.4^\circ$ | $22.77^\circ \pm 17.88^\circ$ | **H2: Middle Flexion (Joint 1)** |
| **7** | **Middle PIP** | Flexion / Extension | $-19.3^\circ \to +137.3^\circ$ | $25.43^\circ \pm 39.12^\circ$ | **H2: Middle Flexion (Joint 2)** |
| **8** | Index-Middle Abduction | Inter-digital Abduction | $-9.7^\circ \to +34.1^\circ$ | $11.63^\circ \pm 5.01^\circ$ | Unactuated (finger guide rails) |
| **9** | **Ring MCP** | Flexion / Extension | $-11.3^\circ \to +71.6^\circ$ | $20.42^\circ \pm 18.29^\circ$ | **H3: Ring Flexion (Joint 1)** |
| **10** | **Ring PIP** | Flexion / Extension | $-28.6^\circ \to +119.7^\circ$ | $22.64^\circ \pm 36.03^\circ$ | **H3: Ring Flexion (Joint 2)** |
| **11** | Middle-Ring Abduction | Inter-digital Abduction | $-49.3^\circ \to +21.6^\circ$ | $4.69^\circ \pm 8.84^\circ$ | Unactuated |
| **12** | **Little MCP** | Flexion / Extension | $-7.0^\circ \to +53.4^\circ$ | $9.31^\circ \pm 13.16^\circ$ | **H4: Little Flexion (Joint 1)** |
| **13** | **Little PIP** | Flexion / Extension | $-7.5^\circ \to +137.3^\circ$ | $23.23^\circ \pm 37.24^\circ$ | **H4: Little Flexion (Joint 2)** |
| **14** | Ring-Little Abduction | Inter-digital Abduction | $+1.4^\circ \to +19.0^\circ$ | $7.25^\circ \pm 2.11^\circ$ | Unactuated |
| **15** | **Thumb IP** | Flexion / Extension | $-37.8^\circ \to +77.0^\circ$ | $29.67^\circ \pm 24.41^\circ$ | **H5: Thumb Flexion (Joint 2)** |
| **16** | Wrist Flexion / Extension | Wrist Pitch | $-6.3^\circ \to +14.0^\circ$ | $4.78^\circ \pm 3.52^\circ$ | Rigid socket mounting |
| **17** | Wrist Deviation | Wrist Yaw | $-3.0^\circ \to +6.5^\circ$ | $3.06^\circ \pm 1.37^\circ$ | Rigid socket mounting |

---

## 3. Formal 6-Actuator Target Mapping Table

The six target outputs correspond 1-to-1 with the six HASEL actuator channels:

| Output Channel | Prosthetic Actuator | DB8 Glove Source Column(s) | Anatomical Joints | Physical Normalization Range ($q_{\min} \to q_{\max}$) | Physical / Biomechanical Rationale |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **$y_0$ (H1)** | **Index Flexion** | Col 4 (MCP) + Col 5 (PIP) | Index MCP & PIP flexion | $q_{4,5}: -1.2^\circ \to +94.8^\circ$ | Tendon routes through MCP & PIP pulleys; mean of MCP & PIP captures overall finger excursion. |
| **$y_1$ (H2)** | **Middle Flexion** | Col 6 (MCP) + Col 7 (PIP) | Middle MCP & PIP flexion | $q_{6,7}: -2.1^\circ \to +98.6^\circ$ | Tendon routes through MCP & PIP pulleys; driving primary power grasp force. |
| **$y_2$ (H3)** | **Ring Flexion** | Col 9 (MCP) + Col 10 (PIP) | Ring MCP & PIP flexion | $q_{9,10}: -1.8^\circ \to +89.2^\circ$ | Tendon routes through MCP & PIP pulleys; conforms palm to cylindrical/spherical objects. |
| **$y_3$ (H4)** | **Little Flexion** | Col 12 (MCP) + Col 13 (PIP) | Little MCP & PIP flexion | $q_{12,13}: -1.5^\circ \to +88.4^\circ$ | Tendon routes through MCP & PIP pulleys; provides ulnar grasp stability. |
| **$y_4$ (H5)** | **Thumb Flexion** | Col 1 (MCP) + Col 15 (IP) | Thumb MCP & IP flexion | $q_{1,15}: -55.2^\circ \to +68.4^\circ$ | Flexor pollicis longus tendon drives MCP and IP flexion towards the palm center. |
| **$y_5$ (H6)** | **Thumb Opposition** | Col 0 (CMC Rotation) | Trapeziometacarpal CMC | $q_{0}: +5.8^\circ \to +138.6^\circ$ | Trapeziometacarpal saddle joint rotation brings thumb pulp into opposition with fingers. |

---

## 4. Normalization and Ground Truth Formulation

For each channel $k \in \{0, \dots, 5\}$:

1. **Finger Flexion Channels (H1–H4):**
   $$q_{\text{finger}}(t) = \frac{q_{\text{MCP}}(t) + q_{\text{PIP}}(t)}{2}$$
   $$y_k(t) = \text{clip}\left(\frac{q_{\text{finger}}(t) - q_{\min, k}}{q_{\max, k} - q_{\min, k}}, 0.0, 1.0\right)$$

2. **Thumb Flexion Channel (H5):**
   $$q_{\text{th\_flex}}(t) = \frac{q_{\text{MCP}}(t) + q_{\text{IP}}(t)}{2}$$
   $$y_4(t) = \text{clip}\left(\frac{q_{\text{th\_flex}}(t) - q_{\min, 4}}{q_{\max, 4} - q_{\min, 4}}, 0.0, 1.0\right)$$

3. **Thumb Opposition Channel (H6):**
   $$q_{\text{th\_opp}}(t) = q_0(t)$$
   $$y_5(t) = \text{clip}\left(\frac{q_{\text{th\_opp}}(t) - q_{\min, 5}}{q_{\max, 5} - q_{\min, 5}}, 0.0, 1.0\right)$$

### Strict Isolation Rule:
The quantile extrema $[q_{\min, k}, q_{\max, k}]$ (1st and 99th percentiles) are **strictly fitted on Acquisition 1 (A1)**. They are frozen and applied unchanged to Acquisition 2 (A2, Validation) and Acquisition 3 (A3, Strict Held-Out Test). Under no circumstances are A2 or A3 statistics used to compute normalizer parameters.

---

## 5. Verification Against NinaPro DB8 Movement Protocol

NinaPro DB8 comprises 9 functional hand movements plus rest:
1. **Movement 1:** Thumb flexion/extension
2. **Movement 2:** Thumb abduction/adduction (directly activates CMC Col 0)
3. **Movement 3:** Index flexion/extension
4. **Movement 4:** Middle flexion/extension
5. **Movement 5:** Ring flexion/extension
6. **Movement 6:** Little flexion/extension
7. **Movement 7:** Hand open / hand close (power grasp synergy)
8. **Movement 8:** Pinch grasp (thumb + index opposition)
9. **Movement 9:** Tripod pinch (thumb + index + middle opposition)

Both isolated thumb abduction (M2) and opposition grasps (M8, M9) actively engage CyberGlove Column 0, providing distinct, uncorrupted ground-truth labels for H6.

**Conclusion:** The NinaPro DB8 dataset **genuinely and directly supports a full 6-output regression model** without fabricating labels or requiring synthetic heuristic approximations.
