# Parameterized Six-HASEL Biomechatronic Prosthetic Hand — Phase 1 Mechanical Digital Twin

This repository contains the deterministic physics-based digital twin and automated design-optimization framework for a six-HASEL-actuator prosthetic hand developed in Python and MuJoCo.

In strict compliance with **Phase 1 Master Requirements**, no machine learning, sEMG/FMG decoding, or sensory feedback loops are included. All kinematics, tendon transmissions, joint compliances, contact kinetics, and robustness metrics are derived deterministically.

---

## Architecture Summary
- **Channels**: 6 Independent HASEL Actuator Channels (No whiffletree differentials).
  1. Index Finger Flexor
  2. Middle Finger Flexor
  3. Ring Finger Flexor
  4. Little Finger Flexor
  5. Thumb Flexion (MP & IP joints)
  6. Thumb Opposition (Oblique CMC joint)
- **Actuator Model**: Electrohydraulic phenomenological force-stroke relationship:
  $$F = F_{\\max} \\cdot V^2 \\cdot \\left(1 - \\frac{s}{s_{\\max}}\\right)$$
- **Joint Compliance**: Passive flexure TPU joints (stiffness = 0.15 N*m/rad, damping = 0.02 N*m*s/rad).
- **Optimal Placement**: Forearm cassette mounting (reduces distal inertia and keeps wrist reaction torque to 0.210 N*m).

---

## Directory Structure
```
prosthetic_hand_phase1/
├── hand.xml                         # Procedurally generated optimal MJCF model
├── optimize.py                      # Master pipeline runner
├── parameters.py                    # Typed parameter schema & reference loaders
├── hasel_model.py                   # Phenomenological HASEL actuator model
├── anthropometry.py                 # Human anthropometric geometry & bounds
├── simulation.py                    # Procedural MuJoCo MJCF generator
├── grasp_tests.py                   # 10 Standardized grasp benchmark evaluator
├── optimization.py                  # Hierarchical optimizer & robustness suite
├── analysis.py                      # Kinematics, efficiencies, and kinetics
├── visualization.py                 # 15 Scientific plot generators
│
├── data/
│   ├── anthropometry.json           # Literature references & bounds
│   ├── hasel_reference.json         # Pouch stack physical targets
│   └── design_requirements.json     # Benchmark performance targets
│
└── results/
    ├── optimization_results.csv     # Target vs achieved metric summary
    ├── design_database.csv          # 46 Evaluated candidate architectures
    ├── best_design.json             # CAD-ready parameters export
    ├── BEST_DESIGN_REPORT.md        # Comprehensive technical report
    └── plots/                       # 15 High-resolution engineering figures
        ├── 01_hand_geometry.png
        ├── 02_hasel_placement.png
        ├── 03_tendon_routing.png
        ├── 04_finger_joint_angles.png
        ├── 05_force_vs_stroke.png
        ├── 06_fingertip_force_vs_pulley.png
        ├── 07_pulley_heatmap.png
        ├── 08_placement_vs_force.png
        ├── 09_palm_vs_performance.png
        ├── 10_thumb_workspace.png
        ├── 11_wrist_torque.png
        ├── 12_volume_vs_performance.png
        ├── 13_robustness_distribution.png
        ├── 14_pareto_front.png
        └── 15_final_best_design.png
```

---

## Measured Performance Highlights
- **Power Grasp Force**: **294.22 N** (Target: ≥ 60.0 N) — **PASSED**
- **Precision Pinch Force**: **113.92 N** (Target: ≥ 15.0 N) — **PASSED**
- **10-Task Grasp Benchmark Success Rate**: **100% (10/10 tasks passed)**
- **Total Hand Assembly Mass**: **506.0 g** (Target: < 650 g) — **PASSED**
- **Static Wrist Reaction Torque**: **0.210 N*m** (Target: < 0.35 N*m) — **PASSED**
- **Monte Carlo Robustness (±10% variations)**: **99.3% pass rate**

---

## Reproducing the Pipeline
Run the master pipeline script in the Python environment:
```bash
python optimize.py
```
This regenerates all simulations, task benchmarks, database records, plots, and reports from a clean state.
