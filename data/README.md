# Data Directory

This directory contains two primary datasets:

1. **Phase 1 Deterministic Mechanical Digital Twin Data (Tracked in Git):**
   - `anthropometry.json`: Anatomical hand dimensions and joint angle limits from literature.
   - `hasel_reference.json`: Physical force, stroke, and geometric specifications for HASEL pouches.
   - `design_requirements.json`: Performance benchmark goals for prosthetic grasp forces.

2. **Machine Learning NinaPro DB8 Dataset (Not Tracked in Git, Local Binary Storage):**
   - Location: `data/ninapro_db8/`
   - Files required: `S1_E1_A1.mat`, `S1_E1_A2.mat`, `S1_E1_A3.mat` (~2.1 GB total).
   - Redistribution Policy & Checksums: See [ml/data/README.md](../ml/data/README.md) and [ml/data/manifests/ninapro_db8_manifest.json](../ml/data/manifests/ninapro_db8_manifest.json).
