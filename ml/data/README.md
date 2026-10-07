# NinaPro DB8 Dataset Handling & Licensing

## Dataset Identification

- **Dataset Name:** NinaPro Database 8 (DB8)
- **Primary Focus:** Non-invasive forearm surface electromyography (sEMG) synchronized with 18-channel CyberGlove II kinematic joint angles.
- **Reference Publication:** Krasoulis et al., *"A database of multi-channel surface electromyographic and hand kinematic signals of amputees and able-bodied individuals"*, Scientific Data, 2017.
- **Host Institution:** NinaPro Consortium / University of Applied Sciences Western Switzerland (HES-SO Valais-Wallis).
- **Official URL:** [http://ninapro.hevs.ch/DB8](http://ninapro.hevs.ch/DB8)

---

## Redistribution Policy & Legal Notice

> [!IMPORTANT]
> **DO NOT COMMIT RAW DATA FILES TO GIT**
> NinaPro database terms of use require researchers to register on the official portal, accept the data agreement, and cite the publication. Redistribution of raw binary `.mat` archives (>2.1 GB) within public or non-consortium Git repositories is restricted.
>
> The project repository includes exact cryptographic SHA-256 hashes, file manifests, and streaming pipelines. Anyone can reproduce the exact pipeline by downloading the files from the official portal and verifying hashes.

---

## Expected Directory Structure

Place downloaded `.mat` archives into either `data/ninapro_db8/` or `ml/data/ninapro_db8/`:

```text
prosthetic_hand_phase1/
└── data/
    └── ninapro_db8/
        ├── S1_E1_A1.mat   (~955 MB)  [Training Split]
        ├── S1_E1_A2.mat   (~969 MB)  [Validation Split]
        └── S1_E1_A3.mat   (~222 MB)  [Strictly Held-Out Test Split]
```

---

## Cryptographic Checksums (Subject 1)

| File | Size (bytes) | SHA-256 Checksum | Split Role |
| :--- | :---: | :--- | :--- |
| `S1_E1_A1.mat` | 1,001,402,376 | `4f2355061b594baffb474350ce6ede5f852911a8871d547b1adb7cadcfa86d5f` | Training (A1) |
| `S1_E1_A2.mat` | 1,016,065,584 | `a431c57ecc6bc8f4689b66e96c2197a696201dfb475ed04794593b3820d6293e` | Validation (A2) |
| `S1_E1_A3.mat` | 232,810,568 | `772d18763ba3a51e0ae50183b16238c90a92bbe4c6dff7d4745b259e33f515fe` | Held-Out Test (A3) |

Verify checksums locally:
```bash
sha256sum data/ninapro_db8/*.mat
```

---

## Session Splitting & Strict Isolation Protocol

1. **Acquisition 1 (`A1`): Training Partition**
   - Duration: $1,146.3\text{ s}$ ($2,292,526$ samples, $35,817$ sliding windows).
   - All normalizer parameters ($\boldsymbol{\mu}, \boldsymbol{\sigma}$ for sEMG; 1st-99th percentiles for glove angles) are fitted strictly on `A1`.
2. **Acquisition 2 (`A2`): Validation Partition**
   - Duration: $1,161.9\text{ s}$ ($2,313,470$ samples, $36,144$ sliding windows).
   - Evaluated for early stopping, learning rate scheduling, and EMA grid search ($\alpha = 0.20$).
3. **Acquisition 3 (`A3`): Strictly Held-Out Test Partition**
   - Duration: $264.5\text{ s}$ ($529,099$ samples, $8,264$ sliding windows).
   - **Never seen during model development.** Evaluated strictly once after model parameters and post-processing hyperparameters were frozen.
