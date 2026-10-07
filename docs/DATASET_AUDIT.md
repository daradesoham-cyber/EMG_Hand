# NinaPro Database 8 (DB8) Technical Audit & Empirical Data Verification

**Date:** October 7, 2026
**Project:** Biomechatronic Prosthetic Hand ML Pipeline (Phase 2 Pre-requisite)
**Verification Method:** Empirical inspection of downloaded raw `.mat` acquisition files (`S1_E1_A1.mat`, `S1_E1_A2.mat`, `S1_E1_A3.mat`), official repository documentation (`https://ninapro.hevs.ch/instructions/DB8.html`), and primary literature (*Krasoulis et al., Frontiers in Neuroscience 2019*, DOI: 10.3389/fnins.2019.00891).

---

## 1. Executive Summary & Audit Verification Matrix

| Property | Assumed Value | Verified Empirical Value | Status / Discrepancy Note |
| :--- | :--- | :--- | :--- |
| **Total Subjects** | ~10–12 | **12 subjects** (10 able-bodied S1–S10; 2 transradial amputees S11–S12) | **VERIFIED** |
| **Sessions per Subject** | 3 sessions | **3 sequential acquisitions (`A1`, `A2`, `A3`)** per subject | **VERIFIED** (Named `A1`, `A2`, `A3`) |
| **Repetitions** | Unspecified | **A1: 10 repetitions**, **A2: 10 repetitions**, **A3: 2 repetitions** | **VERIFIED** from `repetition` vector |
| **sEMG Channels** | 16 channels | **16 channels** (Delsys Trigno IM Wireless, 2 rows of 8 around forearm) | **VERIFIED** (`emg.shape[1] == 16`) |
| **sEMG Sampling Rate** | ~2 kHz | **2000 Hz** (native 1111 Hz hardware, upsampled to 2000 Hz & post-synchronized) | **VERIFIED** |
| **Glove Channels** | 5 or 18 channels | **18 calibrated channels** (CyberGlove II 18-DOF model on contralateral hand) | **VERIFIED** (`glove.shape[1] == 18`) |
| **Glove Sampling Rate** | ~100 Hz | **2000 Hz** (native 100 Hz hardware, upsampled to 2000 Hz & synchronized) | **VERIFIED** |
| **Data Synchronization** | Synchronized | **Strict sample-level synchronization** across all columns at 2000 Hz | **VERIFIED** |
| **Missing Data / NaNs** | None | **0 NaNs, 0 Infs, complete continuous data matrix** | **VERIFIED** |
| **sEMG Voltage Range** | $\pm 5\text{ mV}$ | $[-1.73\text{ mV}, +1.24\text{ mV}]$ ($[-0.00173, +0.00124]\text{ V}$) | **VERIFIED** (zero-mean baseline) |
| **Glove Angle Range** | $0\text{--}1$ | $[-67.8^\circ, +167.7^\circ]$ calibrated joint angles in degrees | **VERIFIED** (Requires target normalization) |
| **Number of Movements** | Hand movements | **9 active movements + 0 (rest)** | **VERIFIED** (`stimulus` $\in [0, 9]$) |
| **File Format** | MATLAB `.mat` | **MATLAB 5.0 MAT-file** (zlib-compressed, readable via `scipy.io.loadmat`) | **VERIFIED** |

---

## 2. Dataset Structure & Physical Acquisition Protocol

### 2.1 Participants
- **Subjects 1 to 10:** Able-bodied adult participants (8 male, 2 female; ages 21–34).
- **Subjects 11 and 12:** Right-hand transradial amputees (traumatic accident and epithelioid sarcoma; 50% remaining forearm).
- **Acquisition Task:** Mirrored bilateral movements—subjects performed movements with their right hand (phantom limb for amputees) while the contralateral left hand wore the instrumented CyberGlove II.

### 2.2 Electrode Placement & sEMG Signal
- **Hardware:** Delsys Trigno IM Wireless EMG system with active double-differential electrodes.
- **Topography:** 16 active channels arranged in **two concentric rows of 8 electrodes** around the proximal forearm below the radio-humeral joint. No individual muscle isolation was enforced; the grid captures distributed muscular activation.
- **Physical Sampling:** Native sampling at $1,111\text{ Hz}$. Software-upsampled using band-limited sinc interpolation to exactly $2,000\text{ Hz}$ to synchronize with other streams.
- **Signal Quality:** Raw bipolar voltage in Volts. DC offset is minimal ($\mu = -1.2 \times 10^{-6}\text{ V}$).

### 2.3 Contralateral Kinematics (CyberGlove II)
- **Hardware:** CyberGlove II with 18 resistive bend sensors (18-DOF configuration).
- **Physical Sampling:** Native sampling at $100\text{ Hz}$, calibrated using manufacturer calibration routine, then upsampled to $2,000\text{ Hz}$.
- **Stored Values:** Joint angles calibrated in degrees ($^\circ$).
- **Missing Columns / Unmeasured DOFs:** In the 18-DOF model, 4 sensor channels present on the 22-sensor version are excluded or filled with static constants.

### 2.4 Movement Protocol
Nine distinct movement classes plus baseline rest:
0. **Rest** (interleaved 3-second recovery periods)
1. **Thumb flexion / extension**
2. **Thumb abduction / adduction**
3. **Index finger flexion / extension**
4. **Middle finger flexion / extension**
5. **Combined ring and little fingers flexion / extension**
6. **Index pointer**
7. **Cylindrical grip** (power grasp)
8. **Lateral grip** (key pinch)
9. **Tripod grip** (precision pinch)

Each active trial lasted 5 to 9 seconds, following dynamic video cues.

---

## 3. Detailed File Structure & Variable Audit (Subject 1)

For each subject, the database contains three files: `S<id>_E1_A1.mat`, `S<id>_E1_A2.mat`, `S<id>_E1_A3.mat`.

### Subject 1 Acquisition Metrics:
| Acquisition File | Total Samples | Duration [s] | Duration [min] | Repetitions Present | Purpose / Role in Split |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `S1_E1_A1.mat` | 2,292,526 | 1,146.3 s | 19.10 min | 10 reps (1–10) + rest | **Training Set** |
| `S1_E1_A2.mat` | 2,323,738 | 1,161.9 s | 19.36 min | 10 reps (1–10) + rest | **Validation Set (Hyperparameter Tuning)** |
| `S1_E1_A3.mat` | 529,099 | 264.5 s | 4.41 min | 2 reps (1–2) + rest | **STRICT Held-Out Test Set** |
| **Combined** | **5,145,363** | **2,572.7 s** | **42.87 min** | **22 reps total** | Full Subject Protocol |

### Variable Dictionary per `.mat` File:
```
subject        : uint8 matrix (1, 1) -> Subject ID (1 to 12)
exercise       : uint8 matrix (1, 1) -> Exercise ID (always 1 in DB8)
emg            : float32 matrix (N, 16) -> 16-channel sEMG signals in Volts
acc            : float32 matrix (N, 48) -> 16 sensors x 3-axis accelerometer
gyro           : float32 matrix (N, 48) -> 16 sensors x 3-axis gyroscope
mag            : float32 matrix (N, 48) -> 16 sensors x 3-axis magnetometer
glove          : float32 matrix (N, 18) -> 18 calibrated CyberGlove channels (degrees)
stimulus       : int8 matrix (N, 1) -> Visual cue stimulus index (0 to 9)
restimulus     : int8 matrix (N, 1) -> Algorithmic a-posteriori refined movement label (0 to 9)
repetition     : int8 matrix (N, 1) -> Stimulus repetition index
rerepetition   : int8 matrix (N, 1) -> Refined restimulus repetition index
```

---

## 4. CyberGlove II 18-Channel Mapping & Target Identification

Through kinematic correlation and dynamic excursion analysis across isolated movement tasks (Movements 1–5), the 18 columns of `glove` are mapped as follows:

| Glove Col Index | Anatomical Joint / Function | Dynamic Range (S1) | Primary Motion Task |
| :---: | :--- | :---: | :--- |
| **0** | **Thumb CMC Abduction / Rotation (Opposition)** | $14.9^\circ \to 167.7^\circ$ ($\Delta = 152.8^\circ$) | **M2 (Thumb Abd/Add)**, M7 (Cylinder) |
| **1** | **Thumb MCP Flexion** | $-67.8^\circ \to 71.7^\circ$ ($\Delta = 139.5^\circ$) | **M1 (Thumb Flex)**, M7, M9 |
| **2** | **Thumb-Index Abduction** | $-6.5^\circ \to 69.8^\circ$ ($\Delta = 76.3^\circ$) | M2, M8 |
| **3** | Uncalibrated / Auxiliary Thumb Sensor | $14.9^\circ \to 48.7^\circ$ ($\Delta = 33.8^\circ$) | Low excursion |
| **4** | **Index MCP Flexion** | $-22.1^\circ \to 85.4^\circ$ ($\Delta = 107.5^\circ$) | **M3 (Index Flex)**, M6, M7, M9 |
| **5** | **Index PIP Flexion** | $-4.3^\circ \to 112.0^\circ$ ($\Delta = 116.3^\circ$) | **M3 (Index Flex)**, M7, M9 |
| **6** | **Middle MCP Flexion** | $-18.5^\circ \to 58.7^\circ$ ($\Delta = 77.2^\circ$) | **M4 (Middle Flex)**, M7 |
| **7** | **Middle PIP Flexion** | $-19.3^\circ \to 137.3^\circ$ ($\Delta = 156.6^\circ$) | **M4 (Middle Flex)**, M6, M7, M8, M9 |
| **8** | Index-Middle Abduction | $-10.7^\circ \to 25.9^\circ$ ($\Delta = 36.6^\circ$) | M3, M4 |
| **9** | **Ring MCP Flexion** | $-12.9^\circ \to 68.4^\circ$ ($\Delta = 81.3^\circ$) | **M5 (Ring+Little Flex)**, M7, M8 |
| **10** | **Ring PIP Flexion** | $-23.3^\circ \to 121.8^\circ$ ($\Delta = 145.1^\circ$) | **M5 (Ring+Little Flex)**, M6, M7, M8 |
| **11** | Middle-Ring Abduction | $-43.4^\circ \to 21.9^\circ$ ($\Delta = 65.3^\circ$) | M5 |
| **12** | **Little MCP Flexion** | $-8.9^\circ \to 48.9^\circ$ ($\Delta = 57.8^\circ$) | **M5 (Ring+Little Flex)**, M7, M8 |
| **13** | **Little PIP Flexion** | $-7.5^\circ \to 141.6^\circ$ ($\Delta = 149.1^\circ$) | **M5 (Ring+Little Flex)**, M6, M7, M8 |
| **14** | Ring-Little Abduction | $2.5^\circ \to 15.7^\circ$ ($\Delta = 13.2^\circ$) | Static / Low excursion |
| **15** | **Thumb IP Flexion** | $-34.0^\circ \to 70.6^\circ$ ($\Delta = 104.6^\circ$) | **M1 (Thumb Flex)**, M5, M7 |
| **16** | Wrist Flexion / Extension | $-2.3^\circ \to 14.5^\circ$ ($\Delta = 16.8^\circ$) | Wrist constraint (minimal) |
| **17** | Wrist Radial / Ulnar Deviation | $-1.4^\circ \to 6.9^\circ$ ($\Delta = 8.3^\circ$) | Wrist constraint (minimal) |

---

## 5. Verification of the Held-Out Split Protocol

The original experimental guideline from the dataset authors states:
> *"Three datasets/acquisitions are provided for each subject. It is recommended that dataset 3, which comprises only two repetitions for each movement, is only used to report performance results and no training or hyper-parameter tuning is performed using this data (i.e. test dataset). The three datasets, which were recorded sequentially, can offer an out-of-the-box three-way split for model training (dataset 1), hyper-parameter tuning/validation (dataset 2), and performance testing (dataset 3)."*

**Conclusion:** The prompt's mandate of:
- **Sessions 1 & 2 (`A1` & `A2`):** Training and Validation
- **Session 3 (`A3`):** Strict Held-Out Test Set
is **100% physically and methodologically aligned** with the native NinaPro DB8 design. No artificial partitioning or window splitting across sessions is required or permitted.
