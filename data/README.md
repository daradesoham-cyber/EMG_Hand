# EMG Dataset Documentation

## 1. Dataset Overview

This project utilizes the benchmark **Dataset of upper limb surface electromyogram (sEMG) signals from 91 participants for hand gesture recognition** (published in *Scientific Data*, Nature Publishing Group).

- **Subjects**: 91 demographically diverse healthy individuals (varying age, BMI, and skin properties).
- **Total Trials**: 32,760 hand gesture trials.
- **Protocol**: Each subject performed 10 distinct upper-limb gestures, repeated 36 times each, in a natural, unconstrained posture.
- **Acquisition Sampling Rate**: 1000 Hz.
- **Original Channels**: 12 surface EMG channels placed across the forearm musculature.

---

## 2. V1 Prototype Gesture Selection

For the V1 prototype, three active functional gestures were selected from the 10-gesture protocol to control the robotic/prosthetic hand:

| Dataset Class Index | Gesture Name | BiDexHand Function |
| :---: | :---: | :---: |
| **1** | **PINCH** | Precision grip: Thumb and index finger flex to contact. |
| **5** | **FIST** | Power grasp: All four fingers and thumb fully flex. |
| **9** | **OPEN** | Full hand extension: All fingers and thumb abduct and extend. |

### REST State Handling
> [!IMPORTANT]
> **REST is NOT present as an active labeled gesture in the source dataset.**  
> In this V1 deployment:
> 1. An **RMS Inactivity Threshold** ($\text{RMS} < 0.01$) detects when the forearm muscles are quiet, classifying the state as `REST`.
> 2. A **Confidence Threshold** ($\text{Confidence} < 0.40$) detects classification uncertainty and safely defaults to `REST` to prevent involuntary prosthesis movement.
> 3. For physical sensor integration, the RMS threshold must be calibrated to the specific hardware electrode noise floor.

---

## 3. Selected EMG Channels

From the 12 available sEMG channels, a 4-channel subset was selected for the compact V1 embedded pipeline:
- **Selected Channels (1-based)**: `[12, 11, 7, 2]`
- **Selected Channels (0-based)**: `[11, 10, 6, 1]`
- **Rationale**: Captures differential activity across both flexor (flexor digitorum superficialis/profundus) and extensor (extensor digitorum communis) muscle compartments while minimizing embedded processing requirements.

---

## 4. Subject Split (Cross-Subject Validation)

To rigorously evaluate generalization across unseen individuals, the 91 subjects were partitioned into disjoint sets (Random Seed 42):
- **Training Set (63 subjects, 69.2%)**: Subjects `[1, 2, 3, 4, 6, 7, 9, 12, 13, 21, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 36, 39, 40, 42, 46, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 61, 64, 65, 66, 68, 69, 70, 71, 72, 76, 77, 78, 79, 80, 81, 83, 84, 87, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **Validation Set (13 subjects, 14.3%)**: Subjects `[8, 15, 22, 37, 43, 53, 58, 63, 67, 82, 85, 88, 90]`
- **Test Set (15 subjects, 16.5%)**: Subjects `[10, 16, 19, 20, 25, 28, 44, 45, 50, 56, 62, 73, 74, 75, 86]`

Full split definitions are preserved in `results/subject_split.json`.

---

## 5. Offline Replay Sample

For immediate demonstration and testing without downloading the multi-gigabyte raw dataset, a small evaluation set of offline trials is included in the repository:
- **Location**: `src/emg_classifier/models/sample_emg_trials.npz` (1.6 MB)
- **Contents**: Pre-windowed trials across PINCH, FIST, and OPEN gestures from test subjects, used directly by `emg_replay_publisher` during replay mode.

---

## 6. How to Download the Full Raw Dataset

If you wish to retrain or extend the machine-learning pipeline:
1. Download the raw sEMG dataset from the official repository:
   - [Nature Scientific Data Publication](https://doi.org/10.1038/s41597-021-00843-9)
   - [Public Data Repository / Figshare](https://figshare.com)
2. Place downloaded files under:
   ```
   EMG_Hand/
   └── data/
       └── raw/
           ├── Subject01/
           ├── Subject02/
           └── ...
   ```
3. Run preprocessing scripts to extract the 4-channel, 400 ms windowed tensors.
