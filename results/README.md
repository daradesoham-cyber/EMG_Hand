# V1 Model Evaluation and Experimental Results

This directory contains the verified evaluation metrics, training history, and confusion matrix for the **V1 Compact 1D CNN** gesture classification model.

---

## 1. Verified Performance Summary

Evaluated across **15 unseen test subjects** (inter-subject generalization benchmark):

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Model Architecture** | Compact 1D CNN | 3 Conv1D layers + BatchNorm + MaxPool + GAP + Dense |
| **Input Shape** | $(400, 4)$ | 400 ms @ 1000 Hz, 4 sEMG channels `[12, 11, 7, 2]` |
| **Number of Classes** | 3 active classes | `PINCH`, `FIST`, `OPEN` (`REST` via inactivity threshold) |
| **Test Accuracy** | **51.05%** | Cross-subject zero-shot generalization across 15 unseen subjects |
| **Balanced Accuracy** | **51.05%** | Uniform performance across the 3 target classes |
| **Macro F1-Score** | **0.5067** | Macro-averaged F1 score on test split |
| **Random Seed** | 42 | Deterministic subject split |

> [!NOTE]
> In surface EMG gesture recognition, cross-subject (subject-independent) classification is notoriously challenging due to inter-individual variations in muscle anatomy, subcutaneous fat layer thickness, electrode contact impedance, and contraction intensity. The 51.05% zero-shot accuracy across 15 previously unseen subjects represents a genuine baseline for a compact 4-channel model prior to user-specific fine-tuning.

---

## 2. Artifacts Included

- `confusion_matrix.png`: Confusion matrix plot illustrating classification distribution across `PINCH`, `FIST`, and `OPEN`.
- `training_history.csv`: Epoch-by-epoch training and validation loss and accuracy curves.
- `final_results.json`: Machine-readable summary of evaluation metrics.
- `deployment_config.json`: Deployment parameters (sampling rate, window size, bandpass frequencies, channels).
- `subject_split.json`: Exact subject IDs assigned to training (63), validation (13), and test (15) splits.
