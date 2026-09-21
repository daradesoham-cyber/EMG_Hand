#!/usr/bin/env python3
"""
Standalone V1 Model Verification Script.
Tests the pure NumPy 1D CNN pipeline against sample offline trials.
"""

import os
import sys
import numpy as np

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'emg_classifier'))

from emg_classifier.v1_model import V1EMGModel


def main():
    print("=" * 60)
    print("EMG Hand V1 - Standalone Inference & Pipeline Verification")
    print("=" * 60)

    model_dir = os.path.join(REPO_ROOT, 'src', 'emg_classifier', 'models')
    weights_path = os.path.join(model_dir, 'emg_hand_cnn_best_weights.npz')
    mean_path = os.path.join(model_dir, 'train_mean.npy')
    std_path = os.path.join(model_dir, 'train_std.npy')
    trials_path = os.path.join(model_dir, 'sample_emg_trials.npz')

    print(f"[1/4] Loading model weights from: {weights_path}")
    model = V1EMGModel(
        model_path=weights_path,
        mean_path=mean_path,
        std_path=std_path,
    )
    print("      Model successfully initialized.")

    print(f"[2/4] Loading sample offline trials from: {trials_path}")
    data = np.load(trials_path)
    trial_keys = [k for k in data.keys()]
    print(f"      Loaded trials for gestures: {trial_keys}")

    print("[3/4] Running inference pass across trials...")
    correct = 0
    total = len(trial_keys)

    for i, gesture_name in enumerate(trial_keys):
        raw_trial = data[gesture_name]  # (400, 4)
        normed, rms = model.preprocess(raw_trial)
        pred_gesture, probs, conf, reason = model.classify(
            normed,
            raw_rms=rms,
            rms_threshold=0.01,
            confidence_threshold=0.40,
        )
        is_match = (pred_gesture == gesture_name)
        if is_match:
            correct += 1

        print(f"  Trial {i+1:02d}: Expected={gesture_name:<6} Pred={pred_gesture:<6} "
              f"Conf={conf:.3f} RMS={rms:.5f} ({reason}) {'[OK]' if is_match else '[MISMATCH]'}")

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print("[4/4] Summary:")
    print(f"      Total Evaluated: {total}")
    print(f"      Correct Matches: {correct}")
    print(f"      Sample Accuracy: {acc:.1f}%")
    print("=" * 60)
    print("Standalone verification complete.")


if __name__ == '__main__':
    main()
