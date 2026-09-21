"""
EMG Evaluation Module.
Computes classification metrics, confusion matrices, and per-class reports.
"""

import numpy as np


def compute_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Computes basic classification accuracy."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def compute_confusion_matrix(y_true: list[str], y_pred: list[str], classes: list[str]) -> np.ndarray:
    """Computes confusion matrix across specified class labels."""
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            matrix[class_to_idx[t], class_to_idx[p]] += 1
    return matrix


def generate_classification_report(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    """Generates per-class precision, recall, and F1 scores."""
    report = {}
    cm = compute_confusion_matrix(y_true, y_pred, classes)
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report[c] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': int(np.sum(cm[i, :]))}
    return report
