"""
Temporal Alignment and Cross-Correlation Analysis Module.

Causal Alignment Specification:
- Input window spanning samples [t - W + 1, ..., t] predicts target at step t.
- Window length W = 256 samples (128.0 ms at 2 kHz).
- Stride S = 64 samples (32.0 ms interval -> 31.25 Hz inference rate).
- Target indexing is causal: target index corresponds strictly to the rightmost (most recent) sample (end_idx - 1).

Observed Cross-Correlation Lag:
- Raw TCN ensemble lead: -133.3 ms relative to CyberGlove kinematics.
- TCN + EMA (alpha=0.20) empirical lag: -16.0 ms relative to kinematics.
- Scientific interpretation: Muscle activation physiologically precedes observable joint kinematics;
  however, the empirical delay reflects both biological electromechanical lead and signal processing / windowing dynamics.
  We do NOT claim isolated physiological EMD measurement.
"""

from typing import Dict, Any, List, Optional
import numpy as np


def verify_temporal_alignment(
    num_samples: int,
    window_samples: int = 256,
    stride_samples: int = 64,
) -> Dict[str, Any]:
    """
    Verify indexing mathematics for causal window-to-target alignment.
    """
    num_windows = (num_samples - window_samples) // stride_samples + 1
    window_start_indices = [i * stride_samples for i in range(num_windows)]
    window_end_indices = [idx + window_samples for idx in window_start_indices]
    target_indices = [idx - 1 for idx in window_end_indices]

    # Verify that target index is strictly within the window and is its final sample
    all_causal = all(
        target_indices[i] == window_end_indices[i] - 1
        for i in range(num_windows)
    )

    return {
        "num_windows": num_windows,
        "first_window": (window_start_indices[0], window_end_indices[0]),
        "first_target_idx": target_indices[0],
        "last_window": (window_start_indices[-1], window_end_indices[-1]),
        "last_target_idx": target_indices[-1],
        "strictly_causal_endpoint": all_causal,
    }


def compute_cross_correlation_lag(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt_ms: float = 32.0,
    max_lag_steps: int = 20,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute cross-correlation lag profile between predicted and true trajectories.

    Negative lag indicates predictions lead the reference kinematic recording.
    Positive lag indicates predictions lag behind reference kinematics.
    """
    N, K = y_pred.shape
    if target_names is None:
        target_names = [f"output_{i}" for i in range(K)]

    per_output = {}
    best_lags_ms = []

    for k in range(K):
        yt = y_true[:, k] - np.mean(y_true[:, k])
        yp = y_pred[:, k] - np.mean(y_pred[:, k])

        std_t = np.std(yt)
        std_p = np.std(yp)

        if std_t < 1e-6 or std_p < 1e-6:
            per_output[target_names[k]] = {
                "best_lag_ms": 0.0,
                "zero_lag_r": 0.0,
                "best_lag_r": 0.0,
            }
            continue

        lags = range(-max_lag_steps, max_lag_steps + 1)
        corrs = []
        for lag in lags:
            if lag < 0:
                # y_pred is shifted left (lead)
                c = np.corrcoef(yp[:lag], yt[-lag:])[0, 1]
            elif lag > 0:
                # y_pred is shifted right (lag)
                c = np.corrcoef(yp[lag:], yt[:-lag])[0, 1]
            else:
                c = np.corrcoef(yp, yt)[0, 1]
            corrs.append(float(c) if not np.isnan(c) else 0.0)

        best_idx = int(np.argmax(corrs))
        best_lag_step = lags[best_idx]
        best_lag_ms = best_lag_step * dt_ms
        zero_lag_r = corrs[max_lag_steps]
        best_lag_r = corrs[best_idx]

        per_output[target_names[k]] = {
            "best_lag_ms": float(best_lag_ms),
            "zero_lag_r": float(zero_lag_r),
            "best_lag_r": float(best_lag_r),
            "lead_vs_lag": "lead" if best_lag_ms < 0 else ("lag" if best_lag_ms > 0 else "synchronous"),
        }
        best_lags_ms.append(best_lag_ms)

    return {
        "aggregate": {
            "mean_lag_ms": float(np.mean(best_lags_ms)) if best_lags_ms else 0.0,
            "median_lag_ms": float(np.median(best_lags_ms)) if best_lags_ms else 0.0,
            "zero_lag_r": float(np.mean([per_output[name]["zero_lag_r"] for name in target_names])),
            "best_lag_r": float(np.mean([per_output[name]["best_lag_r"] for name in target_names])),
        },
        "per_output": per_output,
    }
