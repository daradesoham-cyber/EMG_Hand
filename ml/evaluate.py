"""
Evaluation, Metrics, and EMA Post-Filtering Module.

Metrics Computed:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Pearson Correlation Coefficient (r)
- Per-finger metrics across [thumb, index, middle, ring, pinky]
- Jitter metric (mean absolute sample-to-sample derivative)
- Delay / Response latency under EMA filtering
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive regression metrics per output and aggregated.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    K = y_true.shape[1] if y_true.ndim > 1 else 1
    if target_names is None:
        target_names = [f"output_{i}" for i in range(K)]

    mae_per_channel = np.mean(np.abs(y_true - y_pred), axis=0)
    mse_per_channel = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse_per_channel = np.sqrt(mse_per_channel)

    # R2 per channel
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_channel = np.where(ss_tot > 1e-8, 1.0 - (ss_res / ss_tot), 0.0)

    # Pearson correlation per channel
    corr_per_channel = np.zeros(K, dtype=np.float32)
    for k in range(K):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            r = np.corrcoef(yt, yp)[0, 1]
            corr_per_channel[k] = float(r) if not np.isnan(r) else 0.0
        else:
            corr_per_channel[k] = 0.0

    per_finger = {}
    for i, name in enumerate(target_names):
        per_finger[name] = {
            "mae": float(mae_per_channel[i]),
            "rmse": float(rmse_per_channel[i]),
            "r2": float(r2_per_channel[i]),
            "correlation": float(corr_per_channel[i]),
        }

    return {
        "aggregate": {
            "mae": float(np.mean(mae_per_channel)),
            "rmse": float(np.mean(rmse_per_channel)),
            "r2": float(np.mean(r2_per_channel)),
            "correlation": float(np.mean(corr_per_channel)),
        },
        "per_output": per_finger,
    }


def apply_ema_filter(
    predictions: np.ndarray,
    alpha: float = 0.20,
) -> np.ndarray:
    """
    Causal Exponential Moving Average (EMA) filter.

    y_filtered[t] = alpha * y_pred[t] + (1 - alpha) * y_filtered[t-1]

    Args:
        predictions: (N, K) raw model predictions.
        alpha: Smoothing parameter in (0, 1]. Lower = smoother, higher latency.
    """
    N, K = predictions.shape
    filtered = np.empty_like(predictions)
    filtered[0] = predictions[0]

    for t in range(1, N):
        filtered[t] = alpha * predictions[t] + (1.0 - alpha) * filtered[t - 1]

    return filtered


def evaluate_ema_tradeoff(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    alpha: float = 0.20,
    target_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Quantify trade-offs of EMA filtering:
    - MAE impact
    - Jitter reduction (sample-to-sample difference)
    - Effective group delay / lag
    """
    y_ema = apply_ema_filter(y_raw, alpha=alpha)

    # Jitter: mean absolute first-difference
    jitter_raw = float(np.mean(np.abs(np.diff(y_raw, axis=0))))
    jitter_ema = float(np.mean(np.abs(np.diff(y_ema, axis=0))))
    jitter_reduction_pct = ((jitter_raw - jitter_ema) / max(jitter_raw, 1e-8)) * 100.0

    raw_metrics = compute_regression_metrics(y_true, y_raw, target_names=target_names)
    ema_metrics = compute_regression_metrics(y_true, y_ema, target_names=target_names)

    # Theoretical EMA time constant (tau = -dt / ln(1 - alpha))
    # At 31.25 Hz update rate, dt = 32 ms
    dt_ms = 32.0
    tau_ms = -dt_ms / np.log(1.0 - alpha) if alpha < 1.0 else 0.0

    return {
        "alpha": alpha,
        "raw_mae": raw_metrics["aggregate"]["mae"],
        "ema_mae": ema_metrics["aggregate"]["mae"],
        "mae_delta": ema_metrics["aggregate"]["mae"] - raw_metrics["aggregate"]["mae"],
        "jitter_raw": jitter_raw,
        "jitter_ema": jitter_ema,
        "jitter_reduction_pct": jitter_reduction_pct,
        "effective_time_constant_ms": float(tau_ms),
        "raw_metrics": raw_metrics,
        "ema_metrics": ema_metrics,
    }


def compute_temporal_lag_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt_ms: float = 32.0,
    max_lag_steps: int = 25,
    target_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Compute empirical cross-correlation temporal lag between ground truth and predictions.

    Convention:
      Positive lag k (> 0) means the model prediction lags behind ground truth by k steps (k * dt_ms).
      That is, y_pred[t + k] aligns with y_true[t].
      Negative lag k (< 0) means the model prediction leads ground truth.

    Args:
        y_true: (N, K) ground truth kinematic targets.
        y_pred: (N, K) model predicted kinematic states.
        dt_ms: inference interval (default 32.0 ms for stride=64 at fs=2000 Hz).
        max_lag_steps: maximum step displacement to search (+/- 25 steps = +/- 800 ms).
        target_names: channel names.

    Returns:
        Dictionary containing zero_lag_r, best_lag_r, best_lag_steps, best_lag_ms,
        both aggregated and per output channel.
    """
    N, K = y_true.shape
    if target_names is None:
        target_names = [f"output_{i}" for i in range(K)]

    per_channel = {}
    best_lags = []
    zero_lag_corrs = []
    best_lag_corrs = []

    for k_idx in range(K):
        yt = y_true[:, k_idx]
        yp = y_pred[:, k_idx]

        # Zero-lag correlation
        r0 = np.corrcoef(yt, yp)[0, 1] if (np.std(yt) > 1e-6 and np.std(yp) > 1e-6) else 0.0
        r0 = float(r0) if not np.isnan(r0) else 0.0

        best_r = r0
        best_k = 0

        # Search lags from -max_lag_steps to +max_lag_steps
        for shift in range(-max_lag_steps, max_lag_steps + 1):
            if shift == 0:
                continue
            elif shift > 0:
                # y_pred is shifted backward by shift (i.e. y_pred lags ground truth)
                seg_true = yt[:-shift]
                seg_pred = yp[shift:]
            else:
                # shift < 0: y_pred leads ground truth
                pos_shift = -shift
                seg_true = yt[pos_shift:]
                seg_pred = yp[:-pos_shift]

            if len(seg_true) > 10 and np.std(seg_true) > 1e-6 and np.std(seg_pred) > 1e-6:
                r_val = float(np.corrcoef(seg_true, seg_pred)[0, 1])
                if not np.isnan(r_val) and r_val > best_r:
                    best_r = r_val
                    best_k = shift

        per_channel[target_names[k_idx]] = {
            "zero_lag_r": r0,
            "best_lag_r": float(best_r),
            "best_lag_steps": int(best_k),
            "best_lag_ms": float(best_k * dt_ms),
        }
        best_lags.append(best_k * dt_ms)
        zero_lag_corrs.append(r0)
        best_lag_corrs.append(best_r)

    return {
        "aggregate": {
            "zero_lag_r": float(np.mean(zero_lag_corrs)),
            "best_lag_r": float(np.mean(best_lag_corrs)),
            "mean_lag_ms": float(np.mean(best_lags)),
            "median_lag_ms": float(np.median(best_lags)),
        },
        "per_output": per_channel,
    }
