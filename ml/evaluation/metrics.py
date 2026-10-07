"""
Evaluation Metrics and Post-Filtering Analysis Module.

Computes:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Pearson Correlation Coefficient (r)
- Step-to-Step Jitter (mean derivative |dy/dt|)
- Causal Exponential Moving Average (EMA) post-filtering
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive regression metrics per channel and aggregated.
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


def compute_jitter(predictions: np.ndarray) -> float:
    """Compute step-to-step sample derivative: mean(|y[t] - y[t-1]|)."""
    if len(predictions) < 2:
        return 0.0
    diff = np.abs(np.diff(predictions, axis=0))
    return float(np.mean(diff))


def apply_ema_filter(
    predictions: np.ndarray,
    alpha: float = 0.20,
) -> np.ndarray:
    """
    Causal Exponential Moving Average (EMA) filter.
    y_filtered[t] = alpha * y_pred[t] + (1 - alpha) * y_filtered[t-1]
    """
    N, K = predictions.shape
    filtered = np.empty_like(predictions)
    filtered[0] = predictions[0]

    for t in range(1, N):
        filtered[t] = alpha * predictions[t] + (1.0 - alpha) * filtered[t - 1]

    return filtered


def evaluate_ema_tradeoff(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.20,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate accuracy vs smoothing trade-off for a specific EMA alpha.
    """
    filtered = apply_ema_filter(y_pred, alpha=alpha)

    raw_metrics = compute_regression_metrics(y_true, y_pred, target_names)
    ema_metrics = compute_regression_metrics(y_true, filtered, target_names)

    jitter_raw = compute_jitter(y_pred)
    jitter_ema = compute_jitter(filtered)
    jitter_reduction = (
        (jitter_raw - jitter_ema) / jitter_raw * 100.0 if jitter_raw > 0 else 0.0
    )

    # Theoretical time constant in ms (dt = 32 ms)
    tau_ms = -32.0 / np.log(1.0 - alpha) if alpha < 1.0 else 0.0

    return {
        "alpha": alpha,
        "tau_ms": float(tau_ms),
        "raw_mae": raw_metrics["aggregate"]["mae"],
        "ema_mae": ema_metrics["aggregate"]["mae"],
        "mae_delta": ema_metrics["aggregate"]["mae"] - raw_metrics["aggregate"]["mae"],
        "jitter_raw": jitter_raw,
        "jitter_ema": jitter_ema,
        "jitter_reduction_pct": jitter_reduction,
        "ema_metrics": ema_metrics,
        "raw_metrics": raw_metrics,
    }
