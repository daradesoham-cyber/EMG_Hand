"""
Bootstrap Confidence Interval Evaluation Module.

Computes 95% non-parametric bootstrap confidence intervals for:
- MAE
- RMSE
- R²
- Pearson r
"""

from typing import Dict, Any, List, Optional
import numpy as np
from ml.evaluation.metrics import compute_regression_metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute non-parametric bootstrap confidence intervals across test windows.
    """
    rng = np.random.RandomState(seed)
    N = len(y_true)
    alpha_low = (1.0 - ci_level) / 2.0 * 100.0
    alpha_high = (1.0 + ci_level) / 2.0 * 100.0

    boot_mae = []
    boot_rmse = []
    boot_r2 = []
    boot_corr = []

    for _ in range(n_bootstraps):
        indices = rng.choice(N, size=N, replace=True)
        yt_sample = y_true[indices]
        yp_sample = y_pred[indices]

        m = compute_regression_metrics(yt_sample, yp_sample, target_names)
        boot_mae.append(m["aggregate"]["mae"])
        boot_rmse.append(m["aggregate"]["rmse"])
        boot_r2.append(m["aggregate"]["r2"])
        boot_corr.append(m["aggregate"]["correlation"])

    return {
        "n_bootstraps": n_bootstraps,
        "ci_level": ci_level,
        "mae": {
            "mean": float(np.mean(boot_mae)),
            "ci_low": float(np.percentile(boot_mae, alpha_low)),
            "ci_high": float(np.percentile(boot_mae, alpha_high)),
        },
        "rmse": {
            "mean": float(np.mean(boot_rmse)),
            "ci_low": float(np.percentile(boot_rmse, alpha_low)),
            "ci_high": float(np.percentile(boot_rmse, alpha_high)),
        },
        "r2": {
            "mean": float(np.mean(boot_r2)),
            "ci_low": float(np.percentile(boot_r2, alpha_low)),
            "ci_high": float(np.percentile(boot_r2, alpha_high)),
        },
        "correlation": {
            "mean": float(np.mean(boot_corr)),
            "ci_low": float(np.percentile(boot_corr, alpha_low)),
            "ci_high": float(np.percentile(boot_corr, alpha_high)),
        },
    }
