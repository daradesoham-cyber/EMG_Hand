"""
Plotting and Visualization Module for NinaPro DB8 ML Pipeline.

Generates the 8 required publication-grade figures on the held-out Session 3 test set:
1. Training and validation loss curves (training history)
2. Predicted vs ground truth scatter/correlation plots
3. Continuous time-series trajectory overlay across all 6 actuators
4. Per-digit / per-actuator error breakdown (MAE and RMSE bar chart)
5. Residual distribution histograms (y_true - y_pred)
6. Correlation plot with linear fit and Pearson r
7. Prediction error over time
8. Jitter and Raw vs EMA filter smoothing comparison
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Headless rendering


def plot_loss_curves(
    history: List[Dict[str, float]],
    save_path: str = "results/plots/1_training_validation_loss.png",
) -> None:
    """Plot training and validation loss curves over epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epochs, train_loss, label="Training Loss (A1)", color="#1f77b4", lw=2)
    ax.plot(epochs, val_loss, label="Validation Loss (A2)", color="#ff7f0e", lw=2, linestyle="--")

    best_idx = int(np.argmin(val_loss))
    ax.scatter(
        [epochs[best_idx]],
        [val_loss[best_idx]],
        color="red",
        s=80,
        zorder=5,
        label=f"Best Val Epoch ({epochs[best_idx]}: {val_loss[best_idx]:.4f})",
    )

    ax.set_title("Causal TCN Training & Validation Loss (MAE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (Normalized [0, 1])", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_predicted_vs_ground_truth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    save_path: str = "results/plots/2_predicted_vs_ground_truth.png",
) -> None:
    """Plot Predicted vs Ground Truth scatter plot for each actuator."""
    num_channels = y_true.shape[1]
    cols = 3
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i in range(num_channels):
        ax = axes[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]
        name = target_names[i]

        # Subsample points for scatter clarity
        idx = np.random.choice(len(yt), size=min(3000, len(yt)), replace=False)
        ax.scatter(yt[idx], yp[idx], alpha=0.25, s=12, color="#1f77b4", edgecolors="none")
        ax.plot([0, 1], [0, 1], color="red", linestyle="--", lw=1.8, label="Ideal (y=x)")

        r = np.corrcoef(yt, yp)[0, 1] if np.std(yp) > 1e-6 else 0.0
        mae = np.mean(np.abs(yt - yp))

        ax.set_title(f"{name.capitalize()}\nr={r:.3f}, MAE={mae:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Ground Truth", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)

    # Hide extra subplots
    for j in range(num_channels, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Predicted vs Ground Truth across 6 Actuator Channels (Session 3)", fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_time_series_overlay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    time_window_samples: int = 500,
    start_sample: int = 800,
    save_path: str = "results/plots/3_time_series_overlay.png",
) -> None:
    """Plot continuous multi-digit traces: Ground Truth vs Prediction."""
    end_sample = min(start_sample + time_window_samples, len(y_true))
    t = np.arange(start_sample, end_sample) * 0.032  # 32 ms per prediction
    t = t - t[0]

    num_channels = y_true.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2.0 * num_channels), sharex=True, dpi=300)
    if num_channels == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i in range(num_channels):
        ax = axes[i]
        name = target_names[i]
        col = colors[i % len(colors)]

        ax.plot(t, y_true[start_sample:end_sample, i], label="Ground Truth (CyberGlove)", color="black", lw=2.0)
        ax.plot(t, y_pred[start_sample:end_sample, i], label=f"Causal TCN Pred ({name})", color=col, lw=1.8, linestyle="--")

        ax.set_ylabel(f"{name.capitalize()}\n[0-1]", fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right", frameon=True, ncol=2)

    axes[-1].set_xlabel("Time [seconds] (Inference at 31.25 Hz)", fontsize=11)
    fig.suptitle("Held-Out Session 3: Continuous Trajectory Tracking Overlay", fontsize=13, fontweight="bold", y=0.99)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_per_finger_error(
    per_output_metrics: Dict[str, Dict[str, float]],
    target_names: List[str],
    save_path: str = "results/plots/4_per_finger_error.png",
) -> None:
    """Bar chart comparing MAE and RMSE across all channels."""
    maes = [per_output_metrics[name]["mae"] for name in target_names]
    rmses = [per_output_metrics[name]["rmse"] for name in target_names]

    x = np.arange(len(target_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width/2, maes, width, label="MAE", color="#1f77b4")
    rects2 = ax.bar(x + width/2, rmses, width, label="RMSE", color="#ff7f0e")

    ax.set_title("Kinematic Regression Error Breakdown by Actuator Channel", fontsize=13, fontweight="bold")
    ax.set_ylabel("Error (Normalized [0, 1])", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n").capitalize() for name in target_names], fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(frameon=True)

    # Add numeric labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    save_path: str = "results/plots/5_residual_distribution.png",
) -> None:
    """Plot residual (pred - true) distributions."""
    errors = y_pred - y_true
    num_channels = y_true.shape[1]
    cols = 3
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i in range(num_channels):
        ax = axes[i]
        err = errors[:, i]
        name = target_names[i]

        ax.hist(err, bins=45, color="#1f77b4", alpha=0.7, edgecolor="black", density=True)
        ax.axvline(0, color="red", linestyle="--", lw=1.5)
        ax.set_title(f"{name.capitalize()}\nMean Err: {np.mean(err):.3f}, Std: {np.std(err):.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Residual (Pred - True)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    for j in range(num_channels, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Prediction Residual Distributions on Held-Out Session 3", fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_correlation_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    save_path: str = "results/plots/6_correlation_scatter.png",
) -> None:
    """Plot linear correlation scatter with best-fit lines."""
    num_channels = y_true.shape[1]
    cols = 3
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i in range(num_channels):
        ax = axes[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]
        name = target_names[i]

        idx = np.random.choice(len(yt), size=min(2500, len(yt)), replace=False)
        ax.scatter(yt[idx], yp[idx], alpha=0.2, s=10, color="#2ca02c")

        # Best fit line
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            poly = np.polyfit(yt, yp, 1)
            x_line = np.linspace(0, 1, 100)
            ax.plot(x_line, np.polyval(poly, x_line), color="darkgreen", lw=2, label=f"Fit (slope={poly[0]:.2f})")

        ax.plot([0, 1], [0, 1], color="red", linestyle=":", lw=1.5, label="y=x")
        r = np.corrcoef(yt, yp)[0, 1] if np.std(yp) > 1e-6 else 0.0

        ax.set_title(f"{name.capitalize()}\nPearson r = {r:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Ground Truth", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)

    for j in range(num_channels, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Empirical Pearson Correlation on Held-Out Session 3", fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_prediction_error_over_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    time_window_samples: int = 600,
    start_sample: int = 800,
    save_path: str = "results/plots/7_prediction_error_over_time.png",
) -> None:
    """Plot absolute tracking error |y - y_pred| over time."""
    end_sample = min(start_sample + time_window_samples, len(y_true))
    t = np.arange(start_sample, end_sample) * 0.032
    t = t - t[0]

    abs_err = np.abs(y_true[start_sample:end_sample] - y_pred[start_sample:end_sample])
    mean_err_over_time = np.mean(abs_err, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, dpi=300)

    # Upper panel: individual finger absolute errors
    for i in range(y_true.shape[1]):
        ax1.plot(t, abs_err[:, i], label=target_names[i].capitalize(), lw=1.2, alpha=0.7)
    ax1.set_ylabel("Absolute Error", fontsize=10, fontweight="bold")
    ax1.set_title("Absolute Tracking Error per Actuator Channel Over Time", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", ncol=3, fontsize=8)

    # Lower panel: ensemble mean error
    ax2.plot(t, mean_err_over_time, color="black", lw=2.0, label="Mean Error Across 6 Channels")
    ax2.axhline(np.mean(mean_err_over_time), color="red", linestyle="--", lw=1.5, label=f"Temporal Average ({np.mean(mean_err_over_time):.3f})")
    ax2.set_xlabel("Time [seconds]", fontsize=11)
    ax2.set_ylabel("Mean |Error|", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_jitter_and_ema_comparison(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_ema: np.ndarray,
    target_names: List[str],
    save_path: str = "results/plots/8_jitter_and_ema_comparison.png",
) -> None:
    """Plot jitter reduction bar chart and trajectory comparison for Raw vs EMA."""
    jitter_raw_per_ch = np.mean(np.abs(np.diff(y_raw, axis=0)), axis=0)
    jitter_ema_per_ch = np.mean(np.abs(np.diff(y_ema, axis=0)), axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Left: Jitter Bar Chart
    x = np.arange(len(target_names))
    width = 0.35
    ax1.bar(x - width/2, jitter_raw_per_ch, width, label="Raw TCN Output", color="#1f77b4")
    ax1.bar(x + width/2, jitter_ema_per_ch, width, label="EMA Filtered (alpha=0.20)", color="#d62728")
    ax1.set_title("Sample-to-Sample Jitter |dy/dt| by Channel", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Step Jitter", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace("_", "\n").capitalize() for name in target_names], fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.legend(frameon=True)

    # Right: Trajectory Zoom
    start = 1000
    length = 250
    t = np.arange(length) * 0.032
    ch_idx = 0  # Index finger
    ax2.plot(t, y_true[start:start+length, ch_idx], label="Ground Truth", color="black", lw=2.2)
    ax2.plot(t, y_raw[start:start+length, ch_idx], label="Raw TCN", color="#1f77b4", alpha=0.6, lw=1.2)
    ax2.plot(t, y_ema[start:start+length, ch_idx], label="TCN + EMA (alpha=0.20)", color="#d62728", lw=2.0)
    ax2.set_title(f"Dynamic Smoothing Detail ({target_names[ch_idx].capitalize()})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time [seconds]", fontsize=10)
    ax2.set_ylabel("Command [0, 1]", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_inference_latency(
    cpu_latencies_ms: np.ndarray,
    gpu_latencies_ms: Optional[np.ndarray] = None,
    save_path: str = "results/plots/9_prediction_latency.png",
) -> None:
    """Plot measured inference latency histogram vs 32 ms real-time deadline."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    ax.hist(cpu_latencies_ms, bins=35, color="#1f77b4", alpha=0.6, label=f"CPU Latency (mean={np.mean(cpu_latencies_ms):.2f} ms)", density=True)
    if gpu_latencies_ms is not None and len(gpu_latencies_ms) > 0:
        ax.hist(gpu_latencies_ms, bins=35, color="#2ca02c", alpha=0.6, label=f"GPU Latency (mean={np.mean(gpu_latencies_ms):.2f} ms)", density=True)

    ax.axvline(32.0, color="red", linestyle="--", lw=2, label="32.0 ms Real-Time Deadline (31.25 Hz)")

    ax.set_title("Empirical Model Inference Latency Benchmark", fontsize=12, fontweight="bold")
    ax.set_xlabel("Inference Latency [ms]", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
