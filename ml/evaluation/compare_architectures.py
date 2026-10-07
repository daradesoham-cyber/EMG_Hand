"""
Architecture Comparison and EMA Grid Search on Session 2 (Validation Set).

Strict Rule (Problem 2 & Problem 5):
- Evaluates Model A, Model B, and Model C strictly using Session 1 (Train) and Session 2 (Validation).
- Session 3 (A3) is NEVER touched during this script.
- Generates empirical metrics: MAE, RMSE, R2, Pearson r, jitter, receptive field, parameters, CPU latency.
- Tests EMA alpha grid search on Session 2 only to select optimal alpha.
"""

import os
import sys
import time
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.dataset import NinaProDB8SessionDataset, NumpyTensorDataset
from ml.windowing import WindowConfig
from ml.model import CausalTCN
from ml.evaluate import compute_regression_metrics, apply_ema_filter, compute_temporal_lag_analysis


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def measure_cpu_latency(model: nn.Module, in_channels: int = 16, window_samples: int = 256, runs: int = 500) -> Dict[str, float]:
    """Measure single-window inference latency on CPU."""
    model_cpu = model.to("cpu")
    model_cpu.eval()
    dummy_input = torch.randn(1, in_channels, window_samples)

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model_cpu(dummy_input)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model_cpu(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    return {
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
    }


def train_and_eval_model(
    name: str,
    hidden_channels: List[int],
    dilations: List[int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_Y: np.ndarray,
    target_names: List[str],
    device: torch.device,
    epochs: int = 25,
    patience: int = 6,
    lr: float = 1e-4,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Training & Evaluating: {name}")
    print(f"Dilations: {dilations}")
    print(f"Hidden channels: {hidden_channels}")
    print(f"=======================================================")

    model = CausalTCN(
        in_channels=16,
        hidden_channels=hidden_channels,
        kernel_size=3,
        dilations=dilations,
        dropout=0.2,
        num_outputs=len(target_names),
        convs_per_block=2,
        head_activation="linear",
    )

    rf_info = model.compute_receptive_field()
    param_count = model.count_parameters()
    fp32_size_kb = (param_count * 4) / 1024.0

    print(f"Parameters: {param_count:,} ({fp32_size_kb:.1f} kB FP32)")
    print(f"Receptive field: {rf_info['receptive_field_samples']} samples ({rf_info['receptive_field_ms']:.2f} ms, {rf_info['coverage_ratio']*100:.1f}% window)")

    # Benchmark CPU latency before moving model to GPU
    print("Benchmarking CPU latency (500 iterations)...")
    cpu_perf = measure_cpu_latency(model, in_channels=16, window_samples=256, runs=500)
    print(f"CPU Latency: mean={cpu_perf['latency_mean_ms']:.3f} ms, p95={cpu_perf['latency_p95_ms']:.3f} ms")

    # Move to training device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    best_val_mae = float("inf")
    patience_cnt = 0
    best_weights = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)

        train_loss = total_train_loss / n_train

        # Validation on Session 2 (A2)
        model.eval()
        total_val_loss = 0.0
        n_val = 0
        val_preds_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item() * xb.size(0)
                n_val += xb.size(0)
                val_preds_list.append(pred.cpu().numpy())

        val_loss = total_val_loss / n_val
        scheduler.step(val_loss)

        if val_loss < best_val_mae:
            best_val_mae = val_loss
            patience_cnt = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_preds = np.vstack(val_preds_list)
        else:
            patience_cnt += 1

        print(f"  Epoch {ep:2d}/{epochs:2d} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | Best: {best_val_mae:.4f}")
        if patience_cnt >= patience:
            print(f"  Early stopping triggered at epoch {ep}")
            break

    train_time = time.time() - t0

    # Evaluate best weights on Session 2
    raw_metrics = compute_regression_metrics(val_Y, best_val_preds, target_names)
    jitter = float(np.mean(np.abs(np.diff(best_val_preds, axis=0))))

    return {
        "model_name": name,
        "dilations": str(dilations),
        "rf_samples": rf_info["receptive_field_samples"],
        "rf_ms": rf_info["receptive_field_ms"],
        "rf_coverage_pct": rf_info["coverage_ratio"] * 100.0,
        "parameters": param_count,
        "fp32_size_kb": fp32_size_kb,
        "cpu_latency_mean_ms": cpu_perf["latency_mean_ms"],
        "cpu_latency_p95_ms": cpu_perf["latency_p95_ms"],
        "val_mae": raw_metrics["aggregate"]["mae"],
        "val_rmse": raw_metrics["aggregate"]["rmse"],
        "val_r2": raw_metrics["aggregate"]["r2"],
        "val_correlation": raw_metrics["aggregate"]["correlation"],
        "val_jitter": jitter,
        "train_time_s": train_time,
        "best_preds": best_val_preds,
        "best_weights": best_weights,
    }


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparison executing on device: {device}")

    # Prepare data for A1 and A2 (A3 is kept STRICTLY HELD OUT)
    target_names = ["index", "middle", "ring", "little", "thumb_flexion", "thumb_opposition"]
    win_config = WindowConfig(sampling_rate=2000, window_ms=128.0, stride_ms=32.0)
    dataset = NinaProDB8SessionDataset(
        data_dir="data/ninapro_db8",
        subject_id=1,
        window_config=win_config,
        num_outputs=6,
        target_names=target_names,
    )

    print("\nPreparing Session 1 (Train) and Session 2 (Validation)...")
    (train_X, train_Y), (val_X, val_Y), _ = dataset.prepare_data()
    print(f"Train (A1): {train_X.shape[0]} windows")
    print(f"Val   (A2): {val_X.shape[0]} windows")

    train_loader = DataLoader(NumpyTensorDataset(train_X, train_Y), batch_size=64, shuffle=True)
    val_loader = DataLoader(NumpyTensorDataset(val_X, val_Y), batch_size=64, shuffle=False)

    # Architectures to compare
    configs = [
        {
            "name": "Model A (Current)",
            "hidden_channels": [32, 64, 64, 128],
            "dilations": [1, 2, 4, 8],
        },
        {
            "name": "Model B (Extended Dilation 16)",
            "hidden_channels": [32, 64, 64, 128, 128],
            "dilations": [1, 2, 4, 8, 16],
        },
        {
            "name": "Model C (Full Window RF 32)",
            "hidden_channels": [32, 64, 64, 128, 128, 128],
            "dilations": [1, 2, 4, 8, 16, 32],
        },
    ]

    results = []
    trained_models = {}

    for cfg in configs:
        res = train_and_eval_model(
            name=cfg["name"],
            hidden_channels=cfg["hidden_channels"],
            dilations=cfg["dilations"],
            train_loader=train_loader,
            val_loader=val_loader,
            val_Y=val_Y,
            target_names=target_names,
            device=device,
            epochs=20,
            patience=5,
        )
        trained_models[cfg["name"]] = res
        results.append({k: v for k, v in res.items() if k not in ["best_preds", "best_weights"]})

    # Save architecture comparison table
    df_arch = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df_arch.to_csv("results/architecture_comparison_session2.csv", index=False)
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON ON SESSION 2 (VALIDATION SET)")
    print("=" * 70)
    print(df_arch[["model_name", "rf_ms", "parameters", "cpu_latency_mean_ms", "val_mae", "val_rmse", "val_r2", "val_correlation", "val_jitter"]].to_string(index=False))

    # Evaluate EMA Alpha Grid Search on Session 2
    print("\n" + "=" * 70)
    print("EMA ALPHA GRID SEARCH ON SESSION 2 (VALIDATION SET)")
    print("=" * 70)
    alphas = [0.10, 0.15, 0.20, 0.30, 0.40]
    ema_grid_records = []

    for model_name, m_data in trained_models.items():
        val_preds_raw = m_data["best_preds"]
        raw_jitter = m_data["val_jitter"]
        raw_mae = m_data["val_mae"]

        # Raw lag
        raw_lag = compute_temporal_lag_analysis(val_Y, val_preds_raw, dt_ms=32.0, max_lag_steps=20)

        ema_grid_records.append({
            "model_name": model_name,
            "alpha": "RAW (None)",
            "mae": raw_mae,
            "rmse": m_data["val_rmse"],
            "r2": m_data["val_r2"],
            "correlation": m_data["val_correlation"],
            "jitter": raw_jitter,
            "jitter_reduction_pct": 0.0,
            "zero_lag_r": raw_lag["aggregate"]["zero_lag_r"],
            "best_lag_r": raw_lag["aggregate"]["best_lag_r"],
            "mean_lag_ms": raw_lag["aggregate"]["mean_lag_ms"],
        })

        for a in alphas:
            val_preds_ema = apply_ema_filter(val_preds_raw, alpha=a)
            metrics_ema = compute_regression_metrics(val_Y, val_preds_ema, target_names)
            jitter_ema = float(np.mean(np.abs(np.diff(val_preds_ema, axis=0))))
            jitter_red = ((raw_jitter - jitter_ema) / max(raw_jitter, 1e-8)) * 100.0
            lag_ema = compute_temporal_lag_analysis(val_Y, val_preds_ema, dt_ms=32.0, max_lag_steps=20)

            ema_grid_records.append({
                "model_name": model_name,
                "alpha": a,
                "mae": metrics_ema["aggregate"]["mae"],
                "rmse": metrics_ema["aggregate"]["rmse"],
                "r2": metrics_ema["aggregate"]["r2"],
                "correlation": metrics_ema["aggregate"]["correlation"],
                "jitter": jitter_ema,
                "jitter_reduction_pct": jitter_red,
                "zero_lag_r": lag_ema["aggregate"]["zero_lag_r"],
                "best_lag_r": lag_ema["aggregate"]["best_lag_r"],
                "mean_lag_ms": lag_ema["aggregate"]["mean_lag_ms"],
            })

    df_ema = pd.DataFrame(ema_grid_records)
    df_ema.to_csv("results/ema_grid_session2.csv", index=False)
    print(df_ema.to_string(index=False))

    print("\nArchitecture comparison and EMA grid search complete! Results saved in results/")


if __name__ == "__main__":
    main()
