"""
Final Training & Evaluation Pipeline for 6-Actuator NinaPro DB8 Causal TCN.

Follows strict experimental protocols:
- Dataset: NinaPro DB8, Subject 1.
- Split: Acquisition 1 (A1) = Training, Acquisition 2 (A2) = Validation, Acquisition 3 (A3) = Strictly Held-Out Test.
- Preprocessor & Normalizers fitted strictly on A1 only.
- Architecture: Frozen Model C (Dilations [1, 2, 4, 8, 16, 32], RF = 253 samples / 126.5 ms, 6 outputs).
- Baselines:
    1. Oracle Persistence / Temporal Smoothness Reference
    2. Linear Regression (TD Features)
    3. Simple MLP
    4. Causal TCN (Frozen Model C)
    5. Causal GRU
- Evaluation on strictly held-out Session 3 (A3):
    - Raw TCN
    - TCN + EMA (alpha=0.20)
    - Cross-correlation temporal lag analysis
    - Per-finger metrics across all 6 channels
    - 8 publication-grade visualization plots
"""

import os
import sys
import time
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.dataset import NinaProDB8SessionDataset, NumpyTensorDataset
from ml.windowing import WindowConfig
from ml.model import (
    CausalTCN,
    LinearRegressionBaseline,
    SimpleMLPBaseline,
    CausalGRUBaseline,
    PersistenceBaseline,
)
from ml.evaluate import (
    compute_regression_metrics,
    apply_ema_filter,
    evaluate_ema_tradeoff,
    compute_temporal_lag_analysis,
)
from ml import plots


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    return avg_loss, y_true, y_pred


def train_baseline_linear_regression(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    val_X: np.ndarray,
    val_Y: np.ndarray,
    test_X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    print("\nTraining Baseline 2: Linear Regression (Time-Domain Features)...")
    # Convert numpy windows to PyTorch to extract features
    with torch.no_grad():
        x_tr_feat = LinearRegressionBaseline.extract_features(torch.from_numpy(train_X))
        x_val_feat = LinearRegressionBaseline.extract_features(torch.from_numpy(val_X))
        x_te_feat = LinearRegressionBaseline.extract_features(torch.from_numpy(test_X)).to(device)

    model = LinearRegressionBaseline(in_features=x_tr_feat.size(-1), num_outputs=train_Y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    tr_dataset = DataLoader(NumpyTensorDataset(x_tr_feat.numpy(), train_Y), batch_size=128, shuffle=True)
    best_val = float("inf")
    best_weights = None

    for ep in range(15):
        model.train()
        for xb, yb in tr_dataset:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            l = criterion(model(xb), yb)
            l.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_pred = model(x_val_feat.to(device))
            v_loss = criterion(v_pred, torch.from_numpy(val_Y).to(device)).item()
            if v_loss < best_val:
                best_val = v_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_te_feat).cpu().numpy()
    return y_pred


def train_baseline_mlp(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_outputs: int,
    device: torch.device,
) -> np.ndarray:
    print("\nTraining Baseline 3: Simple MLP...")
    mlp = SimpleMLPBaseline(in_channels=16, window_samples=256, num_outputs=num_outputs).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.L1Loss()
    best_val = float("inf")
    best_weights = None

    for ep in range(15):
        mlp.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            l = criterion(mlp(xb), yb)
            l.backward()
            optimizer.step()

        val_loss, _, _ = evaluate_epoch(mlp, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

    mlp.load_state_dict(best_weights)
    _, _, y_pred_mlp = evaluate_epoch(mlp, test_loader, criterion, device)
    return y_pred_mlp


def train_baseline_gru(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_outputs: int,
    device: torch.device,
) -> np.ndarray:
    print("\nTraining Baseline 5: Causal GRU...")
    gru = CausalGRUBaseline(in_channels=16, hidden_dim=64, num_layers=2, num_outputs=num_outputs).to(device)
    optimizer = torch.optim.Adam(gru.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.L1Loss()
    best_val = float("inf")
    best_weights = None

    for ep in range(15):
        gru.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            l = criterion(gru(xb), yb)
            l.backward()
            optimizer.step()

        val_loss, _, _ = evaluate_epoch(gru, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.cpu().clone() for k, v in gru.state_dict().items()}

    gru.load_state_dict(best_weights)
    _, _, y_pred_gru = evaluate_epoch(gru, test_loader, criterion, device)
    return y_pred_gru


def main(config_path: str = "configs/tcn_db8.yaml") -> None:
    print("=" * 70)
    print("Six-Actuator NinaPro DB8 Causal TCN Hand Kinematic Regression")
    print("=" * 70)

    # 1. Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Window configuration
    w_cfg = cfg["features"]
    win_config = WindowConfig(
        sampling_rate=cfg["dataset"].get("sampling_rate", 2000),
        window_ms=w_cfg.get("window_ms", 128.0),
        stride_ms=w_cfg.get("stride_ms", 32.0),
    )
    print(f"Window Configuration: {win_config}")

    # 3. Load & Preprocess NinaPro DB8 Dataset (A1, A2, A3)
    num_outputs = w_cfg.get("num_outputs", 6)
    target_names = w_cfg.get("target_names", [
        "index", "middle", "ring", "little", "thumb_flexion", "thumb_opposition"
    ])

    print("\nPreparing dataset with strict session isolation...")
    print("Fitting preprocessor and normalizers strictly on Acquisition 1 (A1)...")
    dataset_handler = NinaProDB8SessionDataset(
        data_dir=cfg["dataset"].get("data_dir", "data/ninapro_db8"),
        subject_id=cfg["dataset"].get("subject", 1),
        window_config=win_config,
        num_outputs=num_outputs,
        target_names=target_names,
    )

    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y) = dataset_handler.prepare_data()

    print(f"Train Set (A1): {train_X.shape[0]} windows, shape={train_X.shape}")
    print(f"Val Set   (A2): {val_X.shape[0]} windows, shape={val_X.shape}")
    print(f"Test Set  (A3): {test_X.shape[0]} windows, shape={test_X.shape} (HELD OUT UNTIL FROZEN)")

    batch_size = cfg["training"].get("batch_size", 64)
    train_loader = DataLoader(NumpyTensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NumpyTensorDataset(val_X, val_Y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(NumpyTensorDataset(test_X, test_Y), batch_size=batch_size, shuffle=False)

    # 4. Instantiate Frozen Model C Architecture
    m_cfg = cfg["model"]
    model = CausalTCN(
        in_channels=m_cfg.get("in_channels", 16),
        hidden_channels=m_cfg.get("hidden_channels", [32, 64, 64, 128, 128, 128]),
        kernel_size=m_cfg.get("kernel_size", 3),
        dilations=m_cfg.get("dilations", [1, 2, 4, 8, 16, 32]),
        dropout=m_cfg.get("dropout", 0.2),
        num_outputs=num_outputs,
        convs_per_block=2,
        head_activation=m_cfg.get("head_activation", "linear"),
    ).to(device)

    rf_info = model.compute_receptive_field()
    print("\n--- Model Architecture & Receptive Field Audit ---")
    print(f"Trainable Parameters: {model.count_parameters():,}")
    print(f"Theoretical Receptive Field: {rf_info['receptive_field_samples']} samples ({rf_info['receptive_field_ms']:.2f} ms)")
    print(f"Input Window: {rf_info['input_window_samples']} samples ({rf_info['input_window_ms']:.2f} ms)")
    print(f"Receptive Field Coverage: {rf_info['coverage_ratio']*100:.1f}% of 128 ms buffer")

    # 5. Training Setup
    t_cfg = cfg["training"]
    lr = float(t_cfg.get("learning_rate", 1e-4))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=float(t_cfg.get("weight_decay", 1e-5))
    )
    criterion = nn.L1Loss()  # MAE Loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    max_epochs = t_cfg.get("epochs", 30)
    patience = t_cfg.get("early_stopping_patience", 7)
    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    best_model_path = "results/best_model_6ch.pt"

    history = []
    print("\nStarting training loop for Frozen 6-Actuator Model C...")
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_time = time.time() - t_epoch_start
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": epoch_time,
        })

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                    "target_names": target_names,
                    "target_norm_stats": dataset_handler.target_norm_stats,
                    "preprocessor_stats": dataset_handler.preprocessor.get_normalization_stats(),
                },
                best_model_path,
            )
            # Also update best_model.pt for compatibility
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                    "target_names": target_names,
                    "target_norm_stats": dataset_handler.target_norm_stats,
                    "preprocessor_stats": dataset_handler.preprocessor.get_normalization_stats(),
                },
                "results/best_model.pt",
            )
            improved = " [BEST MODEL SAVED]"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:2d}/{max_epochs:2d} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.2f}s{improved}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs (patience={patience}).")
            break

    total_train_time = time.time() - t_start
    print(f"\nTraining completed in {total_train_time:.1f} seconds.")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv("results/training_history.csv", index=False)

    # 6. Load Frozen Best Model Checkpoint
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"\nLoaded frozen best model from Epoch {checkpoint['epoch']} with Val MAE {checkpoint['val_loss']:.4f}")

    # 7. Evaluate Strictly Once on Held-Out Session 3 (A3)
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON STRICTLY HELD-OUT TEST SET (SESSION 3 / A3)")
    print("=" * 70)

    test_loss, y_true_test, y_pred_test = evaluate_epoch(model, test_loader, criterion, device)
    raw_test_metrics = compute_regression_metrics(y_true_test, y_pred_test, target_names)

    # Evaluate with Frozen EMA alpha=0.20
    alpha = cfg.get("evaluation", {}).get("ema_alpha", 0.20)
    ema_eval = evaluate_ema_tradeoff(y_true_test, y_pred_test, alpha=alpha, target_names=target_names)
    y_pred_ema = apply_ema_filter(y_pred_test, alpha=alpha)

    # Temporal Lag Analysis for Raw TCN and TCN+EMA
    raw_lag = compute_temporal_lag_analysis(y_true_test, y_pred_test, dt_ms=32.0, max_lag_steps=20, target_names=target_names)
    ema_lag = compute_temporal_lag_analysis(y_true_test, y_pred_ema, dt_ms=32.0, max_lag_steps=20, target_names=target_names)

    print("\n--- Raw Causal TCN Performance (Session 3) ---")
    print(f"Aggregate MAE:  {raw_test_metrics['aggregate']['mae']:.4f}")
    print(f"Aggregate RMSE: {raw_test_metrics['aggregate']['rmse']:.4f}")
    print(f"Aggregate R²:   {raw_test_metrics['aggregate']['r2']:.4f}")
    print(f"Aggregate Corr: {raw_test_metrics['aggregate']['correlation']:.4f}")
    print(f"Zero-Lag Corr:  {raw_lag['aggregate']['zero_lag_r']:.4f} | Best-Lag Corr: {raw_lag['aggregate']['best_lag_r']:.4f} (Mean Lag: {raw_lag['aggregate']['mean_lag_ms']:.1f} ms)")

    print(f"\n--- Causal TCN + EMA (alpha={alpha}) Performance (Session 3) ---")
    print(f"Aggregate MAE:  {ema_eval['ema_mae']:.4f} (Delta: {ema_eval['mae_delta']:+.4f})")
    print(f"Aggregate RMSE: {ema_eval['ema_metrics']['aggregate']['rmse']:.4f}")
    print(f"Aggregate R²:   {ema_eval['ema_metrics']['aggregate']['r2']:.4f}")
    print(f"Aggregate Corr: {ema_eval['ema_metrics']['aggregate']['correlation']:.4f}")
    print(f"Step Jitter:    {ema_eval['jitter_raw']:.4f} -> {ema_eval['jitter_ema']:.4f} ({ema_eval['jitter_reduction_pct']:.1f}% reduction)")
    print(f"Zero-Lag Corr:  {ema_lag['aggregate']['zero_lag_r']:.4f} | Best-Lag Corr: {ema_lag['aggregate']['best_lag_r']:.4f} (Mean Lag: {ema_lag['aggregate']['mean_lag_ms']:.1f} ms)")

    print("\n--- Per-Output Performance (Session 3) ---")
    per_output_records = []
    for name in target_names:
        m = raw_test_metrics["per_output"][name]
        m_ema = ema_eval["ema_metrics"]["per_output"][name]
        l_raw = raw_lag["per_output"][name]
        l_ema = ema_lag["per_output"][name]

        print(f"  {name.replace('_', ' ').capitalize():18s}: MAE={m['mae']:.4f} (EMA: {m_ema['mae']:.4f}), RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}, r={m['correlation']:.4f}, lag={l_raw['best_lag_ms']:.0f}ms")
        per_output_records.append({
            "actuator": name,
            "raw_mae": m["mae"],
            "raw_rmse": m["rmse"],
            "raw_r2": m["r2"],
            "raw_correlation": m["correlation"],
            "raw_lag_ms": l_raw["best_lag_ms"],
            "ema_mae": m_ema["mae"],
            "ema_rmse": m_ema["rmse"],
            "ema_r2": m_ema["r2"],
            "ema_correlation": m_ema["correlation"],
            "ema_lag_ms": l_ema["best_lag_ms"],
        })
    pd.DataFrame(per_output_records).to_csv("results/metrics_per_output.csv", index=False)

    # 8. Train & Evaluate Baselines
    print("\n" + "=" * 70)
    print("BENCHMARKING BASELINES ON IDENTICAL HELD-OUT SPLIT (SESSION 3)")
    print("=" * 70)

    # Baseline 1: Oracle Persistence / Temporal Smoothness Reference
    y_pred_persist = np.vstack([test_Y[0:1], test_Y[:-1]])
    persist_metrics = compute_regression_metrics(test_Y, y_pred_persist, target_names)
    persist_jitter = float(np.mean(np.abs(np.diff(y_pred_persist, axis=0))))
    print(f"Baseline 1 (Oracle Persistence Ref): MAE={persist_metrics['aggregate']['mae']:.4f}, RMSE={persist_metrics['aggregate']['rmse']:.4f}, R²={persist_metrics['aggregate']['r2']:.4f}, r={persist_metrics['aggregate']['correlation']:.4f}")

    # Baseline 2: Linear Regression
    y_pred_lr = train_baseline_linear_regression(train_X, train_Y, val_X, val_Y, test_X, device)
    lr_metrics = compute_regression_metrics(test_Y, y_pred_lr, target_names)
    lr_jitter = float(np.mean(np.abs(np.diff(y_pred_lr, axis=0))))
    print(f"Baseline 2 (Linear Regression):     MAE={lr_metrics['aggregate']['mae']:.4f}, RMSE={lr_metrics['aggregate']['rmse']:.4f}, R²={lr_metrics['aggregate']['r2']:.4f}, r={lr_metrics['aggregate']['correlation']:.4f}")

    # Baseline 3: Simple MLP
    y_pred_mlp = train_baseline_mlp(train_loader, val_loader, test_loader, num_outputs, device)
    mlp_metrics = compute_regression_metrics(test_Y, y_pred_mlp, target_names)
    mlp_jitter = float(np.mean(np.abs(np.diff(y_pred_mlp, axis=0))))
    print(f"Baseline 3 (Simple MLP):             MAE={mlp_metrics['aggregate']['mae']:.4f}, RMSE={mlp_metrics['aggregate']['rmse']:.4f}, R²={mlp_metrics['aggregate']['r2']:.4f}, r={mlp_metrics['aggregate']['correlation']:.4f}")

    # Baseline 5: Causal GRU
    y_pred_gru = train_baseline_gru(train_loader, val_loader, test_loader, num_outputs, device)
    gru_metrics = compute_regression_metrics(test_Y, y_pred_gru, target_names)
    gru_jitter = float(np.mean(np.abs(np.diff(y_pred_gru, axis=0))))
    gru_lag = compute_temporal_lag_analysis(test_Y, y_pred_gru, dt_ms=32.0, max_lag_steps=20, target_names=target_names)
    print(f"Baseline 5 (Causal GRU):             MAE={gru_metrics['aggregate']['mae']:.4f}, RMSE={gru_metrics['aggregate']['rmse']:.4f}, R²={gru_metrics['aggregate']['r2']:.4f}, r={gru_metrics['aggregate']['correlation']:.4f}")

    # Consolidate baseline comparison table
    baselines_table = [
        {
            "Model": "Oracle Persistence (Smoothness Reference)",
            "MAE": persist_metrics["aggregate"]["mae"],
            "RMSE": persist_metrics["aggregate"]["rmse"],
            "R2": persist_metrics["aggregate"]["r2"],
            "Correlation": persist_metrics["aggregate"]["correlation"],
            "Jitter": persist_jitter,
            "Causal sEMG-Only": "NO (Oracle Feedback)",
        },
        {
            "Model": "Linear Regression (TD Features)",
            "MAE": lr_metrics["aggregate"]["mae"],
            "RMSE": lr_metrics["aggregate"]["rmse"],
            "R2": lr_metrics["aggregate"]["r2"],
            "Correlation": lr_metrics["aggregate"]["correlation"],
            "Jitter": lr_jitter,
            "Causal sEMG-Only": "YES",
        },
        {
            "Model": "Simple MLP Baseline",
            "MAE": mlp_metrics["aggregate"]["mae"],
            "RMSE": mlp_metrics["aggregate"]["rmse"],
            "R2": mlp_metrics["aggregate"]["r2"],
            "Correlation": mlp_metrics["aggregate"]["correlation"],
            "Jitter": mlp_jitter,
            "Causal sEMG-Only": "YES",
        },
        {
            "Model": "Causal GRU Baseline",
            "MAE": gru_metrics["aggregate"]["mae"],
            "RMSE": gru_metrics["aggregate"]["rmse"],
            "R2": gru_metrics["aggregate"]["r2"],
            "Correlation": gru_metrics["aggregate"]["correlation"],
            "Jitter": gru_jitter,
            "Causal sEMG-Only": "YES",
        },
        {
            "Model": "Causal TCN Model C (Raw)",
            "MAE": raw_test_metrics["aggregate"]["mae"],
            "RMSE": raw_test_metrics["aggregate"]["rmse"],
            "R2": raw_test_metrics["aggregate"]["r2"],
            "Correlation": raw_test_metrics["aggregate"]["correlation"],
            "Jitter": ema_eval["jitter_raw"],
            "Causal sEMG-Only": "YES",
        },
        {
            "Model": "Causal TCN Model C + EMA (alpha=0.20)",
            "MAE": ema_eval["ema_mae"],
            "RMSE": ema_eval["ema_metrics"]["aggregate"]["rmse"],
            "R2": ema_eval["ema_metrics"]["aggregate"]["r2"],
            "Correlation": ema_eval["ema_metrics"]["aggregate"]["correlation"],
            "Jitter": ema_eval["jitter_ema"],
            "Causal sEMG-Only": "YES",
        },
    ]
    df_baselines = pd.DataFrame(baselines_table)
    df_baselines.to_csv("results/baselines_comparison.csv", index=False)
    print("\n" + df_baselines.to_string(index=False))

    # 9. Save Predictions & Summary
    pred_df = pd.DataFrame()
    for i, name in enumerate(target_names):
        pred_df[f"{name}_true"] = y_true_test[:, i]
        pred_df[f"{name}_pred_raw"] = y_pred_test[:, i]
        pred_df[f"{name}_pred_ema"] = y_pred_ema[:, i]
    pred_df.to_csv("results/session3_predictions.csv", index=False)

    test_results_summary = {
        "dataset": "NinaPro DB8",
        "subject": cfg["dataset"].get("subject", 1),
        "split_protocol": "A1=Train, A2=Val, A3=Held-Out Test",
        "num_outputs": num_outputs,
        "target_names": target_names,
        "receptive_field_samples": rf_info["receptive_field_samples"],
        "receptive_field_ms": rf_info["receptive_field_ms"],
        "input_window_ms": 128.0,
        "parameters": model.count_parameters(),
        "total_train_time_s": total_train_time,
        "best_epoch": int(checkpoint["epoch"]),
        "tcn_raw_metrics": raw_test_metrics,
        "tcn_ema_metrics": ema_eval["ema_metrics"],
        "raw_lag_analysis": raw_lag["aggregate"],
        "ema_lag_analysis": ema_lag["aggregate"],
        "ema_analysis": {
            "alpha": alpha,
            "jitter_raw": ema_eval["jitter_raw"],
            "jitter_ema": ema_eval["jitter_ema"],
            "jitter_reduction_pct": ema_eval["jitter_reduction_pct"],
            "effective_time_constant_ms": ema_eval["effective_time_constant_ms"],
        },
        "baselines": {
            "oracle_persistence_reference": persist_metrics["aggregate"],
            "linear_regression": lr_metrics["aggregate"],
            "simple_mlp": mlp_metrics["aggregate"],
            "causal_gru": gru_metrics["aggregate"],
        },
    }
    with open("results/session3_test_results.json", "w") as f:
        json.dump(test_results_summary, f, indent=2)

    # 10. Generate Visual Plots (All 8 Figures)
    print("\nGenerating all 8 publication-grade evaluation plots in results/plots/...")
    plots.plot_loss_curves(history, "results/plots/1_training_validation_loss.png")
    plots.plot_predicted_vs_ground_truth(y_true_test, y_pred_test, target_names, "results/plots/2_predicted_vs_ground_truth.png")
    plots.plot_time_series_overlay(y_true_test, y_pred_test, target_names, save_path="results/plots/3_time_series_overlay.png")
    plots.plot_per_finger_error(raw_test_metrics["per_output"], target_names, "results/plots/4_per_finger_error.png")
    plots.plot_residual_distribution(y_true_test, y_pred_test, target_names, "results/plots/5_residual_distribution.png")
    plots.plot_correlation_scatter(y_true_test, y_pred_test, target_names, "results/plots/6_correlation_scatter.png")
    plots.plot_prediction_error_over_time(y_true_test, y_pred_test, target_names, save_path="results/plots/7_prediction_error_over_time.png")
    plots.plot_jitter_and_ema_comparison(y_true_test, y_pred_test, y_pred_ema, target_names, "results/plots/8_jitter_and_ema_comparison.png")

    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Causal TCN on NinaPro DB8")
    parser.add_argument("--config", type=str, default="configs/tcn_db8.yaml")
    args = parser.parse_args()
    main(args.config)
