"""
Strict Session 3 Held-Out Evaluation Runner.

Loads frozen checkpoint (best_model_6ch.pt), processes strictly held-out Session 3 (A3),
and generates comprehensive raw, EMA, per-output, and temporal lag metrics.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.dataset import NinaProDB8SessionDataset, NumpyTensorDataset
from ml.models.tcn.model import CausalTCN
from ml.evaluation.metrics import (
    compute_regression_metrics,
    apply_ema_filter,
    evaluate_ema_tradeoff,
)
from ml.preprocessing.alignment import compute_cross_correlation_lag


def run_evaluation(
    checkpoint_path: str = "ml/models/checkpoints/best_model_6ch.pt",
    config_path: str = "ml/training/configs/tcn_db8.yaml",
    alpha: float = 0.20,
):
    print("=" * 70)
    print("STRICTLY HELD-OUT SESSION 3 (A3) EVALUATION SUITE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path} on {device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    target_names = ckpt.get("target_names", [
        "index", "middle", "ring", "little", "thumb_flexion", "thumb_opposition"
    ])
    num_outputs = len(target_names)

    # Load test dataset
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["dataset"].get("data_dir", "data/ninapro_db8")
    dataset_handler = NinaProDB8SessionDataset(
        data_dir=data_dir,
        subject_id=cfg["dataset"].get("subject", 1),
        num_outputs=num_outputs,
        target_names=target_names,
    )
    _, _, (test_X, test_Y) = dataset_handler.prepare_data()
    print(f"Loaded Session 3: {len(test_X)} windows (16 channels, 256 samples)")

    # Instantiate model
    m_cfg = ckpt.get("config", {}).get("model", {})
    model = CausalTCN(
        in_channels=m_cfg.get("in_channels", 16),
        hidden_channels=m_cfg.get("hidden_channels", [32, 64, 64, 128, 128, 128]),
        dilations=m_cfg.get("dilations", [1, 2, 4, 8, 16, 32]),
        kernel_size=m_cfg.get("kernel_size", 3),
        num_outputs=num_outputs,
        convs_per_block=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Predict
    test_loader = DataLoader(NumpyTensorDataset(test_X, test_Y), batch_size=64, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds = model(xb.to(device))
            all_preds.append(preds.cpu().numpy())
    y_pred_raw = np.concatenate(all_preds, axis=0)

    # Raw metrics
    raw_metrics = compute_regression_metrics(test_Y, y_pred_raw, target_names)
    raw_lag = compute_cross_correlation_lag(test_Y, y_pred_raw, dt_ms=32.0, target_names=target_names)

    # EMA metrics
    ema_eval = evaluate_ema_tradeoff(test_Y, y_pred_raw, alpha=alpha, target_names=target_names)
    y_pred_ema = apply_ema_filter(y_pred_raw, alpha=alpha)
    ema_lag = compute_cross_correlation_lag(test_Y, y_pred_ema, dt_ms=32.0, target_names=target_names)

    print("\n--- Aggregate Results on Held-Out Session 3 ---")
    print(f"Raw TCN:     MAE={raw_metrics['aggregate']['mae']:.4f}, RMSE={raw_metrics['aggregate']['rmse']:.4f}, R²={raw_metrics['aggregate']['r2']:.4f}, r={raw_metrics['aggregate']['correlation']:.4f}, lead={raw_lag['aggregate']['mean_lag_ms']:.1f}ms")
    print(f"TCN + EMA:   MAE={ema_eval['ema_mae']:.4f}, RMSE={ema_eval['ema_metrics']['aggregate']['rmse']:.4f}, R²={ema_eval['ema_metrics']['aggregate']['r2']:.4f}, r={ema_eval['ema_metrics']['aggregate']['correlation']:.4f}, lag={ema_lag['aggregate']['mean_lag_ms']:.1f}ms")
    print(f"Jitter Red:  {ema_eval['jitter_raw']:.4f} -> {ema_eval['jitter_ema']:.4f} ({ema_eval['jitter_reduction_pct']:.1f}% reduction)")

    print("\n--- Per-Output Breakdown (TCN + EMA alpha=0.20) ---")
    for name in target_names:
        m = ema_eval["ema_metrics"]["per_output"][name]
        l = ema_lag["per_output"][name]
        print(f"  {name:18s}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}, r={m['correlation']:.4f}, lag={l['best_lag_ms']:.0f}ms")

    return {
        "raw_metrics": raw_metrics,
        "ema_metrics": ema_eval["ema_metrics"],
        "raw_lag": raw_lag,
        "ema_lag": ema_lag,
        "jitter_reduction_pct": ema_eval["jitter_reduction_pct"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Causal TCN on Session 3")
    parser.add_argument("--checkpoint", type=str, default="ml/models/checkpoints/best_model_6ch.pt")
    parser.add_argument("--config", type=str, default="ml/training/configs/tcn_db8.yaml")
    parser.add_argument("--alpha", type=float, default=0.20)
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.config, args.alpha)
