#!/usr/bin/env python3
"""
Root execution entrypoint for evaluating on Session 3 held-out test set.
Usage:
    python evaluate.py --checkpoint results/best_model.pt
"""
import argparse
import yaml
import torch
from ml.dataset import NinaProDB8SessionDataset
from ml.windowing import WindowConfig
from ml.model import CausalTCN
from ml.evaluate import compute_regression_metrics, evaluate_ema_tradeoff, apply_ema_filter
from torch.utils.data import DataLoader
from ml.dataset import NumpyTensorDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Causal TCN on Session 3")
    parser.add_argument("--checkpoint", type=str, default="results/best_model_6ch.pt")
    parser.add_argument("--config", type=str, default="configs/tcn_db8.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    num_outputs = ckpt.get("config", {}).get("features", {}).get("num_outputs", 6)
    target_names = ckpt.get("target_names", [
        "index", "middle", "ring", "little", "thumb_flexion", "thumb_opposition"
    ][:num_outputs])

    m_cfg = ckpt.get("config", {}).get("model", {})
    hidden_channels = m_cfg.get("hidden_channels", [32, 64, 64, 128, 128, 128])
    dilations = m_cfg.get("dilations", [1, 2, 4, 8, 16, 32])

    dataset_handler = NinaProDB8SessionDataset(
        data_dir=cfg["dataset"].get("data_dir", "data/ninapro_db8"),
        subject_id=cfg["dataset"].get("subject", 1),
        num_outputs=num_outputs,
        target_names=target_names,
    )
    _, _, (test_X, test_Y) = dataset_handler.prepare_data()

    model = CausalTCN(
        in_channels=16,
        hidden_channels=hidden_channels,
        dilations=dilations,
        num_outputs=num_outputs,
        convs_per_block=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader = DataLoader(NumpyTensorDataset(test_X, test_Y), batch_size=64, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds = model(xb.to(device))
            all_preds.append(preds.cpu().numpy())
    y_pred = torch.cat([torch.from_numpy(p) for p in all_preds], dim=0).numpy()

    raw_metrics = compute_regression_metrics(test_Y, y_pred, target_names)
    print("Aggregate Raw Test Results:", raw_metrics["aggregate"])

    ema_eval = evaluate_ema_tradeoff(test_Y, y_pred, alpha=0.20, target_names=target_names)
    print("Aggregate EMA (alpha=0.20) Results:", ema_eval["ema_metrics"]["aggregate"])
    print(f"Jitter Reduction: {ema_eval['jitter_raw']:.4f} -> {ema_eval['jitter_ema']:.4f} ({ema_eval['jitter_reduction_pct']:.1f}%)")
