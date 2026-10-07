"""
ONNX Model Export Script for Causal TCN.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from ml.models.tcn.model import CausalTCN


def export_model_to_onnx(
    checkpoint_path: str = "ml/models/checkpoints/best_model_6ch.pt",
    output_onnx_path: str = "ml/models/checkpoints/model_causal_tcn.onnx",
    opset_version: int = 14,
):
    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m_cfg = ckpt.get("config", {}).get("model", {})

    model = CausalTCN(
        in_channels=m_cfg.get("in_channels", 16),
        hidden_channels=m_cfg.get("hidden_channels", [32, 64, 64, 128, 128, 128]),
        dilations=m_cfg.get("dilations", [1, 2, 4, 8, 16, 32]),
        kernel_size=m_cfg.get("kernel_size", 3),
        num_outputs=len(ckpt.get("target_names", [0] * 6)),
        convs_per_block=2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 16, 256)
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    print(f"Exporting ONNX graph to: {output_onnx_path} (opset {opset_version})")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["semg_input"],
        output_names=["hasel_actuator_commands"],
        dynamic_axes={
            "semg_input": {0: "batch_size"},
            "hasel_actuator_commands": {0: "batch_size"},
        },
    )
    print("ONNX export completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="ml/models/checkpoints/best_model_6ch.pt")
    parser.add_argument("--output", type=str, default="ml/models/checkpoints/model_causal_tcn.onnx")
    args = parser.parse_args()

    export_model_to_onnx(args.checkpoint, args.output)
