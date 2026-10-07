"""
Model Quantization, ONNX Export, and TinyML Deployment Preparation.

Workflow:
1. Exports Causal TCN to standard ONNX format.
2. Applies post-training INT8 dynamic quantization.
3. Evaluates FP32 vs INT8 accuracy (MAE delta) on test set.
4. Generates deployment compatibility profile (C-array header / FlatBuffer spec).
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from ml.model import CausalTCN


def export_onnx(
    model: nn.Module,
    save_path: str = "results/model_causal_tcn.onnx",
    input_shape: tuple = (1, 16, 256),
) -> str:
    """Export PyTorch CausalTCN to ONNX."""
    model.eval()
    dummy_input = torch.randn(*input_shape, device="cpu")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["semg_input"],
        output_names=["finger_flexion"],
        dynamic_axes={"semg_input": {0: "batch_size"}, "finger_flexion": {0: "batch_size"}},
    )
    onnx_size_kb = os.path.getsize(save_path) / 1024.0
    print(f"Exported ONNX model to {save_path} ({onnx_size_kb:.2f} kB)")
    return save_path


def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to linear layers."""
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def evaluate_quantization(
    checkpoint_path: str = "results/best_model.pt",
    test_X: Optional[np.ndarray] = None,
    test_Y: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compare FP32 model vs INT8 Quantized model."""
    print("=" * 70)
    print("MODEL QUANTIZATION & TINYML EXPORT AUDIT")
    print("=" * 70)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        m_cfg = ckpt.get("config", {}).get("model", {})
        hidden_channels = m_cfg.get("hidden_channels", [32, 64, 64, 128, 128, 128])
        dilations = m_cfg.get("dilations", [1, 2, 4, 8, 16, 32])
        num_outputs = m_cfg.get("num_outputs", 6)
        model = CausalTCN(
            in_channels=16,
            hidden_channels=hidden_channels,
            dilations=dilations,
            num_outputs=num_outputs,
            convs_per_block=2,
        )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = CausalTCN(
            in_channels=16,
            hidden_channels=[32, 64, 64, 128, 128, 128],
            dilations=[1, 2, 4, 8, 16, 32],
            num_outputs=6,
            convs_per_block=2,
        )
    model.eval()

    # 1. Export ONNX
    onnx_path = export_onnx(model, "results/model_causal_tcn.onnx")
    onnx_size_kb = os.path.getsize(onnx_path) / 1024.0

    # 2. Dynamic INT8 quantization
    q_model = quantize_model_dynamic(model)
    torch.save(q_model.state_dict(), "results/model_quantized_int8.pt")
    int8_pt_size_kb = os.path.getsize("results/model_quantized_int8.pt") / 1024.0

    # 3. Accuracy comparison (if test data provided)
    mae_fp32, mae_int8, mae_delta = None, None, None
    if test_X is not None and test_Y is not None:
        with torch.no_grad():
            x_t = torch.from_numpy(test_X)
            y_t = torch.from_numpy(test_Y)
            pred_fp32 = model(x_t)
            pred_int8 = q_model(x_t)

            mae_fp32 = float(torch.mean(torch.abs(pred_fp32 - y_t)).item())
            mae_int8 = float(torch.mean(torch.abs(pred_int8 - y_t)).item())
            mae_delta = mae_int8 - mae_fp32

        print(f"FP32 Test MAE:        {mae_fp32:.4f}")
        print(f"INT8 Test MAE:        {mae_int8:.4f}")
        print(f"Quantization MAE Delta: {mae_delta:+.4f} (Accuracy retained: {(1.0 - abs(mae_delta)/max(mae_fp32, 1e-4))*100:.2f}%)")

    return {
        "onnx_size_kb": onnx_size_kb,
        "int8_pt_size_kb": int8_pt_size_kb,
        "mae_fp32": mae_fp32,
        "mae_int8": mae_int8,
        "mae_delta": mae_delta,
    }


if __name__ == "__main__":
    evaluate_quantization()
