"""
ONNX Graph Verification and PyTorch Numerical Parity Checker.
"""

import os
import sys
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from ml.models.tcn.model import CausalTCN


def verify_onnx_model(
    checkpoint_path: str = "ml/models/checkpoints/best_model_6ch.pt",
    onnx_path: str = "ml/models/checkpoints/model_causal_tcn.onnx",
    tolerance: float = 1e-5,
):
    print("=" * 70)
    print("ONNX MODEL STRUCTURAL AND NUMERICAL VERIFICATION")
    print("=" * 70)

    # 1. Structural check
    print(f"Loading ONNX model: {onnx_path}")
    onnx_proto = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_proto)
    print("✓ Structural validation passed: Valid ONNX computational graph.")

    # 2. Reconstruct PyTorch model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m_cfg = ckpt.get("config", {}).get("model", {})
    num_outputs = len(ckpt.get("target_names", [0] * 6))

    pt_model = CausalTCN(
        in_channels=m_cfg.get("in_channels", 16),
        hidden_channels=m_cfg.get("hidden_channels", [32, 64, 64, 128, 128, 128]),
        dilations=m_cfg.get("dilations", [1, 2, 4, 8, 16, 32]),
        kernel_size=m_cfg.get("kernel_size", 3),
        num_outputs=num_outputs,
        convs_per_block=2,
    )
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()

    # 3. ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    # 4. Deterministic comparison
    np.random.seed(42)
    dummy_input = np.random.randn(10, 16, 256).astype(np.float32)

    with torch.no_grad():
        pt_preds = pt_model(torch.from_numpy(dummy_input)).numpy()

    ort_preds = ort_session.run(None, {input_name: dummy_input})[0]

    max_diff = float(np.max(np.abs(pt_preds - ort_preds)))
    mean_diff = float(np.mean(np.abs(pt_preds - ort_preds)))

    print(f"Max Absolute Difference:  {max_diff:.8e}")
    print(f"Mean Absolute Difference: {mean_diff:.8e}")

    if max_diff <= tolerance:
        print(f"✓ Numerical agreement test PASSED (within tolerance {tolerance}).")
    else:
        print(f"✗ Numerical discrepancy exceeded tolerance {tolerance}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ONNX model against PyTorch")
    parser.add_argument("--checkpoint", type=str, default="ml/models/checkpoints/best_model_6ch.pt")
    parser.add_argument("--onnx", type=str, default="ml/models/checkpoints/model_causal_tcn.onnx")
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    verify_onnx_model(args.checkpoint, args.onnx, args.tol)
