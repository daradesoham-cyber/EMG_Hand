"""
Hardware & Latency Benchmarking Module for Causal TCN.

Measures:
- Parameter count
- FP32, FP16, and INT8 model sizes (kB)
- Empirical CPU inference latency (mean, std, p50, p95, p99 across 500 runs)
- Empirical GPU inference latency (if CUDA available)
- Real-time throughput (predictions/sec)
- Comparison against the 32.0 ms deadline (31.25 Hz)
"""

import os
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch

from ml.model import CausalTCN
from ml import plots


def benchmark_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """Calculate exact parameter counts and memory footprint."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Size in bytes
    fp32_bytes = num_params * 4
    fp16_bytes = num_params * 2
    int8_bytes = num_params * 1

    return {
        "parameters": num_params,
        "fp32_kb": fp32_bytes / 1024.0,
        "fp16_kb": fp16_bytes / 1024.0,
        "int8_kb": int8_bytes / 1024.0,
    }


def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple = (1, 16, 256),
    num_runs: int = 500,
    warmup: int = 50,
) -> np.ndarray:
    """Measure single-window inference latency in milliseconds."""
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    return np.array(latencies, dtype=np.float32)


def run_full_benchmark(
    checkpoint_path: str = "results/best_model.pt",
    num_runs: int = 500,
) -> Dict[str, Any]:
    """Run full CPU & GPU benchmarking suite."""
    print("=" * 70)
    print("EMPIRICAL MODEL SIZE & INFERENCE LATENCY BENCHMARK")
    print("=" * 70)

    # 1. Instantiate model from checkpoint or default Model C
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
        print(f"Loaded weights from {checkpoint_path} (num_outputs={num_outputs}, dilations={dilations})")
    else:
        model = CausalTCN(
            in_channels=16,
            hidden_channels=[32, 64, 64, 128, 128, 128],
            dilations=[1, 2, 4, 8, 16, 32],
            num_outputs=6,
            convs_per_block=2,
        )

    # 2. Size measurements
    size_info = benchmark_model_size(model)
    print("\n--- Memory Footprint ---")
    print(f"Total Trainable Parameters: {size_info['parameters']:,}")
    print(f"FP32 Model Size: {size_info['fp32_kb']:.2f} kB")
    print(f"FP16 Model Size: {size_info['fp16_kb']:.2f} kB")
    print(f"INT8 Quantized Size Target: {size_info['int8_kb']:.2f} kB")

    # 3. CPU Latency
    print("\nBenchmarking CPU Latency (500 runs)...")
    cpu_device = torch.device("cpu")
    cpu_latencies = measure_latency(model, cpu_device, num_runs=num_runs)

    cpu_mean = float(np.mean(cpu_latencies))
    cpu_std = float(np.std(cpu_latencies))
    cpu_p50 = float(np.percentile(cpu_latencies, 50))
    cpu_p95 = float(np.percentile(cpu_latencies, 95))
    cpu_p99 = float(np.percentile(cpu_latencies, 99))
    cpu_fps = 1000.0 / cpu_mean

    print(f"CPU Latency: mean={cpu_mean:.2f} ms ± {cpu_std:.2f} ms | p50={cpu_p50:.2f} ms | p95={cpu_p95:.2f} ms | p99={cpu_p99:.2f} ms")
    print(f"CPU Throughput: {cpu_fps:.1f} inferences/sec (Budget: 31.25 Hz -> 32.0 ms deadline)")

    # 4. GPU Latency
    gpu_latencies = None
    gpu_mean, gpu_std, gpu_p50, gpu_p95, gpu_p99, gpu_fps = (None,) * 6
    if torch.cuda.is_available():
        print("\nBenchmarking GPU Latency (500 runs)...")
        gpu_device = torch.device("cuda")
        gpu_latencies = measure_latency(model, gpu_device, num_runs=num_runs)
        gpu_mean = float(np.mean(gpu_latencies))
        gpu_std = float(np.std(gpu_latencies))
        gpu_p50 = float(np.percentile(gpu_latencies, 50))
        gpu_p95 = float(np.percentile(gpu_latencies, 95))
        gpu_p99 = float(np.percentile(gpu_latencies, 99))
        gpu_fps = 1000.0 / gpu_mean
        print(f"GPU ({torch.cuda.get_device_name(0)}) Latency: mean={gpu_mean:.2f} ms ± {gpu_std:.2f} ms | p50={gpu_p50:.2f} ms | p95={gpu_p95:.2f} ms")
        print(f"GPU Throughput: {gpu_fps:.1f} inferences/sec")

    # 5. Export results
    records = [
        {"device": "CPU", "mean_ms": cpu_mean, "std_ms": cpu_std, "p50_ms": cpu_p50, "p95_ms": cpu_p95, "p99_ms": cpu_p99, "throughput_fps": cpu_fps, "realtime_headroom_ms": 32.0 - cpu_mean},
    ]
    if gpu_mean is not None:
        records.append(
            {"device": "GPU", "mean_ms": gpu_mean, "std_ms": gpu_std, "p50_ms": gpu_p50, "p95_ms": gpu_p95, "p99_ms": gpu_p99, "throughput_fps": gpu_fps, "realtime_headroom_ms": 32.0 - gpu_mean}
        )

    bench_df = pd.DataFrame(records)
    bench_df.to_csv("results/inference_benchmark.csv", index=False)

    # 6. Plot latency
    plots.plot_inference_latency(cpu_latencies, gpu_latencies, save_path="results/plots/7_prediction_latency.png")

    return {
        "size": size_info,
        "cpu_latency": {"mean": cpu_mean, "p50": cpu_p50, "p95": cpu_p95, "p99": cpu_p99},
        "gpu_latency": {"mean": gpu_mean, "p50": gpu_p50, "p95": gpu_p95} if gpu_mean else None,
    }


if __name__ == "__main__":
    run_full_benchmark()
