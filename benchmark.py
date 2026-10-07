#!/usr/bin/env python3
"""
Root execution entrypoint for latency and model size benchmark.
Usage:
    python benchmark.py
"""
from ml.benchmark import run_full_benchmark

if __name__ == "__main__":
    run_full_benchmark("results/best_model.pt")
