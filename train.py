#!/usr/bin/env python3
"""
Root execution entrypoint for NinaPro DB8 Causal TCN Training.
Usage:
    python train.py --config configs/tcn_db8.yaml
"""
import sys
import argparse
from ml.train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Causal TCN on NinaPro DB8")
    parser.add_argument("--config", type=str, default="configs/tcn_db8.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
