# Environment & Hardware Version Specification

## Host Operating System

- **Distribution:** Ubuntu 26.04.1 LTS (Resolute Raccoon)
- **Kernel:** Linux `7.0.0-38-generic` SMP PREEMPT_DYNAMIC
- **Architecture:** x86_64

## Python & Toolchain

- **Python Runtime:** Python 3.14.4 (main, Aug 20 2026, 10:41:58) [GCC 15.2.0]
- **PyTorch:** `2.14.0+cu130`
- **CUDA Toolkit:** CUDA 13.0
- **GPU Device:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Host CPU:** AMD Ryzen 7 5800H / Intel Core Compatible x86_64

## Key Package Versions Audited

| Package | Audited Version | Role |
| :--- | :--- | :--- |
| `torch` | `2.14.0` | Deep learning, automatic differentiation, TCN/GRU execution |
| `torchvision` | `0.29.0` | Vision / tensor utilities |
| `numpy` | `2.5.3` | Vectorized numerical operations, ring buffer management |
| `scipy` | `1.18.1` | Causal Butterworth IIR & Notch filtering (`lfilter`), cross-correlation |
| `pandas` | `3.0.6` | Dataset table ingestion, metric recording |
| `pyyaml` | `6.0.3` | Experiment configuration management |
| `scikit-learn` | `1.9.1` | Baseline linear regression, metric scoring |
| `onnx` | `1.23.2` | ONNX graph verification, export, checker |
| `onnxruntime` | `1.30.0` | Standard CPU ONNX inference engine |
| `matplotlib` | `3.10.0` | Visualization generation |
| `h5py` | `3.15.1` | HDF5 / MATLAB .mat file structure support |

---

## Reproducibility Recommendations

- Python $\ge 3.10$ and PyTorch $\ge 2.1.0$ are compatible.
- The pipeline utilizes standard `torch.nn.Conv1d`, `torch.nn.BatchNorm1d`, and `scipy.signal.lfilter`, which maintain backward and forward compatibility across PyTorch 2.x and SciPy 1.x.
