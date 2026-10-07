# Embedded Deployment Status & Feasibility Analysis

## Current Deployment Status

> [!WARNING]
> **PROSPECTIVE TARGET ONLY — NOT PHYSICALLY VALIDATED ON HARDWARE**
> Embedded microcontroller deployment (e.g., Espressif ESP32-S3, ARM Cortex-M7/M55) is currently a **prospective target**.
> Physical flashing, firmware integration, and on-chip profiling have **NOT** been performed in this phase.
> All reported latencies reflect execution on a local x86_64 CPU and laptop GPU.

---

## Memory Footprint & SRAM Accounting

### Model Weight Storage

| Representation | Bytes per Parameter | Total Weight Memory | Storage Medium |
| :--- | :---: | :---: | :--- |
| **FP32 Baseline** | 4 bytes | $1,300.65\text{ kB}$ ($1.27\text{ MB}$) | Flash / ROM |
| **FP16 Half-Precision** | 2 bytes | $650.32\text{ kB}$ | Flash / ROM |
| **INT8 Quantized Target** | 1 byte | **$325.16\text{ kB}$** | Flash / Internal SRAM |

### Inference RAM Breakdown (Batch Size = 1, Ring Buffer = 256)

1. **Quantized Model Weights (INT8):** $\approx 325.2\text{ kB}$ (if mapped into SRAM rather than execute-in-place Flash).
2. **Intermediate Activation Double-Buffer:** $\approx 262.0\text{ kB}$ (maximum feature tensor across dilated layers: $128\text{ channels} \times 256\text{ samples} \times 4\text{ bytes} \times 2$).
3. **Runtime & Kernel Overhead:** $\approx 60.0\text{ kB}$ (ring buffer, IIR delay registers, EMA filter states, tensor descriptors).
4. **Total Estimated Inference SRAM:** **$\approx 647.2\text{ kB}$**.

---

## Hardware Constraint Evaluation (ESP32-S3)

- **Total Physical SRAM on ESP32-S3:** $512\text{ kB}$.
- **Typical Usable Application SRAM (after FreeRTOS, BLE/WiFi stacks):** $\approx 380\text{ kB}$.
- **Implication:**
  Because the total estimated inference requirement ($\approx 647\text{ kB}$) exceeds internal SRAM, **the full model cannot execute entirely within internal SRAM without memory optimization**.

### Required Engineering for Future Embedded Work:
1. **External PSRAM:** Use an ESP32-S3 module with 2 MB, 8 MB, or 16 MB Octal PSRAM (e.g., ESP32-S3-WROOM-1-N8R8).
2. **Streaming Activation Buffer:** Rather than storing full $256$-sample feature maps across all 6 residual blocks, stream receptive field updates sample-by-sample to reduce activation RAM to $< 40\text{ kB}$.
3. **Int8 Operator Quantization:** Use ESP-NN / ESP-DL vectorized SIMD assembly routines for quantized convolutions.

---

## Host Inference Feasibility (Validated)

Host CPU execution benchmarks demonstrate that algorithmic latency is not a bottleneck:
- **Mean Single-Window Latency:** **$2.23\text{ ms}$**
- **95th Percentile Latency (p95):** **$2.43\text{ ms}$**
- **Inference Interval Deadline:** $32.0\text{ ms}$ ($31.25\text{ Hz}$)
- **Real-Time Headroom:** **$+29.77\text{ ms}$ ($93.0\%$ idle headroom)**
