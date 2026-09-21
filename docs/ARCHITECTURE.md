# EMG Hand V1 Architecture & Signal Pipeline

This document details the engineering architecture, signal preprocessing, deep learning model, and ROS 2 integration for the **EMG-Based Prosthetic/Robotic Hand (V1 Prototype)**.

---

## 1. System Pipeline Overview

```
[ sEMG Electrodes (4 Channels @ 1000 Hz) ]
                   |
                   v
    +------------------------------+
    |      Window Buffering        |  400 ms sliding window (400 samples)
    +--------------+---------------+
                   |
                   v
    +------------------------------+
    |       DC Offset Removal      |  Subtract channel-wise mean
    +--------------+---------------+
                   |
                   v
    +------------------------------+
    |  Butterworth Bandpass Filter |  20 - 450 Hz (4th-order SOS)
    +--------------+---------------+
                   |
                   v
    +------------------------------+
    |    Train-Set Normalization   |  Z-score using training mean and std
    +--------------+---------------+
                   |
                   v
    +------------------------------+
    |      RMS Energy Calculation  |  Evaluate forearm muscular quiescence
    +--------------+---------------+
                   |
         +---------+---------+
         | RMS < threshold   | RMS >= threshold
         v                   v
+------------------+  +--------------------------------+
| Force REST State |  |      1D CNN Forward Pass       |
+------------------+  | (Conv1D -> BN -> Pool -> Dense)|
                      +---------------+----------------+
                                      |
                           +----------+----------+
                           | Softmax < conf_thresh| Softmax >= conf_thresh
                           v                     v
                  +------------------+  +------------------------+
                  |  Fallback REST   |  | Target: PINCH|FIST|OPEN|
                  +--------+---------+  +-----------+------------+
                           |                        |
                           +----------+-------------+
                                      |
                                      v
                      +-------------------------------+
                      | 16-Servo Angle Mapping Table  |
                      |   (BiDexHand joint kinematics)|
                      +---------------+---------------+
                                      |
                                      | /hand_servo_input (degrees)
                                      v
                      +-------------------------------+
                      |  ROS 2 Joint State Publisher  |
                      |    & RViz / Physical Servos   |
                      +-------------------------------+
```

---

## 2. Signal Preprocessing

Raw surface electromyography (sEMG) signals contain low-frequency motion artifacts, baseline drift, and high-frequency noise:
1. **Windowing**: Samples are buffered into $400\text{ ms}$ windows ($400\text{ samples}$ at $1000\text{ Hz}$).
2. **DC Removal**: Baseline offset is removed by subtracting the mean of the window for each channel:
   $$x_{\text{zeroed}}[n] = x[n] - \frac{1}{N}\sum_{k=0}^{N-1} x[k]$$
3. **Bandpass Filtering**: A 4th-order Butterworth bandpass filter ($20\text{ Hz} - 450\text{ Hz}$) implemented using Second-Order Sections (`scipy.signal.sosfiltfilt`) isolates the frequency spectrum of skeletal muscle action potentials while attenuating motion artifacts ($<20\text{ Hz}$) and electromagnetic line interference.
4. **Z-Score Normalization**: Inputs are scaled using the global mean and standard deviation computed strictly from the training dataset:
   $$x_{\text{norm}}[n] = \frac{x_{\text{filt}}[n] - \mu_{\text{train}}}{\sigma_{\text{train}} + 10^{-8}}$$

---

## 3. Compact 1D CNN Architecture

The model is structured for high temporal feature extraction with minimal parameter overhead, enabling real-time embedded deployment without a GPU:

| Layer Index | Layer Type | Kernel / Stride | Output Shape | Parameters | Activation / Function |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | Input Tensor | - | $(400, 4)$ | 0 | Raw normalized sEMG window |
| **1** | Conv1D | $15 \times 4$, 32 filters | $(400, 32)$ | 1,952 | ReLU, 'same' padding |
| **2** | Batch Normalization | - | $(400, 32)$ | 128 | $\epsilon = 10^{-3}$ |
| **3** | MaxPooling1D | Pool=4, Stride=4 | $(100, 32)$ | 0 | Downsampling |
| **4** | Conv1D | $11 \times 32$, 64 filters | $(100, 64)$ | 22,592 | ReLU, 'same' padding |
| **5** | Batch Normalization | - | $(100, 64)$ | 256 | $\epsilon = 10^{-3}$ |
| **6** | MaxPooling1D | Pool=4, Stride=4 | $(25, 64)$ | 0 | Downsampling |
| **7** | Conv1D | $7 \times 64$, 128 filters | $(25, 128)$ | 57,472 | ReLU, 'same' padding |
| **8** | Batch Normalization | - | $(25, 128)$ | 512 | $\epsilon = 10^{-3}$ |
| **9** | GlobalAveragePooling1D | - | $(128)$ | 0 | Temporal average |
| **10**| Dense (Fully Connected) | - | $(64)$ | 8,256 | ReLU |
| **11**| Dense (Output) | - | $(3)$ | 195 | Softmax (`PINCH`, `FIST`, `OPEN`) |

**Total Parameters**: $\approx 91,363$ ($\approx 365\text{ KB}$).

---

## 4. Pure NumPy Inference Engine (`v1_model.py`)

In production ROS 2 environments, loading heavyweight deep learning runtimes (e.g. full TensorFlow/PyTorch) introduces large startup latencies and memory overhead:
- The trained weights and batch normalization parameters were exported into `emg_hand_cnn_best_weights.npz`.
- `V1EMGModel` implements a vectorized forward pass using standard NumPy matrix multiplication (`windows_flat @ W_flat + b`).
- **Inference Latency**: **$< 1.5\text{ ms}$** per window on standard CPU.
- **Zero Heavy Dependencies**: Operates with only `numpy` and `scipy`.

---

## 5. Dual-Threshold REST State Logic

The source 91-subject dataset does not include a labeled `REST` class. To safely control a prosthetic device, two safeguards are implemented:
1. **Inactivity Detection (Energy Threshold)**:
   $$\text{RMS} = \sqrt{\frac{1}{N \cdot C} \sum_{c=1}^C \sum_{n=1}^N x_{c}[n]^2}$$
   If $\text{RMS} < 0.01$, muscles are deemed inactive, and the state defaults to `REST`.
2. **Uncertainty Rejection (Confidence Threshold)**:
   If $\max(P_{\text{PINCH}}, P_{\text{FIST}}, P_{\text{OPEN}}) < 0.40$, the prediction is deemed uncertain, defaulting to `REST` to prevent erroneous grasping motions.
