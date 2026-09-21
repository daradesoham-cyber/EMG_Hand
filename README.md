# EMG-Based Prosthetic/Robotic Hand — V1 (Research Prototype)

[![ROS 2](https://img.shields.io/badge/ROS%202-Lyrical-blue.svg)](https://docs.ros.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized%20CNN-blue.svg)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-lightgrey.svg)](https://scipy.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An open-source **V1 research prototype** interfacing surface electromyography (sEMG) gesture classification with a multi-articulated 16-servo anthropomorphic robotic hand (**BiDexHand**). The system integrates a real-time signal processing pipeline (DC removal, Butterworth bandpass filtering, and train-set normalization), a compact 1D Convolutional Neural Network (CNN) implemented with a vectorized pure NumPy forward pass, dual-threshold `REST` state detection, and a ROS 2 control node.

> [!IMPORTANT]
> **Prototype Classification**: This repository represents a **V1 development / research prototype**. It is **NOT** a certified clinical prosthesis or medical device. It is intended for academic research, benchmarking, educational robotics, and human-machine interface (HMI) prototyping.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Problem Statement](#2-problem-statement)
3. [EMG Signal Acquisition](#3-emg-signal-acquisition)
4. [Dataset Reference & Protocol](#4-dataset-reference--protocol)
5. [Signal Preprocessing Pipeline](#5-signal-preprocessing-pipeline)
6. [Feature Extraction & Energy Metrics](#6-feature-extraction--energy-metrics)
7. [Machine-Learning Pipeline (Compact 1D CNN)](#7-machine-learning-pipeline-compact-1d-cnn)
8. [Gesture Classification & REST Logic](#8-gesture-classification--rest-logic)
9. [Model Evaluation & Verified Results](#9-model-evaluation--verified-results)
10. [V1 Hand Kinematics & Simulation](#10-v1-hand-kinematics--simulation)
11. [Hardware & Software Requirements](#11-hardware--software-requirements)
12. [Installation & Setup](#12-installation--setup)
13. [Running the Code & Operating Modes](#13-running-the-code--operating-modes)
14. [Repository Structure](#14-repository-structure)
15. [Current Limitations](#15-current-limitations)
16. [Future Improvements](#16-future-improvements)

---

## 1. Project Objective

The primary objective of this project is to develop an accessible, computationally lightweight, and reproducible end-to-end pipeline connecting forearm surface electromyography (sEMG) signals to multi-finger robotic hand actuation. Specifically:
- Extract muscle activation signals corresponding to functional grasps (`PINCH`, `FIST`, `OPEN`).
- Ensure robust `REST` handling when the limb is relaxed or when classification confidence is low.
- Deploy real-time inference without heavy deep learning runtimes on embedded robotic hardware.
- Validate kinematic motion on a 16-degree-of-freedom (DoF) anthropomorphic robotic hand in ROS 2 / RViz.

---

## 2. Problem Statement

Upper-limb amputees and individuals with motor impairments face significant barriers in accessing intuitive prosthetic devices. While commercial myoelectric prostheses often rely on binary thresholding across one or two antagonist muscle pairs, modern research aims for proportional and multi-gesture dexterous control. However, deploying deep learning models on embedded prosthetic microcontrollers faces major hurdles:
1. **Computational Footprint**: Standard deep learning frameworks (TensorFlow, PyTorch) require heavy runtime environments and introduce unpredictable latency spikes.
2. **Cross-Subject Variability**: Muscle anatomy, subcutaneous adipose tissue, skin impedance, and electrode displacement cause substantial distribution shifts across individuals.
3. **The "Rest" Dilemma**: Public sEMG datasets frequently omit a dedicated `REST` class, causing naive classifiers to force an active gesture prediction even when the user is completely relaxed.

This V1 prototype directly addresses these challenges with a compact 1D CNN architecture, a standalone pure NumPy inference engine, and a dual-threshold inactivity/uncertainty safeguard.

---

## 3. EMG Signal Acquisition

Surface EMG measures the motor unit action potentials (MUAPs) generated during skeletal muscle contraction:
- **Sampling Frequency**: $1000\text{ Hz}$ (sufficient to capture the physiological sEMG frequency band up to $450\text{ Hz}$ per the Nyquist-Shannon sampling theorem).
- **Target Musculature**: Forearm flexor and extensor compartments.
- **Electrode Channels**: 4 differential channels selected from the standard 12-channel forearm ring (`[12, 11, 7, 2]`, 1-based index):
  - **Channel 12**: *Flexor Digitorum Superficialis* (FDS) — primary driver for finger flexion (FIST, PINCH).
  - **Channel 11**: *Flexor Carpi Radialis* (FCR) — wrist flexion and thumb opposition.
  - **Channel 7**: *Extensor Digitorum Communis* (EDC) — finger extension and hand opening (OPEN).
  - **Channel 2**: *Extensor Carpi Ulnaris* (ECU) — hand stabilization and wrist extension.

---

## 4. Dataset Reference & Protocol

The model was trained and benchmarked on the large-scale **Dataset of upper limb surface electromyogram (sEMG) signals from 91 participants for hand gesture recognition** (*Scientific Data*, Nature Publishing Group):
- **Participants**: 91 healthy, demographically diverse subjects (varying age, BMI, and skin hydration).
- **Total Trials**: 32,760 hand gesture trials (36 repetitions per gesture per subject).
- **Protocol**: Unconstrained natural contractions.
- **Cross-Subject Split**:
  - **Training Set**: 63 subjects (69.2%, 22,680 trials).
  - **Validation Set**: 13 subjects (14.3%, 4,680 trials).
  - **Test Set**: 15 unseen subjects (16.5%, 5,400 trials).

For dataset download details, see [data/README.md](data/README.md).

---

## 5. Signal Preprocessing Pipeline

Raw signals are conditioned in real-time through a 3-step digital signal processing sequence:
1. **Window Buffering**: Fixed $400\text{ ms}$ sliding window ($400\text{ samples}$ at $1000\text{ Hz}$).
2. **DC Offset Removal**: Subtracts channel-wise mean to eliminate electrode baseline offset:
   $$x_{\text{zeroed}}[n] = x[n] - \frac{1}{N}\sum_{k=0}^{N-1} x[k]$$
3. **Bandpass Filtering**: 4th-order Butterworth bandpass filter ($20\text{ Hz} - 450\text{ Hz}$) using Second-Order Sections (`sosfiltfilt`), eliminating motion artifacts ($<20\text{ Hz}$) and high-frequency noise ($>450\text{ Hz}$).
4. **Z-Score Normalization**: Scaled using training-set statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$) preserved in `train_mean.npy` and `train_std.npy`.

---

## 6. Feature Extraction & Energy Metrics

Before neural classification, the window energy is computed via the Root Mean Square (RMS):
$$\text{RMS} = \sqrt{\frac{1}{400 \cdot 4} \sum_{c=1}^4 \sum_{n=1}^{400} x_{c}[n]^2}$$
- The RMS metric serves as a direct measure of muscular effort.
- If $\text{RMS} < 0.01$, the signal is classified as `REST` due to muscular inactivity without executing the neural network.

---

## 7. Machine-Learning Pipeline (Compact 1D CNN)

The model is a compact 1D CNN designed for high temporal feature extraction with minimal parameter count:

```
Input (400 samples x 4 channels)
  │
  ├─► Conv1D (kernel=15, 32 filters, ReLU, same padding)
  ├─► Batch Normalization
  ├─► MaxPooling1D (pool=4, stride=4) ────────► (100 x 32)
  │
  ├─► Conv1D (kernel=11, 64 filters, ReLU, same padding)
  ├─► Batch Normalization
  ├─► MaxPooling1D (pool=4, stride=4) ────────► (25 x 64)
  │
  ├─► Conv1D (kernel=7, 128 filters, ReLU, same padding)
  ├─► Batch Normalization
  ├─► GlobalAveragePooling1D ─────────────────► (128)
  │
  ├─► Dense (64 units, ReLU)
  └─► Dense (3 units, Softmax) ───────────────► [P(PINCH), P(FIST), P(OPEN)]
```

### Pure NumPy Vectorized Inference (`v1_model.py`)
To eliminate heavyweight framework dependencies during ROS 2 execution, the forward pass is implemented entirely in NumPy using `sliding_window_view` and matrix multiplications. Inference latency is **$< 1.5\text{ ms}$** per window on standard CPU hardware.

---

## 8. Gesture Classification & REST Logic

Because the source dataset lacks an explicit `REST` class, the V1 classifier implements a **dual-threshold safety system**:
1. **Inactivity Threshold**: If $\text{RMS} < 0.01$, immediately output `REST` with $1.0$ confidence.
2. **Confidence Threshold**: If $\max(P) < 0.40$, the model rejects the prediction as ambiguous and defaults to `REST`.
3. **Active Classification**: If $\text{RMS} \ge 0.01$ and $\max(P) \ge 0.40$, the highest-probability gesture is selected:
   - `PINCH`: Index and thumb opposition for precision grasping.
   - `FIST`: Full finger flexion for power grasping.
   - `OPEN`: Full finger extension and abduction.

---

## 9. Model Evaluation & Verified Results

Evaluated on the **15 unseen test subjects** (zero-shot cross-subject benchmark):

| Evaluation Metric | Result | Description |
| :--- | :--- | :--- |
| **Model Architecture** | Compact 1D CNN | 91,363 trainable parameters ($\approx 365\text{ KB}$) |
| **Test Accuracy** | **51.05%** | Cross-subject generalization across 15 unseen subjects |
| **Balanced Accuracy** | **51.05%** | Equal performance across `PINCH`, `FIST`, `OPEN` |
| **Macro F1-Score** | **0.5067** | Balanced harmonic mean of precision and recall |
| **Inference Time** | **< 1.5 ms** | Pure NumPy CPU forward pass |

Confusion matrix and full training history are available in the [results/](results/) directory.

---

## 10. V1 Hand Kinematics & Simulation

The V1 prototype interfaces with the **BiDexHand** anthropomorphic robotic hand:
- **Actuation**: 16 independently controlled servo joints.
- **Joint Mapping**:
  - `0-4`: Finger MCP Abduction / Adduction.
  - `5`: Thumb MCP Abduction / Adduction.
  - `6-9`: Finger MCP Flexion / Extension.
  - `10`: Thumb MCP Flexion.
  - `11-14`: Finger PIP Flexion.
  - `15`: Thumb CMC Flexion.
- **ROS 2 Interface**: The `emg_classifier_node` translates classified gestures into 16-element joint angle arrays published to `/hand_servo_input` (`std_msgs/Float32MultiArray`).

For joint angle values and hardware diagrams, see [hardware/README.md](hardware/README.md).

---

## 11. Hardware & Software Requirements

### Hardware:
- **Robotic Hand**: BiDexHand (or simulated URDF in RViz).
- **EMG Acquisition**: 4-channel sEMG frontend amplifier / ADC (1000 Hz sampling rate, 12-bit).
- **Host System**: Standard PC / Single-Board Computer (Raspberry Pi 4/5, Jetson Nano).

### Software:
- **OS**: Linux (tested on Ubuntu 24.04 / 26.04).
- **ROS Distribution**: ROS 2 (Lyrical / Humble / Iron).
- **Python**: $\ge 3.10$.
- **Libraries**: `numpy>=1.22.0`, `scipy>=1.9.0`, `matplotlib>=3.5.0`, `rclpy`.

---

## 12. Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/daradesoham-cyber/EMG_Hand.git ~/EMG_Hand
cd ~/EMG_Hand
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Standalone Pipeline
Run the standalone verification script (requires only NumPy and SciPy, no ROS 2 required):
```bash
python3 scripts/evaluate_v1.py
```

### 4. Build ROS 2 Packages (Optional)
If using with ROS 2 and BiDexHand:
```bash
# Sourced from your ROS 2 workspace:
colcon build --symlink-install --packages-select emg_classifier emg_hand_demo
```

---

## 13. Running the Code & Operating Modes

The `emg_classifier_node` supports four distinct operational modes:

### Mode 1: ML Replay Mode (Default Demonstration)
Feeds sample offline EMG trials through the complete preprocessing and CNN classification pipeline:
```bash
source /opt/ros/lyrical/setup.bash
source ~/robotic_hand_ws/install/setup.bash

ros2 launch emg_classifier emg_hand_ml_demo.launch.py mode:=replay cycle_time:=3.0
```

### Mode 2: Live Monitor Dashboard
Launches a presentation-friendly live terminal dashboard monitoring predictions, confidence, and RMS:
```bash
ros2 run emg_classifier emg_prediction_monitor
```

### Mode 3: Manual Control Mode
Allows manual command of gestures via ROS 2 topic `/manual_gesture` or ROS parameters:
```bash
ros2 launch emg_classifier emg_hand_ml_demo.launch.py mode:=manual
```

### Mode 4: Live EMG Stream Mode
Subscribes to live multi-channel sEMG data on `/emg/raw`:
```bash
ros2 launch emg_classifier emg_hand_ml_demo.launch.py mode:=ml
```

---

## 14. Repository Structure

```
EMG_Hand/
├── README.md                 # Complete project documentation
├── LICENSE                   # Apache 2.0 open-source license
├── requirements.txt          # Python dependencies
│
├── data/
│   └── README.md             # Dataset documentation, channels, and split details
│
├── docs/
│   └── ARCHITECTURE.md       # Pipeline architecture, DSP, and CNN layers
│
├── hardware/
│   └── README.md             # 16-servo angle table and electrode placement
│
├── simulation/
│   └── README.md             # URDF, joint state forwarding, and RViz guide
│
├── results/
│   ├── confusion_matrix.png  # Confusion matrix across test subjects
│   ├── training_history.csv  # Loss and accuracy curves
│   ├── final_results.json    # Machine-readable benchmark metrics
│   ├── deployment_config.json# Runtime DSP parameters
│   └── subject_split.json    # Deterministic subject split IDs
│
├── scripts/
│   ├── evaluate_v1.py        # Standalone verification runner (no ROS required)
│   ├── run_replay_demo.sh    # Launch script for replay demo
│   ├── run_manual_demo.sh    # Launch script for manual demo
│   └── run_monitor.sh        # Launch script for prediction monitor
│
└── src/
    ├── emg_classifier/       # ROS 2 classification package
    │   ├── emg_classifier/   # v1_model.py, classifier_node, replay, monitor
    │   ├── models/           # Trained weights (.keras, .npz), sample trials
    │   ├── launch/           # emg_hand_ml_demo.launch.py
    │   ├── package.xml
    │   └── setup.py
    └── emg_hand_demo/        # ROS 2 demo package
        ├── emg_hand_demo/    # gesture_publisher.py
        ├── launch/           # emg_demo.launch.py
        ├── package.xml
        └── setup.py
```

---

## 15. Current Limitations

- **Cross-Subject Generalization**: The 51.05% zero-shot test accuracy on 15 unseen subjects reflects the inherent challenge of inter-individual sEMG variance without user-specific calibration.
- **Sensor Channel Subset**: The model utilizes 4 channels from the 12-channel dataset; physical electrode placement must match these muscle groups.
- **REST Calibration**: The RMS threshold ($0.01$) must be re-calibrated on physical hardware to account for individual skin impedance and amplifier noise floor.
- **Laboratory vs. Real World**: The training data was collected in seated, controlled conditions; arm posture shifts and fatigue effects are not modeled.

---

## 16. Future Improvements

1. **Few-Shot User Adaptation**: Implement fine-tuning on 3-5 calibration trials per user to elevate accuracy from $51\%$ to $>85\%$.
2. **Temporal Smoothing**: Add a majority voting filter or temporal window smoothing to prevent rapid state chatter.
3. **Physical Hardware Integration**: Connect a dedicated 4-channel sEMG analog frontend (e.g., ADS1299 or MyoWare 2.0) via microcontroller serial bridge.
4. **Proportional Grasp Control**: Map continuous RMS energy to grasp closing velocity/force rather than discrete positional states.

---

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
