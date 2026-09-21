
EMG ROBOTIC HAND FINAL MODEL
============================

Model:
Compact 1D CNN

Dataset:
91 subjects
10 original gesture classes
32,760 total trials

Prototype classes:
PINCH
FIST
OPEN

REST:
REST is NOT present as a labeled class in the source dataset.
The ROS2 deployment should use an inactivity/RMS threshold
calibrated using the real EMG sensor.

Selected channels:
[12, 11, 7, 2]

Sampling rate:
1000 Hz

Input window:
400 ms

Preprocessing:
1. Resample to 1000 Hz
2. Remove DC
3. 20-450 Hz bandpass
4. 400 ms window
5. Training-set normalization
6. CNN inference

Model file:
model/emg_hand_cnn_best.keras

Label mapping:
Dataset label 1 -> PINCH
Dataset label 5 -> FIST
Dataset label 9 -> OPEN

ROS2 target:
Gesture -> 16 servo angles -> /hand_servo_input

IMPORTANT:
The model was trained using 4 channels from the 12-channel dataset.
The physical EMG sensor arrangement must eventually provide compatible
signals/features or the model must be retrained for the actual sensor channels.
