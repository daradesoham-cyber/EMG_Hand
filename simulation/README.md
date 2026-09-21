# BiDexHand Simulation and RViz Visualization

This directory documents the 3D kinematic simulation and visualization setup for the V1 Robotic/Prosthetic Hand.

---

## 1. Kinematic Model & URDF

The system uses the **BiDexHand** multi-articulated anthropomorphic robotic hand model:
- **Total Degrees of Freedom (DoF)**: 16 independently actuated servo joints.
- **Fingers (5)**:
  - **Thumb**: CMC Abduction (`tca_joint`), CMC Flexion (`tcf_joint`), MCP Abduction (`tma_joint`), MCP Flexion (`tmf_joint`), Interphalangeal/Distal Flexion.
  - **Index**: MCP Abduction (`ima_joint`), MCP Flexion (`imf_joint`), PIP Flexion (`ipf_joint`), DIP Flexion.
  - **Middle**: MCP Abduction (`mma_joint`), MCP Flexion (`mmf_joint`), PIP Flexion (`mpf_joint`), DIP Flexion.
  - **Ring**: MCP Abduction (`rma_joint`), MCP Flexion (`rmf_joint`), PIP Flexion (`rpf_joint`), DIP Flexion.
  - **Pinky**: MCP Abduction (`pma_joint`), MCP Flexion (`pmf_joint`), PIP Flexion (`ppf_joint`), DIP Flexion.

---

## 2. ROS 2 Control & Forwarding Architecture

```
+-----------------------------------+
|     EMG Classifier Node           |
| (Classifies gesture -> 16 angles) |
+-----------------+-----------------+
                  |
                  | /hand_servo_input (std_msgs/Float32MultiArray, 16 angles in deg)
                  v
+-----------------+-----------------+
|   Joint State Forwarding Node     |
| (Converts deg -> rad, maps joints)|
+-----------------+-----------------+
                  |
                  | /hand/joint_states (sensor_msgs/JointState)
                  v
+-----------------+-----------------+
|      Robot State Publisher        |
|  (Calculates 3D TF transforms)    |
+-----------------+-----------------+
                  |
                  | /tf, /tf_static
                  v
+-----------------+-----------------+
|             RViz2                 |
|   (Interactive 3D Hand Model)     |
+-----------------------------------+
```

---

## 3. Topic Interface: `/hand_servo_input`

- **Type**: `std_msgs/msg/Float32MultiArray`
- **Data Array**: 16 floating-point values representing servo target angles in **degrees**.
- **Index Ordering**:
  - `[0]`: Thumb CMC Abduction (`tca_joint`)
  - `[1]`: Index MCP Abduction (`ima_joint`)
  - `[2]`: Middle MCP Abduction (`mma_joint`)
  - `[3]`: Ring MCP Abduction (`rma_joint`)
  - `[4]`: Pinky MCP Abduction (`pma_joint`)
  - `[5]`: Thumb MCP Abduction (`tma_joint`)
  - `[6]`: Index MCP Flexion (`imf_joint`)
  - `[7]`: Middle MCP Flexion (`mmf_joint`)
  - `[8]`: Ring MCP Flexion (`rmf_joint`)
  - `[9]`: Pinky MCP Flexion (`pmf_joint`)
  - `[10]`: Thumb MCP Flexion (`tmf_joint`)
  - `[11]`: Index PIP Flexion (`ipf_joint`)
  - `[12]`: Middle PIP Flexion (`mpf_joint`)
  - `[13]`: Ring PIP Flexion (`rpf_joint`)
  - `[14]`: Pinky PIP Flexion (`ppf_joint`)
  - `[15]`: Thumb CMC Flexion (`tcf_joint`)

---

## 4. How to Launch the Simulation

Ensure `robotic_hand_ws` is built and sourced:

```bash
source /opt/ros/lyrical/setup.bash
source ~/robotic_hand_ws/install/setup.bash

# Launch full simulation with ML replay demo
ros2 launch emg_classifier emg_hand_ml_demo.launch.py mode:=replay cycle_time:=3.0
```
