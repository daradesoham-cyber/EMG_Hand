# EMG-Controlled BiDexHand — Simulation (V2)

This merges the two earlier prototypes:

- **Hand:** the 16-servo **BiDexHand** from this repo's ROS 2/RViz demo (same URDF, meshes and the same 16→21 joint mapping as `hand_rviz/joint_state_forwarding_node.py`), now simulated in **MuJoCo**.
- **ML model:** the **CNN-LSTM** from the EMG Prosthetic Hand project (putEMG, 24 channels, 8 gestures, ONNX). On held-out subjects it reaches 84.5 % accuracy and 0.69 macro-F1 over 8 classes. The old V1 1D-CNN reached 51 % on 3 classes.
- **Microcontroller:** `../firmware/emg_hand_servo` (Arduino/ESP32 + PCA9685) receives `G<id>` over UART and drives the 16 servos with a speed limit and a watchdog.

```
EMG (24 ch, 1024 Hz) → HP 20 Hz + 50 Hz notches → 300 ms window / 50 ms step → CNN-LSTM (ONNX)
→ command filter → gesture → 16 servo angles / PWM µs → [UART → MCU → PCA9685 → servos] + MuJoCo
```

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r simulation/requirements.txt
export MUJOCO_GL=egl
.venv/bin/python simulation/app.py                        # GUI (ML + manual modes)
.venv/bin/python simulation/app.py --serial /dev/ttyUSB0  # also drive the real servo board
.venv/bin/python simulation/app.py --selftest 15          # scripted check, writes simulation/selftest/
.venv/bin/python simulation/make_figures.py [--data DIR]  # evaluation + report figures
```
`data/` ships one held-out recording (subject S23, session 1). You can add more putEMG `*_sequential.npz` exports to the folder.

| Path | Content |
|---|---|
| `app.py` | GUI, ML engine, serial link |
| `emg_sim/bidexhand.py` | URDF → MuJoCo, gesture table, servo model, PWM |
| `emg_sim/emg_runtime/` | causal filters and the real-time decoder |
| `model/` | `cnn_lstm.onnx`, `deploy.json`, benchmark |
| `assets/bidexhand/` | BiDexHand URDF + STL (MIT licence, Z. K. Weng) |

The legacy ROS 2/RViz instructions for V1 are still valid in `src/` (see `docs/ARCHITECTURE.md`).
Project report: [`docs/report/REPORT.md`](../docs/report/REPORT.md) (also as `.docx` and `.pdf`).
