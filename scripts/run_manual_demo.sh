#!/usr/bin/env bash
# Launch V1 EMG Manual Demonstration
set -e

if [ -f "/opt/ros/lyrical/setup.bash" ]; then
    source /opt/ros/lyrical/setup.bash
fi

if [ -f "$HOME/robotic_hand_ws/install/setup.bash" ]; then
    source "$HOME/robotic_hand_ws/install/setup.bash"
fi

ros2 launch emg_classifier emg_hand_ml_demo.launch.py mode:=manual
