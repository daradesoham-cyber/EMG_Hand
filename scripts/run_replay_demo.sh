#!/usr/bin/env bash
# Launch V1 EMG Replay Demonstration (BiDexHand RViz + Replay Publisher + Classifier)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Starting EMG Hand V1 ML Replay Demonstration"
echo "=================================================="

# Source ROS 2 and workspace if available
if [ -f "/opt/ros/lyrical/setup.bash" ]; then
    source /opt/ros/lyrical/setup.bash
fi

if [ -f "$HOME/robotic_hand_ws/install/setup.bash" ]; then
    source "$HOME/robotic_hand_ws/install/setup.bash"
elif [ -f "$WORKSPACE_ROOT/install/setup.bash" ]; then
    source "$WORKSPACE_ROOT/install/setup.bash"
fi

ros2 launch emg_classifier emg_hand_ml_demo.launch.py mode:=replay cycle_time:=3.0
