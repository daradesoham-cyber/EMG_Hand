#!/usr/bin/env python3
"""
EMG Replay Publisher Node.

Publishes prerecorded / synthetic raw EMG trials to /emg/raw
to demonstrate live ML inference in ROS 2.
"""

import os
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Float32MultiArray
import numpy as np


class EMGReplayPublisherNode(Node):
    def __init__(self):
        super().__init__('emg_replay_publisher_node')

        default_replay_path = os.path.expanduser(
            '~/robotic_hand_ws/src/emg_classifier/models/sample_emg_trials.npz'
        )

        self.declare_parameter(
            'replay_file',
            default_replay_path,
            ParameterDescriptor(description="Path to .npz file containing EMG replay trials")
        )
        self.declare_parameter(
            'publish_rate_sec',
            3.5,
            ParameterDescriptor(description="Time interval between replayed trials (seconds)")
        )

        self.replay_file = os.path.expanduser(str(self.get_parameter('replay_file').value))
        self.publish_rate_sec = float(self.get_parameter('publish_rate_sec').value)

        self.publisher_ = self.create_publisher(Float32MultiArray, '/emg/raw', 10)

        # Load trials
        self.trials = {}
        self.trial_keys = []
        self._load_trials()

        self.index = 0
        self.timer = self.create_timer(self.publish_rate_sec, self.timer_callback)

        self.get_logger().info(f"EMG Replay Publisher started.")
        self.get_logger().info(f"Replay file: {self.replay_file}")
        self.get_logger().info(f"Loaded sequences: {self.trial_keys}")
        self.get_logger().info(f"Publishing to /emg/raw every {self.publish_rate_sec}s")

    def _load_trials(self):
        if not os.path.exists(self.replay_file):
            self.get_logger().error(f"Replay file not found: {self.replay_file}")
            return

        data = np.load(self.replay_file)
        # Sequence in presentation order
        preferred_order = ['REST', 'PINCH', 'FIST', 'OPEN']
        self.trial_keys = [k for k in preferred_order if k in data]
        if not self.trial_keys:
            self.trial_keys = list(data.keys())

        for k in self.trial_keys:
            self.trials[k] = data[k].astype(np.float32)

    def timer_callback(self):
        if not self.trial_keys:
            return

        key = self.trial_keys[self.index]
        arr = self.trials[key]
        self.index = (self.index + 1) % len(self.trial_keys)

        msg = Float32MultiArray()
        msg.data = [float(v) for v in arr.flatten()]
        self.publisher_.publish(msg)

        rms = float(np.sqrt(np.mean(arr ** 2)))
        self.get_logger().info(
            f"[Replay Publisher] Published '{key}' raw EMG trial to /emg/raw (shape {arr.shape}, RMS {rms:.5f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = EMGReplayPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
