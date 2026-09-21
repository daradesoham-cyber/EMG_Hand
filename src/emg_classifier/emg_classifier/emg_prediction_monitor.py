#!/usr/bin/env python3
"""
EMG Prediction Monitor Node for BiDexHand.

Presentation-friendly real-time terminal dashboard displaying:
  - CNN Prediction
  - Hand Action (verified via /hand_servo_input)
  - Confidence
  - System Status
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray


class EMGPredictionMonitor(Node):
    """
    Subscribes to EMG predictions, probabilities, and servo commands,
    and displays a continuously updating terminal dashboard.
    Updates cleanly when state changes or transitions occur.
    """

    GESTURE_ACTIONS = {
        'PINCH': 'PINCH / PRECISION GRASP',
        'FIST': 'FIST / POWER GRASP',
        'OPEN': 'OPEN / FULL EXTENSION',
        'REST': 'REST / NEUTRAL',
    }

    # Reference 16-servo angles from emg_classifier_node
    GESTURE_ANGLES = {
        'REST': [0.0] * 16,
        'FIST': [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            50.0, 50.0, 50.0, 50.0, -30.0,
            80.0, 80.0, 80.0, 80.0, -45.0
        ],
        'OPEN': [
            10.0, -10.0, 0.0, 5.0, 15.0, -15.0,
            -15.0, -15.0, -15.0, -15.0, -15.0,
            -10.0, -10.0, -10.0, -10.0, 0.0
        ],
        'PINCH': [
            -15.0, 0.0, 0.0, 0.0, 0.0, 35.0,
            45.0, 0.0, 0.0, 0.0, -40.0,
            60.0, 0.0, 0.0, 0.0, -35.0
        ],
    }

    def __init__(self):
        super().__init__('emg_prediction_monitor')

        self.current_prediction = "WAITING..."
        self.verified_servo_gesture = None
        self.confidence_str = "WAITING..."
        self.probabilities = None
        self.status = "INITIALIZING"
        self.transition_timer = None
        self.render_timer = None

        # ROS2 Subscriptions
        self.sub_gesture = self.create_subscription(
            String,
            '/emg/predicted_gesture',
            self.gesture_callback,
            10
        )
        self.sub_probs = self.create_subscription(
            Float32MultiArray,
            '/emg/probabilities',
            self.probs_callback,
            10
        )
        self.sub_servo = self.create_subscription(
            Float32MultiArray,
            '/hand_servo_input',
            self.servo_callback,
            10
        )

        self.render_dashboard()

    def _match_servo_gesture(self, angles):
        if len(angles) < 16:
            return None
        for g_name, g_angles in self.GESTURE_ANGLES.items():
            if all(abs(float(a) - float(b)) < 1.0 for a, b in zip(angles[:16], g_angles)):
                return g_name
        return None

    def _schedule_render(self):
        """Debounce render by 30ms to allow all simultaneous ROS topics to arrive together."""
        if self.render_timer is not None:
            self.render_timer.cancel()
        self.render_timer = self.create_timer(0.03, self._do_debounced_render)

    def _do_debounced_render(self):
        if self.render_timer is not None:
            self.render_timer.cancel()
            self.render_timer = None
        self.render_dashboard()

    def gesture_callback(self, msg: String):
        gesture = msg.data.strip().upper()
        changed = (gesture != self.current_prediction)
        self.current_prediction = gesture

        if changed:
            self.status = "MOVING"
            # Reset transition timer to switch status to steady/idle after hand moves
            if self.transition_timer is not None:
                self.transition_timer.cancel()
            self.transition_timer = self.create_timer(1.8, self._transition_steady)

        self._schedule_render()

    def _transition_steady(self):
        if self.transition_timer is not None:
            self.transition_timer.cancel()
            self.transition_timer = None
        if self.current_prediction == 'REST':
            self.status = "IDLE / REST"
        else:
            self.status = "HOLDING POSE"
        self.render_dashboard()

    def probs_callback(self, msg: Float32MultiArray):
        # Format: [PINCH, FIST, OPEN]
        if len(msg.data) >= 3:
            p_pinch = float(msg.data[0])
            p_fist = float(msg.data[1])
            p_open = float(msg.data[2])
            self.probabilities = (p_pinch, p_fist, p_open)

            max_p = max(p_pinch, p_fist, p_open)
            if self.current_prediction == 'REST':
                if max_p < 0.01:
                    self.confidence_str = "100.0 % (INACTIVITY)"
                else:
                    self.confidence_str = f"{(1.0 - max_p) * 100:.1f} %"
            elif self.current_prediction == 'PINCH':
                self.confidence_str = f"{p_pinch * 100:.1f} %"
            elif self.current_prediction == 'FIST':
                self.confidence_str = f"{p_fist * 100:.1f} %"
            elif self.current_prediction == 'OPEN':
                self.confidence_str = f"{p_open * 100:.1f} %"
            else:
                self.confidence_str = f"{max_p * 100:.1f} %"
            self._schedule_render()

    def servo_callback(self, msg: Float32MultiArray):
        matched = self._match_servo_gesture(msg.data)
        if matched != self.verified_servo_gesture:
            self.verified_servo_gesture = matched
            self._schedule_render()

    def render_dashboard(self):
        # Determine Hand Action vs Hand Command based on verification
        if self.verified_servo_gesture:
            action_label = "HAND ACTION   "
            action_val = self.GESTURE_ACTIONS.get(self.verified_servo_gesture, self.verified_servo_gesture)
        elif self.current_prediction in self.GESTURE_ACTIONS:
            action_label = "HAND COMMAND  "
            action_val = self.current_prediction
        else:
            action_label = "HAND ACTION   "
            action_val = "STANDBY"

        lines = [
            "==================================================",
            "           EMG ROBOTIC HAND — V1",
            "==================================================",
            f"CNN PREDICTION : {self.current_prediction}",
            f"{action_label} : {action_val}",
            f"CONFIDENCE     : {self.confidence_str}",
            f"STATUS         : {self.status}",
        ]

        if self.probabilities is not None and any(p > 0.0 for p in self.probabilities):
            p_p, p_f, p_o = self.probabilities
            lines.extend([
                "--------------------------------------------------",
                f"PROBABILITIES  : PINCH: {p_p*100:4.1f}% | FIST: {p_f*100:4.1f}% | OPEN: {p_o*100:4.1f}%",
            ])

        lines.extend([
            "--------------------------------------------------",
            "Pipeline: EMG -> CNN -> ROS 2 -> BiDexHand",
            "==================================================",
        ])

        dashboard_str = "\033[H\033[2J" + "\n".join(lines) + "\n"
        sys.stdout.write(dashboard_str)
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = EMGPredictionMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
