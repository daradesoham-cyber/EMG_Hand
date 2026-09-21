#!/usr/bin/env python3
"""
EMG Gesture Classifier Node for BiDexHand.

Pipeline Architecture:
  EMG input -> preprocess() -> V1 CNN inference -> gesture_to_servo_angles() -> publish()

Supported Gestures:
  REST, FIST, OPEN, PINCH

Supported Modes:
  'test'   : Deterministically cycles through REST, FIST, OPEN, PINCH every N seconds.
  'manual' : Holds and responds to manual commands via services, /manual_gesture topic, or ROS parameters.
  'ml'     : Live V1 EMG CNN classification from /emg/raw topic.
  'replay' : Prerecorded / offline EMG trials fed through V1 preprocessing and CNN inference.
"""

import json
import os
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory

from emg_classifier.v1_model import V1EMGModel


class EMGClassifierNode(Node):
    """
    ROS2 Node that interfaces EMG gesture classification to BiDexHand /hand_servo_input.
    """

    # 16-Servo Mapping Reference (BiDexHand hardware / joint_state_forwarding_node):
    # Index 0:  Thumb CMC Abduction  (tca_joint)
    # Index 1:  Index MCP Abduction  (ima_joint)
    # Index 2:  Middle MCP Abduction (mma_joint)
    # Index 3:  Ring MCP Abduction   (rma_joint)
    # Index 4:  Pinky MCP Abduction  (pma_joint)
    # Index 5:  Thumb MCP Abduction  (tma_joint)
    # Index 6:  Index MCP Flexion    (imf_joint)
    # Index 7:  Middle MCP Flexion   (mmf_joint)
    # Index 8:  Ring MCP Flexion     (rmf_joint)
    # Index 9:  Pinky MCP Flexion    (pmf_joint)
    # Index 10: Thumb MCP Flexion    (tmf_joint)
    # Index 11: Index PIP Flexion    (ipf_joint)
    # Index 12: Middle PIP Flexion   (mpf_joint)
    # Index 13: Ring PIP Flexion     (rpf_joint)
    # Index 14: Pinky PIP Flexion    (ppf_joint)
    # Index 15: Thumb CMC Flexion    (tcf_joint)

    # Verified safe servo angles (degrees) for each gesture
    GESTURE_ANGLES = {
        'REST': [
            0.0,   # 0:  thumb cmc abd
            0.0,   # 1:  index mcp abd
            0.0,   # 2:  middle mcp abd
            0.0,   # 3:  ring mcp abd
            0.0,   # 4:  pinky mcp abd
            0.0,   # 5:  thumb mcp abd
            0.0,   # 6:  index mcp flex
            0.0,   # 7:  middle mcp flex
            0.0,   # 8:  ring mcp flex
            0.0,   # 9:  pinky mcp flex
            0.0,   # 10: thumb mcp flex
            0.0,   # 11: index pip flex
            0.0,   # 12: middle pip flex
            0.0,   # 13: ring pip flex
            0.0,   # 14: pinky pip flex
            0.0,   # 15: thumb cmc flex
        ],
        'FIST': [
            0.0,    # 0:  thumb cmc abd
            0.0,    # 1:  index mcp abd
            0.0,    # 2:  middle mcp abd
            0.0,    # 3:  ring mcp abd
            0.0,    # 4:  pinky mcp abd
            0.0,    # 5:  thumb mcp abd
            50.0,   # 6:  index mcp flex
            50.0,   # 7:  middle mcp flex
            50.0,   # 8:  ring mcp flex
            50.0,   # 9:  pinky mcp flex
            -30.0,  # 10: thumb mcp flex
            80.0,   # 11: index pip flex
            80.0,   # 12: middle pip flex
            80.0,   # 13: ring pip flex
            80.0,   # 14: pinky pip flex
            -45.0,  # 15: thumb cmc flex
        ],
        'OPEN': [
            10.0,   # 0:  thumb cmc abd
            -10.0,  # 1:  index mcp abd (spread outward)
            0.0,    # 2:  middle mcp abd
            5.0,    # 3:  ring mcp abd
            15.0,   # 4:  pinky mcp abd (spread outward)
            -15.0,  # 5:  thumb mcp abd
            -15.0,  # 6:  index mcp flex (extended)
            -15.0,  # 7:  middle mcp flex (extended)
            -15.0,  # 8:  ring mcp flex (extended)
            -15.0,  # 9:  pinky mcp flex (extended)
            -15.0,  # 10: thumb mcp flex (extended)
            -10.0,  # 11: index pip flex (extended)
            -10.0,  # 12: middle pip flex (extended)
            -10.0,  # 13: ring pip flex (extended)
            -10.0,  # 14: pinky pip flex (extended)
            0.0,    # 15: thumb cmc flex
        ],
        'PINCH': [
            -15.0,  # 0:  thumb cmc abd
            0.0,    # 1:  index mcp abd
            0.0,    # 2:  middle mcp abd
            0.0,    # 3:  ring mcp abd
            0.0,    # 4:  pinky mcp abd
            35.0,   # 5:  thumb mcp abd (opposes index)
            45.0,   # 6:  index mcp flex
            0.0,    # 7:  middle mcp flex
            0.0,    # 8:  ring mcp flex
            0.0,    # 9:  pinky mcp flex
            -40.0,  # 10: thumb mcp flex
            60.0,   # 11: index pip flex
            0.0,    # 12: middle pip flex
            0.0,    # 13: ring pip flex
            0.0,    # 14: pinky pip flex
            -35.0,  # 15: thumb cmc flex
        ],
    }

    GESTURE_NAMES = ['REST', 'FIST', 'OPEN', 'PINCH']

    def __init__(self):
        super().__init__('emg_classifier_node')

        # Default paths to V1 artifacts inside workspace
        default_model = os.path.expanduser(
            '~/robotic_hand_ws/src/emg_classifier/models/emg_hand_cnn_best.keras'
        )
        default_mean = os.path.expanduser(
            '~/robotic_hand_ws/src/emg_classifier/models/train_mean.npy'
        )
        default_std = os.path.expanduser(
            '~/robotic_hand_ws/src/emg_classifier/models/train_std.npy'
        )
        default_replay = os.path.expanduser(
            '~/robotic_hand_ws/src/emg_classifier/models/sample_emg_trials.npz'
        )

        # --- Declare ROS2 Parameters ---
        self.declare_parameter(
            'mode',
            'test',
            ParameterDescriptor(description="Mode: 'test', 'manual', 'ml', or 'replay'")
        )
        self.declare_parameter(
            'cycle_time_sec',
            3.5,
            ParameterDescriptor(description="Time interval between gestures in test/replay mode (seconds)")
        )
        self.declare_parameter(
            'manual_gesture',
            'REST',
            ParameterDescriptor(description="Current gesture in manual mode: REST, FIST, OPEN, PINCH")
        )
        self.declare_parameter(
            'model_path',
            default_model,
            ParameterDescriptor(description="Path to V1 model (.keras or .npz)")
        )
        self.declare_parameter(
            'mean_path',
            default_mean,
            ParameterDescriptor(description="Path to train_mean.npy")
        )
        self.declare_parameter(
            'std_path',
            default_std,
            ParameterDescriptor(description="Path to train_std.npy")
        )
        self.declare_parameter(
            'replay_file',
            default_replay,
            ParameterDescriptor(description="Path to EMG replay dataset (.npz)")
        )
        self.declare_parameter(
            'rms_threshold',
            0.01,
            ParameterDescriptor(description="RMS inactivity threshold for REST state")
        )
        self.declare_parameter(
            'confidence_threshold',
            0.40,
            ParameterDescriptor(description="Minimum confidence threshold for classification")
        )

        self.mode = str(self.get_parameter('mode').value).lower()
        self.cycle_time_sec = float(self.get_parameter('cycle_time_sec').value)
        self.manual_gesture = str(self.get_parameter('manual_gesture').value).upper()
        self.model_path = self._resolve_path(str(self.get_parameter('model_path').value), 'models/emg_hand_cnn_best.keras')
        self.mean_path = self._resolve_path(str(self.get_parameter('mean_path').value), 'models/train_mean.npy')
        self.std_path = self._resolve_path(str(self.get_parameter('std_path').value), 'models/train_std.npy')
        self.replay_file = self._resolve_path(str(self.get_parameter('replay_file').value), 'models/sample_emg_trials.npz')
        self.rms_threshold = float(self.get_parameter('rms_threshold').value)
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)

        self.add_on_set_parameters_callback(self.parameter_changed_callback)

        # --- Initialize V1 EMG Model ---
        self.v1_model = None
        self._init_v1_model()

        # --- Replay Data Buffer ---
        self.replay_trials = {}
        self.replay_keys = []
        self.replay_index = 0
        self._load_replay_data()

        # --- Publisher for /hand_servo_input ---
        self.servo_pub = self.create_publisher(
            Float32MultiArray,
            '/hand_servo_input',
            10
        )

        # --- Observability Publishers ---
        self.gesture_pub = self.create_publisher(
            String,
            '/emg/predicted_gesture',
            10
        )
        self.probs_pub = self.create_publisher(
            Float32MultiArray,
            '/emg/probabilities',
            10
        )

        # --- Subscriber for manual gesture command topic ---
        self.manual_sub = self.create_subscription(
            String,
            '/manual_gesture',
            self.manual_gesture_callback,
            10
        )

        # --- Subscriber for live/recorded EMG data (/emg/raw) ---
        self.emg_sub = self.create_subscription(
            Float32MultiArray,
            '/emg/raw',
            self.emg_callback,
            10
        )

        # --- Services for Manual Gesture Selection ---
        self.srv_rest = self.create_service(
            Trigger, 'set_gesture_rest', self.service_set_rest
        )
        self.srv_fist = self.create_service(
            Trigger, 'set_gesture_fist', self.service_set_fist
        )
        self.srv_open = self.create_service(
            Trigger, 'set_gesture_open', self.service_set_open
        )
        self.srv_pinch = self.create_service(
            Trigger, 'set_gesture_pinch', self.service_set_pinch
        )
        self.srv_resume = self.create_service(
            Trigger, 'resume_test_cycle', self.service_resume_cycle
        )

        # --- Internal State ---
        self.test_sequence = ['REST', 'FIST', 'OPEN', 'PINCH']
        self.test_index = 0
        self.current_gesture = 'REST'

        # --- Timer for Test / Replay Mode Cycling ---
        self.timer = self.create_timer(self.cycle_time_sec, self.timer_callback)

        self.get_logger().info('EMG Classifier Node Initialized.')
        self.get_logger().info(f'Operating Mode: {self.mode}')
        self.get_logger().info(f'Available Gestures: {", ".join(self.GESTURE_NAMES)}')
        self.get_logger().info(f'Model Path: {self.model_path}')
        self.get_logger().info(f'Mean Path: {self.mean_path}')
        self.get_logger().info(f'Std Path: {self.std_path}')
        self.get_logger().info('Publishing to /hand_servo_input (16 servos)')

        # Publish initial REST pose immediately on startup
        self.execute_pipeline(explicit_gesture='REST')

    def _resolve_path(self, path_str: str, relative_fallback: str) -> str:
        """Resolve file path across workspace source or share directories."""
        expanded = os.path.expanduser(path_str)
        if os.path.exists(expanded):
            return expanded
        try:
            pkg_share = get_package_share_directory('emg_classifier')
            cand = os.path.join(pkg_share, relative_fallback)
            if os.path.exists(cand):
                return cand
        except Exception:
            pass
        src_cand = os.path.expanduser(f'~/robotic_hand_ws/src/emg_classifier/{relative_fallback}')
        if os.path.exists(src_cand):
            return src_cand
        return expanded

    def _init_v1_model(self):
        """Instantiate V1 EMG Model for ML and replay modes."""
        try:
            self.v1_model = V1EMGModel(
                model_path=self.model_path,
                mean_path=self.mean_path,
                std_path=self.std_path,
                logger=self.get_logger()
            )
            self.get_logger().info("V1 ML Model loaded and ready.")
        except Exception as e:
            self.get_logger().warn(f"Could not load V1 ML model at startup: {e}")

    def _load_replay_data(self):
        """Load prerecorded replay trials."""
        if not os.path.exists(self.replay_file):
            self.get_logger().warn(f"Replay file not found: {self.replay_file}")
            return
        try:
            data = np.load(self.replay_file)
            preferred = ['REST', 'PINCH', 'FIST', 'OPEN']
            self.replay_keys = [k for k in preferred if k in data]
            if not self.replay_keys:
                self.replay_keys = list(data.keys())
            for k in self.replay_keys:
                self.replay_trials[k] = data[k].astype(np.float32)
            self.get_logger().info(f"Loaded {len(self.replay_keys)} replay trials: {self.replay_keys}")
        except Exception as e:
            self.get_logger().error(f"Error loading replay file {self.replay_file}: {e}")

    # =========================================================================
    # PIPELINE: PREPROCESS -> CNN INFERENCE -> MAPPING -> PUBLISH
    # =========================================================================

    def preprocess(self, raw_emg_data):
        """
        STEP 1: SIGNAL PREPROCESSING
        Filters and normalizes raw multi-channel EMG data using V1 specifications:
          - Channel selection [12, 11, 7, 2]
          - Central 400 ms extraction
          - DC removal
          - 20-450 Hz Butterworth bandpass
          - Normalization using train_mean and train_std
        """
        if raw_emg_data is None:
            return None, 0.0
        if self.v1_model is None:
            self._init_v1_model()
            if self.v1_model is None:
                raise RuntimeError("V1 ML Model not loaded.")

        return self.v1_model.preprocess(raw_emg_data)

    def classify(self, normed_window, raw_rms=None) -> tuple[str, dict, float, str]:
        """
        STEP 2: V1 CNN CLASSIFIER INFERENCE
        Classifies preprocessed window into PINCH, FIST, OPEN, or REST (inactivity).
        """
        if self.v1_model is None:
            self._init_v1_model()
            if self.v1_model is None:
                raise RuntimeError("V1 ML Model not loaded.")

        return self.v1_model.classify(
            normed_window,
            raw_rms=raw_rms,
            rms_threshold=self.rms_threshold,
            confidence_threshold=self.confidence_threshold
        )

    def gesture_to_servo_angles(self, gesture_name: str) -> list:
        """
        STEP 3: GESTURE TO 16-SERVO MAPPING
        Converts gesture name string to 16-element servo angles (degrees).
        """
        name = gesture_name.upper().strip()
        if name not in self.GESTURE_ANGLES:
            self.get_logger().warn(
                f"Unknown gesture '{name}'. Allowed: {self.GESTURE_NAMES}. Falling back to REST."
            )
            name = 'REST'
        return list(self.GESTURE_ANGLES[name])

    def publish_servo_angles(self, angles: list):
        """
        STEP 4: PUBLISH TO /hand_servo_input
        Publishes std_msgs/msg/Float32MultiArray with 16 values.
        """
        msg = Float32MultiArray()
        msg.data = [float(a) for a in angles]
        self.servo_pub.publish(msg)

    def execute_pipeline(self, raw_emg_data=None, explicit_gesture=None, trial_name=None):
        """
        Executes full pipeline:
          input -> preprocess() -> classify() -> mapping -> publish()
        """
        probs_dict = {}
        confidence = 1.0
        reason = "Explicit / Manual"

        if explicit_gesture is not None:
            gesture = explicit_gesture
        elif raw_emg_data is not None:
            try:
                normed_window, raw_rms = self.preprocess(raw_emg_data)
                gesture, probs_dict, confidence, reason = self.classify(normed_window, raw_rms=raw_rms)
            except Exception as e:
                self.get_logger().error(f"Inference error: {e}. Falling back to REST.")
                gesture = 'REST'
        else:
            # Fallback for test mode sequence cycling
            gesture = self.test_sequence[self.test_index]
            self.test_index = (self.test_index + 1) % len(self.test_sequence)
            reason = "Test Sequence"

        angles = self.gesture_to_servo_angles(gesture)

        # Print exactly formatted output to stdout and log
        output_str = f"[EMG] Gesture: {gesture}"
        print(output_str, flush=True)

        log_msg = f"{output_str} (Mode: {self.mode}, Conf: {confidence:.2f}, Reason: {reason})"
        if trial_name:
            log_msg = f"[Trial: {trial_name}] " + log_msg
        self.get_logger().info(log_msg)

        # Publish to /hand_servo_input
        self.publish_servo_angles(angles)
        self.current_gesture = gesture

        # Publish observability topics
        g_msg = String()
        g_msg.data = gesture
        self.gesture_pub.publish(g_msg)

        if probs_dict:
            p_msg = Float32MultiArray()
            p_msg.data = [
                float(probs_dict.get('PINCH', 0.0)),
                float(probs_dict.get('FIST', 0.0)),
                float(probs_dict.get('OPEN', 0.0)),
            ]
            self.probs_pub.publish(p_msg)

    # =========================================================================
    # TIMERS & CALLBACKS
    # =========================================================================

    def timer_callback(self):
        """Timer callback for test and replay modes."""
        if self.mode == 'test':
            self.execute_pipeline()
        elif self.mode == 'replay':
            if not self.replay_keys:
                self.get_logger().warn("Replay buffer empty. Retrying reload...")
                self._load_replay_data()
                return

            trial_key = self.replay_keys[self.replay_index]
            raw_trial = self.replay_trials[trial_key]
            self.replay_index = (self.replay_index + 1) % len(self.replay_keys)

            # Feed the prerecorded raw trial through the full V1 pipeline
            self.execute_pipeline(raw_emg_data=raw_trial, trial_name=trial_key)

    def emg_callback(self, msg: Float32MultiArray):
        """
        Callback for incoming live or recorded EMG data from /emg/raw topic.
        Runs full V1 preprocessing and CNN inference in ML mode.
        """
        if self.mode in ('ml', 'stream'):
            self.execute_pipeline(raw_emg_data=list(msg.data))

    def manual_gesture_callback(self, msg: String):
        """Callback for manual gesture command topic (/manual_gesture)."""
        gesture_name = msg.data.strip().upper()
        if gesture_name in self.GESTURE_ANGLES:
            self.mode = 'manual'
            self.manual_gesture = gesture_name
            self.get_logger().info(f"Switched to MANUAL mode with gesture: {gesture_name}")
            self.execute_pipeline(explicit_gesture=gesture_name)
        else:
            self.get_logger().error(
                f"Invalid manual gesture '{msg.data}'. Must be one of {self.GESTURE_NAMES}"
            )

    # =========================================================================
    # SERVICES FOR MANUAL GESTURE SELECTION
    # =========================================================================

    def _apply_manual_gesture(self, gesture_name: str, response):
        self.mode = 'manual'
        self.manual_gesture = gesture_name
        self.execute_pipeline(explicit_gesture=gesture_name)
        response.success = True
        response.message = f"Switched to MANUAL mode. Gesture set to {gesture_name}"
        return response

    def service_set_rest(self, request, response):
        return self._apply_manual_gesture('REST', response)

    def service_set_fist(self, request, response):
        return self._apply_manual_gesture('FIST', response)

    def service_set_open(self, request, response):
        return self._apply_manual_gesture('OPEN', response)

    def service_set_pinch(self, request, response):
        return self._apply_manual_gesture('PINCH', response)

    def service_resume_cycle(self, request, response):
        self.mode = 'test'
        response.success = True
        response.message = "Switched to TEST mode (auto-cycling gestures)."
        self.get_logger().info("Resumed TEST mode (auto-cycling).")
        return response

    # =========================================================================
    # PARAMETER DYNAMIC RECONFIGURATION
    # =========================================================================

    def parameter_changed_callback(self, params):
        for param in params:
            if param.name == 'mode':
                val = str(param.value).lower()
                if val in ('test', 'manual', 'ml', 'replay'):
                    self.mode = val
                    self.get_logger().info(f"Parameter 'mode' changed to: {self.mode}")
                else:
                    return SetParametersResult(successful=False, reason="Mode must be 'test', 'manual', 'ml', or 'replay'")
            elif param.name == 'manual_gesture':
                val = str(param.value).upper()
                if val in self.GESTURE_ANGLES:
                    self.manual_gesture = val
                    if self.mode == 'manual':
                        self.execute_pipeline(explicit_gesture=val)
                    self.get_logger().info(f"Parameter 'manual_gesture' set to: {val}")
                else:
                    return SetParametersResult(successful=False, reason=f"Gesture must be one of {self.GESTURE_NAMES}")
            elif param.name == 'cycle_time_sec':
                new_time = float(param.value)
                if new_time > 0.1:
                    self.cycle_time_sec = new_time
                    self.timer.cancel()
                    self.timer = self.create_timer(self.cycle_time_sec, self.timer_callback)
                    self.get_logger().info(f"Parameter 'cycle_time_sec' updated to: {self.cycle_time_sec}s")
                else:
                    return SetParametersResult(successful=False, reason="cycle_time_sec must be > 0.1")
            elif param.name == 'model_path':
                self.model_path = self._resolve_path(str(param.value), 'models/emg_hand_cnn_best.keras')
                self._init_v1_model()
            elif param.name == 'mean_path':
                self.mean_path = self._resolve_path(str(param.value), 'models/train_mean.npy')
                self._init_v1_model()
            elif param.name == 'std_path':
                self.std_path = self._resolve_path(str(param.value), 'models/train_std.npy')
                self._init_v1_model()
            elif param.name == 'replay_file':
                self.replay_file = self._resolve_path(str(param.value), 'models/sample_emg_trials.npz')
                self._load_replay_data()
            elif param.name == 'rms_threshold':
                self.rms_threshold = float(param.value)
            elif param.name == 'confidence_threshold':
                self.confidence_threshold = float(param.value)

        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = EMGClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
