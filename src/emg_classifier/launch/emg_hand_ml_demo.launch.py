import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    hand_rviz_dir = get_package_share_directory('hand_rviz')
    hand_rviz_launch = os.path.join(hand_rviz_dir, 'launch', 'hand_rviz.launch.py')

    default_model_path = os.path.expanduser(
        '~/robotic_hand_ws/src/emg_classifier/models/emg_hand_cnn_best.keras'
    )
    default_mean_path = os.path.expanduser(
        '~/robotic_hand_ws/src/emg_classifier/models/train_mean.npy'
    )
    default_std_path = os.path.expanduser(
        '~/robotic_hand_ws/src/emg_classifier/models/train_std.npy'
    )
    default_replay_path = os.path.expanduser(
        '~/robotic_hand_ws/src/emg_classifier/models/sample_emg_trials.npz'
    )

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='test',
        description='Operating mode: manual, test, ml, or replay'
    )
    cycle_time_arg = DeclareLaunchArgument(
        'cycle_time',
        default_value='3.5',
        description='Cycle time in seconds for test and replay modes'
    )
    manual_gesture_arg = DeclareLaunchArgument(
        'manual_gesture',
        default_value='REST',
        description='Initial manual gesture: REST, FIST, OPEN, PINCH'
    )
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='Absolute or relative path to V1 model'
    )
    mean_path_arg = DeclareLaunchArgument(
        'mean_path',
        default_value=default_mean_path,
        description='Absolute or relative path to train_mean.npy'
    )
    std_path_arg = DeclareLaunchArgument(
        'std_path',
        default_value=default_std_path,
        description='Absolute or relative path to train_std.npy'
    )
    replay_file_arg = DeclareLaunchArgument(
        'replay_file',
        default_value=default_replay_path,
        description='Path to replay trials .npz file'
    )
    rms_threshold_arg = DeclareLaunchArgument(
        'rms_threshold',
        default_value='0.01',
        description='RMS inactivity threshold for REST state'
    )
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.40',
        description='Minimum confidence threshold for ML classification'
    )

    return LaunchDescription([
        mode_arg,
        cycle_time_arg,
        manual_gesture_arg,
        model_path_arg,
        mean_path_arg,
        std_path_arg,
        replay_file_arg,
        rms_threshold_arg,
        confidence_threshold_arg,

        # 1. BiDexHand RViz visualization & joint state forwarding
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(hand_rviz_launch),
            launch_arguments={'js_gui': 'false'}.items()
        ),

        # 2. EMG classifier node
        Node(
            package='emg_classifier',
            executable='emg_classifier_node',
            name='emg_classifier_node',
            output='screen',
            parameters=[{
                'mode': LaunchConfiguration('mode'),
                'cycle_time_sec': LaunchConfiguration('cycle_time'),
                'manual_gesture': LaunchConfiguration('manual_gesture'),
                'model_path': LaunchConfiguration('model_path'),
                'mean_path': LaunchConfiguration('mean_path'),
                'std_path': LaunchConfiguration('std_path'),
                'replay_file': LaunchConfiguration('replay_file'),
                'rms_threshold': LaunchConfiguration('rms_threshold'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            }]
        ),
    ])
