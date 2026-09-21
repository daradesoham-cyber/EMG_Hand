import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    hand_rviz_dir = get_package_share_directory('hand_rviz')
    hand_rviz_launch = os.path.join(hand_rviz_dir, 'launch', 'hand_rviz.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(hand_rviz_launch),
            launch_arguments={'js_gui': 'false'}.items()
        ),
        Node(
            package='emg_hand_demo',
            executable='gesture_publisher',
            output='screen'
        ),
    ])
