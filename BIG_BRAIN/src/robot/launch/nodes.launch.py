from launch import LaunchDescription
import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    watchdog_config = os.path.join(
        get_package_share_directory('robot'), 'config', 'watchdog.yaml'
    )
    topic_bridge = Node(
        package='robot',
        executable='topic_bridge.py',
        output='screen',
        arguments=['--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error'],
    )
    llm_bridge = Node(
        package='robot',
        executable='llm_bridge.py',
        output='screen',
        arguments=['--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error'],
    )
    person_lidar_tracker = Node(
        package='robot',
        executable='person_lidar_tracker.py',
        output='screen',
        arguments=['--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error'],
    )
    system_watchdog = Node(
        package='robot',
        executable='system_watchdog.py',
        name='system_watchdog',
        output='screen',
        parameters=[watchdog_config],
        arguments=['--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error'],
    )
    return LaunchDescription([
        topic_bridge,
        llm_bridge,
        person_lidar_tracker,
    ])
