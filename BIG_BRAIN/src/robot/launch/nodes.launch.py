from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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
    return LaunchDescription([
        topic_bridge,
        llm_bridge,
        person_lidar_tracker,
    ])
