import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
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
    yolo_venv_python = 'src/yolo_vision/.venv/bin/python3'
    yolo_script = 'src/yolo_vision/yolo_vision/yolo_node.py'

    yolo_vision_node = ExecuteProcess(
        cmd=[yolo_venv_python, yolo_script],
        output='screen'
    )

    return LaunchDescription([
        topic_bridge,
        llm_bridge,
        yolo_vision_node,
    ])
