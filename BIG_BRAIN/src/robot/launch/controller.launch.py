import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command
from launch_ros.actions import Node

def generate_launch_description():
    robot_description = Command(['xacro ', os.path.join(get_package_share_directory('robot'), 'urdf', 'mobile_robot.xacro')])
    controller_config = os.path.join(get_package_share_directory('robot'), 'config', 'controller.yaml')
    
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[{'robot_description': robot_description},
                    controller_config],
        arguments=['--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error']
    )

    delayed_controller_manager = TimerAction(period=3.0, actions=[controller_manager])

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", '--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error'],
    )

    diff_drive_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_controller", '--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error'],
    )

    delayed_joint_broadcaster = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[joint_state_broadcaster_spawner]
        )
    )

    delayed_diff_drive = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[diff_drive_controller_spawner]
        )
    )

    return LaunchDescription([
        delayed_controller_manager,
        delayed_joint_broadcaster,
        delayed_diff_drive
    ])