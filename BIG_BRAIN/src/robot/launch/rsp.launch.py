import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    robot_description = ParameterValue(
        Command([
            'xacro ', os.path.join(get_package_share_directory('robot'), 'urdf', 'mobile_robot.xacro')
        ]),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        arguments=['--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error'],
    )

    return LaunchDescription([
        robot_state_publisher
    ])