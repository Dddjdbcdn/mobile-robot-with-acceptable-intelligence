import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def include_launch(package_name, launch_file_name, launch_arguments=None, condition=None):
    pkg_path = get_package_share_directory(package_name)
    launch_path = os.path.join(pkg_path, 'launch', launch_file_name)
    args = launch_arguments.items() if launch_arguments else {}
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_path),
        launch_arguments=args,
        condition=condition
    )

def declare_arg(name, default_value, description=""):
    return DeclareLaunchArgument(
        name,
        default_value=default_value,
        description=description
    )

def generate_launch_description():
    launch_args = [
        declare_arg('slam', 'false'),
        declare_arg('amcl', 'false'),
        declare_arg('nav2', 'false'),
        declare_arg('use_sim_time', 'false'),
        declare_arg('nav2_type', 'approach')
    ]

    hardware_launch = include_launch('robot', 'hardware.launch.py')
    rsp_launch = include_launch('robot', 'rsp.launch.py')
    controller_launch = include_launch('robot', 'controller.launch.py')
    localization_launch = include_launch('robot', 'localization.launch.py')
    navigation_launch = include_launch('robot', 'navigation.launch.py',condition=IfCondition(LaunchConfiguration('nav2')))
    nodes_launch = include_launch('robot', 'nodes.launch.py')

    delayed_navigation_launch = TimerAction(period=5.0,actions=[navigation_launch])


    return LaunchDescription([
        *launch_args,
        hardware_launch,
        rsp_launch,
        controller_launch,
        localization_launch,
        nodes_launch,
        delayed_navigation_launch
    ])
