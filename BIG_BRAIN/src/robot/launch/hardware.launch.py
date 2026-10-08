import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression, LaunchConfiguration

def generate_launch_description():
    # Configure serial port for low latency before starting micro-ros-agent
    setup_serial = ExecuteProcess(
        cmd=['stty', '-F', '/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066BFF485270535067113035-if02',
             '921600', 'raw', '-echo', '-crtscts', '-ixon', '-ixoff'],
        output='screen',
    )

    micro_ros_agent = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=['serial', '-b', '921600',
                '--dev', '/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066BFF485270535067113035-if02',
                '--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error'],
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    stm32_reset = ExecuteProcess(
        cmd=['st-flash', '--serial', '066BFF485270535067113035', 'reset'],
        output='screen',
    )

    start_agent_after_serial_setup = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=setup_serial,
            on_exit=[
                micro_ros_agent,
                TimerAction(period=2.0, actions=[stm32_reset]),
            ],
        )
    )

    lidar_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('sllidar_ros2'), 'launch', 'sllidar_c1_launch.py')
        ),
         launch_arguments={'serial_port': '/dev/rplidar', 'frame_id': 'lidar'}.items(),
    )

    depth_camera_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(get_package_share_directory('astra_camera'), 'launch', 'astra.launch.xml')
        )
    )

    return LaunchDescription([
        start_agent_after_serial_setup,
        setup_serial,
        lidar_node,
        depth_camera_node,
    ])
