import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
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
    )

    stm32_reset = ExecuteProcess(
        cmd=['st-flash', '--serial', '066BFF485270535067113035', 'reset'],
        output='screen',
    )

    delayed_stm32_reset = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=micro_ros_agent,
            on_start=[stm32_reset],
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
        ),
        launch_arguments={
            'camera_name': 'camera',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_ir': 'false',
            # Register depth onto the color pixel grid for RGB detections.
            'depth_registration': 'true',
            'color_depth_synchronization': 'true',
            'enable_point_cloud': 'true',
            # robot_state_publisher owns the camera TF declared in the URDF.
            'publish_tf': 'false',
            'color_width': '640',
            'color_height': '480',
            'color_fps': '30',
            'depth_width': '640',
            'depth_height': '480',
            'depth_fps': '30',
            # Avoid immediately reopening Mini-series firmware after failure.
            'connection_delay': '1000',
        }.items(),
    )

    return LaunchDescription([
        setup_serial,
        micro_ros_agent,
        delayed_stm32_reset,
        lidar_node,
        depth_camera_node,
    ])