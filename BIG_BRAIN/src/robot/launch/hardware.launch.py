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
    )

    usb_camera_node = Node(
        package='image_tools',
        executable='cam2image',
        name='cam2image',
        parameters=[{
            'device_id': 0,
            'width': 1280,
            'height': 720,
        }],
        remappings=[
            ('/image', '/usb_camera/image_raw')
        ],
        arguments=['--ros-args', '--log-level', 'WARN'],
        output='screen'
    )

    rqt_image_view_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/usb_camera/image_raw'], 
        additional_env={'DISPLAY': ':0'}, # Changed from env to additional_env         
        output='screen'
    )

    return LaunchDescription([
        setup_serial,
        micro_ros_agent,
        delayed_stm32_reset,
        lidar_node,
        depth_camera_node,
        # usb_camera_node,
        # rqt_image_view_node
    ])