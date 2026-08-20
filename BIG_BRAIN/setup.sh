source install/setup.bash

alias run_lidar="ros2 launch sllidar_ros2 sllidar_c1_launch.py serial_port:=/dev/rplidar"
alias view_lidar="ros2 launch sllidar_ros2 view_sllidar_c1_launch.py serial_port:=/dev/rplidar"
alias run_agent='ros2 run micro_ros_agent micro_ros_agent serial -b 921600 --dev /dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066BFF485270535067113035-if02' 
alias run_depth_cam='ros2 launch astra_camera astra.launch.xml'

alias run='ros2 launch robot bringup.launch.py'
alias auto_run='ros2 launch robot bringup.launch.py slam:=true nav2:=true'
alias sim_run='ros2 launch robot sim_bringup.launch.py'
alias sim_auto_run='ros2 launch robot sim_bringup.launch.py slam:=true nav2:=true'
alias run_yolo='src/yolo_vision/.venv/bin/python3 src/yolo_vision/yolo_vision/yolo_node.py'

alias build='colcon build --packages-select robot' 
alias remove='rm -rf build/robot install/robot'
alias install_dep='rosdep update && rosdep install --from-paths src --ignore-src -r -y'
alias stop="ros2 topic pub --once /diff_drive_controller/cmd_vel geometry_msgs/msg/TwistStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: \"base_link\"}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}'"

alias traj='ros2 run robot trajectory.py'
alias traj_square='ros2 run robot trajectory.py --ros-args -p trajectory_type:=square -p square_length:=1.2 -p square_linear_speed:=0.2 -p square_angular_speed:=0.8 -p clockwise:=true'
alias traj_circle='ros2 run robot trajectory.py --ros-args -p trajectory_type:=circle -p circle_radius:=0.75 -p circle_linear_speed:=0.2'
alias traj_idle='ros2 param set /trajectory_node trajectory_type idle'
alias traj_set_square='ros2 param set /trajectory_node trajectory_type square'
alias traj_set_circle='ros2 param set /trajectory_node trajectory_type circle'
alias traj_len='ros2 param set /trajectory_node square_length'
alias traj_v_square='ros2 param set /trajectory_node square_linear_speed'
alias traj_w_square='ros2 param set /trajectory_node square_angular_speed'
alias traj_radius='ros2 param set /trajectory_node circle_radius'
alias traj_v_circle='ros2 param set /trajectory_node circle_linear_speed'

alias pwm_l='ros2 topic pub --once /stm32/pwm/left std_msgs/msg/Float32'
alias pwm_r='ros2 topic pub --once /stm32/pwm/right std_msgs/msg/Float32'
alias pwm_stop='ros2 topic pub --once /stm32/pwm/left std_msgs/msg/Float32 "{data: 0.0}" && ros2 topic pub --once /stm32/pwm/right std_msgs/msg/Float32 "{data: 0.0}"'
