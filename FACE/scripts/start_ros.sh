#!/usr/bin/env bash
set -eo pipefail

face_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robot_root="$(cd "$face_dir/.." && pwd)"

cd "$robot_root/BIG_BRAIN"
source /opt/ros/jazzy/setup.bash
source install/setup.bash

read -r -a launch_args <<< "${DJ_ROS_LAUNCH_ARGS:-}"
exec ros2 launch robot bringup.launch.py "${launch_args[@]}"
