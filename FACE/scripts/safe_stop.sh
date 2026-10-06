#!/usr/bin/env bash
set +u

face_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robot_root="$(cd "$face_dir/.." && pwd)"

cd "$robot_root/BIG_BRAIN" || exit 0
source /opt/ros/jazzy/setup.bash
source install/setup.bash

timeout 4 ros2 topic pub --once \
    /diff_drive_controller/cmd_vel \
    geometry_msgs/msg/TwistStamped \
    '{header: {frame_id: "base_link"}, twist: {linear: {x: 0.0}, angular: {z: 0.0}}}' \
    >/dev/null 2>&1 || true
exit 0
