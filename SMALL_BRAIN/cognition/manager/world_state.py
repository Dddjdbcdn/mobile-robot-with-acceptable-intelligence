import math,time

robot_state = {
    "pose": None,
    "person": None,
    "lidar_person": {"state": None, "pose_age_seconds": None},
    "camera": {
        "camera_tof_range": 0.0,
        "pan_angle": 0.0,
        "tilt_angle": 0.0,
        "object_x": 0.0,
        "object_y": 0.0,
        "object_angle": 0.0,
        "object_map_x": None,
        "object_map_y": None,
        "timestamp": 0.0,
    }
}

def update_state(message):
    camera_tof_range = float(message.get("camera_tof_range") or 0.0)
    pan_angle = float(message.get("servo_pan_angle") or 95.0)
    tilt_angle = float(message.get("servo_tilt_angle") or 90.0)
    robot_pose = message.get("robot_pose")
    person_pose = message.get("person_pose")
    tracker_status = message.get("person_tracker_status")

    zenith = math.radians(tilt_angle)
    azimuth = math.radians(pan_angle - 95)

    object_x = camera_tof_range * math.sin(zenith) * math.cos(azimuth)
    object_y = camera_tof_range * math.sin(zenith) * math.sin(azimuth)

    object_map_x = None
    object_map_y = None
    if robot_pose is not None:
        robot_yaw = float(robot_pose["yaw"])
        object_map_x = (
            float(robot_pose["x"])
            + object_x * math.cos(robot_yaw)
            - object_y * math.sin(robot_yaw)
        )
        object_map_y = (
            float(robot_pose["y"])
            + object_x * math.sin(robot_yaw)
            + object_y * math.cos(robot_yaw)
        )

    robot_state["pose"] = robot_pose
    if isinstance(person_pose, dict):
        robot_state["person"] = {
            **person_pose,
            "timestamp": time.monotonic() - float(person_pose.get("age_seconds") or 0.0),
            "tracking_state": message.get("person_tracker_state"),
        }

    robot_state["lidar_person"] = {
        **(tracker_status if isinstance(tracker_status, dict) else {}),
        "state": message.get("person_tracker_state"),
        "pose_age_seconds": (
            float(person_pose.get("age_seconds") or 0.0)
            if isinstance(person_pose, dict) else None
        ),
    }

    robot_state["camera"].update({
        "camera_tof_range": camera_tof_range,
        "pan_angle": pan_angle,
        "tilt_angle": tilt_angle,
        "object_x": object_x,
        "object_y": object_y,
        "object_angle": azimuth,
        "object_map_x": object_map_x,
        "object_map_y": object_map_y,
        "timestamp": time.monotonic()
    })
