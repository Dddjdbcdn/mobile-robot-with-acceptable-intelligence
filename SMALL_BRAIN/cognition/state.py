import math,time

robot_state = {
    "pose": None,
    "room_geometry": None,
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
    room_geometry = message.get("room_geometry")

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
    robot_state["room_geometry"] = room_geometry

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

