import math,time

robot_state = {
    "camera": {
        "camera_tof_range": 0.0,
        "pan_angle": 0.0,
        "tilt_angle": 0.0,
        "object_x": 0.0,
        "object_y": 0.0,
        "object_angle": 0.0,
        "timestamp": 0.0,
        "tracking": False,
        "is_stable": False,
        "vision_mode": "dj",
        "yolo_detections": {},
    }
}

def update_camera_state(message,object_tracking_manager):
    camera_tof_range = message.get("camera_tof_range")
    pan_angle = message.get("servo_pan_angle")
    tilt_angle = message.get("servo_tilt_angle")

    zenith = math.radians(tilt_angle)
    azimuth = math.radians(pan_angle - 95)

    object_x = camera_tof_range * math.sin(zenith) * math.cos(azimuth)
    object_y = camera_tof_range * math.sin(zenith) * math.sin(azimuth)

    robot_state["camera"].update({
        "camera_tof_range": camera_tof_range,
        "pan_angle": pan_angle,
        "tilt_angle": tilt_angle,
        "object_x": object_x,
        "object_y": object_y,
        "object_angle": azimuth,
        "tracking": object_tracking_manager.tracking,
        "is_stable": object_tracking_manager.stable,
        "timestamp": time.monotonic()
    })

