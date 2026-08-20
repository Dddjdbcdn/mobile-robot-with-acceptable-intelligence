import numpy as np
import cv2
import time
import math

from database.state import robot_state 

camera_horizontal_fov_deg: float = 85.0
camera_vertical_fov_deg: float = 52.0
tof_fov_deg: float = 2.0
tof_offset_x_m: float = 0.0
tof_offset_y_m: float = -0.012
tof_offset_z_m: float = 0.006
tof_yaw_deg: float = 0.0
tof_pitch_deg: float = 0.0

def _focal_lengths_from_fov(
    width: int,
    height: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> tuple[float, float]:
    fx = width / (2.0 * np.tan(np.deg2rad(horizontal_fov_deg) / 2.0))
    fy = height / (2.0 * np.tan(np.deg2rad(vertical_fov_deg) / 2.0))
    return float(fx), float(fy)

def _distance_to_bgr(distance_m: float) -> tuple[int, int, int]:
    """
    Simple near-to-far visualization:
        near = red
        middle = yellow/green
        far = blue
    """
    normalized = np.clip(distance_m / 4.0, 0.0, 1.0)
    hue = int(normalized * 120)
    hsv_pixel = np.uint8([[[hue, 230, 255]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr_pixel)

def project_tof_region(
    frame_width: int,
    frame_height: int,
    distance_m: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Projects the TFmini-S 2° circular FoV cone onto the camera frame.
    """
    fx, fy = _focal_lengths_from_fov(
        frame_width,
        frame_height,
        camera_horizontal_fov_deg,
        camera_vertical_fov_deg,
    )

    cx = frame_width / 2.0
    cy = frame_height / 2.0

    yaw = np.deg2rad(tof_yaw_deg)
    pitch = np.deg2rad(tof_pitch_deg)

    # Beam center in camera coordinates.
    center_z_m = distance_m + tof_offset_z_m
    center_x_m = tof_offset_x_m + distance_m * np.tan(yaw)
    center_y_m = tof_offset_y_m + distance_m * np.tan(pitch)

    # TFmini-S full FoV = 2°, so half-angle = 1°.
    radius_m = distance_m * np.tan(np.deg2rad(tof_fov_deg) / 2.0)

    # Circular footprint of the cone.
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)

    points_xyz = np.column_stack(
        (
            center_x_m + radius_m * np.cos(angles),
            center_y_m + radius_m * np.sin(angles),
            np.full_like(angles, center_z_m),
        )
    )

    projected_points = []

    for x_m, y_m, z_m in points_xyz:
        u = cx + fx * x_m / z_m
        v = cy + fy * y_m / z_m

        projected_points.append(
            [
                int(round(np.clip(u, 0, frame_width - 1))),
                int(round(np.clip(v, 0, frame_height - 1))),
            ]
        )

    center_u = cx + fx * center_x_m / center_z_m
    center_v = cy + fy * center_y_m / center_z_m

    center = (
        int(round(np.clip(center_u, 0, frame_width - 1))),
        int(round(np.clip(center_v, 0, frame_height - 1))),
    )

    return np.asarray(projected_points, dtype=np.int32), center

def draw_tof_overlay(
    frame: np.ndarray,
    stale_after_s: float = 0.5,
) -> np.ndarray:
    d = robot_state["camera"]
    distance = d["camera_tof_range"]

    if distance is None:
        cv2.putText(frame, "TOF: NO DATA", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return frame

    age = time.monotonic() - d["timestamp"]
    if age > stale_after_s:
        cv2.putText(frame, f"TOF: STALE ({age:.1f}s)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
        return frame

    h, w = frame.shape[:2]

    polygon, center = project_tof_region(
        w, h, max(distance, 0.02)
    )

    color = _distance_to_bgr(distance)

    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, polygon, color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 18, 2)

    # ToF label beside beam
    x, y = center
    camera_center_z = robot_state["camera"]["camera_tof_range"] + tof_offset_z_m
    cv2.putText(
        frame,
        f"TOF {camera_center_z:.3f} m",
        (max(10, min(x + 12, w - 210)), max(25, y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    # Robot/servo state
    lines = [
        f"X {d['object_x']:+.2f}m  Y {d['object_y']:+.2f}m",
        f"Pan {d['pan_angle']:.1f}  Tilt {d['tilt_angle']:.1f}",
        f"Tracking: {'YES' if d['tracking'] else 'NO'}",
    ]

    if d["tracking"]:
        lines.append(f"Stable: {'YES' if d['is_stable'] else 'NO'}")

    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (20, 35 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA
        )

    return frame

def draw_csrt_overlay(frame,target,bbox):
    x, y, w, h = [int(v * 2) for v in bbox]
    
    # Mirror tracker X coordinate
    x = frame.shape[1] - x - w

    cv2.rectangle(frame,(x, y),(x + w, y + h),(0, 255, 0),2,)
    cv2.putText(frame,target.upper(),(x, max(15, y - 10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)

    return frame


def draw_yolo_overlay(frame):
    detections = robot_state["camera"].get("yolo_detections", [])

    if not detections:
        return frame

    frame_h, frame_w = frame.shape[:2]

    # YOLO inference coordinates are based on 640x360 frames
    scale_x = frame_w / 640.0
    scale_y = frame_h / 360.0

    box_color = (255, 0, 0)          # blue
    skeleton_color = (0, 255, 255)  # yellow
    point_color = (0, 255, 0)       # green

    skeleton = [
        ("left_eye", "right_eye"),
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),

        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),

        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),

        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    for detection in detections:
        bbox = detection.get("bbox")

        if not bbox:
            continue

        class_name = detection.get("class", "unknown")
        confidence = detection.get("confidence", 0.0)

        # ------------------------------------------------
        # Bounding box
        # ------------------------------------------------
        x1 = int(bbox["x1"] * scale_x)
        y1 = int(bbox["y1"] * scale_y)
        x2 = int(bbox["x2"] * scale_x)
        y2 = int(bbox["y2"] * scale_y)

        # Display frame is horizontally flipped
        mirrored_x1 = frame_w - x2
        mirrored_x2 = frame_w - x1

        x1 = max(0, min(mirrored_x1, frame_w - 1))
        x2 = max(0, min(mirrored_x2, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(0, min(y2, frame_h - 1))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2,
            cv2.LINE_AA,
        )

        # ------------------------------------------------
        # Label
        # ------------------------------------------------
        label = f"{class_name.upper()} {confidence:.2f}"

        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        label_y = max(y1, text_h + 10)

        cv2.rectangle(
            frame,
            (x1, label_y - text_h - 10),
            (x1 + text_w + 10, label_y),
            box_color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ------------------------------------------------
        # Only persons can have pose skeletons
        # ------------------------------------------------
        if class_name != "person":
            continue

        keypoints = detection.get("keypoints")

        if not keypoints:
            continue

        projected = {}

        # ------------------------------------------------
        # Project visible keypoints to display frame
        # ------------------------------------------------
        for name, kp in keypoints.items():
            kp_conf = kp.get("confidence", 0.0)

            if kp_conf < 0.3:
                continue

            px = int(kp["x"] * scale_x)
            py = int(kp["y"] * scale_y)

            # Mirror because display frame is flipped
            px = frame_w - px

            px = max(0, min(px, frame_w - 1))
            py = max(0, min(py, frame_h - 1))

            projected[name] = (px, py)

        # ------------------------------------------------
        # Skeleton connections
        # ------------------------------------------------
        for start_name, end_name in skeleton:
            start = projected.get(start_name)
            end = projected.get(end_name)

            if start is None or end is None:
                continue

            cv2.line(
                frame,
                start,
                end,
                skeleton_color,
                2,
                cv2.LINE_AA,
            )

        # ------------------------------------------------
        # Keypoint circles
        # ------------------------------------------------
        for point in projected.values():
            cv2.circle(
                frame,
                point,
                4,
                point_color,
                -1,
                cv2.LINE_AA,
            )

    return frame