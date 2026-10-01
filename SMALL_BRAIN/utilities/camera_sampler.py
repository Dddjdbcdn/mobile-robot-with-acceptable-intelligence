import asyncio
import numpy as np
import cv2
import time

from cognition.manager.world_state import robot_state
from services.hand_landmark_service import HAND_CONNECTIONS

camera_horizontal_fov_deg: float = 85.0
camera_vertical_fov_deg: float = 52.0
tof_fov_deg: float = 2.0
tof_offset_x_m: float = 0.0
tof_offset_y_m: float = -0.012
tof_offset_z_m: float = 0.006
tof_yaw_deg: float = 0.0
tof_pitch_deg: float = 0.0


class SequenceFps:
    """Estimate event rate from a monotonically increasing sequence."""

    def __init__(self, smoothing=0.20, stale_after_s=1.0):
        self.smoothing = float(smoothing)
        self.stale_after_s = float(stale_after_s)
        self._last_sequence = None
        self._last_event_at = None
        self._fps = 0.0

    def observe(self, sequence, now=None):
        now = time.monotonic() if now is None else float(now)
        if not isinstance(sequence, int):
            return self.value(now)
        if self._last_sequence is None or sequence < self._last_sequence:
            self._last_sequence = sequence
            self._last_event_at = now
            self._fps = 0.0
        elif sequence > self._last_sequence:
            elapsed = now - self._last_event_at
            if elapsed > 1e-6:
                instant = (sequence - self._last_sequence) / elapsed
                self._fps = (
                    instant
                    if self._fps <= 0.0
                    else self._fps + self.smoothing * (instant - self._fps)
                )
            self._last_sequence = sequence
            self._last_event_at = now
        return self.value(now)

    def value(self, now=None):
        now = time.monotonic() if now is None else float(now)
        if (
            self._last_event_at is None
            or now - self._last_event_at > self.stale_after_s
        ):
            return 0.0
        return self._fps

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

    return frame

def draw_csrt_overlay(frame,target,bbox):
    frame_h, frame_w = frame.shape[:2]
    scale_x = frame_w / 640.0
    scale_y = frame_h / 360.0
    x, y, w, h = bbox
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)
    
    # Mirror tracker X coordinate
    x = frame.shape[1] - x - w

    cv2.rectangle(frame,(x, y),(x + w, y + h),(0, 255, 0),2,)
    cv2.putText(frame,target.upper(),(x, max(15, y - 10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)

    return frame


def pose_body_mask(frame_shape, projected, bbox):
    """Build the same permissive body area used by display and tracking."""
    x1, y1, x2, y2 = bbox
    person_width = max(1, x2 - x1)
    person_height = max(1, y2 - y1)
    limb_width = max(6, int(round(min(person_width, person_height) * 0.08)))
    joint_radius = max(4, limb_width // 2)

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    body_segments = [
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

    torso_names = (
        "left_shoulder", "right_shoulder", "right_hip", "left_hip"
    )
    if all(name in projected for name in torso_names):
        torso = np.asarray(
            [projected[name] for name in torso_names], dtype=np.int32
        )
        cv2.fillConvexPoly(mask, torso, 255, lineType=cv2.LINE_AA)

    for start_name, end_name in body_segments:
        start = projected.get(start_name)
        end = projected.get(end_name)
        if start is not None and end is not None:
            cv2.line(
                mask, start, end, 255, limb_width, cv2.LINE_AA
            )

    for point in projected.values():
        cv2.circle(mask, point, joint_radius, 255, -1, cv2.LINE_AA)

    head_points = [
        projected[name]
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        if name in projected
    ]
    if head_points:
        center = np.mean(head_points, axis=0)
        head_radius = max(
            limb_width,
            int(max(
                np.linalg.norm(np.asarray(point) - center)
                for point in head_points
            ))
            + joint_radius,
        )
        cv2.circle(
            mask,
            tuple(int(round(value)) for value in center),
            head_radius,
            255,
            -1,
            cv2.LINE_AA,
        )

    return mask


def _draw_pose_body_mask(frame, projected, bbox):
    """Draw a translucent visualization of the pose body area."""
    mask = pose_body_mask(frame.shape, projected, bbox)
    if not np.any(mask):
        return

    overlay = frame.copy()
    overlay[mask > 0] = (255, 160, 80)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, dst=frame)


def draw_yolo_overlay(frame,yolo):
    detections = yolo.detections

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

        _draw_pose_body_mask(
            frame,
            projected,
            (x1, y1, x2, y2),
        )

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


def draw_hand_landmark_overlay(frame, hand_interface):
    """Render full-frame hand landmarks."""
    status = hand_interface.gesture_status()
    if status is None:
        return frame

    height, width = frame.shape[:2]
    landmarks = ((status.get("hand") or {}).get("landmarks") or {})
    projected = {}
    for name, point in landmarks.items():
        normalized_x = point.get("normalized_x")
        normalized_y = point.get("normalized_y")
        if not isinstance(normalized_x, (int, float)) or not isinstance(
            normalized_y, (int, float)
        ):
            continue
        # The camera display is mirrored, while inference coordinates are not.
        x = int(round((1.0 - float(normalized_x)) * (width - 1)))
        y = int(round(float(normalized_y) * (height - 1)))
        projected[name] = (
            max(0, min(x, width - 1)),
            max(0, min(y, height - 1)),
        )

    for start_name, end_name in HAND_CONNECTIONS:
        start = projected.get(start_name)
        end = projected.get(end_name)
        if start is not None and end is not None:
            cv2.line(frame, start, end, (0, 255, 255), 3, cv2.LINE_AA)

    for point in projected.values():
        cv2.circle(frame, point, 5, (0, 80, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 5, (255, 255, 255), 1, cv2.LINE_AA)

    for detected_hand in (status.get("hand") or {}).get(
        "mediapipe_hands", []
    ):
        wrist = detected_hand.get("wrist") or {}
        normalized_x = wrist.get("normalized_x")
        normalized_y = wrist.get("normalized_y")
        if not isinstance(normalized_x, (int, float)) or not isinstance(
            normalized_y, (int, float)
        ):
            continue
        x = int(round((1.0 - float(normalized_x)) * (width - 1)))
        y = int(round(float(normalized_y) * (height - 1)))
        label = str(detected_hand.get("handedness") or "UNKNOWN").upper()
        score = detected_hand.get("handedness_score")
        if isinstance(score, (int, float)):
            label = f"MP {label} {float(score):.2f}"
        else:
            label = f"MP {label}"
        color = (
            (60, 230, 60)
            if detected_hand.get("selected")
            else (230, 80, 230)
        )
        cv2.putText(
            frame,
            label,
            (max(4, min(x + 10, width - 210)), max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame


def draw_tracking_status_overlay(frame, track_action, hand_interface):
    """Draw the high-value tracking signals as a readable HUD."""
    hand_status = hand_interface.gesture_status() or {}
    seed_target = (
        "hand"
        if hand_status.get("gesture") in {"welcome", "push"}
        and hand_status.get("state") in {
            "tracking_person", "following",
        }
        else track_action.target
    )
    seed = track_action.stable_seeds.status(
        target=seed_target,
        session_id=track_action.action_id,
    )
    if seed["valid"]:
        seed_value = f"VALID  {seed['valid_for_seconds']:.1f}s"
        seed_color = (70, 240, 70)
    elif track_action.active and seed["sample_count"]:
        seed_value = (
            f"BUILDING  {seed['sample_count']}/{seed['sample_target']}"
        )
        seed_color = (0, 210, 255)
    else:
        seed_value = "NO VALID SEED"
        seed_color = (80, 80, 255)

    lidar = robot_state.get("lidar_person") or {}
    accepted_age = lidar.get("seed_accepted_age_seconds")
    if isinstance(accepted_age, (int, float)):
        accepted_age = float(accepted_age)
        if accepted_age <= 1.0:
            lidar_seed_value = f"ACCEPTED  {accepted_age:.1f}s AGO"
            lidar_seed_color = (70, 240, 70)
        else:
            lidar_seed_value = f"STALE  {accepted_age:.1f}s AGO"
            lidar_seed_color = (80, 80, 255)
    else:
        lidar_seed_value = "WAITING"
        lidar_seed_color = (0, 210, 255)

    lidar_state = str(lidar.get("state") or "offline").upper()
    pose_age = lidar.get("pose_age_seconds")
    lidar_pose_value = lidar_state
    if isinstance(pose_age, (int, float)):
        lidar_pose_value += f"  {float(pose_age):.1f}s"
    lidar_pose_color = (
        (70, 240, 70)
        if lidar_state in {"TRACKING", "COASTING"}
        else (0, 210, 255)
        if lidar_state in {"WAITING", "INITIALIZING", "RECOVERING"}
        else (80, 80, 255)
    )

    hand_value = str(hand_status.get("gesture") or "unavailable").upper()
    hand_state = str(hand_status.get("state") or "watching_person").upper()
    hand_color = (
        (0, 255, 255) if hand_value != "UNAVAILABLE" else (170, 170, 170)
    )

    rows = [
        ("TOF SEED", seed_value, seed_color),
        ("LIDAR SEED", lidar_seed_value, lidar_seed_color),
        ("LIDAR POSE", lidar_pose_value, lidar_pose_color),
        ("HAND", f"{hand_value}  |  {hand_state}", hand_color),
    ]
    width = frame.shape[1]
    scale = max(0.62, min(1.05, width / 1280.0))
    row_height = int(round(43 * scale))
    panel_width = min(width - 24, int(round(610 * scale)))
    panel_height = row_height * len(rows) + int(round(18 * scale))
    x1, y1 = 12, 12

    overlay = frame.copy()
    cv2.rectangle(
        overlay, (x1, y1), (x1 + panel_width, y1 + panel_height),
        (5, 5, 5), -1,
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    label_x = x1 + int(round(20 * scale))
    value_x = x1 + int(round(180 * scale))
    for index, (label, value, color) in enumerate(rows):
        baseline_y = y1 + int(round(33 * scale)) + index * row_height
        cv2.rectangle(
            frame,
            (x1, baseline_y - int(round(25 * scale))),
            (x1 + int(round(7 * scale)), baseline_y + int(round(6 * scale))),
            color,
            -1,
        )
        cv2.putText(
            frame, label, (label_x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX, scale * 0.72,
            (190, 190, 190), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, value, (value_x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX, scale * 0.82,
            color, 2, cv2.LINE_AA,
        )
    return frame


def draw_pipeline_fps_overlay(frame, camera_fps, yolo_fps, hand_fps):
    """Draw measured camera, YOLO, and MediaPipe hand inference rates."""
    rows = (
        f"CAMERA     {float(camera_fps):4.1f} FPS",
        f"YOLO       {float(yolo_fps):4.1f} FPS",
        f"HAND DET   {float(hand_fps):4.1f} FPS",
    )
    width = frame.shape[1]
    scale = max(0.50, min(0.72, width / 1800.0))
    thickness = 2
    text_sizes = [
        cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        for row in rows
    ]
    text_width = max(size[0][0] for size in text_sizes)
    text_height = max(size[0][1] for size in text_sizes)
    baseline = max(size[1] for size in text_sizes)
    padding = max(7, int(round(10 * scale)))
    row_height = text_height + baseline + padding
    panel_height = row_height * len(rows) + padding
    x = max(4, width - text_width - padding * 2 - 12)
    y = 12
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (min(width - 4, x + text_width + padding * 2),
         y + panel_height),
        (5, 5, 5),
        -1,
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    for index, row in enumerate(rows):
        cv2.putText(
            frame,
            row,
            (x + padding, y + padding + text_height + index * row_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (80, 240, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame


async def display_camera_loop(
    camera, csrt_tracker, yolo, track_action, hand_interface
):
    """Render the complete camera debug view."""
    window_name = "Robot Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fullscreen_requested = False
    last_fullscreen_attempt = 0.0
    camera_rate = SequenceFps()
    yolo_rate = SequenceFps()
    hand_rate = SequenceFps()

    while True:
        try:
            snapshot = camera.snapshot()
            now = time.monotonic()
            camera_fps = camera_rate.observe(snapshot.sequence, now)
            yolo_sequence, _ = yolo.detection_snapshot()
            yolo_fps = yolo_rate.observe(yolo_sequence, now)
            hand_status = hand_interface.gesture_status() or {}
            hand_fps = hand_rate.observe(
                hand_status.get("inference_sequence"), now
            )
            display_frame = cv2.flip(snapshot.full_bgr.copy(), 1)
            display_frame = draw_tof_overlay(display_frame)

            tracking_update = csrt_tracker.tracking_update
            if yolo.detections:
                display_frame = draw_yolo_overlay(display_frame, yolo)
            if tracking_update is not None and tracking_update.success:
                display_frame = draw_csrt_overlay(
                    display_frame,
                    tracking_update.target,
                    tracking_update.bbox_xywh,
                )
            display_frame = draw_hand_landmark_overlay(
                display_frame, hand_interface
            )
            display_frame = draw_tracking_status_overlay(
                display_frame, track_action, hand_interface
            )
            display_frame = draw_pipeline_fps_overlay(
                display_frame, camera_fps, yolo_fps, hand_fps
            )
            cv2.imshow(window_name, display_frame)
            cv2.waitKey(1)

            now = time.monotonic()
            if (
                cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                != cv2.WINDOW_FULLSCREEN
                and (
                    not fullscreen_requested
                    or now - last_fullscreen_attempt >= 0.5
                )
            ):
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
                fullscreen_requested = True
                last_fullscreen_attempt = now
        except RuntimeError:
            await asyncio.sleep(0.05)
            continue

        await asyncio.sleep(0.03)
