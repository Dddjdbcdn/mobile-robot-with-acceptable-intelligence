"""Camera visualization and sensor-projection helpers."""

import asyncio
import subprocess
import time

import cv2
import numpy as np

from cognition.manager.world_state import robot_state
from services.vision.hand_landmark_service import HAND_CONNECTIONS

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


class CpuUsage:
    """Measure whole-system CPU utilization from Linux ``/proc/stat``."""

    def __init__(self, sample_interval_s=0.5):
        self.sample_interval_s = float(sample_interval_s)
        self._last_sample_at = None
        self._last_total = None
        self._last_idle = None
        self._usage = 0.0

    @staticmethod
    def _read_times():
        with open("/proc/stat", encoding="ascii") as stat_file:
            fields = stat_file.readline().split()
        if not fields or fields[0] != "cpu" or len(fields) < 5:
            raise ValueError("unexpected /proc/stat CPU data")

        # guest and guest_nice are already included in user and nice, so do
        # not count those final fields twice in the total.
        times = [int(value) for value in fields[1:9]]
        idle = times[3] + (times[4] if len(times) > 4 else 0)
        return sum(times), idle

    def sample(self, now=None):
        now = time.monotonic() if now is None else float(now)
        if (
            self._last_sample_at is not None
            and now - self._last_sample_at < self.sample_interval_s
        ):
            return self._usage

        try:
            total, idle = self._read_times()
        except (OSError, ValueError):
            return self._usage

        if self._last_total is not None:
            total_delta = total - self._last_total
            idle_delta = idle - self._last_idle
            if total_delta > 0:
                self._usage = max(
                    0.0,
                    min(100.0, 100.0 * (total_delta - idle_delta) / total_delta),
                )

        self._last_total = total
        self._last_idle = idle
        self._last_sample_at = now
        return self._usage


class WifiStatus:
    """Read the active NetworkManager Wi-Fi name at a low polling rate."""

    def __init__(self, sample_interval_s=2.0):
        self.sample_interval_s = float(sample_interval_s)
        self._last_sample_at = None
        self._connection_name = None

    def sample(self, now=None):
        now = time.monotonic() if now is None else float(now)
        if (
            self._last_sample_at is not None
            and now - self._last_sample_at < self.sample_interval_s
        ):
            return self._connection_name
        self._last_sample_at = now
        try:
            result = subprocess.run(
                [
                    "nmcli", "-t", "-f", "TYPE,STATE,CONNECTION",
                    "device", "status",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            self._connection_name = None
            return None
        self._connection_name = next((
            fields[2].replace(r"\:", ":")
            for line in result.stdout.splitlines()
            if len(fields := line.split(":", 2)) == 3
            and fields[0] == "wifi"
            and fields[1] == "connected"
            and fields[2]
        ), None)
        return self._connection_name

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


def _detection_box_iou(first, second):
    """Return pixel-box IoU for display-only track annotation matching."""
    try:
        first_box = first["bbox"]
        second_box = second["bbox"]
        first_values = tuple(float(first_box[key]) for key in (
            "x1", "y1", "x2", "y2"))
        second_values = tuple(float(second_box[key]) for key in (
            "x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return 0.0
    left = max(first_values[0], second_values[0])
    top = max(first_values[1], second_values[1])
    right = min(first_values[2], second_values[2])
    bottom = min(first_values[3], second_values[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first_values[2] - first_values[0]) * max(
        0.0, first_values[3] - first_values[1])
    second_area = max(0.0, second_values[2] - second_values[0]) * max(
        0.0, second_values[3] - second_values[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _raw_detection_tracks(detections, person_tracks):
    """Keep every raw YOLO box and attach IDs only to matching tracks."""
    detection_tracks = [[detection, None] for detection in detections]
    available = set(range(len(detection_tracks)))
    for track in person_tracks or ():
        if (track.state != "confirmed" or not track.visible
                or track.person is None):
            continue
        candidates = [
            (_detection_box_iou(track.person, detection_tracks[index][0]),
             index)
            for index in available
            if detection_tracks[index][0].get("class") == "person"
        ]
        if not candidates:
            continue
        overlap, index = max(candidates)
        if overlap <= 0.0:
            continue
        detection_tracks[index][1] = track
        available.remove(index)
    return [(detection, track) for detection, track in detection_tracks]


def draw_yolo_overlay(frame, yolo, track_action, person_tracks=None):
    detection_tracks = _raw_detection_tracks(
        yolo.detections,
        person_tracks,
    )
    detections = [detection for detection, _ in detection_tracks]

    if not detections:
        return frame

    frame_h, frame_w = frame.shape[:2]

    # YOLO inference coordinates are based on 640x360 frames
    scale_x = frame_w / 640.0
    scale_y = frame_h / 360.0

    tracked_box_color = (255, 0, 0)  # blue
    other_person_box_color = (0, 0, 255)  # red
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

    for detection, person_track in detection_tracks:
        bbox = detection.get("bbox")

        if not bbox:
            continue

        class_name = detection.get("class", "unknown")
        confidence = detection.get("confidence", 0.0)
        tracked_person_id = getattr(track_action, "tracked_person_id", None)
        is_tracked_person = (
            class_name == "person"
            and person_track is not None
            and tracked_person_id is not None
            and person_track.track_id == tracked_person_id
        )
        box_color = (
            tracked_box_color
            if is_tracked_person
            else other_person_box_color
        )

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
        keypoints = detection.get("keypoints") or {}
        if class_name == "person":
            has_shoulder = any(name in keypoints for name in (
                "left_shoulder", "right_shoulder"))
            has_hip = any(name in keypoints for name in (
                "left_hip", "right_hip"))
            label += (
                f" | KP:{len(keypoints)} "
                f"S:{int(has_shoulder)} H:{int(has_hip)}"
            )
        if person_track is not None:
            match = person_track.appearance_match_source
            if person_track.reid_similarity is not None:
                match += f":{person_track.reid_similarity:.2f}"
            label += (
                f" | ID {person_track.track_id} "
                f"{match} A:{person_track.appearance_samples}"
            )
        label_lines = [(label, 0.55, 2)]
        if (person_track is not None
                and person_track.association_cost is not None):
            appearance = (
                "--"
                if person_track.appearance_cost is None
                else f"{person_track.appearance_cost:.2f}"
            )
            pose = (
                "--"
                if person_track.pose_cost is None
                else (
                    f"{person_track.pose_cost:.2f}"
                    f"{'*' if person_track.pose_cost_is_fallback else ''}"
                )
            )
            cost_label = (
                f"C{person_track.association_cost:.2f} "
                f"G{person_track.geometry_cost:.2f} "
                f"A{appearance} P{pose}"
            )
            label_lines.append((cost_label, 0.64, 2))

        line_metrics = [
            (*cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                thickness,
            ), text, scale, thickness)
            for text, scale, thickness in label_lines
        ]
        padding_x = 5
        padding_y = 4
        line_gap = 2
        panel_width = min(
            frame_w,
            max(size[0] for size, _, _, _, _ in line_metrics)
            + 2 * padding_x,
        )
        panel_height = (
            2 * padding_y
            + sum(size[1] + baseline
                  for size, baseline, _, _, _ in line_metrics)
            + line_gap * (len(line_metrics) - 1)
        )
        label_y = max(y1, panel_height)
        label_x = max(0, min(x1, frame_w - panel_width))
        panel_top = label_y - panel_height

        cv2.rectangle(
            frame,
            (label_x, panel_top),
            (label_x + panel_width, label_y),
            box_color,
            -1,
        )
        text_y = panel_top + padding_y
        for size, baseline, text, scale, thickness in line_metrics:
            text_y += size[1]
            cv2.putText(
                frame,
                text,
                (label_x + padding_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            text_y += baseline + line_gap

        # ------------------------------------------------
        # Every raw person keeps its pose skeleton visible during acquisition.
        # ------------------------------------------------
        if class_name != "person":
            continue

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

        if is_tracked_person:
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


def compact_stm32_health_message(message):
    """Return a short status and readable diagnostic for firmware health."""
    fields = str(message or "").strip().split()
    if not fields:
        return "", ""
    level = fields[0].upper()
    if level == "OK":
        return level, ""
    details = dict(
        field.split("=", 1) for field in fields[1:] if "=" in field
    )
    detail_parts = []
    failing = str(details.get("failing") or "").replace("_", " ").upper()
    reason = str(details.get("reason") or "").replace("_", " ").upper()
    if failing:
        detail_parts.append(failing)
    if reason:
        detail_parts.append(reason)
    rc = details.get("rc")
    if rc not in {None, "0"}:
        detail_parts.append(f"RC {rc}")
    return level, "  |  ".join(detail_parts)


def stm32_health_display(health):
    """Return STM32 status, secondary diagnostic, and BGR status color."""
    health = health or {}
    message = str(health.get("message") or "").strip()
    age = health.get("age_seconds")
    if not message:
        return "NO DATA", "NO HEALTH MESSAGE", (80, 80, 255)
    if isinstance(age, (int, float)) and age > 5.0:
        return "STALE", f"LAST UPDATE {float(age):.1f}s AGO", (80, 80, 255)
    status, detail = compact_stm32_health_message(message)
    if message == "OK":
        return status, detail, (70, 240, 70)
    if message.startswith("DEGRADED"):
        return status, detail, (0, 210, 255)
    return status, detail, (80, 80, 255)


def navigation_status_display(status):
    """Return the compact Nav2 value and color used by the debug board."""
    status = status or {}
    state = str(status.get("state") or "idle").lower()
    labels = {
        "idle": "IDLE",
        "sending": "SENDING",
        "navigating": "MOVING",
        "waiting_target": "WAIT TARGET",
        "waiting_map": "WAIT MAP",
        "recovery_spin": "RECOVERY SPIN",
        "succeeded": "REACHED",
        "canceled": "CANCELED",
        "rejected": "REJECTED",
        "failed": "FAILED",
    }
    value = labels.get(state, state.replace("_", " ").upper())
    green = (70, 240, 70)
    yellow = (0, 210, 255)
    red = (80, 80, 255)
    gray = (170, 170, 170)
    color = (
        green
        if state in {"navigating", "succeeded"}
        else yellow
        if state in {"sending", "waiting_target", "waiting_map", "recovery_spin"}
        else red
        if state in {"failed", "rejected", "canceled"}
        else gray
    )

    if state == "navigating":
        distance = status.get("distance_remaining")
        if isinstance(distance, (int, float)) and np.isfinite(distance):
            value += f"  |  {max(0.0, float(distance)):.1f}m"
        recoveries = max(0, int(status.get("recoveries") or 0))
        value += f"  |  R{recoveries}"
        if recoveries:
            color = yellow
    elif state in {"failed", "rejected"}:
        error_code = status.get("error_code")
        if error_code not in {None, 0, "0"}:
            value += f"  |  CODE {error_code}"
    return value, color


def voice_status_display(state):
    """Return the bracketed local voice state and its display color."""
    state = str(state or "idle").lower()
    labels = {
        "listening": "[LISTENING]",
        "talking": "[TALKING]",
        "idle": "[IDLE]",
    }
    value = labels.get(state, f"[{state.replace('_', ' ').upper()}]")
    color = (
        (0, 210, 255)
        if state == "listening"
        else (70, 240, 70)
        if state == "talking"
        else (170, 170, 170)
    )
    return value, color


class ResetOverlayState:
    """Hold a short READY confirmation after a blocking reset state clears."""

    def __init__(self, ready_seconds=3.0):
        self.ready_seconds = float(ready_seconds)
        self.previous = None
        self.ready_until = 0.0

    def visible_state(self, state, now):
        state = str(state or "").lower()
        if state != self.previous:
            if state == "ready" and self.previous in {
                "initializing", "lifted", "resetting_lift", "resetting_manual",
            }:
                self.ready_until = float(now) + self.ready_seconds
            self.previous = state
        if state and state != "ready":
            return state
        if state == "ready" and float(now) < self.ready_until:
            return "ready"
        return None


def draw_reset_state_overlay(frame, state):
    """Draw a high-priority centered reset/recovery layer."""
    messages = {
        "initializing": ("INITIALIZING", "KEEP ROBOT STILL", (0, 210, 255)),
        "lifted": ("ROBOT LIFTED", "PLACE ROBOT DOWN", (80, 80, 255)),
        "resetting_lift": ("RESETTING MAP", "PLEASE WAIT", (0, 210, 255)),
        "resetting_manual": ("RESETTING ROBOT", "PLEASE WAIT", (0, 210, 255)),
        "ready": ("READY", "ROBOT CAN MOVE", (70, 240, 70)),
    }
    if state not in messages:
        return frame
    title, detail, color = messages[state]
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (5, 5, 5), -1)
    opacity = 0.45 if state == "ready" else 0.68
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)

    panel_width = min(width - 32, max(420, int(width * 0.52)))
    panel_height = min(height - 32, 150)
    left = (width - panel_width) // 2
    top = (height - panel_height) // 2
    right = left + panel_width
    bottom = top + panel_height
    cv2.rectangle(frame, (left, top), (right, bottom), (12, 15, 20), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 4, cv2.LINE_AA)

    title_scale = max(0.8, min(1.35, width / 1000.0))
    detail_scale = title_scale * 0.58
    title_size = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 3
    )[0]
    detail_size = cv2.getTextSize(
        detail, cv2.FONT_HERSHEY_SIMPLEX, detail_scale, 2
    )[0]
    cv2.putText(
        frame, title,
        ((width - title_size[0]) // 2, top + 62),
        cv2.FONT_HERSHEY_SIMPLEX, title_scale, color, 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, detail,
        ((width - detail_size[0]) // 2, top + 112),
        cv2.FONT_HERSHEY_SIMPLEX, detail_scale,
        (225, 225, 225), 2, cv2.LINE_AA,
    )
    return frame


def tracking_panel_width(rows, frame_width, scale):
    """Size the left debug board to its current text instead of a fixed box."""
    label_offset = int(round(20 * scale))
    value_offset = int(round(180 * scale))
    right_padding = int(round(20 * scale))
    required = 0
    for label, value, _color in rows:
        label_width = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.72, 2
        )[0][0]
        value_width = cv2.getTextSize(
            value, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.82, 2
        )[0][0]
        required = max(
            required,
            label_offset + label_width,
            value_offset + value_width,
        )
    return min(frame_width - 24, required + right_padding)


def person_tracking_status_display(person_tracks):
    """Format the single person identity owned by the target tracker."""
    track = next(iter(person_tracks or ()), None)
    if track is None:
        return "ID --  |  0 IMAGES  |  --", (170, 170, 170)

    samples = int(track.appearance_samples)
    image_label = "IMAGE" if samples == 1 else "IMAGES"
    update_age = track.gallery_update_age_seconds
    updated = "--" if update_age is None else f"{float(update_age):.1f}s"
    value = f"ID {track.track_id}  |  {samples} {image_label}  |  {updated}"
    color = (70, 240, 70) if track.visible else (0, 210, 255)
    return value, color


def draw_tracking_status_overlay(
    frame,
    track_action,
    hand_interface,
    person_tracks=(),
    voice_state="idle",
):
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

    action = hand_status.get("action") or {}
    action_state = str(action.get("state") or "idle").upper()
    action_type = str(action.get("action_type") or "none").replace(
        "_", " "
    ).upper()
    action_target = str(action.get("target") or "-").replace(
        "_", " "
    ).upper()
    action_value = f"{action_state}  |  {action_type}  |  {action_target}"
    action_color = (
        (70, 240, 70)
        if action_state in {"RUNNING", "NAVIGATING", "FOLLOWING"}
        else (170, 170, 170)
        if action_state == "IDLE"
        else (0, 210, 255)
    )
    person_value, person_color = person_tracking_status_display(person_tracks)

    navigation_value, navigation_color = navigation_status_display(
        robot_state.get("navigation")
    )
    voice_value, voice_color = voice_status_display(voice_state)

    rows = [
        ("ACTION", action_value, action_color),
        ("VOICE", voice_value, voice_color),
        ("NAV2", navigation_value, navigation_color),
        ("PERSON", person_value, person_color),
        ("TOF SEED", seed_value, seed_color),
        ("LIDAR SEED", lidar_seed_value, lidar_seed_color),
        ("LIDAR POSE", lidar_pose_value, lidar_pose_color),
        ("HAND", f"{hand_value}  |  {hand_state}", hand_color),
    ]
    width = frame.shape[1]
    scale = max(0.62, min(1.05, width / 1280.0))
    row_height = int(round(43 * scale))
    panel_width = tracking_panel_width(rows, width, scale)
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


def draw_pipeline_fps_overlay(
    frame,
    camera_fps,
    yolo_fps,
    hand_fps,
    cpu_usage=0.0,
    wifi_value="DISCONNECTED",
    wifi_connected=False,
    stm32_status="NO DATA",
    stm32_detail="NO HEALTH MESSAGE",
    stm32_color=(80, 80, 255),
):
    """Draw performance, Wi-Fi, and firmware status on the right panel."""
    yellow = (80, 240, 255)
    rows = (
        (f"CAMERA     {float(camera_fps):4.1f} FPS", yellow),
        (f"YOLO       {float(yolo_fps):4.1f} FPS", yellow),
        (f"HAND DET   {float(hand_fps):4.1f} FPS", yellow),
        (f"CPU        {float(cpu_usage):4.1f} %", yellow),
        (
            f"WIFI       {wifi_value}",
            yellow if wifi_connected else (80, 80, 255),
        ),
        (f"STM32      {stm32_status}", stm32_color),
    )
    width = frame.shape[1]
    scale = max(0.50, min(0.72, width / 1800.0))
    thickness = 2
    padding = max(7, int(round(10 * scale)))
    max_content_width = min(width - 24, max(220, int(width * 0.30)))

    def fit_text(text, font_scale, font_thickness):
        if cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )[0][0] <= max_content_width:
            return text
        suffix = "..."
        while text and cv2.getTextSize(
                text + suffix,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
        )[0][0] > max_content_width:
            text = text[:-1]
        return text.rstrip() + suffix

    fitted_rows = tuple(
        (fit_text(row, scale, thickness), color) for row, color in rows
    )
    detail_scale = scale * 0.70
    detail_thickness = 1
    fitted_detail = fit_text(
        str(stm32_detail or ""), detail_scale, detail_thickness
    )
    text_sizes = [
        cv2.getTextSize(row[0], cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        for row in fitted_rows
    ]
    text_width = max(size[0][0] for size in text_sizes)
    if fitted_detail:
        detail_size, detail_baseline = cv2.getTextSize(
            fitted_detail,
            cv2.FONT_HERSHEY_SIMPLEX,
            detail_scale,
            detail_thickness,
        )
        text_width = max(text_width, detail_size[0])
    else:
        detail_size = (0, 0)
        detail_baseline = 0
    text_height = max(size[0][1] for size in text_sizes)
    baseline = max(size[1] for size in text_sizes)
    row_height = text_height + baseline + padding
    main_last_y = padding + text_height + (len(fitted_rows) - 1) * row_height
    detail_y = main_last_y + detail_size[1] + max(4, padding // 2)
    panel_height = (
        detail_y + detail_baseline + padding
        if fitted_detail else main_last_y + baseline + padding
    )
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
    for index, (row, color) in enumerate(fitted_rows):
        cv2.putText(
            frame,
            row,
            (x + padding, y + padding + text_height + index * row_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    if fitted_detail:
        cv2.putText(
            frame,
            fitted_detail,
            (x + padding, y + detail_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            detail_scale,
            stm32_color,
            detail_thickness,
            cv2.LINE_AA,
        )
    return frame


def draw_connection_lost_warning(frame):
    """Draw an always-visible warning when the AI connection is gone."""
    label = "WIFI / AI CONNECTION LOST - PRESS STOP ROBOT"
    scale = max(0.58, min(0.92, frame.shape[1] / 1500.0))
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    padding_x, padding_y = 18, 10
    left = max(8, (frame.shape[1] - text_width) // 2 - padding_x)
    right = min(frame.shape[1] - 8, left + text_width + padding_x * 2)
    top = 8
    bottom = top + text_height + baseline + padding_y * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (20, 20, 110), -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
    cv2.rectangle(frame, (left, top), (right, bottom), (80, 80, 255), 2)
    cv2.putText(
        frame,
        label,
        (left + padding_x, bottom - padding_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (220, 220, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def draw_debug_board_toggle(frame, enabled, above_bounds=None):
    """Draw the small always-visible control for the optional debug boards."""
    label = f"DEBUG {'ON' if enabled else 'OFF'}"
    scale = max(0.48, min(0.68, frame.shape[1] / 1900.0))
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    padding_x = 10
    padding_y = 7
    margin = 12
    left = margin if above_bounds is None else above_bounds[0]
    bottom = (
        frame.shape[0] - margin
        if above_bounds is None
        else above_bounds[1] - 10
    )
    right = min(frame.shape[1], left + text_width + padding_x * 2)
    top = max(0, bottom - text_height - baseline - padding_y * 2)
    color = (70, 240, 70) if enabled else (170, 170, 170)

    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (5, 5, 5), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (left + padding_x, bottom - padding_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return left, top, right, bottom


def debug_board_toggle_clicked(event, x, y, bounds):
    """Return whether an OpenCV mouse event activates the debug control."""
    if event != cv2.EVENT_LBUTTONUP or bounds is None:
        return False
    left, top, right, bottom = bounds
    return left <= x <= right and top <= y <= bottom


def _draw_control_button(frame, label, left, bottom, color):
    scale = max(0.48, min(0.68, frame.shape[1] / 1900.0))
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    padding_x, padding_y = 10, 7
    right = left + text_width + padding_x * 2
    top = bottom - text_height - baseline - padding_y * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (5, 5, 5), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 1, cv2.LINE_AA)
    cv2.putText(
        frame, label, (left + padding_x, bottom - padding_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
    )
    return left, top, right, bottom


def draw_audio_controls(frame, output_muted, input_muted):
    """Draw independent speaker and microphone controls."""
    bottom = frame.shape[0] - 12
    speaker_label = "UNMUTE" if output_muted else "MUTE"
    speaker_color = (80, 80, 255) if output_muted else (70, 240, 70)
    speaker = _draw_control_button(
        frame, speaker_label, 12, bottom, speaker_color
    )
    ear_label = "UNCOVER EAR [E]" if input_muted else "COVER EAR"
    ear_color = (80, 80, 255) if input_muted else (70, 240, 70)
    ear = _draw_control_button(
        frame, ear_label, speaker[2] + 10, bottom, ear_color
    )
    return {"speaker": speaker, "ear": ear}


def draw_vision_mode_control(frame, object_mode, enabled, left=12):
    """Draw the idle-only pose/object inference switch."""
    bottom = frame.shape[0] - 12
    mode = "OBJECT" if object_mode else "POSE"
    label = f"YOLO {mode} [O]" if enabled else f"YOLO {mode} LOCKED"
    color = (
        (0, 190, 255) if object_mode else (70, 240, 70)
    ) if enabled else (120, 120, 120)
    return _draw_control_button(frame, label, left, bottom, color)


def control_button_clicked(event, x, y, bounds):
    if event != cv2.EVENT_LBUTTONUP or bounds is None:
        return False
    left, top, right, bottom = bounds
    return left <= x <= right and top <= y <= bottom


async def display_camera_loop(
    camera,
    csrt_tracker,
    yolo,
    track_action,
    hand_interface,
    person_tracking_service=None,
    audio_app=None,
    connection_status=None,
    cognition_manager=None,
):
    """Render the complete camera debug view."""
    window_name = "Robot Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fullscreen_requested = False
    last_fullscreen_attempt = 0.0
    camera_rate = SequenceFps()
    yolo_rate = SequenceFps()
    hand_rate = SequenceFps()
    cpu_usage = CpuUsage()
    wifi_status = WifiStatus()
    reset_overlay = ResetOverlayState()
    person_tracks = ()
    debug_control = {
        "visible": True,
        "button_bounds": None,
        "audio_bounds": {},
        "vision_bounds": None,
    }

    def toggle_vision_mode():
        if cognition_manager is None:
            return False
        return cognition_manager.toggle_debug_object_mode()

    def handle_mouse(event, x, y, _flags, _parameter):
        if debug_board_toggle_clicked(
                event, x, y, debug_control["button_bounds"]):
            debug_control["visible"] = not debug_control["visible"]
        elif audio_app is not None and control_button_clicked(
                event, x, y, debug_control["audio_bounds"].get("speaker")):
            audio_app.set_output_muted(not audio_app.output_muted)
        elif audio_app is not None and control_button_clicked(
                event, x, y, debug_control["audio_bounds"].get("ear")):
            audio_app.set_input_muted(not audio_app.input_muted)
        elif control_button_clicked(
                event, x, y, debug_control["vision_bounds"]):
            toggle_vision_mode()

    cv2.setMouseCallback(window_name, handle_mouse)

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

            if person_tracking_service is not None:
                _, person_tracks = person_tracking_service.tracking_snapshot()
            display_frame = cv2.flip(snapshot.full_bgr.copy(), 1)
            display_frame = draw_tof_overlay(display_frame)

            tracking_update = csrt_tracker.tracking_update
            if yolo.detections or person_tracks:
                display_frame = draw_yolo_overlay(
                    display_frame,
                    yolo,
                    track_action,
                    (person_tracks
                     if person_tracking_service is not None else None),
                )
            if tracking_update is not None and tracking_update.success:
                display_frame = draw_csrt_overlay(
                    display_frame,
                    tracking_update.target,
                    tracking_update.bbox_xywh,
                )
            display_frame = draw_hand_landmark_overlay(
                display_frame, hand_interface
            )
            cpu_percent = cpu_usage.sample(now)
            wifi_name = wifi_status.sample(now)
            connection_lost = bool(
                connection_status is not None
                and connection_status.get("lost")
            )
            wifi_connected = wifi_name is not None and not connection_lost
            wifi_value = (
                "LOST CONNECTION"
                if connection_lost or wifi_name is None
                else wifi_name.upper()
            )
            stm32_status, stm32_detail, stm32_color = stm32_health_display(
                robot_state.get("stm32_health")
            )
            if debug_control["visible"]:
                display_frame = draw_tracking_status_overlay(
                    display_frame,
                    track_action,
                    hand_interface,
                    person_tracks,
                    audio_app.voice_state() if audio_app is not None else "idle",
                )
                display_frame = draw_pipeline_fps_overlay(
                    display_frame,
                    camera_fps,
                    yolo_fps,
                    hand_fps,
                    cpu_percent,
                    wifi_value,
                    wifi_connected,
                    stm32_status,
                    stm32_detail,
                    stm32_color,
                )
            if connection_lost or wifi_name is None:
                display_frame = draw_connection_lost_warning(display_frame)
            visible_reset_state = reset_overlay.visible_state(
                robot_state.get("reset_state"), now
            )
            if visible_reset_state is not None:
                display_frame = draw_reset_state_overlay(
                    display_frame, visible_reset_state
                )
            if audio_app is not None:
                debug_control["audio_bounds"] = draw_audio_controls(
                    display_frame,
                    audio_app.output_muted,
                    audio_app.input_muted,
                )
            vision_enabled = bool(
                cognition_manager is not None
                and cognition_manager.debug_object_mode_available()
            )
            audio_ear = debug_control["audio_bounds"].get("ear")
            vision_left = audio_ear[2] + 10 if audio_ear is not None else 12
            debug_control["vision_bounds"] = draw_vision_mode_control(
                display_frame,
                yolo.object_mode,
                vision_enabled,
                vision_left,
            )
            debug_control["button_bounds"] = draw_debug_board_toggle(
                display_frame,
                debug_control["visible"],
                debug_control["audio_bounds"].get("speaker")
                or debug_control["vision_bounds"],
            )
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("d"), ord("D")):
                debug_control["visible"] = not debug_control["visible"]
            elif audio_app is not None and key in (ord("m"), ord("M")):
                audio_app.set_output_muted(not audio_app.output_muted)
            elif audio_app is not None and key in (ord("e"), ord("E")):
                audio_app.set_input_muted(not audio_app.input_muted)
            elif key in (ord("o"), ord("O")):
                toggle_vision_mode()

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
