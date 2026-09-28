"""Right-hand landmark inference on a pose-guided dynamic crop."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
import time

import cv2
import numpy as np


LANDMARK_NAMES = {
    0: "wrist", 1: "thumb_cmc", 2: "thumb_mcp", 3: "thumb_ip",
    4: "thumb_tip", 5: "index_mcp", 6: "index_pip", 7: "index_dip",
    8: "index_tip", 9: "middle_mcp", 10: "middle_pip",
    11: "middle_dip", 12: "middle_tip", 13: "ring_mcp",
    14: "ring_pip", 15: "ring_dip", 16: "ring_tip", 17: "pinky_mcp",
    18: "pinky_pip", 19: "pinky_dip", 20: "pinky_tip",
}

HAND_CONNECTIONS = (
    ("wrist", "thumb_cmc"), ("thumb_cmc", "thumb_mcp"),
    ("thumb_mcp", "thumb_ip"), ("thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp"), ("index_mcp", "index_pip"),
    ("index_pip", "index_dip"), ("index_dip", "index_tip"),
    ("index_mcp", "middle_mcp"), ("middle_mcp", "middle_pip"),
    ("middle_pip", "middle_dip"), ("middle_dip", "middle_tip"),
    ("middle_mcp", "ring_mcp"), ("ring_mcp", "ring_pip"),
    ("ring_pip", "ring_dip"), ("ring_dip", "ring_tip"),
    ("ring_mcp", "pinky_mcp"), ("wrist", "pinky_mcp"),
    ("pinky_mcp", "pinky_pip"), ("pinky_pip", "pinky_dip"),
    ("pinky_dip", "pinky_tip"),
)


class HandLandmarkService:
    """Detect a person's right hand and return full-frame landmarks."""

    CROP_MIN_SIZE_PX = 224
    CROP_FOREARM_SCALE = 2.6
    CROP_WRIST_FORWARD_OFFSET = 0.35

    def __init__(self, camera, model_path=None, landmarker=None, image_factory=None):
        self.camera = camera
        self._inference_lock = asyncio.Lock()
        self._last_mediapipe_timestamp_ms = 0
        self.model_path = Path(model_path) if model_path else (
            Path(__file__).resolve().parents[1]
            / "vision_models" / "hand_landmark_tools" / "hand_landmarker.task"
        )
        if landmarker is not None:
            self.landmarker = landmarker
            self._image_factory = image_factory
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmark model is missing: {self.model_path}. "
                "Run `source setup.sh` once to download it."
            )

        import mediapipe as mp

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.30,
            min_hand_presence_confidence=0.30,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._image_factory = lambda rgb: mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb
        )

    @staticmethod
    def _point_pixels(point, width, height):
        normalized_x = point.get("normalized_x")
        normalized_y = point.get("normalized_y")
        if isinstance(normalized_x, (int, float)) and isinstance(
            normalized_y, (int, float)
        ):
            return float(normalized_x) * width, float(normalized_y) * height
        return float(point["x"]), float(point["y"])

    @staticmethod
    def _square_crop_box(width, height, center_x, center_y, size):
        size = max(1, min(int(round(size)), int(width), int(height)))
        x1 = int(round(float(center_x) - size / 2.0))
        y1 = int(round(float(center_y) - size / 2.0))
        x1 = min(max(0, x1), int(width) - size)
        y1 = min(max(0, y1), int(height) - size)
        return x1, y1, x1 + size, y1 + size

    @classmethod
    def _crop_box(cls, frame_shape, person):
        height, width = frame_shape[:2]
        keypoints = person.get("keypoints") or {}
        wrist = keypoints.get("right_wrist")
        elbow = keypoints.get("right_elbow")
        if not wrist or not elbow:
            return None

        wx, wy = cls._point_pixels(wrist, width, height)
        ex, ey = cls._point_pixels(elbow, width, height)
        crop_size = max(
            cls.CROP_MIN_SIZE_PX,
            math.hypot(wx - ex, wy - ey) * cls.CROP_FOREARM_SCALE,
        )
        center_x = wx + cls.CROP_WRIST_FORWARD_OFFSET * (wx - ex)
        center_y = wy + cls.CROP_WRIST_FORWARD_OFFSET * (wy - ey)
        return cls._square_crop_box(
            width, height, center_x, center_y, crop_size
        )

    @staticmethod
    def _save_debug_crop(crop, crop_box, points, debug_path):
        if debug_path is None:
            return None
        output = crop.copy()
        x1, y1, _, _ = crop_box
        if points:
            projected = {
                name: (int(point["x"] - x1), int(point["y"] - y1))
                for name, point in points.items()
            }
            for start_name, end_name in HAND_CONNECTIONS:
                start = projected.get(start_name)
                end = projected.get(end_name)
                if start is not None and end is not None:
                    cv2.line(output, start, end, (0, 255, 0), 2, cv2.LINE_AA)
            for point in projected.values():
                cv2.circle(output, point, 3, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            cv2.putText(
                output, "NO HAND LANDMARKS", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            )
        path = Path(debug_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), output):
            raise RuntimeError(f"Could not save hand landmark debug image: {path}")
        return str(path)

    def _detect(self, frame_bgr, person, debug_path=None):
        crop_box = self._crop_box(frame_bgr.shape, person)
        if crop_box is None:
            return None
        x1, y1, x2, y2 = crop_box
        crop = frame_bgr[y1:y2, x1:x2]
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        timestamp_ms = max(
            self._last_mediapipe_timestamp_ms + 1,
            time.monotonic_ns() // 1_000_000,
        )
        self._last_mediapipe_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(
            self._image_factory(rgb), timestamp_ms
        )
        if not result.hand_landmarks:
            self._save_debug_crop(crop, crop_box, None, debug_path)
            return None

        frame_height, frame_width = frame_bgr.shape[:2]
        crop_height, crop_width = crop.shape[:2]
        wrist_x, wrist_y = self._point_pixels(
            person["keypoints"]["right_wrist"], frame_width, frame_height
        )

        def full_point(point):
            pixel_x = x1 + float(point.x) * crop_width
            pixel_y = y1 + float(point.y) * crop_height
            transformed = {
                "x": pixel_x,
                "y": pixel_y,
                "normalized_x": pixel_x / frame_width,
                "normalized_y": pixel_y / frame_height,
            }
            if hasattr(point, "z"):
                transformed["z"] = float(point.z)
            return transformed

        hands = []
        for landmarks in result.hand_landmarks:
            points = {
                name: full_point(landmarks[index])
                for index, name in LANDMARK_NAMES.items()
            }
            distance = math.hypot(
                points["wrist"]["x"] - wrist_x,
                points["wrist"]["y"] - wrist_y,
            )
            hands.append((distance, points))
        points = min(hands, key=lambda item: item[0])[1]
        saved_debug_path = self._save_debug_crop(
            crop, crop_box, points, debug_path
        )
        return {
            "landmarks": points,
            "image_width": frame_width,
            "image_height": frame_height,
            "crop_size_px": crop.shape[1],
            "crop_box": list(crop_box),
            "mediapipe_timestamp_ms": timestamp_ms,
            "debug_image_path": saved_debug_path,
        }

    async def detect_right_hand(self, frame_bgr, person, debug_path=None):
        async with self._inference_lock:
            return await asyncio.to_thread(
                self._detect, frame_bgr, person, debug_path
            )

    async def close(self):
        async with self._inference_lock:
            await asyncio.to_thread(self.landmarker.close)
