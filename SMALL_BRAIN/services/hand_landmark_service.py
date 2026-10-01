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

    MIN_HAND_DETECTION_CONFIDENCE = 0.30
    MIN_HAND_PRESENCE_CONFIDENCE = 0.30
    CROP_MIN_SIZE_PX = 224
    CROP_FOREARM_SCALE = 2.6
    CROP_WRIST_FORWARD_OFFSET = 0.35
    CROP_SMOOTHING_ALPHA = 0.25
    CROP_HAND_FOLLOW_ALPHA = 0.20
    CROP_EXPANSION_FACTOR = 1.45
    CROP_EXPANSION_DECAY = 0.97
    RIGHT_ARM_CROP_MIN_CONFIDENCE = 0.50

    def __init__(self, camera, model_path=None, landmarker=None, image_factory=None):
        self.camera = camera
        self._inference_lock = asyncio.Lock()
        self._last_mediapipe_timestamp_ms = 0
        self._crop_state = None
        self._crop_frame_shape = None
        self._crop_expansion = 1.0
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
            num_hands=2,
            min_hand_detection_confidence=self.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=self.MIN_HAND_PRESENCE_CONFIDENCE,
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
        size = max(1, min(int(round(size)), max(int(width), int(height))))
        x1 = int(round(float(center_x) - size / 2.0))
        y1 = int(round(float(center_y) - size / 2.0))
        return x1, y1, x1 + size, y1 + size

    @staticmethod
    def _extract_padded_crop(frame, crop_box):
        x1, y1, x2, y2 = crop_box
        size = x2 - x1
        output = np.zeros((size, size, frame.shape[2]), dtype=frame.dtype)
        frame_height, frame_width = frame.shape[:2]
        source_x1 = max(0, x1)
        source_y1 = max(0, y1)
        source_x2 = min(frame_width, x2)
        source_y2 = min(frame_height, y2)
        if source_x1 >= source_x2 or source_y1 >= source_y2:
            return output
        destination_x1 = source_x1 - x1
        destination_y1 = source_y1 - y1
        destination_x2 = destination_x1 + source_x2 - source_x1
        destination_y2 = destination_y1 + source_y2 - source_y1
        output[
            destination_y1:destination_y2,
            destination_x1:destination_x2,
        ] = frame[source_y1:source_y2, source_x1:source_x2]
        return output

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

    @classmethod
    def _right_arm_confidence(cls, person, name):
        confidences = person.get("keypoint_confidences") or {}
        value = confidences.get(name)
        if isinstance(value, (int, float)):
            return float(value)
        point = (person.get("keypoints") or {}).get(name)
        if not point:
            return 0.0
        value = point.get("confidence")
        # Compatibility for pose sources created before per-joint confidence
        # was retained. An existing joint from such a source remains usable.
        return float(value) if isinstance(value, (int, float)) else 1.0

    @classmethod
    def _right_arm_crop_is_reliable(cls, person):
        return all(
            cls._right_arm_confidence(person, name)
            >= cls.RIGHT_ARM_CROP_MIN_CONFIDENCE
            for name in ("right_wrist", "right_elbow")
        )

    def _stabilized_crop_box(self, frame_shape, person):
        raw_box = self._crop_box(frame_shape, person)
        if raw_box is None:
            self._crop_state = None
            self._crop_frame_shape = None
            self._crop_expansion = 1.0
            return None

        height, width = frame_shape[:2]
        shape = (height, width)
        raw_x1, raw_y1, raw_x2, raw_y2 = raw_box
        desired = {
            "center_x": (raw_x1 + raw_x2) * 0.5,
            "center_y": (raw_y1 + raw_y2) * 0.5,
            "size": min(
                max(width, height),
                (raw_x2 - raw_x1) * self._crop_expansion,
            ),
        }
        crop_jump = False
        if self._crop_state is not None:
            crop_jump = math.hypot(
                desired["center_x"] - self._crop_state["center_x"],
                desired["center_y"] - self._crop_state["center_y"],
            ) > max(desired["size"], self._crop_state["size"]) * 1.5
        if (
            self._crop_state is None
            or self._crop_frame_shape != shape
            or crop_jump
        ):
            self._crop_state = desired
            self._crop_frame_shape = shape
        else:
            alpha = self.CROP_SMOOTHING_ALPHA
            for name, value in desired.items():
                self._crop_state[name] += alpha * (
                    value - self._crop_state[name]
                )

        return self._square_crop_box(
            width,
            height,
            self._crop_state["center_x"],
            self._crop_state["center_y"],
            self._crop_state["size"],
        )

    def _expand_next_crop(self, frame_shape):
        if self._crop_state is None:
            return
        maximum = max(frame_shape[:2])
        self._crop_expansion = min(
            maximum / self.CROP_MIN_SIZE_PX,
            self._crop_expansion * self.CROP_EXPANSION_FACTOR,
        )
        self._crop_state["size"] = min(
            maximum,
            self._crop_state["size"] * self.CROP_EXPANSION_FACTOR,
        )

    def _relax_crop_expansion(self):
        self._crop_expansion = max(
            1.0, self._crop_expansion * self.CROP_EXPANSION_DECAY
        )

    def _follow_detected_hand(self, points):
        if self._crop_state is None or not points:
            return
        xs = [point["x"] for point in points.values()]
        ys = [point["y"] for point in points.values()]
        hand_center_x = (min(xs) + max(xs)) * 0.5
        hand_center_y = (min(ys) + max(ys)) * 0.5
        alpha = self.CROP_HAND_FOLLOW_ALPHA
        self._crop_state["center_x"] += alpha * (
            hand_center_x - self._crop_state["center_x"]
        )
        self._crop_state["center_y"] += alpha * (
            hand_center_y - self._crop_state["center_y"]
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
        frame_height, frame_width = frame_bgr.shape[:2]
        use_full_frame = not self._right_arm_crop_is_reliable(person)
        if use_full_frame:
            crop_box = (0, 0, frame_width, frame_height)
        else:
            crop_box = self._stabilized_crop_box(frame_bgr.shape, person)
            if crop_box is None:
                return None

        wrist = (person.get("keypoints") or {}).get("right_wrist")
        if wrist:
            wrist_x, wrist_y = self._point_pixels(
                wrist, frame_width, frame_height
            )
        else:
            wrist_x, wrist_y = frame_width * 0.5, frame_height * 0.5
        x1, y1, _, _ = crop_box
        crop = (
            frame_bgr
            if use_full_frame
            else self._extract_padded_crop(frame_bgr, crop_box)
        )
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
            if not use_full_frame:
                self._expand_next_crop(frame_bgr.shape)
            self._save_debug_crop(crop, crop_box, None, debug_path)
            return None

        crop_height, crop_width = crop.shape[:2]

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
        handedness_hands = getattr(result, "handedness", None) or []
        for hand_index, landmarks in enumerate(result.hand_landmarks):
            points = {
                name: full_point(landmarks[index])
                for index, name in LANDMARK_NAMES.items()
            }
            handedness = None
            handedness_score = None
            if (
                hand_index < len(handedness_hands)
                and handedness_hands[hand_index]
            ):
                category = handedness_hands[hand_index][0]
                handedness = getattr(category, "category_name", None)
                score = getattr(category, "score", None)
                if score is not None:
                    handedness_score = float(score)
            distance = math.hypot(
                points["wrist"]["x"] - wrist_x,
                points["wrist"]["y"] - wrist_y,
            )
            hands.append((
                distance, points, handedness, handedness_score, landmarks,
            ))
        selectable_indices = [
            index
            for index, item in enumerate(hands)
            if str(item[2] or "").lower() != "left"
        ]
        selected_index = (
            min(
                selectable_indices,
                key=lambda index: hands[index][0],
            )
            if selectable_indices
            else None
        )
        detected_hands = [
            {
                "handedness": item[2],
                "handedness_score": item[3],
                "wrist": item[1]["wrist"],
                "selected": index == selected_index,
            }
            for index, item in enumerate(hands)
        ]
        if selected_index is None:
            saved_debug_path = self._save_debug_crop(
                crop, crop_box, None, debug_path
            )
            return {
                "landmarks": {},
                "handedness": None,
                "handedness_score": None,
                "mediapipe_hands": detected_hands,
                "image_width": frame_width,
                "image_height": frame_height,
                "inference_scope": "full_frame" if use_full_frame else "crop",
                "crop_size_px": crop.shape[1],
                "crop_box": list(crop_box),
                "mediapipe_timestamp_ms": timestamp_ms,
                "debug_image_path": saved_debug_path,
            }
        selected = hands[selected_index]
        crop_landmarks = selected[4]
        touches_edge = any(
            float(point.x) <= 0.03
            or float(point.x) >= 0.97
            or float(point.y) <= 0.03
            or float(point.y) >= 0.97
            for point in crop_landmarks
        )
        if not use_full_frame:
            if touches_edge:
                self._expand_next_crop(frame_bgr.shape)
            else:
                self._relax_crop_expansion()

        _, points, handedness, handedness_score, _ = selected
        self._follow_detected_hand(points)
        saved_debug_path = self._save_debug_crop(
            crop, crop_box, points, debug_path
        )
        return {
            "landmarks": points,
            "handedness": handedness,
            "handedness_score": handedness_score,
            "mediapipe_hands": detected_hands,
            "image_width": frame_width,
            "image_height": frame_height,
            "inference_scope": "full_frame" if use_full_frame else "crop",
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
