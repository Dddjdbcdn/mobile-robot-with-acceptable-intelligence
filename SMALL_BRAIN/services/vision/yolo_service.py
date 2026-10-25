import asyncio
from contextlib import asynccontextmanager
import threading
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


class YoloService:
    """Run either person-pose or household-object inference continuously."""

    def __init__(self, camera):
        model_dir = (
            Path(__file__).resolve().parents[2]
            / "vision_models"
            / "yolo_tools"
        )
        pose_model_path = model_dir / "yolo11n-pose_openvino_model"
        object_model_path = model_dir / "yoloe-11m_openvino_model"

        self.pose_model = YOLO(pose_model_path, task="pose")
        self.object_model = YOLO(object_model_path, task="detect")

        self.conf_threshold = 0.3
        self.camera = camera

        self.detections = []
        self.inference_thread = threading.Thread(target=self.timer_callback, daemon=True)
        self.running = False
        self.fps = 10.0
        self._state_condition = threading.Condition()
        self._inference_sequence = 0
        self._inference_frame_bgr = None
        self._inference_full_frame_bgr = None
        self._inference_source_captured_at = None
        self._inference_camera_sequence = None
        self._inference_started_at = None
        self._inference_completed_at = None
        self._inference_mode = None
        self._object_mode = False
        self._debug_object_mode = False

    @property
    def active(self):
        return self.running

    @property
    def object_mode(self):
        with self._state_condition:
            return self._object_mode

    @object_mode.setter
    def object_mode(self, enabled):
        """Select object inference and invalidate results from the old mode."""
        enabled = bool(enabled)
        with self._state_condition:
            if enabled == self._object_mode:
                return
            self._object_mode = enabled
            self.detections = []
            self._inference_frame_bgr = None
            self._inference_full_frame_bgr = None
            self._inference_source_captured_at = None
            self._inference_camera_sequence = None
            self._inference_started_at = None
            self._inference_completed_at = None
            self._inference_mode = None
            self._state_condition.notify_all()

    @property
    def debug_object_mode(self):
        with self._state_condition:
            return self._debug_object_mode

    def set_debug_object_mode(self, enabled):
        """Set the idle-only debug mode through the guarded UI entry point."""
        enabled = bool(enabled)
        with self._state_condition:
            mode_is_debug_owned = self._debug_object_mode
            self._debug_object_mode = enabled
        # Disabling an already-disabled debug mode must not interrupt an
        # action-owned object_detection_mode context.
        if enabled or mode_is_debug_owned:
            self.object_mode = enabled

    @asynccontextmanager
    async def object_detection_mode(self):
        """Enable object inference for one action and always restore pose mode."""
        self.object_mode = True
        try:
            yield
        finally:
            self.object_mode = False

    def detection_snapshot(self):
        """Return one inference generation and its detections atomically."""
        with self._state_condition:
            return self._inference_sequence, list(self.detections)

    def detection_snapshot_with_timing(self):
        """Return detections and monotonic timestamps from the same inference."""
        with self._state_condition:
            return (
                self._inference_sequence,
                list(self.detections),
                {
                    "source_captured_at": self._inference_source_captured_at,
                    "camera_sequence": self._inference_camera_sequence,
                    "inference_started_at": self._inference_started_at,
                    "inference_completed_at": self._inference_completed_at,
                },
            )

    def detection_snapshot_with_frame(self):
        """Return detections and their exact full-resolution source frame."""
        with self._state_condition:
            frame = self._inference_full_frame_bgr
            return (
                self._inference_sequence,
                list(self.detections),
                None if frame is None else frame.copy(),
                {
                    "source_captured_at": self._inference_source_captured_at,
                    "camera_sequence": self._inference_camera_sequence,
                    "inference_started_at": self._inference_started_at,
                    "inference_completed_at": self._inference_completed_at,
                },
            )

    def detection_snapshot_with_tracking_frame(self):
        """Return detections with the exact tracking frame used by YOLO."""
        with self._state_condition:
            frame = self._inference_frame_bgr
            return (
                self._inference_sequence,
                list(self.detections),
                None if frame is None else frame.copy(),
                {
                    "source_captured_at": self._inference_source_captured_at,
                    "camera_sequence": self._inference_camera_sequence,
                    "inference_started_at": self._inference_started_at,
                    "inference_completed_at": self._inference_completed_at,
                },
            )

    def _wait_for_inference_after(self, sequence, timeout):
        with self._state_condition:
            return self._state_condition.wait_for(
                lambda: (
                    self._inference_sequence > sequence
                    or not self.running
                ),
                timeout=timeout,
            ) and self._inference_sequence > sequence

    async def wait_for_inference_after(self, sequence, timeout=2.0):
        """Wait without blocking asyncio for one fresh accepted inference."""
        return await asyncio.to_thread(
            self._wait_for_inference_after, sequence, timeout
        )

    def start_background(self):
        dummy_frame = np.zeros(
            (320, 640, 3),
            dtype=np.uint8
        )
        # Multiple passes help ensure compilation/caching is complete.
        for model in (self.pose_model, self.object_model):
            for _ in range(3):
                model.predict(
                    source=dummy_frame,
                    device="intel:gpu",
                    verbose=False,
                )

        self.running = True
        self.inference_thread.start()

    async def wait_until_ready(self):
        while not self.running:
            await asyncio.sleep(0.05)

    def timer_callback(self):
        while self.running:
            loop_start = time.perf_counter()

            latest = self.camera.latest

            if latest is None:
                time.sleep(0.01)
                continue

            frame = latest.tracking_bgr
            full_frame = latest.full_bgr
            source_captured_at = latest.captured_at
            camera_sequence = latest.sequence
            inference_started_at = time.monotonic()

            with self._state_condition:
                object_mode = self._object_mode

            model = self.object_model if object_mode else self.pose_model
            results = model.predict(
                source=frame,
                device="intel:gpu",
                verbose=False,
            )
            detections = (
                self.parse_objects(results[0])
                if object_mode
                else self.parse_pose(results[0])
            )

            inference_completed_at = time.monotonic()
            current = self.camera.latest
            with self._state_condition:
                if (
                    current is not None
                    and self.running
                    and object_mode == self._object_mode
                ):
                    self.detections = detections
                    self._inference_frame_bgr = frame
                    self._inference_full_frame_bgr = full_frame
                    self._inference_source_captured_at = source_captured_at
                    self._inference_camera_sequence = camera_sequence
                    self._inference_started_at = inference_started_at
                    self._inference_completed_at = inference_completed_at
                    self._inference_mode = (
                        "object" if object_mode else "pose"
                    )
                    self._inference_sequence += 1
                    self._state_condition.notify_all()

            elapsed = time.perf_counter() - loop_start
            remaining = 1/self.fps - elapsed

            if remaining > 0:
                time.sleep(remaining)

    async def close(self):
        self.running = False
        with self._state_condition:
            self._state_condition.notify_all()
        await asyncio.to_thread(self.inference_thread.join, 2)

    def parse_objects(self, result):
        """Convert YOLOE household detections to the shared detection shape."""
        detections = []
        frame_height, frame_width = result.orig_shape

        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < self.conf_threshold:
                continue

            class_id = int(box.cls[0])
            names = result.names
            class_name = (
                names.get(class_id, str(class_id))
                if isinstance(names, dict)
                else names[class_id]
            )
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": str(class_name),
                "class_id": class_id,
                "confidence": round(confidence, 3),
                "bbox": self._bbox(
                    x1, y1, x2, y2, frame_width, frame_height
                ),
            })

        return detections

    @staticmethod
    def _bbox(x1, y1, x2, y2, frame_width, frame_height):
        return {
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1),
            "normalized_x1": round(float(x1) / frame_width, 4),
            "normalized_y1": round(float(y1) / frame_height, 4),
            "normalized_x2": round(float(x2) / frame_width, 4),
            "normalized_y2": round(float(y2) / frame_height, 4),
            "normalized_center_x": round(
                ((x1 + x2) * 0.5) / frame_width, 4
            ),
            "normalized_center_y": round(
                ((y1 + y2) * 0.5) / frame_height, 4
            ),
        }

    def parse_pose(self, result):
        keypoint_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        detections = []
        frame_height, frame_width = result.orig_shape

        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])

            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            keypoints_xy = result.keypoints.xy[i].cpu().numpy()
            keypoints_xyn = result.keypoints.xyn[i].cpu().numpy()
            keypoints_conf = result.keypoints.conf[i].cpu().numpy()

            human_keypoints = {}
            human_keypoint_confidences = {}

            for name, (x, y), (nx, ny), kp_conf in zip(
                keypoint_names,
                keypoints_xy,
                keypoints_xyn,
                keypoints_conf,
            ):
                human_keypoint_confidences[name] = round(
                    float(kp_conf), 3
                )

                if kp_conf < self.conf_threshold: continue
                
                human_keypoints[name] = {
                    "x": round(float(x), 1),
                    "y": round(float(y), 1),
                    "normalized_x": round(float(nx), 4),
                    "normalized_y": round(float(ny), 4),
                    "confidence": round(float(kp_conf), 3),
                }

            detections.append({
                "class": "person",
                "class_id": 0,
                "confidence": round(conf, 3),

                "bbox": self._bbox(
                    x1, y1, x2, y2, frame_width, frame_height
                ),

                "keypoints": human_keypoints,
                "keypoint_confidences": human_keypoint_confidences,
            })

        return detections
