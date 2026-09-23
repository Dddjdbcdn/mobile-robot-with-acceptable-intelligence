from pathlib import Path
from ultralytics import YOLO
import json
import struct
import math
import zmq
import threading
import numpy as np
import time
import asyncio

class YoloService():
    def __init__(self, camera):
        model_dir = (
            Path(__file__).resolve().parents[1]
            / "vision_models"
            / "yolo_tools"
        )
        pose_model_path = model_dir / "yolo11m-pose_openvino_model"
        DJ_custom_model_path = model_dir / "yoloe-11m_openvino_model"

        self.DJ_custom_model = YOLO(DJ_custom_model_path, task="detect")
        self.pose_model = YOLO(pose_model_path, task="pose")

        self.conf_threshold = 0.3
        self.camera = camera

        self.detections = []
        self.inference_thread = threading.Thread(target=self.timer_callback, daemon=True)
        self.running = False
        self.fps = 10.0
        self._active_event = threading.Event()
        self._state_condition = threading.Condition()
        self._owner_modes = {}
        self._mode_revision = 0
        self._inference_sequence = 0

    @property
    def active(self):
        with self._state_condition:
            return bool(self._owner_modes)

    def detection_snapshot(self):
        """Return one inference generation and its detections atomically."""
        with self._state_condition:
            return self._inference_sequence, list(self.detections)

    def _effective_mode_locked(self):
        modes = set(self._owner_modes.values()) - {"none"}
        if "both" in modes or {"dj", "pose"}.issubset(modes):
            return "both"
        if "pose" in modes:
            return "pose"
        if "dj" in modes:
            return "dj"
        return "none"

    @property
    def vision_mode(self):
        with self._state_condition:
            return self._effective_mode_locked()

    @vision_mode.setter
    def vision_mode(self, mode):
        # Compatibility for older callers. Action code should use owner modes.
        self.set_owner_mode("legacy", mode)

    def activate(self, owner, mode="dj"):
        """Enable inference for one action and return the current sequence."""
        owner = str(owner)
        if mode not in {"dj", "pose", "both", "none"}:
            raise ValueError(f"Unknown YOLO mode: {mode}")
        with self._state_condition:
            self._owner_modes[owner] = mode
            self._mode_revision += 1
            self.detections = []
            if self._effective_mode_locked() == "none":
                self._active_event.clear()
            else:
                self._active_event.set()
            return self._inference_sequence

    def set_owner_mode(self, owner, mode):
        return self.activate(owner, mode)

    def deactivate(self, owner):
        """Release one action's inference lease."""
        with self._state_condition:
            if self._owner_modes.pop(str(owner), None) is None:
                return
            self._mode_revision += 1
            self.detections = []
            if self._effective_mode_locked() == "none":
                self._active_event.clear()
            else:
                self._active_event.set()
            self._state_condition.notify_all()

    def _wait_for_inference_after(self, sequence, timeout):
        with self._state_condition:
            return self._state_condition.wait_for(
                lambda: (
                    self._inference_sequence > sequence
                    or not self._owner_modes
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
        # Multiple passes help ensure compilation/caching is complete
        for _ in range(3):
            self.DJ_custom_model.predict(
                source=dummy_frame,
                device="intel:gpu",
                verbose=False,
            )
        for _ in range(3):
            self.pose_model.predict(
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
            if not self._active_event.wait(timeout=0.1):
                continue
            with self._state_condition:
                mode = self._effective_mode_locked()
                mode_revision = self._mode_revision
            if mode == "none":
                continue
            loop_start = time.perf_counter()

            latest = self.camera.latest

            if latest is None:
                time.sleep(0.01)
                continue

            frame = latest.tracking_bgr
            source_generation = latest.source_generation

            detections = []

            if mode in ("dj", "both"):
                dj_results = self.DJ_custom_model.predict(
                    source=frame,
                    device="intel:gpu",
                    verbose=False,
                )
                detections.extend(
                    self.parse_dj(dj_results[0])
                )

            if mode in ("pose", "both"):
                pose_results = self.pose_model.predict(
                    source=frame,
                    device="intel:gpu",
                    verbose=False,
                )
                detections.extend(
                    self.parse_pose(pose_results[0])
                )

            current = self.camera.latest
            with self._state_condition:
                if (
                    current is not None
                    and current.source_generation == source_generation
                    and mode_revision == self._mode_revision
                    and self._effective_mode_locked() != "none"
                ):
                    self.detections = detections
                    self._inference_sequence += 1
                    self._state_condition.notify_all()

            elapsed = time.perf_counter() - loop_start
            remaining = 1/self.fps - elapsed

            if remaining > 0:
                time.sleep(remaining)

    async def close(self):
        self.running = False
        self._active_event.set()
        with self._state_condition:
            self._state_condition.notify_all()
        await asyncio.to_thread(self.inference_thread.join, 2)

    def parse_dj(self, result):
        detections = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < self.conf_threshold:
                continue

            class_name = result.names[cls_id]

            # Pose model owns person detections
            if self.vision_mode == "both" and class_name == "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": class_name,
                "class_id": cls_id,
                "confidence": round(conf, 3),

                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                },
            })

        return detections

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

        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])

            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            keypoints_xy = result.keypoints.xy[i].cpu().numpy()
            keypoints_xyn = result.keypoints.xyn[i].cpu().numpy()
            keypoints_conf = result.keypoints.conf[i].cpu().numpy()

            human_keypoints = {}

            for name, (x, y), (nx, ny), kp_conf in zip(
                keypoint_names,
                keypoints_xy,
                keypoints_xyn,
                keypoints_conf,
            ):

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

                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                },

                "keypoints": human_keypoints,
            })

        return detections
