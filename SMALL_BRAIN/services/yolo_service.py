from pathlib import Path

from ultralytics import YOLO
import json
import struct
import math
import zmq
import threading
import numpy as np

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
        self.vision_mode = "dj"
        self.camera = camera
        self.warmup_models()

        self.detections = []

        self.frame_lock = threading.Lock()

        timer_period = 0.1
        self.inference_timer = threading.Thread(target=self.timer_callback, daemon=True)

    def warmup_models(self):
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

    def timer_callback(self):
        frame = self.camera.latest.tracking_bgr if self.camera.latest else None

        detections = []

        if self.vision_mode in ("dj", "both"):
            dj_results = self.DJ_custom_model.predict(
                source=frame,
                device="intel:gpu",
                verbose=False,
            )
            detections.extend(
                self.parse_dj(dj_results[0])
            )

        if self.vision_mode in ("pose", "both"):
            pose_results = self.pose_model.predict(
                source=frame,
                device="intel:gpu",
                verbose=False,
            )
            detections.extend(
                self.parse_pose(pose_results[0])
            )

        self.detections = detections

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
