from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time


class SemanticMemory:
    """Persistent, confidence-weighted map observations of tracked objects."""

    def __init__(self, path, max_range_m=5.0, state_max_age_s=0.5):
        self.path = Path(path)
        self.max_range_m = float(max_range_m)
        self.state_max_age_s = float(state_max_age_s)
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self):
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            data = {}
        objects = data.get("objects")
        return {"version": 1, "objects": objects if isinstance(objects, dict) else {}}

    def _save_locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self._data, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary_path, self.path)

    def remember(self, target, robot_state, confidence=1.0):
        target = str(target or "").strip().lower()
        camera = robot_state.get("camera") or {}
        pose = robot_state.get("pose") or {}
        now_monotonic = time.monotonic()

        required = (
            camera.get("object_map_x"),
            camera.get("object_map_y"),
            camera.get("camera_tof_range"),
            camera.get("timestamp"),
            pose.get("x"),
            pose.get("y"),
            pose.get("yaw"),
        )
        if not target or not all(isinstance(value, (int, float)) for value in required):
            return None

        map_x, map_y, distance, sample_time, robot_x, robot_y, robot_yaw = (
            float(value) for value in required
        )
        if (
            not all(math.isfinite(value) for value in (
                map_x, map_y, distance, sample_time, robot_x, robot_y, robot_yaw
            ))
            or distance <= 0.05
            or distance > self.max_range_m
            or now_monotonic - sample_time > self.state_max_age_s
        ):
            return None

        confidence = max(0.0, min(1.0, float(confidence)))
        with self._lock:
            previous = self._data["objects"].get(target)
            observations = 1
            if isinstance(previous, dict):
                displacement = math.hypot(
                    map_x - float(previous["map_x"]),
                    map_y - float(previous["map_y"]),
                )
                if displacement <= 0.5:
                    observations = int(previous.get("observations", 0)) + 1
                    weight = max(0.25, min(0.75, confidence))
                    map_x = (1.0 - weight) * float(previous["map_x"]) + weight * map_x
                    map_y = (1.0 - weight) * float(previous["map_y"]) + weight * map_y

            record = {
                "target": target,
                "map_x": map_x,
                "map_y": map_y,
                "confidence": confidence,
                "range_m": distance,
                "observed_at": time.time(),
                "observations": observations,
                "consecutive_misses": 0,
                "observer_pose": {
                    "x": robot_x,
                    "y": robot_y,
                    "yaw": robot_yaw,
                },
            }
            self._data["objects"][target] = record
            self._save_locked()
            return dict(record)

    def recall(self, target):
        target = str(target or "").strip().lower()
        with self._lock:
            record = self._data["objects"].get(target)
            return dict(record) if isinstance(record, dict) else None

    def forget(self, target):
        target = str(target or "").strip().lower()
        with self._lock:
            removed = self._data["objects"].pop(target, None)
            if removed is not None:
                self._save_locked()
            return removed is not None

    def mark_miss(self, target, forget_after=3):
        """Record a failed verification and discard repeatedly stale memory."""
        target = str(target or "").strip().lower()
        with self._lock:
            record = self._data["objects"].get(target)
            if not isinstance(record, dict):
                return False
            misses = int(record.get("consecutive_misses", 0)) + 1
            if misses >= int(forget_after):
                del self._data["objects"][target]
            else:
                record["consecutive_misses"] = misses
                record["confidence"] = float(record.get("confidence", 1.0)) * 0.5
            self._save_locked()
            return True

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(self._data))
