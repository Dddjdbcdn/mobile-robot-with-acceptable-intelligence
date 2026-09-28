"""Owner-scoped validation and storage for tracked-target ToF seeds."""

from collections import deque
import math
from statistics import median
import time


class StableTargetSeedTracker:
    SAMPLE_COUNT = 3
    MAX_SPREAD_M = 0.12
    MAX_AGE_SECONDS = 0.75
    INPUT_MAX_AGE_SECONDS = 0.50
    HAND_PERSON_MAX_DISTANCE_M = 0.85

    def __init__(self):
        self._samples = {}
        self._stable = {}

    def clear(self, owner=None):
        if owner is None:
            self._samples.clear()
            self._stable.clear()
            return
        owner = str(owner)
        self._samples.pop(owner, None)
        self._stable.pop(owner, None)

    def sample_count(self, owner):
        return len(self._samples.get(str(owner), ()))

    def update(
        self,
        owner,
        camera,
        *,
        target=None,
        session_id=None,
        person=None,
        require_person_proximity=False,
    ):
        owner = str(owner)
        camera = dict(camera or {})
        now = time.monotonic()
        values = [
            camera.get(key)
            for key in (
                "camera_tof_range", "object_x", "object_y", "object_angle",
                "timestamp",
            )
        ]
        valid = all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values
        )
        if valid:
            valid = (
                float(camera["camera_tof_range"]) > 0.05
                and now - float(camera["timestamp"])
                <= self.INPUT_MAX_AGE_SECONDS
            )

        map_x = camera.get("object_map_x")
        map_y = camera.get("object_map_y")
        map_valid = all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (map_x, map_y)
        )
        if valid and require_person_proximity:
            person_x = person.get("x") if isinstance(person, dict) else None
            person_y = person.get("y") if isinstance(person, dict) else None
            valid = map_valid and all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in (person_x, person_y)
            )
            if valid:
                valid = math.hypot(
                    float(map_x) - float(person_x),
                    float(map_y) - float(person_y),
                ) <= self.HAND_PERSON_MAX_DISTANCE_M

        if not valid:
            self.clear(owner)
            return None

        samples = self._samples.setdefault(
            owner, deque(maxlen=self.SAMPLE_COUNT)
        )
        if samples and (
            samples[-1]["target"] != target
            or samples[-1]["session_id"] != session_id
        ):
            samples.clear()
            self._stable.pop(owner, None)
        timestamp = float(camera["timestamp"])
        if samples and timestamp <= samples[-1]["timestamp"]:
            return self.get(owner)

        sample = {
            "x": float(camera["object_x"]),
            "y": float(camera["object_y"]),
            "angle": float(camera["object_angle"]),
            "tof_range": float(camera["camera_tof_range"]),
            "map_x": float(map_x) if map_valid else None,
            "map_y": float(map_y) if map_valid else None,
            "stability_frame": "map" if map_valid else "base_footprint",
            "stability_x": (
                float(map_x) if map_valid else float(camera["object_x"])
            ),
            "stability_y": (
                float(map_y) if map_valid else float(camera["object_y"])
            ),
            "target": target,
            "session_id": session_id,
            "timestamp": timestamp,
        }
        if samples and samples[-1]["stability_frame"] != sample["stability_frame"]:
            samples.clear()
        samples.append(sample)
        self._stable.pop(owner, None)
        if len(samples) < self.SAMPLE_COUNT:
            return None

        center_x = median(item["stability_x"] for item in samples)
        center_y = median(item["stability_y"] for item in samples)
        spread = max(
            math.hypot(
                item["stability_x"] - center_x,
                item["stability_y"] - center_y,
            )
            for item in samples
        )
        if spread > self.MAX_SPREAD_M:
            return None

        stable = {
            key: median(item[key] for item in samples)
            for key in ("x", "y", "angle", "tof_range")
        }
        stable["map_x"] = (
            median(item["map_x"] for item in samples) if map_valid else None
        )
        stable["map_y"] = (
            median(item["map_y"] for item in samples) if map_valid else None
        )
        stable.update({
            "owner": owner,
            "target": target,
            "session_id": session_id,
            "captured_at": max(item["timestamp"] for item in samples),
            "validated_at": now,
        })
        self._stable[owner] = stable
        return dict(stable)

    def get(
        self,
        owner,
        max_age_seconds=None,
        *,
        target=None,
        session_id=None,
    ):
        owner = str(owner)
        seed = self._stable.get(owner)
        if seed is None:
            return None
        if target is not None and seed.get("target") != target:
            return None
        if session_id is not None and seed.get("session_id") != session_id:
            return None
        max_age = (
            self.MAX_AGE_SECONDS
            if max_age_seconds is None else float(max_age_seconds)
        )
        if time.monotonic() - seed["validated_at"] > max_age:
            self.clear(owner)
            return None
        return dict(seed)
