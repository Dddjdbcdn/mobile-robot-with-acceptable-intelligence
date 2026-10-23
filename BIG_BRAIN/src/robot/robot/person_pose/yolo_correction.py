from __future__ import annotations

from dataclasses import dataclass, fields
import math
import threading
import time
from typing import Callable

import numpy as np

from robot.person_pose.lidar_clusters import transform_points, transform_yaw


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def bearing_from(origin: np.ndarray, position: np.ndarray) -> float:
    offset = position - origin
    return math.atan2(float(offset[1]), float(offset[0]))


@dataclass
class VisualObservation:
    base_bearing_rad: float | None
    camera_bearing_rad: float
    horizontal_fov_rad: float
    person_visible: bool
    captured_at: float
    received_at: float
    session_id: str


@dataclass(frozen=True)
class VisualConstraint:
    origin: np.ndarray
    bearing: float
    gate: float


@dataclass(frozen=True)
class CameraView:
    origin: np.ndarray
    bearing: float
    half_fov: float
    person_visible: bool

    def contains(self, position: np.ndarray) -> bool:
        return abs(
            wrap_angle(bearing_from(self.origin, position) - self.bearing)
        ) <= self.half_fov


@dataclass(frozen=True)
class VisualContext:
    """One capture-consistent view of raw and transformed camera evidence."""

    observation: VisualObservation
    camera_view: CameraView
    constraint: VisualConstraint | None


@dataclass(frozen=True)
class YoloSettings:
    visual_fresh_seconds: float = 0.50
    visual_bearing_gate_deg: float = 25.0
    visual_min_confidence: float = 0.40

    @classmethod
    def from_node(cls, node) -> YoloSettings:
        defaults = cls()
        for item in fields(defaults):
            node.declare_parameter(item.name, getattr(defaults, item.name))
        return cls(**{
            item.name: node.get_parameter(item.name).value
            for item in fields(defaults)
        })


class YoloCorrection:
    """Maintain the latest camera view and selected-person bearing."""

    def __init__(self, settings: YoloSettings, base_frame: str):
        self.settings = settings
        self.base_frame = base_frame
        self._lock = threading.Lock()
        self._latest: VisualObservation | None = None

    def reset(self) -> None:
        """Discard visual observations from the previous tracking session."""
        with self._lock:
            self._latest = None

    def accept(self, message: dict) -> bool:
        if message.get("type") != "person_visual_observation":
            return False
        person_visible = bool(message["person_visible"])
        observation = VisualObservation(
            base_bearing_rad=(
                wrap_angle(float(message["base_bearing_rad"]))
                if person_visible else None
            ),
            camera_bearing_rad=wrap_angle(float(message["camera_bearing_rad"])),
            horizontal_fov_rad=float(message["horizontal_fov_rad"]),
            person_visible=person_visible,
            captured_at=float(message["captured_at_monotonic"]),
            received_at=time.monotonic(),
            session_id=message["session_id"],
        )

        with self._lock:
            latest = self._latest
            same_session = latest is not None and (
                latest.session_id == observation.session_id
            )
            if same_session and observation.captured_at <= latest.captured_at:
                return True
            self._latest = observation
        return True

    def snapshot(self) -> VisualObservation | None:
        now = time.monotonic()
        with self._lock:
            observation = self._latest
        if observation is None:
            return None
        age = max(
            0.0,
            now - observation.received_at,
            now - observation.captured_at,
        )
        if age > self.settings.visual_fresh_seconds:
            return None
        return observation

    def context(
        self,
        lookup_transform: Callable,
    ) -> VisualContext | None:
        observation = self.snapshot()
        if observation is None:
            return None

        transform = lookup_transform(
            self.base_frame,
            observation.captured_at,
        )
        if transform is None:
            return None

        transform_angle = transform_yaw(transform)
        origin = transform_points(
            np.zeros((1, 2), dtype=float),
            transform,
        )[0]
        camera_view = CameraView(
            origin,
            wrap_angle(observation.camera_bearing_rad + transform_angle),
            observation.horizontal_fov_rad * 0.5,
            observation.person_visible,
        )
        if not observation.person_visible:
            return VisualContext(observation, camera_view, None)

        bearing = wrap_angle(
            observation.base_bearing_rad + transform_angle
        )
        gate = math.radians(self.settings.visual_bearing_gate_deg)
        return VisualContext(
            observation,
            camera_view,
            VisualConstraint(origin, bearing, gate),
        )

    def constraint(
        self,
        lookup_transform: Callable,
    ) -> tuple[VisualConstraint | None, CameraView | None]:
        """Compatibility view for renderers and callers needing only geometry."""
        context = self.context(lookup_transform)
        if context is None:
            return None, None
        return context.constraint, context.camera_view
