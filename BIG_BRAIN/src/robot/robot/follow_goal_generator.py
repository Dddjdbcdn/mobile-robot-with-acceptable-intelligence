from __future__ import annotations

from dataclasses import dataclass
import math


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class FollowGoalSettings:
    standoff_m: float = 0.50
    camera_center_deg: float = 95.0
    camera_weight: float = 0.35
    camera_agreement_deg: float = 25.0
    heading_comfort_deg: float = 20.0
    heading_critical_deg: float = 45.0


@dataclass(frozen=True)
class FollowGoal:
    x: float
    y: float
    yaw: float
    person_bearing: float
    heading_correction: float
    used_camera: bool


class FollowGoalGenerator:
    """Create a standoff goal while keeping the person in a camera cone."""

    def __init__(self, settings: FollowGoalSettings | None = None):
        self.settings = settings or FollowGoalSettings()

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def _fused_deviation(
        self,
        lidar_deviation: float,
        camera_pan_deg: float | None,
        camera_fresh: bool,
    ) -> tuple[float, bool]:
        if not camera_fresh or camera_pan_deg is None:
            return lidar_deviation, False

        camera_deviation = math.radians(
            camera_pan_deg - self.settings.camera_center_deg
        )
        disagreement = wrap_angle(camera_deviation - lidar_deviation)
        if abs(disagreement) > math.radians(
            self.settings.camera_agreement_deg
        ):
            return lidar_deviation, False

        fused = wrap_angle(
            lidar_deviation + self.settings.camera_weight * disagreement
        )
        return fused, True

    def _heading_correction(self, deviation: float) -> float:
        magnitude = abs(deviation)
        comfort = math.radians(self.settings.heading_comfort_deg)
        critical = math.radians(self.settings.heading_critical_deg)
        if magnitude <= comfort:
            return 0.0

        span = max(critical - comfort, 1e-6)
        urgency = self._smoothstep((magnitude - comfort) / span)
        # The camera absorbs small errors. As the target approaches the
        # critical angle, progressively ask the base to face it completely.
        correction = magnitude * urgency
        return math.copysign(correction, deviation)

    def generate(
        self,
        *,
        person_x: float,
        person_y: float,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        camera_pan_deg: float | None = None,
        camera_fresh: bool = False,
    ) -> FollowGoal | None:
        dx = person_x - robot_x
        dy = person_y - robot_y
        person_range = math.hypot(dx, dy)
        if not math.isfinite(person_range) or person_range < 1e-3:
            return None

        lidar_bearing = math.atan2(dy, dx)
        lidar_deviation = wrap_angle(lidar_bearing - robot_yaw)
        fused_deviation, used_camera = self._fused_deviation(
            lidar_deviation,
            camera_pan_deg,
            camera_fresh,
        )
        fused_bearing = wrap_angle(robot_yaw + fused_deviation)

        # Lidar remains authoritative for position: camera angle may bias
        # heading, but must not pull the navigation goal sideways. Signed
        # travel also backs away when the person is inside the standoff.
        travel = person_range - self.settings.standoff_m
        goal_x = robot_x + travel * math.cos(lidar_bearing)
        goal_y = robot_y + travel * math.sin(lidar_bearing)
        correction = self._heading_correction(fused_deviation)

        return FollowGoal(
            x=goal_x,
            y=goal_y,
            yaw=wrap_angle(robot_yaw + correction),
            person_bearing=fused_bearing,
            heading_correction=correction,
            used_camera=used_camera,
        )
