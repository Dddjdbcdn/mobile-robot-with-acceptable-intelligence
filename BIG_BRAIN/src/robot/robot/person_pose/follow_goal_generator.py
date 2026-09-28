from __future__ import annotations

from dataclasses import dataclass
import math


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class _FollowGoalSettings:
    standoff_m: float = 0.50
    camera_center_deg: float = 95.0
    camera_weight: float = 0.35
    camera_agreement_deg: float = 25.0
    heading_comfort_deg: float = 20.0
    heading_critical_deg: float = 45.0
    position_update_m: float = 0.10
    yaw_update_deg: float = 5.0
    refresh_interval_s: float = 0.80


@dataclass(frozen=True)
class FollowGoal:
    x: float
    y: float
    yaw: float
    person_bearing: float
    heading_correction: float
    camera_pan_target_deg: float
    used_camera: bool


class FollowGoalGenerator:
    """Create a standoff goal while keeping the person in a camera cone."""

    def __init__(
        self,
        *,
        standoff_m: float = 0.50,
        camera_center_deg: float = 95.0,
        camera_weight: float = 0.35,
        camera_agreement_deg: float = 25.0,
        heading_comfort_deg: float = 20.0,
        heading_critical_deg: float = 45.0,
        position_update_m: float = 0.10,
        yaw_update_deg: float = 5.0,
        refresh_interval_s: float = 0.80,
    ):
        # Callers provide values without depending on this utility's config
        # representation.
        self.settings = _FollowGoalSettings(
            standoff_m=float(standoff_m),
            camera_center_deg=float(camera_center_deg),
            camera_weight=float(camera_weight),
            camera_agreement_deg=float(camera_agreement_deg),
            heading_comfort_deg=float(heading_comfort_deg),
            heading_critical_deg=float(heading_critical_deg),
            position_update_m=float(position_update_m),
            yaw_update_deg=float(yaw_update_deg),
            refresh_interval_s=float(refresh_interval_s),
        )

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
            camera_pan_target_deg=(
                self.settings.camera_center_deg
                + math.degrees(lidar_deviation)
            ),
            used_camera=used_camera,
        )

    def should_publish(
        self,
        goal: FollowGoal,
        previous_goal: tuple[float, float, float] | None,
        elapsed_s: float | None,
    ) -> bool:
        """Return whether movement or elapsed time warrants a goal update."""
        if previous_goal is None:
            return True
        previous_x, previous_y, previous_yaw = previous_goal
        position_change = math.hypot(
            goal.x - previous_x,
            goal.y - previous_y,
        )
        yaw_change = abs(wrap_angle(goal.yaw - previous_yaw))
        tolerance = 1e-9
        return (
            position_change + tolerance >= self.settings.position_update_m
            or yaw_change + tolerance
            >= math.radians(self.settings.yaw_update_deg)
            or elapsed_s is None
            or elapsed_s >= self.settings.refresh_interval_s
        )
