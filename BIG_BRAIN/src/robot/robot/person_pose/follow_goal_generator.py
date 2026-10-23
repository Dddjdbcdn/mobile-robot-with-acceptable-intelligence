from __future__ import annotations

from dataclasses import dataclass
import math


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class _FollowGoalSettings:
    standoff_m: float = 0.60
    move_start_distance_m: float = 0.80
    hand_standoff_m: float = 0.25
    position_update_m: float = 0.10
    yaw_update_deg: float = 20.0


@dataclass(frozen=True)
class FollowGoal:
    x: float
    y: float
    yaw: float


class FollowGoalGenerator:
    """Create a hysteretic standoff goal that faces the tracked person."""

    def __init__(self):
        self.settings = _FollowGoalSettings()
        self._person_approach_active = False

    def reset(self) -> None:
        """Reset state carried by the person-follow distance hysteresis."""
        self._person_approach_active = False

    def generate(
        self,
        *,
        person_x: float,
        person_y: float,
        robot_x: float,
        robot_y: float,
        target_mode: str = "person",
    ) -> FollowGoal | None:
        dx = person_x - robot_x
        dy = person_y - robot_y
        person_range = math.hypot(dx, dy)
        if person_range < 1e-3:
            return None

        lidar_bearing = math.atan2(dy, dx)

        if target_mode not in {"person", "recovery", "welcome", "push"}:
            raise ValueError(f"unknown follow target mode: {target_mode}")
        if target_mode == "recovery":
            self._person_approach_active = False
            travel = person_range
        elif target_mode == "person":
            if self._person_approach_active:
                if person_range <= self.settings.standoff_m:
                    self._person_approach_active = False
            elif person_range > self.settings.move_start_distance_m:
                self._person_approach_active = True

            travel = (
                max(0.0, person_range - self.settings.standoff_m)
                if self._person_approach_active else 0.0
            )
        else:
            self._person_approach_active = False
            effective_standoff = (
                self.settings.hand_standoff_m
                if target_mode == "welcome" else self.settings.standoff_m
            )
            travel = person_range - effective_standoff
            if target_mode != "push":
                travel = max(0.0, travel)
        goal_x = robot_x + travel * math.cos(lidar_bearing)
        goal_y = robot_y + travel * math.sin(lidar_bearing)
        destination_bearing = (
            lidar_bearing
            if target_mode == "recovery"
            else math.atan2(
                person_y - goal_y,
                person_x - goal_x,
            )
        )

        return FollowGoal(
            x=goal_x,
            y=goal_y,
            yaw=wrap_angle(destination_bearing),
        )

    def should_publish(
        self,
        goal: FollowGoal,
        previous_goal: tuple[float, float, float] | None,
    ) -> bool:

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
        )
