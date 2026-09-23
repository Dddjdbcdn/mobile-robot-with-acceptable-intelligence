from __future__ import annotations

import asyncio
import math

from actions.action_result import ActionResult
from actions.track_action import normalize_human_target, normalize_object_target
from cognition.state import robot_state


class ApproachAction:
    """Dispatch one tracked-target Nav2 approach and report its result."""

    def __init__(self, send_robot_command, track_action):
        self.send_robot_command = send_robot_command
        self.track_action = track_action
        self.active = False
        self.action_id: str | None = None
        self.target: str | None = None
        self.completion_future: asyncio.Future[ActionResult] | None = None
        self._navigation_attempts = 0
        self._last_destination: dict | None = None
        self.following = False

    async def start_approaching(
        self, target, action_id, standoff_m=None, *, follow=False
    ):
        normalized_target = (
            normalize_human_target(target) or normalize_object_target(target)
        )
        updating_follow = (
            self.active
            and follow
            and self.following
            and normalized_target == self.track_action.target
        )
        if self.active and not updating_follow:
            return ActionResult(
                action_id=action_id,
                action_type="approach_action",
                status="already_running",
                target=target,
                outcome="already_running",
                reason_code="APPROACH_BUSY",
                retryable=True,
                data={"active_action_id": self.action_id},
            )

        if (
            not self.track_action.active
            or normalized_target != self.track_action.target
        ):
            return ActionResult(
                action_id=action_id,
                action_type="approach_action",
                status="failed",
                target=target,
                outcome="precondition_failed",
                reason_code="TARGET_NOT_TRACKED",
                retryable=True,
            )

        quick_person_stability = follow and normalized_target == "person"
        if (
            quick_person_stability
            and not self.track_action.person_stable
        ):
            stable = await self.track_action.wait_until_person_stable(
                timeout=2.0
            )
            if not stable:
                return ActionResult(
                    action_id=action_id,
                    action_type="approach_action",
                    status="failed",
                    target=target,
                    outcome="precondition_failed",
                    reason_code="PERSON_TRACKING_STABILITY_TIMEOUT",
                    retryable=True,
                    data={"person_tracking_stable": False},
                )
        elif not quick_person_stability and not self.track_action.stable:
            stable = await self.track_action.wait_until_stable(timeout=10.0)
            if not stable:
                return ActionResult(
                    action_id=action_id,
                    action_type="approach_action",
                    status="failed",
                    target=target,
                    outcome="precondition_failed",
                    reason_code="TRACKING_STABILITY_TIMEOUT",
                    retryable=True,
                    data={"tracking_stable": False},
                )

        camera_state = robot_state.get("camera") or {}
        values = (
            camera_state.get("object_x"),
            camera_state.get("object_y"),
            camera_state.get("object_angle"),
            camera_state.get("camera_tof_range"),
        )
        if (
            not all(isinstance(value, (int, float)) for value in values)
            or not all(math.isfinite(value) for value in values)
            or float(values[3]) <= 0.05
        ):
            return ActionResult(
                action_id=action_id,
                action_type="approach_action",
                status="failed",
                target=target,
                outcome="range_invalid",
                reason_code="APPROACH_RANGE_INVALID",
                retryable=True,
            )

        x, y, angle, tof_range = (float(value) for value in values)
        raw_destination = {
            "x": x,
            "y": y,
            "angle": angle,
            "tof_range": tof_range,
        }

        if not updating_follow:
            self.completion_future = asyncio.get_running_loop().create_future()
            self.active = True
            self.action_id = action_id
            self.target = target
            self._navigation_attempts = 0
            self._last_destination = None
            self.following = follow

        command = {
            "command": "navigate_to_follow" if follow else "navigate_to_approach",
            "action_id": self.action_id,
            "frame_id": "base_footprint",
            **raw_destination,
        }
        if standoff_m is not None:
            command["standoff_m"] = float(standoff_m)
        feedback = await self.send_robot_command(command)
        if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
            message = (
                feedback.get("message", "navigation_rejected")
                if isinstance(feedback, dict)
                else "invalid_navigation_feedback"
            )
            if updating_follow:
                await self.send_robot_command({
                    "command": "stop_moving",
                    "action_id": self.action_id,
                })
            return self.complete_approaching(
                status="failed",
                outcome="rejected",
                reason_code="NAVIGATION_REJECTED",
                data={"message": message},
            )

        self._navigation_attempts += 1
        resolved = feedback.get("destination")
        self._last_destination = (
            dict(resolved)
            if isinstance(resolved, dict)
            else dict(raw_destination)
        )

        return ActionResult(
            action_id=self.action_id,
            action_type="approach_action",
            status="running",
            target=target,
            outcome="follow_goal_updated" if updating_follow else (
                "following" if follow else "approaching"
            ),
            data={
                "tracking_stable": self.track_action.stable,
                "person_tracking_stable": getattr(
                    self.track_action, "person_stable", False
                ),
                "raw_destination": dict(raw_destination),
                "destination": dict(self._last_destination or raw_destination),
            },
        )


    async def wait_until_finished(self):
        return await self.completion_future

    async def stop_approaching(self, reason=None):
        if not self.active:
            return None
        await self.send_robot_command({
            "command": "stop_moving",
            "action_id": self.action_id,
        })
        return self.complete_approaching(
            status="cancelled",
            outcome="stopped",
            reason_code=reason,
        )

    def handle_navigation_event(self, payload):
        if not self.active or payload.get("event") != "navigation":
            return False
        if payload.get("action_id") not in {None, self.action_id}:
            return False

        navigation_status = str(payload.get("status", ""))
        if navigation_status == "Goal Reached":
            self.complete_approaching(
                status="succeeded",
                outcome="navigation_reached",
                data={
                    "robot_status": navigation_status,
                    "navigation_attempts": self._navigation_attempts,
                    "destination": dict(self._last_destination or {}),
                },
            )
        else:
            self.complete_approaching(
                status="failed",
                outcome="navigation_failed",
                reason_code="NAVIGATION_FAILED",
                data={
                    "robot_status": navigation_status or "unknown",
                    "navigation_attempts": self._navigation_attempts,
                    "destination": dict(self._last_destination or {}),
                },
            )
        return True

    def complete_approaching(self, status, outcome, reason_code=None, data=None):
        result = ActionResult(
            action_id=self.action_id or "unassigned",
            action_type="approach_action",
            status=status,
            target=self.target,
            outcome=outcome,
            reason_code=reason_code,
            retryable=status == "failed",
            data=data or {},
        )
        completion_future = self.completion_future
        self.active = False
        self.action_id = None
        self.target = None
        self.following = False
        if completion_future is not None and not completion_future.done():
            completion_future.set_result(result)
        return result
