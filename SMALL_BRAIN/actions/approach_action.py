from __future__ import annotations

import asyncio
from actions.action_result import ActionResult
from actions.tracking.target_catalog import (
    normalize_human_target,
    normalize_object_target,
)


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

    async def start_approaching(self, target, action_id, standoff_m=None):
        normalized_target = (
            normalize_human_target(target) or normalize_object_target(target)
        )
        if self.active:
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

        tracking_session_id = self.track_action.action_id
        seed = await self.track_action.wait_for_stable_target_seed(
            self.track_action.TRACKED_TARGET_SEED_OWNER,
            target=normalized_target,
            session_id=tracking_session_id,
            timeout=10.0,
        )
        if seed is None:
            return ActionResult(
                action_id=action_id,
                action_type="approach_action",
                status="failed",
                target=target,
                outcome="stable_seed_timeout",
                reason_code="STABLE_SEED_TIMEOUT",
                retryable=True,
            )

        raw_destination = {
            "x": seed["x"],
            "y": seed["y"],
            "angle": seed["angle"],
            "tof_range": seed["tof_range"],
        }

        self.completion_future = asyncio.get_running_loop().create_future()
        self.active = True
        self.action_id = action_id
        self.target = target
        self._navigation_attempts = 0
        self._last_destination = None

        command = {
            "command": "navigate_to_approach",
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
            outcome="approaching",
            data={
                "stable_seed": dict(seed),
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
        if completion_future is not None and not completion_future.done():
            completion_future.set_result(result)
        return result
