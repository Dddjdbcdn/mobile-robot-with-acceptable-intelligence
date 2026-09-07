from __future__ import annotations

import asyncio

from actions.action_result import ActionResult
from cognition.state import robot_state

from actions.track_action import normalize_human_target, normalize_object_target

class ApproachAction:
    def __init__(self, send_robot_command, track_action):
        self.send_robot_command = send_robot_command
        self.track_action = track_action
        self.active = False
        self.action_id: str | None = None
        self.target: str | None = None
        self.completion_future: asyncio.Future[ActionResult] | None = None

    async def start_approaching(self, target, action_id):
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

        normalized_target = (
            normalize_human_target(target) or normalize_object_target(target)
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

        if not self.track_action.stable:
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

        navigate_payload = {
            "command": "navigate_to_pose",
            "action_id": action_id,
            "x": robot_state["camera"]["object_x"],
            "y": robot_state["camera"]["object_y"],
            "angle": robot_state["camera"]["object_angle"],
        }

        self.completion_future = asyncio.get_running_loop().create_future()
        self.active = True
        self.action_id = action_id
        self.target = target

        feedback = await self.send_robot_command(navigate_payload)
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
                data={"message": str(message)},
            )
            
        return ActionResult(
            action_id=action_id,
            action_type="approach_action",
            status="running",
            target=target,
            outcome="approaching",
            data={
                "tracking_stable": True,
                "destination": {
                    "x": navigate_payload["x"],
                    "y": navigate_payload["y"],
                    "angle": navigate_payload["angle"],
                },
            },
        )

    async def wait_until_finished(self):
        return await self.completion_future

    async def stop_approaching(self, reason=None):
        if not self.active:
            return

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
        succeeded = navigation_status == "Goal Reached"
        self.complete_approaching(
            status="succeeded" if succeeded else "failed",
            outcome="reached" if succeeded else "navigation_failed",
            reason_code=None if succeeded else "NAVIGATION_FAILED",
            data={"robot_status": navigation_status or "unknown"},
        )
        return True

    def complete_approaching(self,status,outcome,reason_code=None,data=None):
        target = self.target
        result = ActionResult(
            action_id=self.action_id,
            action_type="approach_action",
            status=status,
            target=target,
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
