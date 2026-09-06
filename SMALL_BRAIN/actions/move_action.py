from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from actions.action_result import ActionResult

class MoveAction:
    """Own the lifecycle of one timed robot movement."""

    def __init__(
        self,
        send_robot_command: Callable[[dict], Awaitable[dict]],
    ) -> None:
        self.send_robot_command = send_robot_command
        self.active = False
        self.action_id: str | None = None
        self.completion_future: asyncio.Future[ActionResult] | None = None

    async def start_moving(
        self,
        linear_velocity,
        distance,
        angular_velocity,
        angle,
        action_id,
    ):
        if self.active:
            return ActionResult(
                action_id=action_id,
                action_type="move_action",
                status="already_running",
                outcome="already_running",
                reason_code="MOVE_BUSY",
                retryable=True,
                data={"active_action_id": self.action_id},
            )

        payload = {
            "command": "move_action",
            "action_id": action_id,
            "linear_velocity": linear_velocity,
            "distance": distance,
            "angular_velocity": angular_velocity,
            "angle": angle,
        }

        self.completion_future = asyncio.get_running_loop().create_future()
        self.active = True
        self.action_id = action_id

        try:
            feedback = await self.send_robot_command(payload)
        except Exception as error:
            return self.complete_moving(
                status="failed",
                outcome="command_failed",
                reason_code="MOVE_COMMAND_FAILED",
                data={"error": f"{type(error).__name__}: {error}"},
            )

        if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
            message = (
                feedback.get("message", "movement_rejected")
                if isinstance(feedback, dict)
                else "invalid_movement_feedback"
            )
            return self.complete_moving(
                status="failed",
                outcome="rejected",
                reason_code="MOVE_REJECTED",
                data={"message": str(message)},
            )
            
        return ActionResult(
            action_id=action_id,
            action_type="move_action",
            status="running",
            outcome="moving",
            data={
                "linear_velocity": linear_velocity,
                "distance": distance,
                "angular_velocity": angular_velocity,
                "angle": angle,
            },
        )

    async def wait_until_finished(self) -> ActionResult:
        return await self.completion_future

    async def stop_moving(self, reason_code=None):
        if not self.active:
            return ActionResult(
                action_id="unassigned",
                action_type="move_action",
                status="failed",
                outcome="not_active",
                reason_code="MOVE_NOT_ACTIVE",
            )

        await self.send_robot_command({
            "command": "stop_moving",
            "action_id": self.action_id,
        })
        return self.complete_moving(
            status="cancelled",
            outcome="stopped",
            reason_code=reason_code,
        )

    def handle_moving_event(self, payload: dict) -> bool:
        if not self.active or payload.get("event") != "move_action":
            return False
        if payload.get("action_id") not in {None, self.action_id}:
            return False

        moving_status = str(payload.get("status", ""))
        succeeded = moving_status.lower() == "completed"
        self.complete_moving(
            status="succeeded" if succeeded else "failed",
            outcome="completed" if succeeded else "movement_failed",
            reason_code=None if succeeded else "MOVE_FAILED",
            data={"robot_status": moving_status or "unknown"},
        )
        return True

    def complete_moving(
        self,
        status: str,
        outcome: str,
        reason_code: str | None = None,
        data: dict | None = None,
    ) -> ActionResult:
        result = ActionResult(
            action_id=self.action_id or "unassigned",
            action_type="move_action",
            status=status,
            outcome=outcome,
            reason_code=reason_code,
            retryable=status == "failed",
            data=data or {},
        )
        completion_future = self.completion_future
        self.active = False
        self.action_id = None

        if completion_future is not None and not completion_future.done():
            completion_future.set_result(result)

        return result
