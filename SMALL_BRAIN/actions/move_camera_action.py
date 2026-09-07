from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from actions.action_result import ActionResult
from actions.search_action import (
    CAMERA_SETTLE_SECONDS,
    PAN_POSITION_ANGLE,
    TILT_POSITION_ANGLE,
)
from cognition.state import robot_state


SEMANTIC_CAMERA_REGIONS = {
    "upper_left": ("leftmost", "upmost"),
    "up": ("center", "upmost"),
    "upper_right": ("rightmost", "upmost"),
    "left": ("leftmost", "center"),
    "center": ("center", "center"),
    "right": ("rightmost", "center"),
    "lower_left": ("leftmost", "downmost"),
    "down": ("center", "downmost"),
    "lower_right": ("rightmost", "downmost"),
}


class MoveCameraAction:
    """Move the camera to one absolute semantic region of its search grid."""

    def __init__(
        self,
        send_robot_command: Callable[[dict], Awaitable[dict]],
        settle_seconds: float = CAMERA_SETTLE_SECONDS,
    ):
        self.send_robot_command = send_robot_command
        self.settle_seconds = settle_seconds
        self.active = False
        self.action_id: str | None = None
        self.target: str | None = None

    async def move_to_region(self, region, action_id) -> ActionResult:
        normalized_region = str(region or "").strip().lower().replace("-", "_")
        if normalized_region not in SEMANTIC_CAMERA_REGIONS:
            return ActionResult(
                action_id=action_id,
                action_type="move_camera",
                status="failed",
                target=normalized_region or None,
                outcome="invalid_request",
                reason_code="UNKNOWN_CAMERA_REGION",
                data={"valid_regions": list(SEMANTIC_CAMERA_REGIONS)},
            )
        if self.active:
            return ActionResult(
                action_id=action_id,
                action_type="move_camera",
                status="already_running",
                target=normalized_region,
                outcome="already_running",
                reason_code="CAMERA_MOVE_BUSY",
                retryable=True,
                data={"active_action_id": self.action_id},
            )

        pan_position, tilt_position = SEMANTIC_CAMERA_REGIONS[normalized_region]
        target_pan = float(PAN_POSITION_ANGLE[pan_position])
        target_tilt = float(TILT_POSITION_ANGLE[tilt_position])
        camera_state = robot_state["camera"]
        state_is_initialized = float(camera_state.get("timestamp", 0.0)) > 0.0
        current_pan = float(
            camera_state.get("pan_angle", PAN_POSITION_ANGLE["center"])
            if state_is_initialized
            else PAN_POSITION_ANGLE["center"]
        )
        current_tilt = float(
            camera_state.get("tilt_angle", TILT_POSITION_ANGLE["center"])
            if state_is_initialized
            else TILT_POSITION_ANGLE["center"]
        )
        delta_pan = target_pan - current_pan
        delta_tilt = target_tilt - current_tilt

        self.active = True
        self.action_id = action_id
        self.target = normalized_region
        try:
            feedback = await self.send_robot_command({
                "command": "move_camera",
                "action_id": action_id,
                "delta_pan_angle": delta_pan,
                "delta_tilt_angle": delta_tilt,
            })
            if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
                message = (
                    feedback.get("message", "camera_move_rejected")
                    if isinstance(feedback, dict)
                    else "invalid_camera_feedback"
                )
                return self._complete(
                    status="failed",
                    outcome="rejected",
                    reason_code="CAMERA_MOVE_REJECTED",
                    data={"message": str(message)},
                )

            await asyncio.sleep(self.settle_seconds)
            return self._complete(
                status="succeeded",
                outcome="camera_positioned",
                data={
                    "region": normalized_region,
                    "pan_position": pan_position,
                    "tilt_position": tilt_position,
                    "target_pan_angle": target_pan,
                    "target_tilt_angle": target_tilt,
                    "delta_pan_angle": delta_pan,
                    "delta_tilt_angle": delta_tilt,
                },
            )
        except asyncio.CancelledError:
            self.active = False
            self.action_id = None
            self.target = None
            raise
        except Exception as error:
            return self._complete(
                status="failed",
                outcome="command_failed",
                reason_code="CAMERA_MOVE_COMMAND_FAILED",
                data={"error": f"{type(error).__name__}: {error}"},
            )

    def _complete(self, status, outcome, reason_code=None, data=None):
        result = ActionResult(
            action_id=self.action_id or "unassigned",
            action_type="move_camera",
            status=status,
            target=self.target,
            outcome=outcome,
            reason_code=reason_code,
            retryable=status == "failed",
            data=data or {},
        )
        self.active = False
        self.action_id = None
        self.target = None
        return result
