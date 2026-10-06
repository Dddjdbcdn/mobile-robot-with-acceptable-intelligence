"""Exclusive explicit camera movement built on top of shared tracking."""

from __future__ import annotations

import asyncio
import time

from actions.action_result import ActionResult
from cognition.manager.world_state import robot_state


CAMERA_SETTLE_SECONDS = 1.5
CAMERA_RECOVERY_TIMEOUT_SECONDS = 3.0

PAN_POSITION_ANGLE = {
    "center": 95.0,
    "leftmost": 155.0,
    "rightmost": 35.0,
}

TILT_POSITION_ANGLE = {
    "center": 90.0,
    "upmost": 30.0,
    "downmost": 120.0,
}

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
    """Move the camera explicitly, preempting continuous tracking first."""

    def __init__(
        self,
        *,
        camera,
        send_robot_command,
        tracking=None,
        settle_seconds=CAMERA_SETTLE_SECONDS,
    ):
        self.camera = camera
        self.send_robot_command = send_robot_command
        self.tracking = tracking
        self.settle_seconds = settle_seconds
        self._lock = asyncio.Lock()

    @property
    def active(self):
        return self._lock.locked()

    async def move_to_region(self, region, action_id):
        normalized = str(region or "").strip().lower().replace("-", "_")
        if normalized not in SEMANTIC_CAMERA_REGIONS:
            return ActionResult(
                action_id=action_id,
                action_type="move_camera",
                status="failed",
                target=normalized or None,
                outcome="invalid_request",
                reason_code="UNKNOWN_CAMERA_REGION",
                data={"valid_regions": list(SEMANTIC_CAMERA_REGIONS)},
            )

        pan_position, tilt_position = SEMANTIC_CAMERA_REGIONS[normalized]
        result = await self.move_to_angles(
            PAN_POSITION_ANGLE[pan_position],
            TILT_POSITION_ANGLE[tilt_position],
            action_id=action_id,
            target=normalized,
        )
        if result.status == "succeeded":
            result.data.update({
                "region": normalized,
                "pan_position": pan_position,
                "tilt_position": tilt_position,
            })
        return result

    async def move_to_angles(
        self,
        target_pan,
        target_tilt,
        *,
        action_id,
        target=None,
        current_pan=None,
        current_tilt=None,
    ):
        if self._lock.locked():
            return ActionResult(
                action_id=action_id,
                action_type="move_camera",
                status="already_running",
                target=target,
                outcome="already_running",
                reason_code="CAMERA_MOVE_BUSY",
                retryable=True,
            )

        async with self._lock:
            camera_state = robot_state.get("camera") or {}
            state_initialized = float(camera_state.get("timestamp") or 0.0) > 0.0
            if current_pan is None:
                current_pan = (
                    camera_state.get("pan_angle", PAN_POSITION_ANGLE["center"])
                    if state_initialized else PAN_POSITION_ANGLE["center"]
                )
            if current_tilt is None:
                current_tilt = (
                    camera_state.get("tilt_angle", TILT_POSITION_ANGLE["center"])
                    if state_initialized else TILT_POSITION_ANGLE["center"]
                )

            target_pan = float(target_pan)
            target_tilt = float(target_tilt)
            delta_pan = target_pan - float(current_pan)
            delta_tilt = target_tilt - float(current_tilt)
            result_data = {
                "target_pan_angle": target_pan,
                "target_tilt_angle": target_tilt,
                "delta_pan_angle": delta_pan,
                "delta_tilt_angle": delta_tilt,
            }
            if abs(delta_pan) < 1e-6 and abs(delta_tilt) < 1e-6:
                return ActionResult(
                    action_id=action_id,
                    action_type="move_camera",
                    status="succeeded",
                    target=target,
                    outcome="camera_already_positioned",
                    data=result_data,
                )

            if self.tracking is not None and self.tracking.active:
                await self.tracking.stop(
                    reason_code="EXPLICIT_CAMERA_MOVEMENT",
                    status="cancelled",
                    outcome="replaced_by_camera_movement",
                    reset_camera=False,
                )

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
                    return ActionResult(
                        action_id=action_id,
                        action_type="move_camera",
                        status="failed",
                        target=target,
                        outcome="rejected",
                        reason_code="CAMERA_MOVE_REJECTED",
                        retryable=True,
                        data={"message": str(message)},
                    )

                await asyncio.sleep(self.settle_seconds)
                moved_at = time.monotonic()
                frame_ready = await asyncio.to_thread(
                    self.camera.wait_for_frame_captured_after,
                    moved_at,
                    CAMERA_RECOVERY_TIMEOUT_SECONDS,
                )
                if not frame_ready:
                    return ActionResult(
                        action_id=action_id,
                        action_type="move_camera",
                        status="failed",
                        target=target,
                        outcome="camera_unavailable",
                        reason_code="CAMERA_RECOVERY_TIMEOUT",
                        retryable=True,
                    )
                return ActionResult(
                    action_id=action_id,
                    action_type="move_camera",
                    status="succeeded",
                    target=target,
                    outcome="camera_positioned",
                    data=result_data,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                return ActionResult(
                    action_id=action_id,
                    action_type="move_camera",
                    status="failed",
                    target=target,
                    outcome="command_failed",
                    reason_code="CAMERA_MOVE_COMMAND_FAILED",
                    retryable=True,
                    data={"error": f"{type(error).__name__}: {error}"},
                )
