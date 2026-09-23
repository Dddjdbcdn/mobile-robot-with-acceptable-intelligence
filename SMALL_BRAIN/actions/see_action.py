import asyncio
import base64
import json
import time

import cv2

from actions.action_result import ActionResult
from actions.search_action import (
    CAMERA_RECOVERY_TIMEOUT_SECONDS,
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

class SeeAction:
    def __init__(
        self,
        ws,
        camera,
        send_robot_command,
        settle_seconds=CAMERA_SETTLE_SECONDS,
    ):
        self.ws = ws
        self.camera = camera
        self.send_robot_command = send_robot_command
        self.settle_seconds = settle_seconds
        self.camera_moving = False

    async def see(self, query, action_id, region=None, autonomous=False):
        normalized_query = (query or "").strip()
        if not normalized_query:
            return ActionResult(
                action_id=action_id,
                action_type="see_action",
                status="failed",
                outcome="invalid_request",
                reason_code="VISION_QUERY_REQUIRED",
            )

        camera_result = None
        if region is not None:
            camera_result = await self.move_to_region(
                region=region,
                action_id=f"{action_id}:move_camera",
            )
            if camera_result.status != "succeeded":
                return ActionResult(
                    action_id=action_id,
                    action_type="see_action",
                    status="failed",
                    target=str(region),
                    outcome="camera_move_failed",
                    reason_code=camera_result.reason_code or "CAMERA_MOVE_FAILED",
                    retryable=camera_result.retryable,
                    data={"camera_move": camera_result.data},
                )

        try:
            jpeg_bytes = await asyncio.to_thread(
                self.camera.jpeg_bytes_snapshot,
                70,
                False,
                "results/action_results/see_snapshot.jpg",
            )
            snapshot = self.camera.snapshot()
            image_instruction = normalized_query
            if autonomous:
                image_instruction = (
                    "[INTERNAL AUTONOMOUS PERCEPTION] DJ initiated this observation "
                    "without a user request. Treat the attached image as DJ's own "
                    "current visual perception. Do not say or imply that the user "
                    f"asked for this inspection. {normalized_query}"
                )
            image_result = await self.send_camera_image(
                self.ws,
                jpeg_bytes,
                instruction=image_instruction,
            )
        except Exception as error:
            return ActionResult(
                action_id=action_id,
                action_type="see_action",
                status="failed",
                outcome="vision_capture_failed",
                reason_code="VISION_CAPTURE_FAILED",
                retryable=True,
                data={"error": f"{type(error).__name__}: {error}"},
            )

        return ActionResult(
            action_id=action_id,
            action_type="see_action",
            status="succeeded",
            outcome="image_context_added",
            data={
                "query": normalized_query,
                "source": "autonomy" if autonomous else "user",
                "frame_sequence": getattr(snapshot, "sequence", None),
                "captured_at": getattr(snapshot, "captured_at", None),
                "region": str(region) if region is not None else None,
                "camera_move": camera_result.data if camera_result is not None else None,
                **image_result,
            },
        )

    async def move_to_region(self, region, action_id):
        """Aim the camera for a see/search operation."""
        normalized_region = str(region or "").strip().lower().replace("-", "_")
        if normalized_region not in SEMANTIC_CAMERA_REGIONS:
            return ActionResult(
                action_id=action_id,
                action_type="see_action",
                status="failed",
                target=normalized_region or None,
                outcome="invalid_request",
                reason_code="UNKNOWN_CAMERA_REGION",
                data={"valid_regions": list(SEMANTIC_CAMERA_REGIONS)},
            )
        if self.camera_moving:
            return ActionResult(
                action_id=action_id,
                action_type="see_action",
                status="already_running",
                target=normalized_region,
                outcome="already_running",
                reason_code="CAMERA_MOVE_BUSY",
                retryable=True,
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

        self.camera_moving = True
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
                    action_type="see_action",
                    status="failed",
                    target=normalized_region,
                    outcome="rejected",
                    reason_code="CAMERA_MOVE_REJECTED",
                    retryable=True,
                    data={"message": str(message)},
                )

            await asyncio.sleep(self.settle_seconds)
            move_completed_at = time.monotonic()
            frame_ready = await asyncio.to_thread(
                self.camera.wait_for_frame_captured_after,
                move_completed_at,
                CAMERA_RECOVERY_TIMEOUT_SECONDS,
            )
            if not frame_ready:
                return ActionResult(
                    action_id=action_id,
                    action_type="see_action",
                    status="failed",
                    target=normalized_region,
                    outcome="camera_unavailable",
                    reason_code="CAMERA_RECOVERY_TIMEOUT",
                    retryable=True,
                )
            return ActionResult(
                action_id=action_id,
                action_type="see_action",
                status="succeeded",
                target=normalized_region,
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
            raise
        except Exception as error:
            return ActionResult(
                action_id=action_id,
                action_type="see_action",
                status="failed",
                target=normalized_region,
                outcome="command_failed",
                reason_code="CAMERA_MOVE_COMMAND_FAILED",
                retryable=True,
                data={"error": f"{type(error).__name__}: {error}"},
            )
        finally:
            self.camera_moving = False

    async def send_camera_image(self, ws, jpeg_bytes, instruction):
        jpeg_bytes = bytes(jpeg_bytes)
        if not jpeg_bytes or not jpeg_bytes.startswith(b"\xff\xd8"):
            raise ValueError("Invalid JPEG bytes. Encode the OpenCV frame with cv2.imencode('.jpg', frame).")

        encoded_image = base64.b64encode(jpeg_bytes).decode("ascii")
        image_url = f"data:image/jpeg;base64,{encoded_image}"

        image_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction.strip()},
                    {"type": "input_image", "image_url": image_url},
                ],
            },
        }

        await ws.send(json.dumps(image_event))
        print(f"\n[Camera] IMAGE ADDED")

        return {"jpeg_bytes": len(jpeg_bytes), "base64_characters": len(encoded_image)}
