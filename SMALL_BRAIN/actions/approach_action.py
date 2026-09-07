from __future__ import annotations

import asyncio
import base64
import json
import math
from pathlib import Path
import uuid

from actions.action_result import ActionResult
from cognition.state import robot_state
from utilities.database_functions import load_json

from actions.track_action import normalize_human_target, normalize_object_target

REPO_ROOT = Path(__file__).resolve().parent.parent
VISION_OOB_TOOLS_PATH = str(REPO_ROOT / "tools" / "vision_oob_tools.json")
vision_oob_tools = {
    tool["name"]: tool for tool in load_json(VISION_OOB_TOOLS_PATH)
}
ASSESS_APPROACH_VERIFICATION_TOOL = vision_oob_tools.get(
    "assess_approach_verification"
)
VERIFICATION_CONFIDENCE_THRESHOLD = 0.65
VERIFICATION_TIMEOUT_SECONDS = 15.0

class ApproachAction:
    def __init__(
        self,
        send_robot_command,
        track_action,
        ws=None,
        camera=None,
        max_verification_retries=1,
    ):
        self.send_robot_command = send_robot_command
        self.track_action = track_action
        self.ws = ws
        self.camera = camera
        self.max_verification_retries = max(0, int(max_verification_retries))
        self.active = False
        self.action_id: str | None = None
        self.target: str | None = None
        self.completion_future: asyncio.Future[ActionResult] | None = None
        self._verification_task: asyncio.Task | None = None
        self._verification_future: asyncio.Future[dict] | None = None
        self._verification_request_id: str | None = None
        self._verification_retries = 0
        self._navigation_attempts = 0
        self._verification_history: list[dict] = []
        self._last_destination: dict | None = None

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

        destination = self._current_destination()
        if destination is None:
            return ActionResult(
                action_id=action_id,
                action_type="approach_action",
                status="failed",
                target=target,
                outcome="range_invalid",
                reason_code="APPROACH_RANGE_INVALID",
                retryable=True,
            )

        self.completion_future = asyncio.get_running_loop().create_future()
        self.active = True
        self.action_id = action_id
        self.target = target
        self._verification_retries = 0
        self._navigation_attempts = 0
        self._verification_history = []
        self._last_destination = None

        accepted, message = await self._dispatch_navigation(destination)
        if not accepted:
            return self.complete_approaching(
                status="failed",
                outcome="rejected",
                reason_code="NAVIGATION_REJECTED",
                data={"message": message},
            )
            
        return ActionResult(
            action_id=action_id,
            action_type="approach_action",
            status="running",
            target=target,
            outcome="approaching",
            data={
                "tracking_stable": True,
                "destination": dict(destination),
                "post_navigation_verification": True,
            },
        )

    def _current_destination(self):
        camera_state = robot_state.get("camera") or {}
        values = (
            camera_state.get("object_x"),
            camera_state.get("object_y"),
            camera_state.get("object_angle"),
            camera_state.get("camera_tof_range"),
        )
        if not all(isinstance(value, (int, float)) for value in values):
            return None
        x, y, angle, tof_range = (float(value) for value in values)
        if (
            not all(math.isfinite(value) for value in (x, y, angle, tof_range))
            or tof_range <= 0.05
        ):
            return None
        return {"x": x, "y": y, "angle": angle, "tof_range": tof_range}

    async def _dispatch_navigation(self, destination):
        payload = {
            "command": "navigate_to_pose",
            "action_id": self.action_id,
            "x": destination["x"],
            "y": destination["y"],
            "angle": destination["angle"],
        }
        feedback = await self.send_robot_command(payload)
        if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
            message = (
                feedback.get("message", "navigation_rejected")
                if isinstance(feedback, dict)
                else "invalid_navigation_feedback"
            )
            return False, str(message)
        self._navigation_attempts += 1
        self._last_destination = dict(destination)
        return True, "accepted"

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
        if succeeded:
            if self._verification_task is None:
                self._verification_task = asyncio.create_task(
                    self._verify_navigation_result(),
                    name=f"approach-verify-{self.action_id}",
                )
            return True

        self.complete_approaching(
            status="failed",
            outcome="navigation_failed",
            reason_code="NAVIGATION_FAILED",
            data={
                "robot_status": navigation_status or "unknown",
                "navigation_attempts": self._navigation_attempts,
                "verification_history": list(self._verification_history),
            },
        )
        return True

    async def _verify_navigation_result(self):
        try:
            assessment = await self._request_approach_verification()
            self._verification_history.append(dict(assessment))
            result = assessment["result"]
            confidence = assessment["confidence"]

            if result == "reached" and confidence >= VERIFICATION_CONFIDENCE_THRESHOLD:
                self.complete_approaching(
                    status="succeeded",
                    outcome="reached_and_verified",
                    data=self._verification_data(assessment),
                )
                return

            if (
                result == "too_far"
                and confidence >= VERIFICATION_CONFIDENCE_THRESHOLD
                and self._verification_retries < self.max_verification_retries
            ):
                if not self.track_action.active:
                    self._complete_verification_required(
                        assessment, "TARGET_NOT_TRACKED_AFTER_NAVIGATION"
                    )
                    return
                if not self.track_action.stable:
                    stable = await self.track_action.wait_until_stable(timeout=5.0)
                    if not stable:
                        self._complete_verification_required(
                            assessment, "TRACKING_STABILITY_TIMEOUT"
                        )
                        return

                destination = self._current_destination()
                if destination is None:
                    self._complete_verification_required(
                        assessment, "APPROACH_RANGE_INVALID"
                    )
                    return

                self._verification_retries += 1
                accepted, message = await self._dispatch_navigation(destination)
                if not accepted:
                    self.complete_approaching(
                        status="failed",
                        outcome="rejected",
                        reason_code="NAVIGATION_REJECTED",
                        data={
                            **self._verification_data(assessment),
                            "message": message,
                        },
                    )
                return

            self._complete_verification_required(
                assessment, "APPROACH_VERIFICATION_REQUIRED"
            )
        except asyncio.CancelledError:
            return
        except TimeoutError:
            self._complete_verification_required(
                {
                    "result": "uncertain",
                    "confidence": 0.0,
                    "target_visible": False,
                    "occlusion": "not_visible",
                    "evidence": "Vision verification timed out.",
                },
                "APPROACH_VERIFICATION_TIMEOUT",
            )
        except Exception as error:
            self.complete_approaching(
                status="failed",
                outcome="verification_error",
                reason_code="APPROACH_VERIFICATION_ERROR",
                data={
                    "error": f"{type(error).__name__}: {error}",
                    "navigation_attempts": self._navigation_attempts,
                    "verification_history": list(self._verification_history),
                },
            )
        finally:
            if asyncio.current_task() is self._verification_task:
                self._verification_task = None

    async def _request_approach_verification(self):
        if self.ws is None or self.camera is None or ASSESS_APPROACH_VERIFICATION_TOOL is None:
            raise RuntimeError("Approach verification is not configured")

        request_id = uuid.uuid4().hex
        self._verification_request_id = request_id
        self._verification_future = asyncio.get_running_loop().create_future()
        jpeg_bytes = await asyncio.to_thread(
            self.camera.jpeg_bytes_snapshot,
            tracking_bgr=True,
            save_path="results/action_results/approach_verification.jpg",
        )
        tof_range = (robot_state.get("camera") or {}).get("camera_tof_range")
        content = [
            {
                "type": "input_text",
                "text": (
                    f"Verify whether the robot has actually reached this target: {self.target}.\n"
                    f"Current ToF reading: {tof_range!r} meters.\n"
                    "Judge primarily from visible target scale, perspective, and scene context. "
                    "The ToF value may be false when the target is partially occluded or the beam "
                    "hits another surface. Use reached only when the target is identifiable and "
                    "clearly near. Use too_far only when it is identifiable and clearly still far. "
                    "Use uncertain when occlusion or perspective prevents reliable verification. "
                    "Always call assess_approach_verification exactly once."
                ),
            },
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,"
                + base64.b64encode(jpeg_bytes).decode("ascii"),
            },
        ]
        await self.ws.send(json.dumps({
            "event_id": f"approach_verification_{request_id}",
            "type": "response.create",
            "response": {
                "conversation": "none",
                "metadata": {
                    "kind": "approach_verification",
                    "action_id": str(self.action_id),
                    "request_id": request_id,
                    "navigation_attempt": str(self._navigation_attempts),
                },
                "output_modalities": ["text"],
                "tools": [ASSESS_APPROACH_VERIFICATION_TOOL],
                "tool_choice": "required",
                "input": [{"type": "message", "role": "user", "content": content}],
            },
        }))
        try:
            return await asyncio.wait_for(
                self._verification_future,
                timeout=VERIFICATION_TIMEOUT_SECONDS,
            )
        finally:
            self._verification_request_id = None
            self._verification_future = None

    async def assess_approach_verification(self, args, response_metadata):
        if not self.active or response_metadata.get("kind") != "approach_verification":
            return
        if str(response_metadata.get("action_id")) != str(self.action_id):
            return
        if response_metadata.get("request_id") != self._verification_request_id:
            return
        future = self._verification_future
        if future is None or future.done():
            return

        result = args.get("result")
        occlusion = args.get("occlusion")
        if result not in {"reached", "too_far", "uncertain"}:
            result = "uncertain"
        if occlusion not in {"clear", "partial", "heavy", "not_visible"}:
            occlusion = "not_visible"
        try:
            confidence = max(0.0, min(1.0, float(args.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        target_visible = args.get("target_visible") is True
        if not target_visible and result in {"reached", "too_far"}:
            result = "uncertain"
        future.set_result({
            "result": result,
            "confidence": confidence,
            "target_visible": target_visible,
            "occlusion": occlusion,
            "evidence": str(args.get("evidence") or "No evidence provided."),
            "tof_range": (robot_state.get("camera") or {}).get("camera_tof_range"),
        })

    def _verification_data(self, assessment):
        verified = (
            assessment.get("result") == "reached"
            and assessment.get("target_visible") is True
            and float(assessment.get("confidence", 0.0))
            >= VERIFICATION_CONFIDENCE_THRESHOLD
        )
        return {
            "robot_status": "Goal Reached",
            "verified": verified,
            "verification": dict(assessment),
            "verification_history": list(self._verification_history),
            "navigation_attempts": self._navigation_attempts,
            "automatic_retries": self._verification_retries,
            "destination": dict(self._last_destination or {}),
        }

    def _complete_verification_required(self, assessment, reason_code):
        self.complete_approaching(
            status="failed",
            outcome="verification_required",
            reason_code=reason_code,
            data={
                **self._verification_data(assessment),
                "user_guidance": {
                    "question": (
                        f"I reached the navigation goal but couldn't verify that I am "
                        f"really at {self.target}. Should I continue carefully, or can "
                        "you uncover or reposition the target?"
                    ),
                    "continue_requires_new_approach": True,
                },
            },
        )

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
        verification_future = self._verification_future
        verification_task = self._verification_task
        self.active = False
        self.action_id = None
        self.target = None
        self._verification_request_id = None
        self._verification_future = None
        if verification_future is not None and not verification_future.done():
            verification_future.cancel()
        if (
            verification_task is not None
            and verification_task is not asyncio.current_task()
            and not verification_task.done()
        ):
            verification_task.cancel()
        self._verification_task = None

        if completion_future is not None and not completion_future.done():
            completion_future.set_result(result)

        return result
