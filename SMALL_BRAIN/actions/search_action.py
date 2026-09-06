from __future__ import annotations
from io import BytesIO
from PIL import Image, ImageDraw
import asyncio
import base64
import json
from pathlib import Path
import time
from typing import Any
import uuid

# Assuming these are available in your environment
from utilities.database_functions import load_json

from database.state import robot_state

REPO_ROOT = Path(__file__).resolve().parent.parent
VISION_OOB_TOOLS_PATH = str(REPO_ROOT / "tools" / "vision_oob_tools.json")

tools = load_json(VISION_OOB_TOOLS_PATH)
vision_oob_tools = {tool["name"]: tool for tool in tools}
ASSESS_BATCH_SEARCH_TOOL = vision_oob_tools.get("assess_batch_search")
ASSESS_FRAME_SEARCH_TOOL = vision_oob_tools.get("assess_frame_search")

CANDIDATE_CONFIDENCE_THRESHOLD = 0.5
CAMERA_HORIZONTAL_FOV_DEG = 85
CAMERA_VERTICAL_FOV_DEG = 52
AIM_GAIN = 1.0
CAMERA_MOVE_TIMEOUT_SECONDS = 2.0
CAMERA_SETTLE_SECONDS = 0.8

PAN_POSITION_ANGLE = {
    "center": 95.0,
    "leftmost": 160.0,
    "rightmost": 30.0,
}

TILT_POSITION_ANGLE = {
    "center": 90.0,
    "upmost": 30.0,
    "downmost": 120.0,
}

PAN_SWEEP_ORDER = ("leftmost", "center", "rightmost")
SWEEP_TILT_ORDER = ("center", "upmost", "downmost")
SEARCH_EFFORT_ROWS = {
    "center": ("center",),
    "high": ("upmost",),
    "low": ("downmost",),
    "best_effort": ("upmost", "downmost"),
}

from actions.action_result import ActionResult

class SearchAction:
    def __init__(self, ws, send_robot_command, camera):
        self.ws = ws
        self.send_robot_command = send_robot_command
        self.camera = camera
        self.search_task = None
        self.action_id = None

        self.completion_future = None
        self.last_found_target = None
        self.reset_state()

    def reset_state(self) -> None:
        self.active = False
        self.search_task = None
        self.action_id = None
        self.search_id = None
        self.target = None
        self.effort = "center"
        self.sweep_tilt_order = SEARCH_EFFORT_ROWS[self.effort]
        self.sweep_index = 0
        self.pan_angle = PAN_POSITION_ANGLE["center"]
        self.tilt_angle = TILT_POSITION_ANGLE["center"]
        self.current_batch_frames = []
        self.current_candidate = None
        self.first_frame_result = None
        self.batch_results = []
        self.pending_request_id = None

    def target_to_servo_angles(self, center_x: float, center_y: float, base_pan_angle: float, base_tilt_angle: float) -> tuple[float, float]:
        image_x_error = center_x - 0.5
        image_y_error = center_y - 0.5

        pan_correction = -image_x_error * CAMERA_HORIZONTAL_FOV_DEG * AIM_GAIN
        tilt_correction = image_y_error * CAMERA_VERTICAL_FOV_DEG * AIM_GAIN

        target_pan = base_pan_angle + pan_correction
        target_tilt = base_tilt_angle + tilt_correction

        return target_pan, target_tilt

    async def move_camera_angles(self, target_pan: float, target_tilt: float) -> None:
        delta_pan = target_pan - float(self.pan_angle)
        delta_tilt = target_tilt - float(self.tilt_angle)

        payload = {
            "command": "move_camera",
            "delta_pan_angle": delta_pan,
            "delta_tilt_angle": delta_tilt,
        }

        await self.send_robot_command(payload)

        self.pan_angle += delta_pan
        self.tilt_angle += delta_tilt

        await asyncio.sleep(CAMERA_SETTLE_SECONDS)

    async def capture_sweep_batch(self) -> None:
        if not self.active:
            return

        sweep_index = int(self.sweep_index)
        if sweep_index >= len(self.sweep_tilt_order):
            raise RuntimeError("Sweep index exceeded configured tilt rows")

        tilt_position = self.sweep_tilt_order[sweep_index]
        frames: list[dict[str, Any]] = []

        CURRENT_ORDER = PAN_SWEEP_ORDER if tilt_position != "upmost" else ("rightmost", "center", "leftmost")

        for image_number, pan_position in enumerate(CURRENT_ORDER, start=1):
            await self.move_camera_angles(
                PAN_POSITION_ANGLE[pan_position],
                TILT_POSITION_ANGLE[tilt_position]
            )
            jpeg_bytes = await asyncio.to_thread(
                self.camera.jpeg_bytes_snapshot,
                tracking_bgr=True,
                save_path=f"results/search_results/row_{sweep_index}_{image_number}.jpg",
            )
            frames.append({
                "image_id": f"image_{image_number}",
                "pan_position": pan_position,
                "tilt_position": tilt_position,
                "pan_angle": float(self.pan_angle),
                "tilt_angle": float(self.tilt_angle),
                "jpeg_bytes": jpeg_bytes,
            })

        self.current_batch_frames = frames
        await self.request_batch_assessment(frames)

    async def request_frame_assessment(self):
        request_id = uuid.uuid4().hex
        self.pending_request_id = request_id

        jpeg_bytes = await asyncio.to_thread(
            self.camera.jpeg_bytes_snapshot,
            tracking_bgr=True,
            save_path=f"results/search_results/first_search_frame.jpg",
        )

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Search for this object: {self.target}\n\n"
                    "Use candidate when an object is visually clear or plausibly resembles the target.\n"
                    "Use not_found when the target is not visible.\n"
                    "Always call assess_frame_search exactly once."
                ),
            },
            {
                "type": "input_image",
                "image_url": jpeg_to_data_url(jpeg_bytes),
            }
        ]

        event = {
            "event_id": f"visual_search_scan_{request_id}",
            "type": "response.create",
            "response": {
                "conversation": "none",
                "metadata": {
                    "kind": "visual_search_assessment",
                    "search_id": str(self.search_id),
                    "request_id": request_id,
                },
                "output_modalities": ["text"],
                "tools": [ASSESS_FRAME_SEARCH_TOOL],
                "tool_choice": "required",
                "input": [{"type": "message", "role": "user", "content": content}],
            },
        }
        await self.ws.send(json.dumps(event))

    async def request_batch_assessment(self, frames):
        request_id = uuid.uuid4().hex
        self.pending_request_id = request_id

        tilt_position = self.sweep_tilt_order[int(self.sweep_index)]

        images_order = "image_1: leftmost, image_2: center, image_3: rightmost"

        if tilt_position == "upmost":
            images_order = "image_1: rightmost, image_2: center, image_3: leftmost"

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Search for this object: {self.target}\n\n"
                    "These three images form one horizontal camera sweep.\n"
                    f"Current tilt: {tilt_position}\n"
                    f"{images_order}\n"
                    "Examine all three images.\n"
                    "Use candidate when an object is visually clear or plausibly resembles the target.\n"
                    "Use not_found when the target is not visible.\n"
                    "For candidate, return the best image and a normalized "
                    "center position of the target. For other results, return "
                    "candidate_image=none and candidate_position=null.\n"
                    "Always call assess_batch_search exactly once."
                ),
            }
        ]

        for frame in frames:
            content.extend([
                {
                    "type": "input_text",
                    "text": f"{frame['image_id']}: pan={frame['pan_position']}, tilt={frame['tilt_position']}",
                },
                {
                    "type": "input_image",
                    "image_url": jpeg_to_data_url(frame["jpeg_bytes"]),
                },
            ])

        event = {
            "event_id": f"visual_search_scan_{request_id}",
            "type": "response.create",
            "response": {
                "conversation": "none",
                "metadata": {
                    "kind": "visual_search_assessment",
                    "search_id": str(self.search_id),
                    "request_id": request_id,
                    "sweep_index": str(self.sweep_index),
                },
                "output_modalities": ["text"],
                "tools": [ASSESS_BATCH_SEARCH_TOOL],
                "tool_choice": "required",
                "input": [{"type": "message", "role": "user", "content": content}],
            },
        }
        await self.ws.send(json.dumps(event))

    async def assess_frame_search(self, args, response_metadata) -> None:
        if not self.active:
            return

        if response_metadata.get("kind") != "visual_search_assessment": return
        if str(response_metadata.get("search_id")) != str(self.search_id): return
        if response_metadata.get("request_id") != self.pending_request_id: return

        self.pending_request_id = None

        self.first_frame_result = {
            "initial_pan_position": self.pan_angle,
            "initial_tilt_position": self.tilt_angle,
            "assessment": args,
        }

        result = args.get("result")
        confidence = float(args.get("confidence", 0.0))

        if result == "candidate" and confidence >= CANDIDATE_CONFIDENCE_THRESHOLD:
            await self.complete_searching(
                status="succeeded",
                outcome="found"
            )
            return

        else:
            await self.capture_sweep_batch()
        return

    async def assess_batch_search(self, args, response_metadata) -> None:
        if not self.active:
            return

        if response_metadata.get("kind") != "visual_search_assessment": return
        if str(response_metadata.get("search_id")) != str(self.search_id): return
        if response_metadata.get("request_id") != self.pending_request_id: return

        self.pending_request_id = None

        batch_result = {
            "sweep_index": int(self.sweep_index),
            "tilt_position": self.sweep_tilt_order[int(self.sweep_index)],
            "assessment": args,
        }
        self.batch_results.append(batch_result)

        result = args.get("result")
        confidence = float(args.get("confidence", 0.0))

        if result == "candidate" and confidence >= CANDIDATE_CONFIDENCE_THRESHOLD:
            position_validated = get_valid_position(args.get("candidate_position"))
            frame_validated = get_valid_frame(args.get("candidate_image"), self.current_batch_frames)

            if position_validated is not None and frame_validated is not None:
                candidate_position = position_validated
                candidate_frame = frame_validated

                save_candidate_debug_image(
                    jpeg_bytes=candidate_frame["jpeg_bytes"],
                    candidate_position=candidate_position,
                    save_path=(REPO_ROOT / "results" / "search_results" / "candidate.jpg"),
                )

                self.current_candidate = {
                    "assessment": {
                        **args,
                        "candidate_position": candidate_position,
                    },
                    "frame": candidate_frame,
                    "sweep_index": int(self.sweep_index),
                }

                await self.move_to_candidate()
                await self.complete_searching(
                    status="succeeded",
                    outcome="found"
                )
                return

        self.sweep_index += 1
        if self.sweep_index < len(self.sweep_tilt_order):
            await self.capture_sweep_batch()
            return
        else:
            await self.move_camera_angles(PAN_POSITION_ANGLE["center"], TILT_POSITION_ANGLE["center"])
            await self.complete_searching(
                status="failed",
                outcome=(
                    "search_guidance_required"
                    if self.effort == "center"
                    else "not_found"
                ),
                reason_code=(
                    "SEARCH_GUIDANCE_REQUIRED"
                    if self.effort == "center"
                    else "TARGET_NOT_VISIBLE"
                ),
            )
        return

    async def move_to_candidate(self) -> None:
        candidate = self.current_candidate
        if not isinstance(candidate, dict):
            raise RuntimeError("Verification requested without a candidate")

        candidate_x_position = candidate["assessment"].get("candidate_position").get("x")
        candidate_y_position = candidate["assessment"].get("candidate_position").get("y")
        candidate_frame = candidate.get("frame")

        target_pan, target_tilt = self.target_to_servo_angles(
            candidate_x_position,
            candidate_y_position,
            base_pan_angle=float(candidate_frame["pan_angle"]),
            base_tilt_angle=float(candidate_frame["tilt_angle"]),
        )

        await self.move_camera_angles(target_pan, target_tilt)

    async def start_searching(self, target, action_id, effort="center"):
        if self.active:
            return ActionResult(
                action_id=action_id,
                action_type="search_action",
                status="already_running",
                target=target,
                outcome="already_running",
                reason_code="SEARCH_BUSY",
                retryable=True,
                data={
                    "active_action_id": self.action_id,
                    "active_target": self.target,
                },
            )

        normalized_target = (target or "").strip()
        if not normalized_target:
            return ActionResult(
                action_id=action_id,
                action_type="search_action",
                status="failed",
                outcome="invalid_request",
                reason_code="SEARCH_TARGET_REQUIRED",
            )

        normalized_effort = str(effort or "center").strip().lower().replace("-", "_")
        if normalized_effort not in SEARCH_EFFORT_ROWS:
            return ActionResult(
                action_id=action_id,
                action_type="search_action",
                status="failed",
                target=normalized_target,
                outcome="invalid_request",
                reason_code="UNKNOWN_SEARCH_EFFORT",
            )

        self.reset_state()
        self.active = True
        self.action_id = action_id
        self.search_id = action_id
        self.target = normalized_target
        self.effort = normalized_effort
        self.sweep_tilt_order = SEARCH_EFFORT_ROWS[normalized_effort]
        self.pan_angle = robot_state["camera"]["pan_angle"]
        self.tilt_angle = robot_state["camera"]["tilt_angle"]

        clear_images_folder()
        self.completion_future = asyncio.get_running_loop().create_future()
        self.search_task = asyncio.create_task(self.request_frame_assessment())

        return ActionResult(
            action_id=action_id,
            action_type="search_action",
            status="running",
            target=normalized_target,
            outcome="searching",
            data={"effort": self.effort},
        )

    async def wait_until_finished(self):
        return await self.completion_future

    async def stop_searching(self,reason=None):
        if not self.active:
            return

        self.pending_request_id = None
        if self.search_task is not None and not self.search_task.done():
            self.search_task.cancel()
        self.search_task = None

        await self.move_camera_angles(
            PAN_POSITION_ANGLE["center"],
            TILT_POSITION_ANGLE["center"],
        )
        await self.complete_searching(
            status="cancelled",
            outcome="search_cancelled",
            reason_code=reason,
        )

    async def complete_searching(self,status,outcome,reason_code=None):
        if not self.active:
            return None

        action_id = self.action_id
        target = str(self.target)
        effort = self.effort
        configured_rows = tuple(self.sweep_tilt_order)
        scan_results = list(self.batch_results)
        first_frame_result = self.first_frame_result
        final_camera_pose = {
            "pan_angle": self.pan_angle,
            "tilt_angle": self.tilt_angle,
        }
        searched_rows = len(scan_results)
        coverage = min(1.0, searched_rows / max(1, len(configured_rows)))

        data = {
            "initial_frame_assessment": first_frame_result,
            "scan_results": scan_results,
            "searched_rows": searched_rows,
            "configured_sweep_rows": len(configured_rows),
            "configured_tilt_positions": list(configured_rows),
            "search_coverage": coverage,
            "final_camera_pose": final_camera_pose,
            "effort": effort,
        }
        if reason_code == "SEARCH_GUIDANCE_REQUIRED":
            data["user_guidance"] = {
                "question": (
                    f"I couldn't find {target} at center height. "
                    "Is it high, low, not in frame, or should I try my best?"
                ),
                "effort_options": ["high", "low", "best_effort"],
                "not_in_frame_requires_no_retry": True,
            }

        action_result = ActionResult(
            action_id=action_id,
            action_type="search_action",
            status=status,
            target=target,
            outcome=outcome,
            reason_code=(
                reason_code
                if reason_code is not None
                else "TARGET_NOT_VISIBLE" if status == "failed" else None
            ),
            retryable=status == "failed",
            data=data,
        )

        completion_future = self.completion_future
        self.last_found_target = self.target if status == "succeeded" else None
        self.reset_state()

        if completion_future is not None and not completion_future.done():
            completion_future.set_result(action_result)

        return action_result

# Helper Functions

def jpeg_to_data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"

def get_valid_position(position):
    if isinstance(position, dict):
        x = position.get("x")
        y = position.get("y")
    elif isinstance(position, (list, tuple)) and len(position) == 2:
        x, y = position
    else: return None
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0: return None
    return {"x": float(x),"y": float(y)}

def get_valid_frame(image_id, frames):
    if image_id not in {"image_1", "image_2", "image_3"}:
        return None
    return next((frame for frame in frames if frame.get("image_id") == image_id), None)

def get_valid_box(box):
    if isinstance(box, dict):
        x_min = box.get("x_min")
        x_max = box.get("x_max")
        y_min = box.get("y_min")
        y_max = box.get("y_max")
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        x_min, x_max, y_min, y_max = box
    else:
        return None

    if (
        not 0.0 <= x_min <= 1.0 or not 0.0 <= x_max <= 1.0
        or not 0.0 <= y_min <= 1.0 or not 0.0 <= y_max <= 1.0
        or not y_min < y_max or not x_min < x_max
    ):
        return None

    return {"x_min": float(x_min),"x_max": float(x_max),"y_min": float(y_min),"y_max": float(y_max)}

def save_candidate_debug_image(
    jpeg_bytes: bytes,
    candidate_position: dict[str, float],
    save_path: str | Path,
    *,
    radius: int = 8,
):
    x_normalized = float(candidate_position["x"])
    y_normalized = float(candidate_position["y"])

    with Image.open(BytesIO(jpeg_bytes)) as image:
        image = image.convert("RGB")
        width, height = image.size

        x_pixel = round(x_normalized * (width - 1))
        y_pixel = round(y_normalized * (height - 1))

        draw = ImageDraw.Draw(image)

        draw.ellipse(
            (
                x_pixel - radius,
                y_pixel - radius,
                x_pixel + radius,
                y_pixel + radius,
            ),
            fill="red",
            outline="white",
            width=2,
        )

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=95)

def clear_images_folder(folder_path="results/search_results", extensions=(".jpg", ".jpeg", ".png", ".webp")):
    target_dir = Path(folder_path)
    if not target_dir.is_dir():
        return 0

    deleted_count = 0
    for item in target_dir.iterdir():
        if item.is_file() and (extensions is None or item.suffix.lower() in extensions):
            try:
                item.unlink()
                deleted_count += 1
            except OSError as e:
                pass

    return deleted_count
