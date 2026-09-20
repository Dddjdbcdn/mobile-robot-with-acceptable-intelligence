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

from cognition.state import robot_state
from actions.track_action import dj_yolo_classes, normalize_object_target

REPO_ROOT = Path(__file__).resolve().parent.parent
VISION_OOB_TOOLS_PATH = str(REPO_ROOT / "tools" / "vision_oob_tools.json")

tools = load_json(VISION_OOB_TOOLS_PATH)
vision_oob_tools = {tool["name"]: tool for tool in tools}
ASSESS_BATCH_SEARCH_TOOL = vision_oob_tools.get("assess_batch_search")
ASSESS_FRAME_SEARCH_TOOL = vision_oob_tools.get("assess_frame_search")

CANDIDATE_CONFIDENCE_THRESHOLD = 0.5
SEARCH_CANDIDATE_TYPES = {
    "visual", "local", "destination", "speculative",
}
CANDIDATE_TYPE_GUIDANCE = (
    "Classify candidate_type strictly. visual means an object closely matches the "
    "target visually, or a target-specific part is visible, but it is not yet "
    "definitive. local means specific target-linked evidence is visible nearby and "
    "a short movement may reveal the target; generic containers and ordinary nearby "
    "objects are not local evidence. destination means a recognized or mapped place "
    "or route has a strong semantic relationship to the target; an unidentified "
    "hallway alone is not a destination. speculative means only a generic possibility "
    "with no target-specific evidence. Use visual, local, or destination only when "
    "the visible evidence closely supports that definition; otherwise use speculative."
)
CAMERA_HORIZONTAL_FOV_DEG = 85
CAMERA_VERTICAL_FOV_DEG = 52
AIM_GAIN = 1.0
CAMERA_MOVE_TIMEOUT_SECONDS = 2.0
CAMERA_SETTLE_SECONDS = 0.8

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

PAN_SWEEP_ORDER = ("leftmost", "center", "rightmost")
SIDE_SWEEP_ORDER = ("leftmost", "rightmost")
SWEEP_TILT_ORDER = ("center", "upmost", "downmost")
SEARCH_EFFORT_ROWS = {
    "center": ("center",),
    "high": ("upmost",),
    "low": ("downmost",),
    "best_effort": ("center", "upmost", "downmost"),
}

from actions.action_result import ActionResult

class SearchAction:
    def __init__(self, ws, send_robot_command, camera, yolo=None):
        self.ws = ws
        self.send_robot_command = send_robot_command
        self.camera = camera
        self.yolo = yolo
        self.search_task = None
        self.action_id = None

        self.completion_future = None
        self.last_found_target = None
        # A contextual candidate keeps only its selected frame, not the full sweep.
        self.last_observation_frame: dict[str, Any] | None = None
        self.last_coverage_frames: list[dict[str, Any]] = []
        self.last_contextual_clue: dict[str, Any] | None = None
        self.last_detection_source: str | None = None
        self._yolo_owner = None
        self.reset_state()

    def reset_state(self) -> None:
        self.active = False
        self.search_task = None
        self.action_id = None
        self.search_id = None
        self.target = None
        self.effort = "center"
        self.initial_view_only = False
        self.sweep_tilt_order = SEARCH_EFFORT_ROWS[self.effort]
        self.sweep_pan_positions = SIDE_SWEEP_ORDER
        self.sweep_index = 0
        self.pan_angle = PAN_POSITION_ANGLE["center"]
        self.tilt_angle = TILT_POSITION_ANGLE["center"]
        self.current_batch_frames = []
        self.current_initial_frame = None
        self.captured_frames = []
        self.current_candidate = None
        self.first_frame_result = None
        self.batch_results = []
        self.pending_request_id = None
        self.candidate_context = None

    def _candidate_context_instruction(self) -> str:
        context = self.candidate_context
        if not isinstance(context, dict):
            return ""
        reassessment = ""
        if (
            context.get("candidate_type") == "visual"
            and int(context.get("reassessments_used") or 0)
            < int(context.get("reassessment_limit") or 0)
        ):
            reassessment = (
                "This is the single same-view reassessment before any movement. "
            )
        return (
            "\nContinue the active candidate hypothesis instead of starting a new "
            f"one: id={context.get('hypothesis_id')}; "
            f"type={context.get('candidate_type')}; "
            f"original clue={context.get('original_contextual_clue') or context.get('contextual_clue')}; "
            f"latest clue={context.get('contextual_clue')}; "
            f"remaining movements={context.get('remaining_waypoints')}. "
            f"{reassessment}"
            "Return candidate only if the current view still supports that same "
            "intention; otherwise return not_found.\n"
        )

    @staticmethod
    def _robot_pose_at_capture() -> dict[str, Any] | None:
        """Copy the map pose that belongs to a camera observation."""
        pose = robot_state.get("pose") or {}
        try:
            return {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "yaw": float(pose["yaw"]),
                "frame_id": str(pose.get("frame_id") or "map"),
            }
        except (KeyError, TypeError, ValueError):
            return None

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

        if tilt_position == "center" and self.current_initial_frame is not None:
            current_order = self.sweep_pan_positions
        elif tilt_position == "upmost":
            current_order = ("rightmost", "center", "leftmost")
        else:
            current_order = PAN_SWEEP_ORDER

        for image_number, pan_position in enumerate(current_order, start=1):
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
                "robot_pose": self._robot_pose_at_capture(),
            })
            self.captured_frames.append(frames[-1])

            yolo_candidate = self._best_yolo_candidate()
            if yolo_candidate is not None:
                self.current_batch_frames = frames
                self.current_candidate = {
                    "assessment": {
                        "result": "found",
                        "target_position": yolo_candidate["position"],
                        "contextual_clue": None,
                    },
                    "frame": frames[-1],
                    "sweep_index": sweep_index,
                }
                self.last_detection_source = "yolo"
                await self.move_to_found_target()
                await self.complete_searching("succeeded", "found")
                return

        self.current_batch_frames = self._comparison_frames(frames, tilt_position)
        await self.request_batch_assessment(self.current_batch_frames)

    def _comparison_frames(self, sweep_frames, tilt_position):
        """Combine the already-assessed center with newly captured side views."""
        frames = [dict(frame) for frame in sweep_frames]
        if tilt_position == "center" and self.current_initial_frame is not None:
            center = dict(self.current_initial_frame)
            center["pan_position"] = "center"
            center["tilt_position"] = "center"
            by_pan = {frame.get("pan_position"): frame for frame in frames}
            frames = [
                frame for frame in (
                    by_pan.get("leftmost"), center, by_pan.get("rightmost")
                ) if frame is not None
            ]
        for index, frame in enumerate(frames, start=1):
            frame["image_id"] = f"image_{index}"
        return frames

    async def request_frame_assessment(self):
        request_id = uuid.uuid4().hex
        self.pending_request_id = request_id

        jpeg_bytes = await asyncio.to_thread(
            self.camera.jpeg_bytes_snapshot,
            tracking_bgr=True,
            save_path=f"results/search_results/first_search_frame.jpg",
        )
        self.current_initial_frame = {
            "image_id": "image_1",
            "pan_position": "current",
            "tilt_position": "current",
            "pan_angle": float(self.pan_angle),
            "tilt_angle": float(self.tilt_angle),
            "jpeg_bytes": jpeg_bytes,
            "robot_pose": self._robot_pose_at_capture(),
        }
        self.captured_frames.append(self.current_initial_frame)

        yolo_candidate = self._best_yolo_candidate()
        if yolo_candidate is not None:
            self.current_candidate = {
                "assessment": {
                    "result": "found",
                    "target_position": yolo_candidate["position"],
                    "contextual_clue": None,
                },
                "frame": self.current_initial_frame,
                "sweep_index": -1,
            }
            self.last_detection_source = "yolo"
            await self.move_to_found_target()
            await self.complete_searching("succeeded", "found")
            return

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Search for this object: {self.target}\n\n"
                    f"{self._candidate_context_instruction()}"
                    "Use found only when the requested target is definitively visible.\n"
                    "Use candidate only for one specific but not yet definitive piece "
                    "of visible evidence.\n"
                    f"{CANDIDATE_TYPE_GUIDANCE}\n"
                    "Use not_found when neither the target nor a useful clue is visible.\n"
                    "Only candidate needs candidate_type and contextual_clue. "
                    "Use null candidate_type for found/not_found. "
                    "Only found needs target_position.\n"
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

    async def _start_frame_assessment(self, yolo_sequence):
        if (
            yolo_sequence is not None
            and hasattr(self.yolo, "wait_for_inference_after")
        ):
            await self.yolo.wait_for_inference_after(yolo_sequence)
        if self.active:
            await self.request_frame_assessment()

    def _best_yolo_candidate(self):
        """Return the best current YOLO match before spending an LLM call."""
        if self.yolo is None:
            return None
        normalized_target = normalize_object_target(self.target)
        if normalized_target not in dj_yolo_classes:
            return None
        detections = [
            detection for detection in list(self.yolo.detections)
            if detection.get("class") == normalized_target
        ]
        if not detections:
            return None
        detection = max(
            detections, key=lambda item: float(item.get("confidence") or 0.0)
        )
        confidence = float(detection.get("confidence") or 0.0)
        if confidence < CANDIDATE_CONFIDENCE_THRESHOLD:
            return None
        bbox = detection.get("bbox") or {}
        try:
            width, height = self.camera.tracking_size
            x = (float(bbox["x1"]) + float(bbox["x2"])) / (2.0 * width)
            y = (float(bbox["y1"]) + float(bbox["y2"])) / (2.0 * height)
            position = get_valid_position({"x": x, "y": y})
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
        if position is None:
            return None
        return {
            "position": position,
            "confidence": confidence,
        }

    async def request_batch_assessment(self, frames):
        request_id = uuid.uuid4().hex
        self.pending_request_id = request_id

        tilt_position = self.sweep_tilt_order[int(self.sweep_index)]

        images_order = ", ".join(
            f"{frame['image_id']}: {frame['pan_position']}"
            for frame in frames
        )
        initial_assessment = (
            (self.first_frame_result or {}).get("assessment") or {}
        )
        prior_candidate = ""
        if initial_assessment.get("result") == "candidate":
            prior_candidate = (
                "\nThe reused center frame was previously a candidate: "
                f"type={initial_assessment.get('candidate_type')}; "
                f"clue={initial_assessment.get('contextual_clue')}. "
                "Compare it with any side clues and select the best one.\n"
            )
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Search for this object: {self.target}\n\n"
                    f"{self._candidate_context_instruction()}"
                    "These images together cover the available horizontal search view. "
                    "The center image was assessed first and is reused here; the other "
                    "images are only the left/right directions whose 60-degree, 2-meter "
                    "Rank all supplied images together.\n"
                    f"Current tilt: {tilt_position}\n"
                    f"{images_order}\n"
                    f"{prior_candidate}"
                    "Examine every supplied image. Use found only when the requested "
                    "target is definitively visible; provide its image and target_position.\n"
                    "Use candidate only for one specific but not yet definitive piece "
                    "of visible evidence; provide its image, candidate_type, and "
                    "contextual_clue.\n"
                    f"{CANDIDATE_TYPE_GUIDANCE}\n"
                    "Use not_found when neither the target nor a useful clue is visible.\n"
                    "Use null candidate_type and contextual_clue for found/not_found, "
                    "and null target_position for candidate/not_found.\n"
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
        if result == "found":
            target_position = get_valid_position(args.get("target_position"))
            if target_position is not None:
                self.current_candidate = {
                    "assessment": {
                        **args,
                        "target_position": target_position,
                    },
                    "frame": self.current_initial_frame,
                    "sweep_index": -1,
                }
                await self.move_to_found_target()
                await self.complete_searching("succeeded", "found")
                return

        if result == "candidate" and self._contextual_clue_is_valid(args):
            self.current_candidate = {
                "assessment": args,
                "frame": self.current_initial_frame,
                "sweep_index": -1,
            }
            if self.sweep_pan_positions:
                await self.capture_sweep_batch()
            else:
                await self.move_to_candidate_frame()
                await self.complete_searching(
                    status="failed",
                    outcome="contextual_clue",
                    reason_code="SEARCH_CONTEXTUAL_CLUE",
                )
            return

        if not self.sweep_pan_positions:
            await self.complete_searching(
                status="failed",
                outcome="not_found",
                reason_code="TARGET_NOT_VISIBLE",
            )
            return

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
        frame_validated = get_valid_frame(
            args.get("candidate_image"), self.current_batch_frames
        )

        if result == "found":
            target_position = get_valid_position(args.get("target_position"))
            if target_position is not None and frame_validated is not None:

                save_candidate_debug_image(
                    jpeg_bytes=frame_validated["jpeg_bytes"],
                    candidate_position=target_position,
                    save_path=(REPO_ROOT / "results" / "search_results" / "candidate.jpg"),
                )

                self.current_candidate = {
                    "assessment": {
                        **args,
                        "target_position": target_position,
                    },
                    "frame": frame_validated,
                    "sweep_index": int(self.sweep_index),
                }

                await self.move_to_found_target()
                await self.complete_searching("succeeded", "found")
                return

        if (
            result == "candidate"
            and frame_validated is not None
            and self._contextual_clue_is_valid(args)
        ):
            self.current_candidate = {
                "assessment": args,
                "frame": frame_validated,
                "sweep_index": int(self.sweep_index),
            }
            await self.move_to_candidate_frame()
            await self.complete_searching(
                status="failed",
                outcome="contextual_clue",
                reason_code="SEARCH_CONTEXTUAL_CLUE",
            )
            return

        self.sweep_index += 1
        if self.sweep_index < len(self.sweep_tilt_order):
            await self.capture_sweep_batch()
            return
        await self.move_camera_angles(
            PAN_POSITION_ANGLE["center"], TILT_POSITION_ANGLE["center"]
        )
        await self.complete_searching(
            status="failed", outcome="not_found", reason_code="TARGET_NOT_VISIBLE"
        )
        return

    @staticmethod
    def _contextual_clue_is_valid(args) -> bool:
        clue = args.get("contextual_clue")
        return (
            isinstance(clue, str)
            and bool(clue.strip())
            and args.get("candidate_type") in SEARCH_CANDIDATE_TYPES
        )

    async def move_to_found_target(self) -> None:
        candidate = self.current_candidate
        if not isinstance(candidate, dict):
            raise RuntimeError("Target aiming requested without a found target")

        target_position = candidate["assessment"]["target_position"]
        candidate_frame = candidate.get("frame")

        target_pan, target_tilt = self.target_to_servo_angles(
            target_position["x"],
            target_position["y"],
            base_pan_angle=float(candidate_frame["pan_angle"]),
            base_tilt_angle=float(candidate_frame["tilt_angle"]),
        )

        await self.move_camera_angles(target_pan, target_tilt)

    async def move_to_candidate_frame(self) -> None:
        """Aim at the clue frame without treating the clue as the target."""
        frame = self.current_candidate["frame"]
        await self.move_camera_angles(
            float(frame["pan_angle"]),
            float(frame["tilt_angle"]),
        )

    async def start_searching(
        self,
        target,
        action_id,
        effort="center",
        initial_view_only=False,
        sweep_directions=None,
        candidate_context=None,
    ):
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
        self.last_observation_frame = None
        self.last_coverage_frames = []
        self.last_contextual_clue = None
        self.last_detection_source = None
        self.active = True
        self.action_id = action_id
        self.search_id = action_id
        self.target = normalized_target
        self.effort = normalized_effort
        self.initial_view_only = initial_view_only is True
        self.candidate_context = (
            dict(candidate_context)
            if isinstance(candidate_context, dict) else None
        )
        self.sweep_tilt_order = SEARCH_EFFORT_ROWS[normalized_effort]
        if self.initial_view_only:
            self.sweep_pan_positions = ()
        elif sweep_directions is None:
            self.sweep_pan_positions = SIDE_SWEEP_ORDER
        else:
            direction_to_pan = {
                "left": "leftmost",
                "right": "rightmost",
            }
            self.sweep_pan_positions = tuple(
                direction_to_pan[direction] for direction in sweep_directions
                if direction in direction_to_pan
            )
        self.pan_angle = robot_state["camera"]["pan_angle"]
        self.tilt_angle = robot_state["camera"]["tilt_angle"]

        yolo_sequence = None
        yolo_target = normalize_object_target(normalized_target)
        if (
            self.yolo is not None
            and yolo_target in dj_yolo_classes
            and hasattr(self.yolo, "activate")
        ):
            self._yolo_owner = f"search:{action_id}"
            yolo_sequence = self.yolo.activate(self._yolo_owner, "dj")

        clear_images_folder()
        self.completion_future = asyncio.get_running_loop().create_future()
        self.search_task = asyncio.create_task(
            self._start_frame_assessment(yolo_sequence)
        )

        return ActionResult(
            action_id=action_id,
            action_type="search_action",
            status="running",
            target=normalized_target,
            outcome="searching",
            data={
                "effort": self.effort,
                "initial_view_only": self.initial_view_only,
            },
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
        initial_view_only = self.initial_view_only
        configured_rows = tuple(self.sweep_tilt_order)
        scan_results = list(self.batch_results)
        first_frame_result = self.first_frame_result
        final_camera_pose = {
            "pan_angle": self.pan_angle,
            "tilt_angle": self.tilt_angle,
        }
        searched_rows = len(scan_results)
        coverage = min(1.0, searched_rows / max(1, len(configured_rows)))

        clue = None
        clue_frame = None
        if (
            outcome == "contextual_clue"
            and isinstance(self.current_candidate, dict)
        ):
            assessment = self.current_candidate.get("assessment") or {}
            frame = self.current_candidate.get("frame")
            if isinstance(frame, dict):
                clue = {
                    "text": str(assessment.get("contextual_clue") or "").strip(),
                    "candidate_type": str(assessment.get("candidate_type") or ""),
                }
                clue_frame = dict(frame)
                clue_frame["contextual_clue"] = clue["text"]
                clue_frame["candidate_type"] = clue["candidate_type"]

        data = {
            "initial_frame_assessment": first_frame_result,
            "scan_results": scan_results,
            "searched_rows": searched_rows,
            "configured_sweep_rows": len(configured_rows),
            "configured_tilt_positions": list(configured_rows),
            "search_coverage": coverage,
            "final_camera_pose": final_camera_pose,
            "effort": effort,
            "initial_view_only": initial_view_only,
            "detection_source": self.last_detection_source or "vision_assessment",
            "contextual_clue": clue,
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

        allowed_frame_keys = {
            "image_id", "pan_position", "tilt_position", "pan_angle",
            "tilt_angle", "jpeg_bytes", "robot_pose", "contextual_clue",
            "candidate_type", "hypothesis_id", "original_candidate_type",
            "original_contextual_clue",
            "movement_limit", "movements_used", "remaining_waypoints",
            "reassessment_limit", "reassessments_used", "allow_frontier",
        }
        self.last_coverage_frames = [
            {
                key: value for key, value in frame.items()
                if key in allowed_frame_keys
            }
            for frame in self.captured_frames
        ]
        self.last_observation_frame = (
            {
                key: value for key, value in clue_frame.items()
                if key in allowed_frame_keys
            }
            if clue_frame is not None else None
        )
        self.last_contextual_clue = clue
        self.last_found_target = self.target if status == "succeeded" else None
        if self._yolo_owner is not None:
            self.yolo.deactivate(self._yolo_owner)
            self._yolo_owner = None
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
