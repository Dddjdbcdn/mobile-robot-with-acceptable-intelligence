"""Select one vision-grounded map pose for the find loop."""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid

import cv2
import zmq

from actions.action_result import ActionResult


SEARCH_WAYPOINT_QUERY = (
    "Search for {target}. Select exactly one {mode} waypoint for the search "
    "controller; this navigation call ends when that waypoint is reached."
)

SEMANTIC_HEADINGS = {
    "forward": 0.0,
    "forward_left": math.pi / 4.0,
    "left": math.pi / 2.0,
    "back_left": 3.0 * math.pi / 4.0,
    "backward": math.pi,
    "back_right": -3.0 * math.pi / 4.0,
    "right": -math.pi / 2.0,
    "forward_right": -math.pi / 4.0,
}

MAP_GUIDANCE = """MAP AND OUTPUT CONTRACT
The map is robot-relative: the red robot points UP. UP is forward, LEFT and
RIGHT match the centered camera image, and DOWN is behind. Black is occupied,
light cells are known traversable space, and gray is unknown or unavailable.
Blue labeled markers are the only selectable poses. H holds position and only
changes heading. An orange outline, when present, identifies the map region
associated with the supplied clue frame.

Choose exactly one listed pose and one final semantic heading. The heading is
relative to the displayed robot-upright map and controls where the base and
centered camera face after arrival. Nav2 owns path planning and obstacle
avoidance. Never invent coordinates or an ID. Return blocked only when no
listed pose can satisfy the mode contract. There is no later navigation
assessment, correction, confirmation move, or second chance in this call."""

MODE_GUIDANCE = {
    "visual": """MODE: tracker fallback for a visually confirmed target
Image 2 is the frame where the vision model found the requested target, but the
metric tracker could not produce a stable approach pose. Select one conservative
pose inside that observation cone that makes progress toward the visible target
while keeping it observable. This is a recovery waypoint only; the search
controller must detect and track the target again after arrival.""",
    "context": """MODE: object-search context
Image 2 is a contextual clue chosen by the object-search controller. Select one
nearby investigative pose that follows that specific clue. C poses investigate
the clue cone and CF is the furthest reachable point on its center ray. This
call performs only one search waypoint; the
object-search controller decides what happens afterward. Do not claim that the
object has been found.""",
    "exploration": """MODE: search exploration
Image 2 is the fresh centered view. Select exactly one waypoint that exposes
useful unobserved traversable space for the search controller. E is the best
nearby inspection pose; EF, when offered, extends forward through known free
space. Prefer E when a farther move could skip openings or useful detail. This
call ends at the selected waypoint and never decides whether the search target
has been found.""",
}


class NavigationError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class MapNavigationAction:
    """One selection, at most one Nav2 movement, then a terminal result."""

    VISION_TIMEOUT = 15.0
    NAVIGATION_TIMEOUT = 90.0
    CAMERA_MAX_AGE = 2.0
    JPEG_QUALITY = 75

    def __init__(
        self,
        ws,
        camera,
        send_robot_command,
        request_map_snapshot,
        save_map_snapshot,
        see_action,
        send_map_overlay,
        debug_dir=None,
        map_crop_size_m=None,
        move_camera_action=None,
    ):
        self.ws = ws
        self.camera = camera
        self.send_robot_command = send_robot_command
        self.request_map_snapshot = request_map_snapshot
        self.save_map_snapshot = save_map_snapshot
        self.send_map_overlay = send_map_overlay
        self.see_action = see_action
        self.move_camera_action = (
            move_camera_action
            or getattr(see_action, "move_camera_action", see_action)
        )
        definitions_path = Path(__file__).resolve().parents[1] / "tools" / "vision_oob_tools.json"
        definitions = json.loads(definitions_path.read_text())
        self.selection_tool_template = next(
            tool for tool in definitions if tool["name"] == "select_navigation_pose"
        )
        configured_debug_dir = debug_dir or os.environ.get("NAVIGATION_DEBUG_DIR")
        self.debug_dir = Path(configured_debug_dir) if configured_debug_dir else (
            Path(__file__).resolve().parents[1] / "results" / "navigation"
        )
        self.map_crop_size_m = float(
            map_crop_size_m if map_crop_size_m is not None
            else os.environ.get("NAVIGATION_MAP_CROP_SIZE_M", "6.0")
        )
        if not 1.0 <= self.map_crop_size_m <= 20.0:
            raise ValueError("navigation map crop size must be between 1 and 20 metres")

        self.active = False
        self.action_id = self.target = None
        self.stage = "idle"
        self.completion_future = None
        self._worker = self._selection_future = self._navigation_future = None
        self._request_id = self._snapshot_id = None
        self._overlay_action_id = None
        self._overlay_revision = 0
        self._owns_overlay = False
        self._candidates = {}
        self._observation = None
        self._mode = "exploration"
        self._stop_requested = self._command_sent = False
        self._command_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._result_data = {}

    async def start(
        self, query, action_id, *, mode="exploration", observation=None,
        overlay_action_id=None, overlay_revision=0, **_legacy_options,
    ):
        if self.active:
            return ActionResult(
                action_id, "navigate_action", "already_running", target=query,
                reason_code="NAVIGATION_BUSY", retryable=True,
            )
        if not isinstance(query, str) or not query.strip():
            return ActionResult(action_id, "navigate_action", "failed",
                                reason_code="NAVIGATION_QUERY_REQUIRED")
        normalized_mode = str(mode)
        if normalized_mode not in MODE_GUIDANCE:
            return ActionResult(action_id, "navigate_action", "failed",
                                reason_code="UNKNOWN_NAVIGATION_MODE")
        supplied_observation = (
            dict(observation)
            if isinstance(observation, dict)
            and isinstance(observation.get("jpeg_bytes"), (bytes, bytearray))
            else None
        )
        if normalized_mode != "exploration" and supplied_observation is None:
            return ActionResult(
                action_id, "navigate_action", "failed", target=query,
                reason_code="NAVIGATION_OBSERVATION_REQUIRED",
            )

        self.active = True
        self.action_id, self.target = action_id, query.strip()
        self._mode = normalized_mode
        self._observation = supplied_observation
        self._owns_overlay = False
        self._overlay_action_id = overlay_action_id
        self._overlay_revision = int(overlay_revision or 0)
        self._result_data = {"mode": self._mode}
        self.stage = "selecting"
        self._stop_requested = self._command_sent = False
        self.completion_future = asyncio.get_running_loop().create_future()
        self._worker = asyncio.create_task(self._run(), name=f"navigate-{action_id}")
        return ActionResult(
            action_id, "navigate_action", "running", target=self.target,
            outcome="selecting_one_pose", data={"mode": self._mode},
        )

    async def start_find_target_step(
        self, *, target, action_id, mode, observation, overlay_action_id,
        overlay_revision,
    ):
        """Execute one waypoint owned by the object/place find loop."""
        return await self.start(
            query=SEARCH_WAYPOINT_QUERY.format(target=target, mode=mode),
            action_id=action_id,
            mode=mode,
            observation=observation,
            overlay_action_id=overlay_action_id,
            overlay_revision=overlay_revision,
        )

    async def _run(self):
        try:
            snapshot, camera_jpeg = await self._capture()
            selection = await self._choose_pose(snapshot, camera_jpeg)
            self._result_data.update(selection=selection, snapshot_id=self._snapshot_id)
            if selection["decision"] == "blocked":
                raise NavigationError("NAVIGATION_BLOCKED", selection["reasoning"])

            destination = self._resolve_destination(selection, snapshot)
            self._navigation_future = asyncio.get_running_loop().create_future()
            async with self._command_lock:
                if self._stop_requested:
                    return
                self.stage = "dispatching"
                self._command_sent = True
                feedback = await self.send_robot_command({
                    "command": "navigate_to_pose", "action_id": self.action_id,
                    "frame_id": destination["frame_id"], "x": destination["x"],
                    "y": destination["y"], "angle": destination["yaw"],
                })
            if self._stop_requested:
                return
            if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
                raise NavigationError("NAVIGATION_REJECTED", str(feedback))
            await self._save_pending_map_render(destination)

            self.stage = "navigating"
            try:
                event = await asyncio.wait_for(self._navigation_future, self.NAVIGATION_TIMEOUT)
            except TimeoutError as error:
                raise NavigationError(
                    "NAVIGATION_TIMEOUT", "No terminal Nav2 event received"
                ) from error
            if event.get("status") != "Goal Reached":
                raise NavigationError(
                    "NAVIGATION_FAILED", str(event.get("status") or "unknown")
                )
            self._result_data.update(
                destination=dict(destination), robot_status=event.get("status")
            )
            self._finish("succeeded", "waypoint_reached")
        except asyncio.CancelledError:
            return
        except Exception as error:
            if self._stop_requested:
                return
            error_type = type(error).__name__
            error_message = str(error) or repr(error)
            print(
                "\n[MapNavigationAction error] "
                f"action_id={self.action_id} stage={self.stage} "
                f"{error_type}: {error_message}"
            )
            traceback.print_exc()
            if self._command_sent:
                await self._stop_motion()
            if not self._stop_requested:
                self._result_data.update(
                    error_type=error_type, error=error_message, error_stage=self.stage
                )
                reason_code = (
                    error.code
                    if isinstance(error, NavigationError)
                    else "NAVIGATION_ERROR"
                )
                self._finish(
                    "failed", "navigation_failed", reason_code
                )

    async def _center_camera(self):
        camera_motion = getattr(
            self, "move_camera_action", self.see_action
        )
        centered = await camera_motion.move_to_region(
            region="center", action_id=f"{self.action_id}:center-camera"
        )
        if centered.status != "succeeded":
            raise NavigationError(
                "CAMERA_CENTER_FAILED", centered.reason_code or "Could not center the camera"
            )

    async def _capture(self):
        # Center first so the rendered CURRENT VIEW and the camera image agree.
        if self._mode == "exploration":
            await self._center_camera()

        map_request_id = uuid.uuid4().hex
        request = {
            "schema_version": 1, "operation": "snapshot",
            "action_id": self.action_id, "request_id": map_request_id,
            "overlay_action_id": self._overlay_action_id,
            "overlay_revision": self._overlay_revision,
        }
        if self._mode != "exploration":
            request["crop_size_m"] = self.map_crop_size_m
        try:
            snapshot = await self.request_map_snapshot(request)
        except (asyncio.TimeoutError, TimeoutError, RuntimeError, zmq.ZMQError) as error:
            raise NavigationError("MAP_STREAM_UNAVAILABLE", str(error)) from error

        metadata = snapshot["metadata"]
        if str(metadata.get("snapshot_request_id")) != map_request_id:
            raise NavigationError("MAP_REQUEST_MISMATCH",
                                  "Map service returned the wrong request")
        overlay = metadata.get("search_overlay")
        if self._overlay_action_id is not None and not (
            isinstance(overlay, dict)
            and str(overlay.get("action_id")) == str(self._overlay_action_id)
            and int(overlay.get("revision", 0)) >= self._overlay_revision
        ):
            raise NavigationError(
                "MAP_OVERLAY_TIMEOUT",
                "Map service did not render the requested search overlay revision",
            )

        self._snapshot_id = metadata["snapshot_id"]
        try:
            saved = self.save_map_snapshot(snapshot)
            if saved is not None:
                self._result_data["map_renders"] = saved
        except (OSError, KeyError, TypeError, ValueError) as error:
            self._result_data["debug_save_error"] = str(error)
        candidates = [dict(item) for item in metadata.get("candidates") or []]
        self._candidates = {item["id"]: item for item in candidates}
        if not self._candidates:
            raise NavigationError("NO_NAVIGATION_CANDIDATES",
                                  "No reachable candidates for this mode")
        if self._mode in {"context", "visual"}:
            return snapshot, None

        frame = self.camera.snapshot()
        if time.monotonic() - frame.captured_at > self.CAMERA_MAX_AGE:
            raise NavigationError("STALE_CAMERA", "Current camera frame is stale")
        ok, jpeg = await asyncio.to_thread(
            cv2.imencode, ".jpg", frame.tracking_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY],
        )
        if not ok:
            raise NavigationError("CAMERA_ENCODING_FAILED",
                                  "Could not encode camera image")
        return snapshot, jpeg.tobytes()

    async def _save_pending_map_render(self, destination):
        """Save a diagnostic render with all frontiers and the dispatched pose."""
        request_id = uuid.uuid4().hex
        request = {
            "schema_version": 1,
            "operation": "snapshot",
            "action_id": self.action_id,
            "request_id": request_id,
            "overlay_action_id": self._overlay_action_id,
            "overlay_revision": self._overlay_revision,
            "render_frontiers": True,
            "pending_navigation_pose": destination,
        }
        if self._mode != "exploration":
            request["crop_size_m"] = self.map_crop_size_m
        try:
            snapshot = await self.request_map_snapshot(request)
            metadata = snapshot.get("metadata") or {}
            if str(metadata.get("snapshot_request_id")) != request_id:
                raise RuntimeError("Map service returned the wrong request")
            saved = self.save_map_snapshot(snapshot)
            if saved is not None:
                self._result_data["map_renders"] = saved
        except Exception as error:
            self._result_data["debug_save_error"] = (
                f"{type(error).__name__}: {error}"
            )

    @staticmethod
    def _normalize_yaw(yaw):
        return math.atan2(math.sin(yaw), math.cos(yaw))

    async def _choose_pose(self, snapshot, camera_jpeg):
        if self._mode == "exploration" and len(self._candidates) == 1:
            pose_id = next(iter(self._candidates))
            return {
                "decision": "move",
                "pose_id": pose_id,
                "heading": "map_candidate",
                "reasoning": "Only one exploration pose was offered by map logic",
            }
        return await self._select_pose(snapshot, camera_jpeg)

    def _resolve_destination(self, selection, snapshot):
        destination = dict(self._candidates[selection["pose_id"]])
        if selection["heading"] == "map_candidate":
            destination["sampled_yaw"] = destination["yaw"]
            destination["heading"] = selection["heading"]
            destination["yaw"] = self._normalize_yaw(float(destination["yaw"]))
            return destination

        robot_pose = (snapshot.get("metadata") or {}).get("robot_pose") or {}
        try:
            robot_yaw = float(robot_pose["yaw"])
            offset = SEMANTIC_HEADINGS[selection["heading"]]
        except (KeyError, TypeError, ValueError) as error:
            raise NavigationError(
                "INVALID_NAVIGATION_HEADING",
                "Map snapshot cannot resolve the selected semantic heading",
            ) from error
        destination["sampled_yaw"] = destination.get("yaw")
        destination["heading"] = selection["heading"]
        destination["yaw"] = self._normalize_yaw(robot_yaw + offset)
        return destination

    def _selection_context(self):
        observation = None
        if self._observation is not None:
            observation = {
                key: self._observation.get(key)
                for key in (
                    "contextual_clue", "candidate_type", "movement_limit",
                    "movements_used", "remaining_waypoints",
                )
                if self._observation.get(key) is not None
            }
        return {
            "query": self.target, "mode": self._mode,
            "one_selection": True, "one_movement_maximum": True,
            "candidates": {
                pose_id: {
                    key: candidate.get(key)
                    for key in (
                        "kind", "distance_m", "selection_reason",
                        "ray_extension_m", "uncovered_cell_count",
                    )
                    if candidate.get(key) is not None
                }
                for pose_id, candidate in self._candidates.items()
            },
            "observation": observation,
        }

    def _selection_prompt(self, context):
        return (
            MAP_GUIDANCE + "\n\n" + MODE_GUIDANCE[self._mode]
            + "\n\nLIVE CONTEXT\n" + json.dumps(context, allow_nan=False)
        )

    @staticmethod
    def _selection_content(prompt, map_jpeg, vision_jpeg):
        content = [{"type": "input_text", "text": prompt}]
        for label, jpeg_bytes in (
            ("ROBOT-RELATIVE MAP (Image 1)", map_jpeg),
            ("CAMERA OR GROUNDED CLUE (Image 2)", vision_jpeg),
        ):
            content.extend((
                {"type": "input_text", "text": label},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,"
                 + base64.b64encode(jpeg_bytes).decode("ascii")},
            ))
        return content

    async def _select_pose(self, snapshot, camera_jpeg):
        self._request_id = uuid.uuid4().hex
        self._selection_future = asyncio.get_running_loop().create_future()
        tool = self._build_selection_tool()
        vision_jpeg = (
            camera_jpeg if self._mode == "exploration"
            else bytes(self._observation["jpeg_bytes"])
        )
        context = self._selection_context()
        prompt = self._selection_prompt(context)
        self._save_debug_inputs(
            vision_jpeg, snapshot["metadata"], context, prompt, tool, self._request_id
        )
        content = self._selection_content(prompt, snapshot["jpeg_bytes"], vision_jpeg)
        try:
            await self.ws.send(json.dumps({
                "event_id": f"navigate_vision_{self._request_id}",
                "type": "response.create",
                "response": {
                    "conversation": "none",
                    "metadata": {
                        "kind": "navigation_selection", "action_id": self.action_id,
                        "request_id": self._request_id, "snapshot_id": self._snapshot_id,
                    },
                    "output_modalities": ["text"], "tools": [tool],
                    "tool_choice": "required",
                    "input": [{"type": "message", "role": "user", "content": content}],
                },
            }))
            try:
                return await asyncio.wait_for(self._selection_future, self.VISION_TIMEOUT)
            except TimeoutError as error:
                raise NavigationError("NAVIGATION_VISION_TIMEOUT",
                                      "Pose selection timed out") from error
        finally:
            self._request_id = None
            self._selection_future = None

    def _build_selection_tool(self):
        tool = copy.deepcopy(self.selection_tool_template)
        properties = tool["parameters"]["properties"]
        properties["decision"]["enum"] = ["move", "blocked"]
        properties["pose_id"]["enum"] = [*self._candidates, None]
        return tool

    async def select_navigation_pose(self, args, response_metadata):
        if (
            not self.active or self._stop_requested
            or response_metadata.get("kind") != "navigation_selection"
            or response_metadata.get("action_id") != self.action_id
            or response_metadata.get("request_id") != self._request_id
            or response_metadata.get("snapshot_id") != self._snapshot_id
        ):
            return
        future = self._selection_future
        if future is None or future.done():
            return
        decision, pose_id, heading = (
            args.get("decision"), args.get("pose_id"), args.get("heading")
        )
        if decision not in {"move", "blocked"}:
            future.set_exception(NavigationError(
                "INVALID_NAVIGATION_DECISION", "Selection must be move or blocked"
            ))
            return
        if "pose_id" not in args or (
            pose_id is not None
            and (not isinstance(pose_id, str) or pose_id not in self._candidates)
        ):
            future.set_exception(NavigationError(
                "INVALID_POSE_ID", "Vision returned an unknown pose ID"
            ))
            return
        if (decision == "move") != (pose_id is not None):
            future.set_exception(NavigationError(
                "INVALID_NAVIGATION_DECISION",
                "move requires a pose ID; blocked requires pose_id=null",
            ))
            return
        if decision == "move" and heading not in SEMANTIC_HEADINGS:
            future.set_exception(NavigationError(
                "INVALID_NAVIGATION_HEADING", "move requires one valid semantic heading"
            ))
            return
        if decision == "blocked" and heading is not None:
            future.set_exception(NavigationError(
                "INVALID_NAVIGATION_HEADING", "blocked requires heading=null"
            ))
            return
        reasoning = str(args.get("reasoning") or "").strip()
        if not reasoning:
            future.set_exception(NavigationError(
                "INVALID_NAVIGATION_RATIONALE", "Every selection requires a concrete reason"
            ))
            return
        future.set_result({
            "decision": decision, "pose_id": pose_id,
            "heading": heading, "reasoning": reasoning,
        })

    @staticmethod
    def _atomic_write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)

    def _save_debug_inputs(self, vision_jpeg, map_metadata, context, prompt, tool,
                           request_id):
        manifest = {
            "action_id": self.action_id, "request_id": request_id,
            "snapshot_id": self._snapshot_id, "mode": self._mode,
            "map_metadata": map_metadata, "selection_context": context,
            "prompt": prompt, "tool": tool,
            "image_order": ["map", "vision_or_clue"],
        }
        try:
            self._atomic_write(self.debug_dir / "latest_vision.jpg", vision_jpeg)
            self._atomic_write(
                self.debug_dir / "latest_request.json",
                json.dumps(manifest, indent=2, allow_nan=False).encode("utf-8"),
            )
            self._result_data["debug_dir"] = str(self.debug_dir)
        except (OSError, TypeError, ValueError) as error:
            self._result_data["debug_save_error"] = str(error)

    def handle_navigation_event(self, payload):
        if (
            not self.active or self._stop_requested
            or self.stage not in {"dispatching", "navigating"}
            or payload.get("event") != "navigation"
            or payload.get("action_id") != self.action_id
        ):
            return False
        if self._navigation_future is not None and not self._navigation_future.done():
            self._navigation_future.set_result(dict(payload))
        return True

    async def wait_until_finished(self):
        return await asyncio.shield(self.completion_future)

    async def _stop_motion(self):
        try:
            async with self._command_lock:
                feedback = await self.send_robot_command({
                    "command": "stop_moving", "action_id": self.action_id,
                })
            if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
                raise RuntimeError(str(feedback))
            return True
        except Exception as error:
            self._result_data["stop_error"] = str(error)
            return False

    async def stop(self, reason_code="USER_REQUESTED"):
        async with self._stop_lock:
            if not self.active:
                return None
            self._stop_requested = True
            stopped = await self._stop_motion() if self._command_sent else True
            if self._worker is not None and not self._worker.done():
                self._worker.cancel()
                await asyncio.gather(self._worker, return_exceptions=True)
            return self._finish(
                "cancelled" if stopped else "failed",
                "stopped" if stopped else "stop_failed",
                reason_code if stopped else "NAVIGATION_STOP_FAILED",
            )

    def _finish(self, status, outcome, reason_code=None):
        result = ActionResult(
            self.action_id or "unassigned", "navigate_action", status,
            target=self.target, outcome=outcome, reason_code=reason_code,
            retryable=status == "failed", data=dict(self._result_data),
        )
        self.active = False
        self.stage = "idle"
        self._request_id = None
        if self._owns_overlay:
            asyncio.create_task(self.send_map_overlay({
                "schema_version": 1,
                "operation": "clear",
                "action_id": self.action_id,
                "revision": self._overlay_revision,
            }))
        self._owns_overlay = False
        for future in (self._selection_future, self._navigation_future):
            if future is not None and not future.done():
                future.cancel()
        if self.completion_future is not None and not self.completion_future.done():
            self.completion_future.set_result(result)
        return result
