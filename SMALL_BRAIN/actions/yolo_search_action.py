"""Fast closed-set camera search using the shared YOLO object model."""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any

import cv2

from actions.action_result import ActionResult
from actions.move_camera_action import (
    MoveCameraAction,
    PAN_POSITION_ANGLE,
    TILT_POSITION_ANGLE,
)
from actions.tracking.target_catalog import (
    YOLO_OBJECT_CLASSES,
    normalize_object_target,
)
from cognition.manager.world_state import robot_state


CAMERA_HORIZONTAL_FOV_DEG = 85.0
CAMERA_VERTICAL_FOV_DEG = 52.0
YOLO_INFERENCE_TIMEOUT_SECONDS = 1.0
VIEW_OBSERVATION_SECONDS = 1.0

SEARCH_EFFORT_ROWS = {
    "center": ("center",),
    "high": ("upmost",),
    "low": ("downmost",),
    "best_effort": ("center", "upmost", "downmost"),
}


class YoloSearchAction:
    """Search configured camera views for one supported YOLO object class."""

    def __init__(
        self,
        ws,
        send_robot_command,
        camera,
        move_camera_action=None,
        yolo=None,
    ):
        # ``ws`` is accepted so this action can replace SearchAction without
        # changing construction call sites. YOLO search has no model requests.
        self.ws = ws
        self.send_robot_command = send_robot_command
        self.camera = camera
        self.yolo = yolo
        self.move_camera_action = move_camera_action or MoveCameraAction(
            camera=camera,
            send_robot_command=send_robot_command,
        )

        self.active = False
        self.search_task: asyncio.Task | None = None
        self.completion_future: asyncio.Future[ActionResult] | None = None
        self.action_id: str | None = None
        self.search_id: str | None = None
        self.target: str | None = None
        self.effort = "center"
        self.initial_view_only = False
        self.search_mode = "specific"
        self.person_search = False
        self.candidate_context: dict[str, Any] | None = None
        self.pan_angle = PAN_POSITION_ANGLE["center"]
        self.tilt_angle = TILT_POSITION_ANGLE["center"]

        self.last_found_target: str | None = None
        self.last_observation_frame: dict[str, Any] | None = None
        self.last_coverage_frames: list[dict[str, Any]] = []
        self.last_contextual_clue = None
        self.last_detection_source: str | None = None
        self.last_person_detection = None
        self.current_candidate: dict[str, Any] | None = None

        self._sweep_directions: tuple[str, ...] | None = None
        self._observations: list[dict[str, Any]] = []
        self._configured_view_count = 1

    async def start_searching(
        self,
        target,
        action_id,
        effort="center",
        initial_view_only=False,
        sweep_directions=None,
        candidate_context=None,
        search_mode="specific",
        person_search=False,
    ):
        """Start a YOLO-only search with SearchAction-compatible arguments."""
        if self.active:
            return ActionResult(
                action_id=action_id,
                action_type="yolo_search_action",
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

        raw_target = str(target or "").strip()
        if not raw_target:
            return self._invalid_result(
                action_id,
                None,
                "SEARCH_TARGET_REQUIRED",
            )

        normalized_target = normalize_object_target(raw_target)
        if normalized_target not in YOLO_OBJECT_CLASSES:
            return self._invalid_result(
                action_id,
                normalized_target,
                "YOLO_TARGET_UNSUPPORTED",
                data={"supported_targets": sorted(YOLO_OBJECT_CLASSES)},
            )

        normalized_effort = (
            str(effort or "center").strip().lower().replace("-", "_")
        )
        if normalized_effort not in SEARCH_EFFORT_ROWS:
            return self._invalid_result(
                action_id,
                normalized_target,
                "UNKNOWN_SEARCH_EFFORT",
            )

        normalized_mode = str(search_mode or "specific").strip().lower()
        if normalized_mode != "specific" or person_search:
            return self._invalid_result(
                action_id,
                normalized_target,
                "YOLO_OBJECT_SEARCH_ONLY",
            )

        if self.yolo is None:
            return self._invalid_result(
                action_id,
                normalized_target,
                "YOLO_SERVICE_UNAVAILABLE",
            )

        self.active = True
        self.action_id = action_id
        self.search_id = action_id
        self.target = normalized_target
        self.effort = normalized_effort
        self.initial_view_only = initial_view_only is True
        self.search_mode = normalized_mode
        self.person_search = False
        self.candidate_context = (
            dict(candidate_context)
            if isinstance(candidate_context, dict) else None
        )
        self._sweep_directions = self._normalize_sweep_directions(
            sweep_directions
        )
        self._observations = []
        self._configured_view_count = len(self._search_views())
        self.last_found_target = None
        self.last_observation_frame = None
        self.last_coverage_frames = []
        self.last_contextual_clue = None
        self.last_detection_source = None
        self.last_person_detection = None
        self.current_candidate = None

        camera_state = robot_state.get("camera") or {}
        self.pan_angle = float(camera_state.get(
            "pan_angle", PAN_POSITION_ANGLE["center"]
        ))
        self.tilt_angle = float(camera_state.get(
            "tilt_angle", TILT_POSITION_ANGLE["center"]
        ))

        self.completion_future = asyncio.get_running_loop().create_future()
        self.search_task = asyncio.create_task(
            self._run_search(),
            name=f"yolo-search-{action_id}",
        )
        return ActionResult(
            action_id=action_id,
            action_type="yolo_search_action",
            status="running",
            target=normalized_target,
            outcome="searching",
            data={
                "effort": self.effort,
                "initial_view_only": self.initial_view_only,
                "search_mode": self.search_mode,
                "detection_source": "yoloe-11m",
            },
        )

    async def wait_until_finished(self):
        if self.completion_future is None:
            return None
        return await asyncio.shield(self.completion_future)

    async def stop(self, reason_code="USER_REQUESTED"):
        return await self.stop_searching(reason_code)

    async def stop_searching(self, reason=None):
        if not self.active:
            return None

        search_task = self.search_task
        if (
            search_task is not None
            and search_task is not asyncio.current_task()
            and not search_task.done()
        ):
            search_task.cancel()
            await asyncio.gather(search_task, return_exceptions=True)

        try:
            await self._move_camera_angles(
                PAN_POSITION_ANGLE["center"],
                TILT_POSITION_ANGLE["center"],
            )
        except Exception:
            pass

        return await self.complete_searching(
            status="cancelled",
            outcome="search_cancelled",
            reason_code=reason,
        )

    async def complete_searching(self, status, outcome, reason_code=None):
        if not self.active:
            return None

        action_id = self.action_id or "unassigned"
        target = self.target
        observation_count = len(self._observations)
        self.last_coverage_frames = list(self._observations)
        data = {
            "searched_frames": observation_count,
            "configured_views": self._configured_view_count,
            "search_coverage": min(
                1.0,
                observation_count / max(1, self._configured_view_count),
            ),
            "final_camera_pose": {
                "pan_angle": self.pan_angle,
                "tilt_angle": self.tilt_angle,
            },
            "effort": self.effort,
            "initial_view_only": self.initial_view_only,
            "search_mode": self.search_mode,
            "detection_source": self.last_detection_source or "yoloe-11m",
            "detection": copy.deepcopy(
                (self.last_observation_frame or {}).get("detection")
            ),
            "person_detection": None,
            "contextual_clue": None,
        }
        result = ActionResult(
            action_id=action_id,
            action_type="yolo_search_action",
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
        self.active = False
        self.search_task = None
        self.action_id = None
        self.search_id = None
        self.target = None
        self.candidate_context = None

        if completion_future is not None and not completion_future.done():
            completion_future.set_result(result)
        return result

    async def _run_search(self):
        found = None
        try:
            async with self.yolo.object_detection_mode():
                for index, view in enumerate(self._search_views(), start=1):
                    if not self.active:
                        return
                    if view is not None:
                        await self._move_camera_angles(
                            PAN_POSITION_ANGLE[view[0]],
                            TILT_POSITION_ANGLE[view[1]],
                        )
                    observation, detection = await self._inspect_current_view(
                        index=index,
                        pan_position=("current" if view is None else view[0]),
                        tilt_position=("current" if view is None else view[1]),
                    )
                    self._observations.append(observation)
                    if detection is not None:
                        found = observation, detection
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            if self.active:
                await self.complete_searching(
                    status="failed",
                    outcome="search_error",
                    reason_code="YOLO_SEARCH_FAILED",
                )
            return

        if not self.active:
            return
        if found is None:
            self.last_coverage_frames = list(self._observations)
            await self.complete_searching(
                status="failed",
                outcome="not_found",
                reason_code="TARGET_NOT_VISIBLE",
            )
            return

        observation, detection = found
        self.last_detection_source = "yoloe-11m"
        self.last_found_target = self.target
        self.last_observation_frame = observation
        self.last_coverage_frames = list(self._observations)
        bbox = detection["bbox"]
        self.current_candidate = {
            "assessment": {
                "result": "found",
                "target_position": {
                    "x": float(bbox["normalized_center_x"]),
                    "y": float(bbox["normalized_center_y"]),
                },
            },
            "frame": observation,
        }

        target_pan, target_tilt = self._target_servo_angles(detection)
        try:
            await self._move_camera_angles(target_pan, target_tilt)
        except Exception:
            await self.complete_searching(
                status="failed",
                outcome="camera_aim_failed",
                reason_code="CAMERA_MOVE_FAILED",
            )
            return

        await self.complete_searching(
            status="succeeded",
            outcome="found",
        )

    async def _inspect_current_view(self, *, index, pan_position, tilt_position):
        sequence, _ = self.yolo.detection_snapshot()
        deadline = time.monotonic() + VIEW_OBSERVATION_SECONDS
        latest_snapshot = None

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            fresh = await self.yolo.wait_for_inference_after(
                sequence,
                timeout=min(YOLO_INFERENCE_TIMEOUT_SECONDS, remaining),
            )
            if not fresh:
                break

            current_sequence, detections, frame, timing = (
                self.yolo.detection_snapshot_with_tracking_frame()
            )
            if current_sequence <= sequence or frame is None:
                raise RuntimeError("YOLO returned no fresh tracking frame")
            sequence = current_sequence

            matches = [
                detection for detection in detections
                if detection.get("class") == self.target
                and detection.get("bbox")
            ]
            detection = max(
                matches,
                key=lambda item: float(item.get("confidence") or 0.0),
                default=None,
            )
            latest_snapshot = (
                current_sequence,
                frame,
                timing,
                detection,
            )
            if detection is not None:
                break

        if latest_snapshot is None:
            raise TimeoutError("Timed out waiting for YOLO object inference")

        current_sequence, frame, timing, detection = latest_snapshot
        observation = {
            "image_id": f"yolo_{current_sequence}",
            "pan_position": pan_position,
            "tilt_position": tilt_position,
            "pan_angle": float(self.pan_angle),
            "tilt_angle": float(self.tilt_angle),
            "jpeg_bytes": self._encode_jpeg(frame),
            "robot_pose": self._robot_pose_at_capture(),
            "camera_sequence": timing.get("camera_sequence"),
            "sweep_index": index,
            "detection": copy.deepcopy(detection),
        }
        if detection is not None:
            observation.update({
                "candidate_type": "found",
                "contextual_clue": f"YOLO detected {self.target}",
            })
        return observation, detection

    async def _move_camera_angles(self, target_pan, target_tilt):
        result = await self.move_camera_action.move_to_angles(
            target_pan,
            target_tilt,
            action_id=f"{self.action_id or 'yolo-search'}:move-camera",
            current_pan=self.pan_angle,
            current_tilt=self.tilt_angle,
        )
        if result.status != "succeeded":
            raise RuntimeError(result.reason_code or "Camera movement failed")
        self.pan_angle = float(target_pan)
        self.tilt_angle = float(target_tilt)

    def _target_servo_angles(self, detection):
        bbox = detection["bbox"]
        center_x = float(bbox["normalized_center_x"])
        center_y = float(bbox["normalized_center_y"])
        pan_correction = -(center_x - 0.5) * CAMERA_HORIZONTAL_FOV_DEG
        tilt_correction = (center_y - 0.5) * CAMERA_VERTICAL_FOV_DEG
        return (
            self.pan_angle + pan_correction,
            self.tilt_angle + tilt_correction,
        )

    def _search_views(self):
        views: list[tuple[str, str] | None] = [None]
        if self.initial_view_only:
            return views

        center_pans = (
            self._sweep_directions
            if self._sweep_directions is not None
            else ("leftmost", "rightmost")
        )
        for tilt_position in SEARCH_EFFORT_ROWS[self.effort]:
            if tilt_position == "center":
                pan_positions = center_pans
            elif tilt_position == "upmost":
                pan_positions = ("rightmost", "center", "leftmost")
            else:
                pan_positions = ("leftmost", "center", "rightmost")
            views.extend(
                (pan_position, tilt_position)
                for pan_position in pan_positions
            )
        return views

    @staticmethod
    def _normalize_sweep_directions(sweep_directions):
        if sweep_directions is None:
            return None
        direction_to_pan = {
            "left": "leftmost",
            "right": "rightmost",
        }
        return tuple(
            direction_to_pan[direction]
            for direction in sweep_directions
            if direction in direction_to_pan
        )

    @staticmethod
    def _robot_pose_at_capture():
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

    @staticmethod
    def _encode_jpeg(frame):
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70],
        )
        if not ok:
            raise RuntimeError("Failed to encode YOLO search frame")
        return encoded.tobytes()

    @staticmethod
    def _invalid_result(action_id, target, reason_code, data=None):
        return ActionResult(
            action_id=action_id,
            action_type="yolo_search_action",
            status="failed",
            target=target,
            outcome="invalid_request",
            reason_code=reason_code,
            data=data or {},
        )
