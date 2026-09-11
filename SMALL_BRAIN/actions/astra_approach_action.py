from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

import numpy as np

from actions.action_result import ActionResult
from actions.track_action import (
    dj_yolo_classes,
    normalize_human_target,
    normalize_object_target,
)


class AstraApproachAction:
    """Astra-only visual tracking and depth-based local approach."""

    def __init__(
        self,
        csrt_tracker,
        yolo,
        grounding_dino,
        camera,
        send_robot_command,
        standoff_m=0.2,
    ):
        self.csrt_tracker = csrt_tracker
        self.yolo = yolo
        self.grounding_dino = grounding_dino
        self.camera = camera
        self.send_robot_command = send_robot_command
        self.standoff_m = max(0.1, float(standoff_m))

        self.tracking_active = False
        self.approaching_active = False
        self.target = None
        self.stable = False
        self._source_generation = None
        self._detection_sequence = None
        self._bbox_xywh = None
        self._tracking_action_id = None
        self._approach_action_id = None
        self._tracking_task = None
        self._tracking_future = None
        self._approach_future = None
        self._last_depth_sample = None
        self._last_destination = None

    @property
    def active(self):
        return self.tracking_active or self.approaching_active

    @property
    def action_id(self):
        if self.approaching_active:
            return self._approach_action_id
        return self._tracking_action_id

    async def start_tracking(
        self,
        target,
        action_id,
        require_search=False,
        allow_grounding_dino=True,
    ):
        del require_search
        normalized_target = (
            normalize_human_target(target) or normalize_object_target(target)
        )
        if self.camera.source != "astra":
            return self._failure(
                action_id,
                "track_action",
                target,
                "ASTRA_SOURCE_REQUIRED",
                "precondition_failed",
            )

        if self.tracking_active:
            if normalized_target == self.target:
                return ActionResult(
                    action_id=action_id,
                    action_type="track_action",
                    status="running",
                    target=normalized_target,
                    outcome="tracking",
                    data={"camera_source": "astra", "tracking_stable": True},
                )
            await self.stop_tracking(
                reason_code="REPLACED",
                status="cancelled",
                outcome="replaced_by_new_goal",
            )

        snapshot = self.camera.snapshot()
        if snapshot.source != "astra":
            return self._failure(
                action_id,
                "track_action",
                target,
                "ASTRA_SOURCE_CHANGED",
                "precondition_failed",
            )
        jpeg_bytes = await asyncio.to_thread(
            self.camera.jpeg_bytes_snapshot,
            70,
            False,
            "results/action_results/astra_track_snapshot.jpg",
        )

        tracking_bbox, detection_source, confidence = await self._detect_bbox(
            normalized_target,
            snapshot,
            jpeg_bytes,
            allow_grounding_dino,
        )
        if tracking_bbox is None:
            return self._failure(
                action_id,
                "track_action",
                normalized_target,
                "OBJECT_DETECTION_FAILED",
                "target_not_detected",
            )

        current = self.camera.snapshot()
        if (
            current.source != "astra"
            or current.source_generation != snapshot.source_generation
        ):
            return self._failure(
                action_id,
                "track_action",
                normalized_target,
                "ASTRA_SOURCE_CHANGED",
                "precondition_failed",
            )

        self.csrt_tracker.begin_tracking(
            detection_sequence=snapshot.sequence,
            initialization_frame=snapshot.tracking_bgr,
            bbox_xywh=tracking_bbox,
            target=normalized_target,
        )
        self.target = normalized_target
        self._tracking_action_id = action_id
        self._source_generation = snapshot.source_generation
        self._detection_sequence = snapshot.sequence
        self._bbox_xywh = tuple(float(value) for value in tracking_bbox)
        self.tracking_active = True
        # Stability here means the box/depth pairing is usable. No centering
        # motion is required for a fixed Astra camera.
        self.stable = True
        self._tracking_future = asyncio.get_running_loop().create_future()
        self._tracking_task = asyncio.create_task(
            self._tracking_loop(),
            name=f"astra-track-{action_id}",
        )
        return ActionResult(
            action_id=action_id,
            action_type="track_action",
            status="running",
            target=normalized_target,
            outcome="tracking",
            data={
                "camera_source": "astra",
                "tracking_stable": True,
                "robot_motion": False,
                "detection_source": detection_source,
                "detection_confidence": confidence,
                "normalized_bbox": self.normalized_bbox(),
            },
        )

    async def _detect_bbox(
        self,
        target,
        snapshot,
        jpeg_bytes,
        allow_grounding_dino,
    ):
        if target in dj_yolo_classes:
            matching = [
                detection
                for detection in self.yolo.detections
                if detection.get("class") == target
            ]
            if matching:
                detection = max(
                    matching, key=lambda item: item.get("confidence", 0.0)
                )
                bbox = detection["bbox"]
                return (
                    (
                        int(bbox["x1"]),
                        int(bbox["y1"]),
                        int(bbox["x2"] - bbox["x1"]),
                        int(bbox["y2"] - bbox["y1"]),
                    ),
                    "yolo",
                    detection.get("confidence"),
                )

        if not allow_grounding_dino:
            return None, None, None

        previous_mode = self.yolo.vision_mode
        self.yolo.vision_mode = "none"
        await asyncio.sleep(0.1)
        try:
            result = await self.grounding_dino.detect(
                image_source=jpeg_bytes,
                target=target,
                box_threshold=0.25,
                text_threshold=0.25,
                nms_threshold=0.80,
                output_root=Path("results/grounding_results"),
            )
        finally:
            self.yolo.vision_mode = previous_mode

        detection = result.best
        if detection is None:
            return None, None, None
        x, y, width, height = detection.tracker_box_xywh
        full_height, full_width = snapshot.full_bgr.shape[:2]
        tracking_height, tracking_width = snapshot.tracking_bgr.shape[:2]
        scale_x = tracking_width / full_width
        scale_y = tracking_height / full_height
        return (
            (
                int(x * scale_x),
                int(y * scale_y),
                int(width * scale_x),
                int(height * scale_y),
            ),
            "grounding_dino",
            detection.score,
        )

    async def _tracking_loop(self):
        while self.tracking_active:
            if (
                self.camera.source != "astra"
                or self.camera.source_generation != self._source_generation
            ):
                await self.stop_tracking(
                    reason_code="CAMERA_SOURCE_CHANGED",
                    status="cancelled",
                    outcome="camera_source_changed",
                )
                return

            update = self.csrt_tracker.tracking_update
            if (
                update is None
                or update.target != self.target
                or update.sequence is None
                or update.sequence <= self._detection_sequence
            ):
                await asyncio.sleep(0.05)
                continue
            if update.success:
                if update.bbox_xywh is not None:
                    self._bbox_xywh = tuple(
                        float(value) for value in update.bbox_xywh
                    )
            else:
                await self.stop_tracking(
                    reason_code="OBJECT_LOST",
                    status="failed",
                    outcome="target_lost",
                )
                return
            await asyncio.sleep(0.05)

    def normalized_bbox(self):
        if not self.tracking_active or self._bbox_xywh is None:
            raise RuntimeError("No Astra target box is being tracked")
        snapshot = self.camera.snapshot()
        if (
            snapshot.source != "astra"
            or snapshot.source_generation != self._source_generation
        ):
            raise RuntimeError("Astra camera source changed during tracking")
        height, width = snapshot.tracking_bgr.shape[:2]
        x, y, box_width, box_height = self._bbox_xywh
        return [
            max(0.0, min(1.0, x / width)),
            max(0.0, min(1.0, y / height)),
            max(0.0, min(1.0, (x + box_width) / width)),
            max(0.0, min(1.0, (y + box_height) / height)),
        ]

    def _sample_depth(
        self,
        normalized_bbox,
        max_age=0.75,
        minimum_depth=0.15,
        maximum_depth=8.0,
    ):
        try:
            x1, y1, x2, y2 = (float(value) for value in normalized_bbox)
        except (TypeError, ValueError):
            raise ValueError("normalized_bbox must contain four numbers")
        if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("normalized_bbox contains a non-finite value")

        x1, x2 = sorted(
            (max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2)))
        )
        y1, y2 = sorted(
            (max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2)))
        )
        if x2 <= x1 or y2 <= y1:
            raise ValueError("normalized_bbox has no area")

        snapshot = self.camera.astra_depth_snapshot()
        age = time.monotonic() - snapshot["captured_at"]
        if age > max_age:
            raise RuntimeError(f"Astra depth frame is stale ({age:.2f}s)")

        metadata = snapshot["metadata"]
        camera_info = snapshot["camera_info"]
        depth_frame = metadata.get("frame_id")
        color_frame = camera_info.get("frame_id")
        if depth_frame and color_frame and depth_frame != color_frame:
            raise RuntimeError(
                "Astra depth is not registered to the color optical frame"
            )

        depth = snapshot["image"]
        height, width = depth.shape
        px1 = max(0, min(width - 1, int(np.floor(x1 * width))))
        py1 = max(0, min(height - 1, int(np.floor(y1 * height))))
        px2 = max(px1 + 1, min(width, int(np.ceil(x2 * width))))
        py2 = max(py1 + 1, min(height, int(np.ceil(y2 * height))))
        region = depth[py1:py2, px1:px2]
        valid = region[
            np.isfinite(region)
            & (region >= float(minimum_depth))
            & (region <= float(maximum_depth))
        ]
        if valid.size == 0:
            raise RuntimeError("No valid Astra depth pixels inside target box")

        depth_m = float(np.median(valid))
        k = camera_info["k"]
        calibration_width = float(camera_info.get("width") or width)
        fx = float(k[0]) * width / calibration_width
        cx = float(k[2]) * width / calibration_width
        if not math.isfinite(fx) or fx <= 0.0:
            raise RuntimeError("Astra color focal length is invalid")
        center_u = ((x1 + x2) * 0.5) * width

        # Optical +Z is robot-forward; optical +X is robot-right, while
        # base_footprint +Y is robot-left.
        forward_m = depth_m
        left_m = -((center_u - cx) / fx) * depth_m
        planar_range_m = math.hypot(forward_m, left_m)
        bearing_rad = math.atan2(left_m, forward_m)
        return {
            "depth_m": depth_m,
            "forward_m": forward_m,
            "left_m": left_m,
            "planar_range_m": planar_range_m,
            "bearing_rad": bearing_rad,
            "valid_pixel_count": int(valid.size),
            "depth_frame_age_s": age,
            "pixel_bbox": [px1, py1, px2, py2],
            "frame_id": depth_frame,
        }

    async def start_approaching(self, target, action_id):
        normalized_target = (
            normalize_human_target(target) or normalize_object_target(target)
        )
        if self.camera.source != "astra":
            return self._failure(
                action_id,
                "approach_action",
                target,
                "ASTRA_SOURCE_REQUIRED",
                "precondition_failed",
            )
        if not self.tracking_active or self.target != normalized_target:
            return self._failure(
                action_id,
                "approach_action",
                target,
                "TARGET_NOT_TRACKED",
                "precondition_failed",
            )
        if self.approaching_active:
            return ActionResult(
                action_id=action_id,
                action_type="approach_action",
                status="already_running",
                target=normalized_target,
                outcome="already_running",
                reason_code="APPROACH_BUSY",
                retryable=True,
            )

        try:
            sample = self._sample_depth(self.normalized_bbox())
        except (RuntimeError, ValueError) as error:
            return self._failure(
                action_id,
                "approach_action",
                normalized_target,
                "ASTRA_DEPTH_INVALID",
                "range_invalid",
                data={"message": str(error)},
            )

        planar_range = sample["planar_range_m"]
        travel = max(0.0, planar_range - self.standoff_m)
        self._last_depth_sample = dict(sample)
        if travel <= 0.05:
            await self._finish_tracking_acquisition()
            return ActionResult(
                action_id=action_id,
                action_type="approach_action",
                status="succeeded",
                target=normalized_target,
                outcome="already_within_standoff",
                data={
                    "camera_source": "astra",
                    "depth_sample": sample,
                    "standoff_m": self.standoff_m,
                    "robot_motion": False,
                },
            )

        scale = travel / planar_range
        destination = {
            "x": sample["forward_m"] * scale,
            "y": sample["left_m"] * scale,
            "angle": sample["bearing_rad"],
        }
        self._approach_action_id = action_id
        self.approaching_active = True
        self._approach_future = asyncio.get_running_loop().create_future()
        self._last_destination = dict(destination)
        await self._finish_tracking_acquisition()
        feedback = await self.send_robot_command(
            {
                "command": "navigate_to_pose",
                "action_id": action_id,
                "frame_id": "base_footprint",
                **destination,
            }
        )
        if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
            message = (
                feedback.get("message", "navigation_rejected")
                if isinstance(feedback, dict)
                else "invalid_navigation_feedback"
            )
            return self._complete_approach(
                status="failed",
                outcome="rejected",
                reason_code="NAVIGATION_REJECTED",
                data={"message": str(message)},
            )

        return ActionResult(
            action_id=action_id,
            action_type="approach_action",
            status="running",
            target=normalized_target,
            outcome="approaching",
            data={
                "camera_source": "astra",
                "depth_sample": sample,
                "destination": destination,
                "standoff_m": self.standoff_m,
            },
        )

    def handle_navigation_event(self, payload):
        if not self.approaching_active or payload.get("event") != "navigation":
            return False
        if payload.get("action_id") not in {None, self._approach_action_id}:
            return False
        navigation_status = str(payload.get("status", ""))
        if navigation_status == "Goal Reached":
            self._complete_approach(
                status="succeeded",
                outcome="reached",
                data={
                    "robot_status": navigation_status,
                    "camera_source": "astra",
                    "depth_sample": dict(self._last_depth_sample or {}),
                    "destination": dict(self._last_destination or {}),
                    "standoff_m": self.standoff_m,
                },
            )
        else:
            self._complete_approach(
                status="failed",
                outcome="navigation_failed",
                reason_code="NAVIGATION_FAILED",
                data={"robot_status": navigation_status or "unknown"},
            )
        return True

    async def wait_until_finished(self):
        if self.approaching_active and self._approach_future is not None:
            return await self._approach_future
        if self._tracking_future is not None:
            return await self._tracking_future
        raise RuntimeError("No Astra action is active")

    async def wait_until_tracking_finished(self):
        if self._tracking_future is None:
            raise RuntimeError("No Astra tracking action is active")
        return await self._tracking_future

    async def stop_approaching(self, reason_code="USER_REQUESTED"):
        if not self.approaching_active:
            return None
        await self.send_robot_command(
            {"command": "stop_moving", "action_id": self._approach_action_id}
        )
        return self._complete_approach(
            status="cancelled",
            outcome="stopped",
            reason_code=reason_code,
        )

    async def _finish_tracking_acquisition(self):
        """Release the tracker after its box has produced a fixed Nav2 goal."""
        if not self.tracking_active:
            return None
        task = self._tracking_task
        result = self._complete_tracking("succeeded", "target_acquired", None)
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return result

    async def stop_tracking(
        self,
        reason_code="USER_REQUESTED",
        status="cancelled",
        outcome="tracking_stopped",
    ):
        if not self.tracking_active:
            return None
        task = self._tracking_task
        result = self._complete_tracking(status, outcome, reason_code)
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return result

    async def source_changed(self, previous_source, new_source):
        if previous_source != "astra" or new_source == "astra":
            return
        if self.tracking_active:
            await self.stop_tracking(
                reason_code="CAMERA_SOURCE_CHANGED",
                status="cancelled",
                outcome="camera_source_changed",
            )

    def _complete_tracking(self, status, outcome, reason_code):
        result = ActionResult(
            action_id=self._tracking_action_id or "unassigned",
            action_type="track_action",
            status=status,
            target=self.target,
            outcome=outcome,
            reason_code=reason_code,
            retryable=status == "failed",
            data={"camera_source": "astra", "tracking_was_stable": self.stable},
        )
        future = self._tracking_future
        self.csrt_tracker.stop_tracking()
        self.tracking_active = False
        self.stable = False
        if not self.approaching_active:
            self.target = None
        self._source_generation = None
        self._detection_sequence = None
        self._bbox_xywh = None
        self._tracking_action_id = None
        self._tracking_task = None
        self._tracking_future = None
        if future is not None and not future.done():
            future.set_result(result)
        return result

    def _complete_approach(
        self, status, outcome, reason_code=None, data=None
    ):
        result = ActionResult(
            action_id=self._approach_action_id or "unassigned",
            action_type="approach_action",
            status=status,
            target=self.target,
            outcome=outcome,
            reason_code=reason_code,
            retryable=status == "failed",
            data=data or {},
        )
        future = self._approach_future
        self.approaching_active = False
        self._approach_action_id = None
        self._approach_future = None
        if not self.tracking_active:
            self.target = None
        if future is not None and not future.done():
            future.set_result(result)
        return result

    @staticmethod
    def _failure(
        action_id,
        action_type,
        target,
        reason_code,
        outcome,
        data=None,
    ):
        return ActionResult(
            action_id=action_id,
            action_type=action_type,
            status="failed",
            target=target,
            outcome=outcome,
            reason_code=reason_code,
            retryable=True,
            data=data or {},
        )
