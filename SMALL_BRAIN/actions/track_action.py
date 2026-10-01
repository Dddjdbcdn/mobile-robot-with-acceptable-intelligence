import asyncio
import math
from pathlib import Path
import time

from actions.action_result import ActionResult
from actions.tracking.person_pose import (
    find_best_person_path,
    person_keypoint,
    predict_next_keypoint,
)
from actions.tracking.stable_seed import StableTargetSeedTracker
from actions.tracking.target_catalog import (
    HUMAN_TRACKABLE_PARTS,
    normalize_human_target,
    normalize_object_target,
)
from cognition.manager.world_state import robot_state
from utilities.camera_sampler import pose_body_mask, project_tof_region

HORIZONTAL_FOV_DEG = 85
VERTICAL_FOV_DEG = 52


class TrackAction:
    OBJECT_FAILURE_GRACE_FRAMES = 3
    OBJECT_REACQUIRE_ATTEMPTS = 3
    OBJECT_REACQUIRE_DELAY_SECONDS = 0.25
    TARGET_SEED_SAMPLES = StableTargetSeedTracker.SAMPLE_COUNT
    PERSON_POINT_MAX_AGE_SECONDS = StableTargetSeedTracker.INPUT_MAX_AGE_SECONDS
    PERSON_SEED_UPDATE_HZ = 5.0

    def __init__(
        self,
        csrt_tracker,
        yolo,
        grounding_dino,
        camera,
        zmq_pub_socket,
        send_robot_command,
        STABLE_THRESHOLD=0.05,
        person_tracker_pub_socket=None,
    ):
        self.csrt_tracker = csrt_tracker
        self.grounding_dino = grounding_dino
        self.yolo = yolo
        self.zmq_pub_socket = zmq_pub_socket
        self.person_tracker_pub_socket = person_tracker_pub_socket
        self.send_robot_command = send_robot_command
        self.camera = camera

        self.active = False
        self.target = None
        self.stable_threshold = STABLE_THRESHOLD
        self._tracking_task = None
        self.completion_future = None
        self.action_id = None
        self.detection_confidence = 1.0
        self._memory_recorded_for_session = False
        self._allow_grounding_dino = True
        self._continuous_person_reacquisition = False
        self._visual_target_override = None
        self.stable_seeds = StableTargetSeedTracker()
        # Persist across tracker loss so approach can reacquire after moving.
        self.last_stable_target_location = None

        self.person_path = []
        self.person_path_index = 0

    def set_visual_target_override(self, normalized_x, normalized_y):
        """Temporarily aim person tracking at an externally observed point."""
        override_started = self._visual_target_override is None
        self._visual_target_override = {
            "x": max(0.0, min(1.0, float(normalized_x))),
            "y": max(0.0, min(1.0, float(normalized_y))),
            "updated_at": time.monotonic(),
        }
        if override_started:
            self.stable_seeds.clear()

        center_tolerance = getattr(self, "stable_threshold", 0.05)
        if (
            abs(self._visual_target_override["x"] - 0.5)
            < center_tolerance
            and abs(self._visual_target_override["y"] - 0.5)
            < center_tolerance
        ):
            self.stable_seeds.update(
                robot_state.get("camera"),
                target="hand",
                session_id=self.action_id,
                person=robot_state.get("person"),
                require_person_proximity=True,
            )
        else:
            self.stable_seeds.clear()

    def clear_visual_target_override(self):
        if self._visual_target_override is None:
            return
        self._visual_target_override = None
        self.stable_seeds.clear()

    def _current_visual_target_override(self, max_age_seconds=0.5):
        override = self._visual_target_override
        if override is None:
            return None
        if time.monotonic() - override["updated_at"] > max_age_seconds:
            self.stable_seeds.clear()
            self._visual_target_override = None
            return None
        return override["x"], override["y"]

    async def wait_for_stable_target_seed(
        self,
        *,
        target=None,
        session_id=None,
        check_interval=0.05,
        timeout=10.0,
    ):
        """Wait only for the target and tracking session captured at entry."""
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        expected_target = self.target if target is None else target
        expected_session_id = (
            self.action_id if session_id is None else session_id
        )
        while (
            self.active
            and self.target == expected_target
            and self.action_id == expected_session_id
        ):
            seed = self.stable_seeds.get(
                target=expected_target,
                session_id=expected_session_id,
            )
            if seed is not None:
                return seed
            if timeout is not None and loop.time() - started_at >= timeout:
                return None
            await asyncio.sleep(check_interval)
        return None

    async def start_tracking(
        self,
        target,
        action_id,
        allow_grounding_dino=True,
        continuous_person_reacquisition=False,
    ):
        if self.active:
            await self.stop_tracking(
                reason_code="REPLACED",
                status="cancelled",
                outcome="replaced_by_new_goal",
            )

        normalized_target = (
            normalize_human_target(target) or normalize_object_target(target)
        )

        self.action_id = action_id
        self.target = normalized_target
        self.stable_seeds.clear()
        self.detection_confidence = 1.0
        self._memory_recorded_for_session = False
        self._allow_grounding_dino = allow_grounding_dino
        self._continuous_person_reacquisition = bool(
            continuous_person_reacquisition
        )
        self.last_stable_target_location = None

        if normalized_target in HUMAN_TRACKABLE_PARTS:
            return await self._start_person_tracking(normalized_target)

        jpeg_bytes = None
        snapshot = None
        if allow_grounding_dino:
            jpeg_bytes = await asyncio.to_thread(
                self.camera.jpeg_bytes_snapshot,
                70,
                False,
                "results/action_results/track_snapshot.jpg",
            )
            snapshot = self.camera.snapshot()

        return await self._start_object_tracking(
            jpeg_bytes,
            normalized_target,
            snapshot,
            allow_grounding_dino=allow_grounding_dino,
        )

    async def _start_object_tracking(
        self,
        jpeg_bytes,
        target,
        snapshot,
        allow_grounding_dino=True,
    ):
        detection = await self._detect_and_start_object_tracker(
            target,
            jpeg_bytes=jpeg_bytes,
            snapshot=snapshot,
            allow_grounding_dino=allow_grounding_dino,
        )

        if detection is None:
            action_id = self.action_id
            self.action_id = None
            self.target = None
            return ActionResult(
                action_id=action_id,
                action_type="track_action",
                status="failed",
                target=target,
                outcome="target_not_detected",
                reason_code="OBJECT_DETECTION_FAILED",
                retryable=True,
            )

        detection_source, detection_confidence = detection

        await self.send_robot_command({
            "command": "track_action",
            "action_id": self.action_id,
        })

        self.active = True
        self.completion_future = asyncio.get_running_loop().create_future()
        self._tracking_task = asyncio.create_task(self._object_tracking_loop())

        return ActionResult(
            action_id=self.action_id,
            action_type="track_action",
            status="running",
            target=target,
            outcome="tracking",
            data={
                "detection_source": detection_source,
                "detection_confidence": detection_confidence,
            },
        )

    async def _detect_and_start_object_tracker(
        self,
        target,
        *,
        jpeg_bytes=None,
        snapshot=None,
        allow_grounding_dino=True,
    ):
        """Detect an object in the current view and initialize CSRT for it."""
        tracking_bbox = None
        detection_source = None
        detection_confidence = None

        if allow_grounding_dino:
            if jpeg_bytes is None:
                jpeg_bytes = await asyncio.to_thread(
                    self.camera.jpeg_bytes_snapshot, 70, False
                )
            grounding_result = await self.grounding_dino.detect(
                image_source=jpeg_bytes,
                target=target,
                box_threshold=0.25,
                text_threshold=0.25,
                nms_threshold=0.80,
                output_root=Path("results/grounding_results"),
            )

            detection = grounding_result.best
            if detection is not None:
                orig_x, orig_y, orig_w, orig_h = detection.tracker_box_xywh
                tracking_bbox = (
                    int(orig_x / 2.0),
                    int(orig_y / 2.0),
                    int(orig_w / 2.0),
                    int(orig_h / 2.0),
                )
                detection_source = "grounding_dino"
                detection_confidence = detection.score

        if tracking_bbox is None:
            return None

        self.detection_confidence = float(detection_confidence or 1.0)
        if snapshot is None:
            snapshot = self.camera.snapshot()

        self.csrt_tracker.begin_tracking(
            detection_sequence=snapshot.sequence,
            initialization_frame=snapshot.tracking_bgr,
            bbox_xywh=tracking_bbox,
            target=target,
        )

        return detection_source, detection_confidence

    async def _object_tracking_loop(self):
        failure_frames = 0
        last_sequence = None

        while self.active:
            tracking_update = self.csrt_tracker.tracking_update
            if tracking_update is None:
                await asyncio.sleep(0.05)
                continue
            if tracking_update.sequence == last_sequence:
                await asyncio.sleep(0.01)
                continue
            last_sequence = tracking_update.sequence

            if tracking_update.success:
                failure_frames = 0
                target_x = tracking_update.normalized_x
                target_y = tracking_update.normalized_y
                delta_pan_angle, delta_tilt_angle = self.target_to_angles(
                    target_x, target_y
                )

                centered = (
                    abs(target_x - 0.5) < self.stable_threshold
                    and abs(target_y - 0.5) < self.stable_threshold
                )
                if not centered:
                    await self.zmq_pub_socket.send_json({
                        "delta_pan_angle": delta_pan_angle,
                        "delta_tilt_angle": delta_tilt_angle,
                        "tracking_sequence": f"csrt:{tracking_update.sequence}",
                    })

                if centered:
                    seed = self.stable_seeds.update(
                        robot_state.get("camera"),
                        target=self.target,
                        session_id=self.action_id,
                    )
                    if seed is not None:
                        self._remember_stable_target()
                else:
                    self.stable_seeds.clear()
            else:
                self.stable_seeds.clear()
                failure_frames += 1
                if failure_frames >= self.OBJECT_FAILURE_GRACE_FRAMES:
                    # Keep the camera at its last tracked angle. Re-detect there and
                    # initialize a fresh CSRT instance instead of recentering/searching.
                    reacquired = False
                    for _attempt in range(self.OBJECT_REACQUIRE_ATTEMPTS):
                        detection = await self._detect_and_start_object_tracker(
                            self.target,
                            allow_grounding_dino=self._allow_grounding_dino,
                        )
                        await asyncio.sleep(self.OBJECT_REACQUIRE_DELAY_SECONDS)
                        if detection is not None:
                            reacquired = True
                            break

                    if reacquired:
                        failure_frames = 0
                    else:
                        await self.stop_tracking(
                            reason_code="OBJECT_LOST",
                            status="failed",
                            outcome="target_lost_after_reacquisition",
                        )
                        return

            await asyncio.sleep(0.05)

    def _remember_stable_target(self):
        if self._memory_recorded_for_session:
            return
        state_snapshot = {
            "pose": dict(robot_state.get("pose") or {}),
            "camera": dict(robot_state.get("camera") or {}),
        }

        camera = state_snapshot["camera"]
        pose = state_snapshot["pose"]
        required_location = (
            camera.get("object_map_x"),
            camera.get("object_map_y"),
            camera.get("camera_tof_range"),
        )
        if (
            not all(isinstance(value, (int, float)) for value in required_location)
            or not all(math.isfinite(float(value)) for value in required_location)
            or float(camera["camera_tof_range"]) <= 0.05
        ):
            return

        self.last_stable_target_location = {
            "target": self.target,
            "map_x": camera.get("object_map_x"),
            "map_y": camera.get("object_map_y"),
            "range_m": camera.get("camera_tof_range"),
            "camera_pan_angle": camera.get("pan_angle"),
            "camera_tilt_angle": camera.get("tilt_angle"),
            "observer_pose": pose,
            "confidence": self.detection_confidence,
            "captured_at": camera.get("timestamp"),
        }
        self._memory_recorded_for_session = True

    async def _start_person_tracking(self, target):
        detections = [
            detection
            for detection in self.yolo.detections
            if detection["class"] == "person"
            and "keypoints" in detection
        ]

        if not detections:
            action_id = self.action_id or "unassigned"
            self.action_id = None
            self.target = None
            return ActionResult(
                action_id=action_id,
                action_type="track_action",
                status="failed",
                target=target,
                outcome="target_not_detected",
                reason_code="PERSON_DETECTION_FAILED",
                retryable=True,
            )

        person = max(detections, key=lambda detection: detection["confidence"])
        self.detection_confidence = float(person.get("confidence") or 1.0)
        self.person_path = find_best_person_path(person, target)

        self.person_path_index = 0
        feedback = await self.send_robot_command({
            "command": "track_action",
            "action_id": self.action_id,
        })

        self.active = True
        self.completion_future = asyncio.get_running_loop().create_future()
        self._tracking_task = asyncio.create_task(self._person_tracking_loop())

        return ActionResult(
            action_id=self.action_id,
            action_type="track_action",
            status="running",
            target=target,
            outcome="tracking",
            data={
                "detection_source": "yolo_pose",
                "detection_confidence": person.get("confidence"),
                "keypoint_path": list(self.person_path),
            },
        )

    async def _publish_lidar_seed(self, seed):
        """Send the exact validated visual seed to the lidar person tracker."""
        if self.person_tracker_pub_socket is None or seed is None:
            return
        await self.person_tracker_pub_socket.send_json({
            "type": "person_tof_position",
            "x": seed["x"],
            "y": seed["y"],
            "frame_id": "base_footprint",
            "target": seed.get("target"),
            "session_id": seed.get("session_id"),
        })

    def _update_person_seed(self, person):
        """Validate a centered person's ToF point against the pose mask."""
        camera_state = robot_state.get("camera") or {}
        timestamp = camera_state.get("timestamp")
        distance = camera_state.get("camera_tof_range")
        if (
            not isinstance(timestamp, (int, float))
            or not isinstance(distance, (int, float))
            or not math.isfinite(float(timestamp))
            or not math.isfinite(float(distance))
            or float(distance) <= 0.05
            or time.monotonic() - float(timestamp)
            > self.PERSON_POINT_MAX_AGE_SECONDS
        ):
            self.stable_seeds.clear()
            return None

        try:
            frame_height, frame_width = self.camera.snapshot().tracking_bgr.shape[:2]
            bbox = person["bbox"]
            projected = {
                name: (int(point["x"]), int(point["y"]))
                for name, point in person["keypoints"].items()
                if float(point.get("confidence") or 0.0) >= 0.3
            }
            mask = pose_body_mask(
                (frame_height, frame_width),
                projected,
                (
                    int(bbox["x1"]), int(bbox["y1"]),
                    int(bbox["x2"]), int(bbox["y2"]),
                ),
            )
            _, tof_center = project_tof_region(
                frame_width, frame_height, max(float(distance), 0.02)
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            self.stable_seeds.clear()
            return None

        tof_x, tof_y = tof_center
        if mask[tof_y, tof_x] == 0:
            self.stable_seeds.clear()
            return None

        return self.stable_seeds.update(
            camera_state,
            target=self.target,
            session_id=self.action_id,
        )

    async def _person_tracking_loop(self):
        current_reached = False
        predicted_target = None
        missing_keypoint_since = None
        missing_person_since = None
        keypoint_timeout_seconds = 2.0
        loop = asyncio.get_running_loop()
        last_inference_sequence = None
        last_visual_override_updated_at = None
        last_seed_update_at = None
        last_seed_target = None

        while self.active:
            (
                inference_sequence,
                inference_detections,
                inference_timing,
            ) = (
                self.yolo.detection_snapshot_with_timing()
            )
            visual_override = self._current_visual_target_override()
            override_state = self._visual_target_override
            visual_override_updated_at = (
                override_state.get("updated_at")
                if visual_override is not None
                and isinstance(override_state, dict)
                else None
            )
            inference_is_new = inference_sequence != last_inference_sequence
            override_is_new = (
                visual_override_updated_at
                != last_visual_override_updated_at
            )
            if not inference_is_new and not override_is_new:
                if (
                    missing_person_since is not None
                    and loop.time() - missing_person_since
                    >= keypoint_timeout_seconds
                    and not self._continuous_person_reacquisition
                ):
                    await self.stop_tracking(
                        reason_code="PERSON_LOST",
                        status="failed",
                        outcome="target_lost_after_reacquisition",
                    )
                    return
                await asyncio.sleep(0.01)
                continue
            if inference_is_new:
                last_inference_sequence = inference_sequence
            last_visual_override_updated_at = visual_override_updated_at
            detections = [
                detection
                for detection in inference_detections
                if detection["class"] == "person"
                and "keypoints" in detection
            ]

            if not detections:
                self.stable_seeds.clear()
                if missing_person_since is None:
                    missing_person_since = loop.time()
                elif (
                    loop.time() - missing_person_since
                    >= keypoint_timeout_seconds
                    and not self._continuous_person_reacquisition
                ):
                    await self.stop_tracking(
                        reason_code="PERSON_LOST",
                        status="failed",
                        outcome="target_lost_after_reacquisition",
                    )
                    return
                await asyncio.sleep(0.05)
                continue
            missing_person_since = None

            person = max(
                detections,
                key=lambda detection: detection["confidence"]
            )
            current_name = self.person_path[self.person_path_index]
            current_keypoint = person_keypoint(person, current_name)

            if not current_reached:
                if current_keypoint is None:
                    if missing_keypoint_since is None:
                        missing_keypoint_since = loop.time()
                    elif (
                        loop.time() - missing_keypoint_since
                        >= keypoint_timeout_seconds
                        and not self._continuous_person_reacquisition
                    ):
                        await self.stop_tracking(
                            reason_code="PERSON_KEYPOINT_TIMEOUT",
                            status="failed",
                            outcome="target_keypoint_lost",
                        )
                        return
                    await asyncio.sleep(0.05)
                    continue

                missing_keypoint_since = None
                target_x = current_keypoint["normalized_x"]
                target_y = current_keypoint["normalized_y"]

            if self.person_path_index < len(self.person_path) - 1:
                if current_reached:
                    next_name = self.person_path[self.person_path_index + 1]
                    next_keypoint = person_keypoint(person, next_name)

                    if next_keypoint is not None:
                        self.person_path_index += 1
                        current_reached = False
                        predicted_target = None
                        missing_keypoint_since = None

                        print(f"\nTRACING TO: {next_name}\n")

                    else:
                        if predicted_target is not None:
                            target_x,target_y = predicted_target
                        else:
                            await asyncio.sleep(0.05)
                            continue

                else:
                    if (abs(target_x - 0.5) < self.stable_threshold and abs(target_y - 0.5) < self.stable_threshold):
                        predicted_target = predict_next_keypoint(
                            person,
                            self.person_path,
                            self.person_path_index,
                        )
                        current_reached = True

            if visual_override is not None:
                target_x, target_y = visual_override

            target_centered = (
                abs(target_x - 0.5) < self.stable_threshold
                and abs(target_y - 0.5) < self.stable_threshold
            )

            delta_pan_angle,delta_tilt_angle = self.target_to_angles(target_x,target_y)

            command_created_at = time.monotonic()
            captured_at = inference_timing.get("source_captured_at")
            inference_started_at = inference_timing.get("inference_started_at")
            inference_completed_at = inference_timing.get("inference_completed_at")
            tracking_timing = {"sent_at_unix_ns": time.time_ns()}
            if all(
                isinstance(value, (int, float))
                for value in (
                    captured_at,
                    inference_started_at,
                    inference_completed_at,
                )
            ):
                tracking_timing.update({
                    "capture_to_send_ms": max(
                        0.0, (command_created_at - captured_at) * 1000.0
                    ),
                    "capture_wait_ms": max(
                        0.0, (inference_started_at - captured_at) * 1000.0
                    ),
                    "inference_ms": max(
                        0.0,
                        (inference_completed_at - inference_started_at) * 1000.0,
                    ),
                    "post_inference_ms": max(
                        0.0,
                        (command_created_at - inference_completed_at) * 1000.0,
                    ),
                })

            if visual_override is not None or not target_centered:
                tracking_sequence = (
                    f"hand:{visual_override_updated_at}"
                    if visual_override is not None
                    else f"pose:{inference_sequence}"
                )
                await self.zmq_pub_socket.send_json({
                    "delta_pan_angle": delta_pan_angle,
                    "delta_tilt_angle": delta_tilt_angle,
                    "tracking_sequence": tracking_sequence,
                    "tracking_timing": tracking_timing,
                })

            final_keypoint = self.person_path_index == len(self.person_path) - 1
            seed_target = (
                "hand"
                if visual_override is not None
                else self.target
                if final_keypoint and target_centered
                else None
            )
            if seed_target != last_seed_target:
                last_seed_update_at = None
                last_seed_target = seed_target

            if seed_target is not None:
                now = loop.time()
                seed_update_interval = 1.0 / self.PERSON_SEED_UPDATE_HZ
                if (
                    last_seed_update_at is None
                    or now - last_seed_update_at >= seed_update_interval
                ):
                    seed = (
                        self.stable_seeds.get(
                            target="hand",
                            session_id=self.action_id,
                        )
                        if visual_override is not None
                        else self._update_person_seed(person)
                    )
                    if seed is not None:
                        if visual_override is None:
                            self._remember_stable_target()
                        await self._publish_lidar_seed(seed)
                    last_seed_update_at = now
            elif visual_override is None:
                self.stable_seeds.clear()

            await asyncio.sleep(0.01)

    async def adopt_person_tracking(
        self,
        action_id,
        continuous_person_reacquisition=False,
    ):
        """Transfer an active human track without restarting the camera servo."""
        if not self.active or self.target not in HUMAN_TRACKABLE_PARTS:
            return False

        self.action_id = action_id
        self.stable_seeds.clear()
        if continuous_person_reacquisition:
            self._continuous_person_reacquisition = True
        return True


    async def wait_until_finished(self):
        return await self.completion_future

    async def stop_tracking(
        self,
        reason_code="USER_REQUESTED",
        status="cancelled",
        outcome="tracking_stopped",
        reset_camera=True,
    ):
        if not self.active:
            return

        tracking_task = self._tracking_task
        was_person_tracking = self.target in HUMAN_TRACKABLE_PARTS
        preserve_person_tracker = (
            was_person_tracking
            and status == "failed"
            and self._continuous_person_reacquisition
        )
        effective_reset_camera = (
            reset_camera is not False and not preserve_person_tracker
        )

        await self.send_robot_command({
            "command": "stop_tracking",
            "action_id": self.action_id,
            "reset_camera": effective_reset_camera,
        })

        result = self.complete_tracking(
            status=status,
            outcome=outcome,
            reason_code=reason_code,
        )

        if (
            tracking_task is not None
            and tracking_task is not asyncio.current_task()
            and not tracking_task.done()
        ):
            tracking_task.cancel()
            await asyncio.gather(tracking_task, return_exceptions=True)

        return result

    def complete_tracking(
        self,
        status,
        outcome="tracking_stopped",
        reason_code=None,
        data=None,
    ):
        action_id = self.action_id or "unassigned"
        target = self.target
        had_validated_seed = self.stable_seeds.get(
            target=target,
            session_id=action_id,
        ) is not None

        result = ActionResult(
            action_id=action_id,
            action_type="track_action",
            status=status,
            target=target,
            outcome=outcome,
            reason_code=reason_code,
            retryable=status == "failed",
            data={
                "had_validated_seed": had_validated_seed,
                **(data or {}),
            },
        )

        completion_future = self.completion_future

        self.active = False
        self._continuous_person_reacquisition = False
        self._visual_target_override = None
        self.stable_seeds.clear()
        self.csrt_tracker.stop_tracking()
        self.action_id = None
        self.target = None
        self._tracking_task = None

        if completion_future is not None and not completion_future.done():
            completion_future.set_result(result)

        return result

    def target_to_angles(self, x, y):
        center_x_error = x - 0.5
        center_y_error = y - 0.5

        tan_half_fov_h = math.tan(math.radians(HORIZONTAL_FOV_DEG / 2.0))
        tan_half_fov_v = math.tan(math.radians(VERTICAL_FOV_DEG / 2.0))

        pan_angle = math.degrees(math.atan((center_x_error * 2.0) * tan_half_fov_h))
        tilt_angle = math.degrees(math.atan((center_y_error * 2.0) * tan_half_fov_v))

        delta_pan = -pan_angle
        delta_tilt = tilt_angle

        return delta_pan, delta_tilt
