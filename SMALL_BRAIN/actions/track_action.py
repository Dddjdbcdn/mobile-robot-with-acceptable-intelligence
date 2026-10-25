import asyncio
import math
from pathlib import Path
import time

from actions.action_result import ActionResult
from actions.move_camera_action import PAN_POSITION_ANGLE
from actions.tracking.person_pose import (
    TorsoTargetEstimator,
    find_best_person_path,
    person_keypoint,
    predict_next_keypoint,
)
from actions.tracking.stable_seed import StableTargetSeedTracker
from actions.tracking.target_catalog import (
    HUMAN_TRACKABLE_PARTS,
    YOLO_OBJECT_CLASSES,
    normalize_human_target,
    normalize_object_target,
)
from cognition.manager.world_state import robot_state
from services.display.camera_display import pose_body_mask, project_tof_region

HORIZONTAL_FOV_DEG = 85
VERTICAL_FOV_DEG = 52


class TrackingService:
    """One shareable visual-tracking session for all high-level actions."""
    OBJECT_FAILURE_GRACE_FRAMES = 3
    OBJECT_REACQUIRE_ATTEMPTS = 3
    OBJECT_REACQUIRE_DELAY_SECONDS = 0.25
    YOLO_OBJECT_DETECTION_TIMEOUT_SECONDS = 1.0
    TARGET_SEED_SAMPLES = StableTargetSeedTracker.SAMPLE_COUNT
    PERSON_POINT_MAX_AGE_SECONDS = StableTargetSeedTracker.INPUT_MAX_AGE_SECONDS
    PERSON_SEED_UPDATE_HZ = 5.0
    PERSON_TARGET_SMOOTHING_ALPHA = 0.35
    PERSON_ID_ACQUISITION_TIMEOUT_SECONDS = 1.0

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
        person_tracking_service=None,
    ):
        self.csrt_tracker = csrt_tracker
        self.grounding_dino = grounding_dino
        self.yolo = yolo
        self.zmq_pub_socket = zmq_pub_socket
        self.person_tracker_pub_socket = person_tracker_pub_socket
        self.person_tracking_service = person_tracking_service
        self.send_robot_command = send_robot_command
        self.camera = camera

        self.active = False
        self.target = None
        self.stable_threshold = STABLE_THRESHOLD
        self._tracking_task = None
        self.completion_future = None
        self.action_id = None
        self.detection_confidence = 1.0
        self._allow_grounding_dino = True
        self._follow_person_active = False
        self.move_camera_action = None
        self._last_person_lidar_seed_key = None
        self._visual_target_override = None
        self.stable_seeds = StableTargetSeedTracker()

        self.person_path = []
        self.person_path_index = 0
        self._tracked_person_id = None

    @property
    def session_id(self):
        return self.action_id

    @property
    def tracked_person_id(self):
        return self._tracked_person_id

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

    def clear_visual_target_override(self):
        if self._visual_target_override is None:
            return
        self._visual_target_override = None
        self.stable_seeds.clear()

    def set_follow_person_active(self, active):
        """Mirror follow-action state for tracking timeout and seed behavior."""
        self._follow_person_active = bool(active)
        self._last_person_lidar_seed_key = None
        person_tracking = getattr(self, "person_tracking_service", None)
        set_persistent = getattr(
            person_tracking, "set_target_persistent", None)
        if callable(set_persistent):
            set_persistent(self._follow_person_active)

    def set_move_camera_action(self, move_camera_action):
        """Attach shared camera movement after both services are constructed."""
        self.move_camera_action = move_camera_action

    async def _reset_pan_for_follow(self):
        """Center pan after visual loss without interrupting shared tracking."""
        if self.move_camera_action is None:
            return None
        return await self.move_camera_action.reset_pan(
            action_id=f"{self.action_id or 'unassigned'}:person-absent-reset-pan",
            stop_tracking=False,
        )

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
        expected_session_id = (self.action_id
                               if session_id is None else session_id)
        while (self.active and self.target == expected_target
               and self.action_id == expected_session_id):
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
        initial_person=None,
    ):
        normalized_target = (normalize_human_target(target)
                             or normalize_object_target(target))

        if self.active and self.target == normalized_target:
            self._allow_grounding_dino = (self._allow_grounding_dino
                                          or bool(allow_grounding_dino))
            return ActionResult(
                action_id=action_id,
                action_type="track_action",
                status="running",
                target=self.target,
                outcome="tracking_shared",
                data={
                    "tracking_session_id": self.action_id,
                    "reused_tracking": True,
                },
            )

        if self.active:
            await self.stop(
                reason_code="TARGET_CHANGED",
                status="cancelled",
                outcome="replaced_by_new_target",
                reset_camera=False,
            )

        self.action_id = action_id
        self.target = normalized_target
        self.stable_seeds.clear()
        self.detection_confidence = 1.0
        self._allow_grounding_dino = allow_grounding_dino
        self._tracked_person_id = None

        if normalized_target in HUMAN_TRACKABLE_PARTS:
            return await self._start_person_tracking(
                normalized_target,
                initial_person=initial_person,
            )

        jpeg_bytes = None
        snapshot = None
        if (
            allow_grounding_dino
            and normalized_target not in YOLO_OBJECT_CLASSES
        ):
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
        detection = await self._detect_and_initialize_object_tracker(
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
        self._tracking_task = self._create_tracking_task(
            self._object_tracking_loop,
            kind="object",
        )

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

    async def _detect_and_initialize_object_tracker(
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

        if target in YOLO_OBJECT_CLASSES:
            yolo_detection = await self._detect_yolo_object(target)
            if yolo_detection is not None:
                return yolo_detection

        if allow_grounding_dino:
            if jpeg_bytes is None:
                jpeg_bytes = await asyncio.to_thread(
                    self.camera.jpeg_bytes_snapshot, 70, False)
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

    async def _detect_yolo_object(self, target):
        """Initialize CSRT from one fresh object-mode inference."""
        async with self.yolo.object_detection_mode():
            sequence, _ = self.yolo.detection_snapshot()
            if not await self.yolo.wait_for_inference_after(
                sequence,
                timeout=self.YOLO_OBJECT_DETECTION_TIMEOUT_SECONDS,
            ):
                return None

            (
                current_sequence,
                detections,
                frame,
                timing,
            ) = self.yolo.detection_snapshot_with_tracking_frame()
            if current_sequence <= sequence or frame is None:
                return None

            matches = [
                detection for detection in detections
                if detection.get("class") == target
                and detection.get("bbox")
            ]
            if not matches:
                return None

            selected = max(
                matches,
                key=lambda item: float(item.get("confidence") or 0.0),
            )
            camera_sequence = timing.get("camera_sequence")
            if not isinstance(camera_sequence, int):
                return None

            bbox = selected["bbox"]
            frame_height, frame_width = frame.shape[:2]
            x1 = max(
                0, min(round(float(bbox["x1"])), frame_width - 1)
            )
            y1 = max(
                0, min(round(float(bbox["y1"])), frame_height - 1)
            )
            x2 = max(0, min(round(float(bbox["x2"])), frame_width))
            y2 = max(0, min(round(float(bbox["y2"])), frame_height))
            width = x2 - x1
            height = y2 - y1
            if width < 2 or height < 2:
                return None

            confidence = float(selected.get("confidence") or 0.0)
            self.detection_confidence = confidence
            self.csrt_tracker.begin_tracking(
                detection_sequence=camera_sequence,
                initialization_frame=frame,
                bbox_xywh=(x1, y1, width, height),
                target=target,
            )
            return "yoloe-11m", confidence

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
                    target_x, target_y)

                centered = (abs(target_x - 0.5) < self.stable_threshold
                            and abs(target_y - 0.5) < self.stable_threshold)

                if not centered:
                    await self.zmq_pub_socket.send_json({
                        "delta_pan_angle":
                        delta_pan_angle,
                        "delta_tilt_angle":
                        delta_tilt_angle,
                        "tracking_sequence":
                        f"csrt:{tracking_update.sequence}",
                        "tracking_valid": True,
                    })

                if centered:
                    self.stable_seeds.update(
                        robot_state["camera"],
                        target=self.target,
                        session_id=self.action_id,
                    )
                else:
                    self.stable_seeds.clear()
            else:
                self.stable_seeds.clear()
                failure_frames += 1
                if failure_frames >= self.OBJECT_FAILURE_GRACE_FRAMES:
                    reacquired = False
                    for _attempt in range(self.OBJECT_REACQUIRE_ATTEMPTS):
                        detection = await (
                            self._detect_and_initialize_object_tracker(
                                self.target,
                                allow_grounding_dino=self._allow_grounding_dino,
                            ))
                        await asyncio.sleep(self.OBJECT_REACQUIRE_DELAY_SECONDS
                                            )
                        if detection is not None:
                            reacquired = True
                            break

                    if reacquired:
                        failure_frames = 0
                    else:
                        await self.stop(
                            reason_code="OBJECT_LOST",
                            status="failed",
                            outcome="target_lost_after_reacquisition",
                        )
                        return

            await asyncio.sleep(0.05)

    async def _start_person_tracking(self, target, initial_person=None):
        if self.person_tracking_service is None:
            return self._person_detection_failure(
                target,
                reason_code="PERSON_TRACKING_SERVICE_UNAVAILABLE",
            )

        start_target_tracking = getattr(
            self.person_tracking_service, "start_target_tracking", None)
        if callable(start_target_tracking):
            start_target_tracking()

        try:
            person_track = await self._wait_for_initial_person_track(
                initial_person,
            )
        except BaseException:
            stop_target_tracking = getattr(
                self.person_tracking_service, "stop_target_tracking", None)
            if callable(stop_target_tracking):
                stop_target_tracking()
            raise
        if person_track is None or person_track.person is None:
            return self._person_detection_failure(target)

        person = person_track.person
        self._tracked_person_id = person_track.track_id
        self.detection_confidence = person["confidence"]
        self.person_path = find_best_person_path(person, target)

        self.person_path_index = 0
        await self.send_robot_command({
            "command": "track_action",
            "action_id": self.action_id,
        })

        self.active = True
        self.completion_future = asyncio.get_running_loop().create_future()
        self._tracking_task = self._create_tracking_task(
            self._person_tracking_loop,
            kind="person",
        )

        return ActionResult(
            action_id=self.action_id,
            action_type="track_action",
            status="running",
            target=target,
            outcome="tracking",
            data={
                "detection_source": "person_tracking_service",
                "detection_confidence": person["confidence"],
                "keypoint_path": list(self.person_path),
                "person_track_id": person_track.track_id,
            },
        )

    def _person_detection_failure(
        self,
        target,
        reason_code="PERSON_DETECTION_FAILED",
    ):
        stop_target_tracking = getattr(
            self.person_tracking_service, "stop_target_tracking", None)
        if callable(stop_target_tracking):
            stop_target_tracking()
        action_id = self.action_id or "unassigned"
        self.action_id = None
        self.target = None
        self._tracked_person_id = None
        return ActionResult(
            action_id=action_id,
            action_type="track_action",
            status="failed",
            target=target,
            outcome="target_not_detected",
            reason_code=reason_code,
            retryable=True,
        )

    async def _wait_for_initial_person_track(self, initial_person=None):
        """Wait for a confirmed persistent ID without moving the camera."""
        wait_for_track = getattr(
            self.person_tracking_service,
            "wait_for_confirmed_track",
            None,
        )
        if callable(wait_for_track):
            return await wait_for_track(
                initial_person,
                timeout=self.PERSON_ID_ACQUISITION_TIMEOUT_SECONDS,
            )

        # Compatibility path for lightweight service doubles.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.PERSON_ID_ACQUISITION_TIMEOUT_SECONDS
        while True:
            if initial_person is not None:
                track = self.person_tracking_service.track_for_detection(
                    initial_person,
                )
                if (track is not None and track.visible
                        and track.person is not None):
                    return track
            else:
                _, tracks = self.person_tracking_service.tracking_snapshot()
                visible_tracks = [
                    track for track in tracks
                    if track.visible and track.person is not None
                ]
                if visible_tracks:
                    return max(
                        visible_tracks,
                        key=lambda track: track.person.get("confidence", 0.0),
                    )

            if loop.time() >= deadline:
                return None
            await asyncio.sleep(0.01)

    async def _publish_lidar_seed(self, seed):
        """Publish each captured ToF seed at most once."""
        if self.person_tracker_pub_socket is None:
            return False
        captured_at = float(seed["captured_at"])
        seed_key = (
            str(seed["session_id"]),
            str(seed["target"]),
            captured_at,
        )
        if seed_key == self._last_person_lidar_seed_key:
            return False
        await self.person_tracker_pub_socket.send_json({
            "type":
            "person_tof_position",
            "x":
            seed["x"],
            "y":
            seed["y"],
            "frame_id":
            "base_footprint",
            "target":
            seed["target"],
            "session_id":
            seed["session_id"],
            "captured_at_monotonic":
            captured_at,
        })
        self._last_person_lidar_seed_key = seed_key
        return True

    def _create_tracking_task(self, tracking_loop, *, kind):
        """Run a tracking loop without allowing stale active state on failure."""
        session_id = self.action_id
        return asyncio.create_task(
            self._supervise_tracking_loop(tracking_loop, session_id),
            name=f"track-{kind}-{session_id}",
        )

    async def _supervise_tracking_loop(self, tracking_loop, session_id):
        try:
            await tracking_loop()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            error_detail = f"{type(error).__name__}: {error}"
            print(
                f"[TRACKING][{session_id}] loop_failed "
                f"error={error_detail!r}",
                flush=True,
            )
            if not self.active or self.action_id != session_id:
                return
            try:
                await self.stop(
                    reason_code="TRACKING_LOOP_ERROR",
                    status="failed",
                    outcome="tracking_loop_error",
                    data={"error": error_detail},
                )
            except Exception as stop_error:
                # Local action state must still become terminal even if the
                # robot command channel fails at the same time as tracking.
                print(
                    f"[TRACKING][{session_id}] cleanup_command_failed "
                    f"error={type(stop_error).__name__}: {stop_error}",
                    flush=True,
                )
                if self.active and self.action_id == session_id:
                    self.complete_tracking(
                        status="failed",
                        outcome="tracking_loop_error",
                        reason_code="TRACKING_LOOP_ERROR",
                        data={
                            "error": error_detail,
                            "cleanup_error": (
                                f"{type(stop_error).__name__}: {stop_error}"
                            ),
                        },
                    )

    async def _publish_person_visual_observation(
        self,
        person,
        *,
        captured_at,
    ):
        """Publish the camera view and optional person bearing."""
        if self.person_tracker_pub_socket is None:
            return

        camera_angle = math.radians(
            robot_state["camera"]["pan_angle"]
            - PAN_POSITION_ANGLE["center"]
        )
        observation = {
            "type": "person_visual_observation",
            "captured_at_monotonic": captured_at,
            "camera_bearing_rad": camera_angle,
            "horizontal_fov_rad": math.radians(HORIZONTAL_FOV_DEG),
            "person_visible": person is not None,
            "session_id": self.action_id,
        }
        if person is not None:
            center_x = person["bbox"]["normalized_center_x"]
            image_angle = math.atan(
                (center_x - 0.5)
                * 2.0
                * math.tan(math.radians(HORIZONTAL_FOV_DEG) * 0.5)
            )
            observation["base_bearing_rad"] = camera_angle - image_angle
        await self.person_tracker_pub_socket.send_json(observation)
        return observation

    def _valid_tof_coordinate(self, camera_state):
        """Return whether the current base-frame ToF coordinate is usable."""
        return (
            camera_state["camera_tof_range"] > 0.05
            and time.monotonic() - camera_state["timestamp"]
            <= self.PERSON_POINT_MAX_AGE_SECONDS
        )

    def _person_tof_intersects_mask(self, person, camera_state):
        """Return whether the current valid ToF point lies on the person."""
        if not self._valid_tof_coordinate(camera_state):
            return False
        distance = camera_state["camera_tof_range"]

        try:
            frame_height, frame_width = self.camera.snapshot(
            ).tracking_bgr.shape[:2]
        except RuntimeError as error:
            if str(error) != "No camera frame is available.":
                raise
            # CameraStream deliberately clears its latest frame while a USB
            # camera reconnects. That short gap invalidates this ToF sample,
            # but it must not terminate the continuous tracking session.
            return False
        bbox = person["bbox"]
        projected = {
            name: (int(point["x"]), int(point["y"]))
            for name, point in person["keypoints"].items()
            if point["confidence"] >= 0.3
        }
        mask = pose_body_mask(
            (frame_height, frame_width),
            projected,
            (
                int(bbox["x1"]),
                int(bbox["y1"]),
                int(bbox["x2"]),
                int(bbox["y2"]),
            ),
        )
        _, tof_center = project_tof_region(
            frame_width,
            frame_height,
            distance,
        )

        tof_x, tof_y = tof_center
        return bool(mask[tof_y, tof_x])

    async def _person_tracking_loop(self):
        current_reached = False
        predicted_target = None
        missing_person_since = None
        person_timeout_seconds = float(getattr(
            self.person_tracking_service,
            "lost_timeout_seconds",
            2.0,
        ))
        loop = asyncio.get_running_loop()
        last_inference_sequence = None
        last_visual_override_updated_at = None
        last_seed_update_at = None
        last_seed_target = None
        filtered_target = None
        filtered_target_name = None
        pan_reset_for_current_absence = False
        prediction_attempted = False
        torso_estimator = TorsoTargetEstimator()

        while self.active:
            (
                yolo_sequence,
                _inference_detections,
                _inference_frame_bgr,
                inference_timing,
            ) = self.yolo.detection_snapshot_with_frame()
            if self._tracked_person_id is not None:
                (
                    inference_sequence,
                    person_tracks,
                    person_captured_at,
                ) = self.person_tracking_service.tracking_snapshot_with_time()
                person_track = next(
                    (
                        track for track in person_tracks
                        if track.track_id == self._tracked_person_id
                    ),
                    None,
                )
                person = (
                    person_track.person
                    if (person_track is not None and person_track.visible)
                    else None
                )
                if yolo_sequence != inference_sequence:
                    # The ID service owns this detection. Preserve its source
                    # timestamp instead of attributing it to a newer YOLO frame.
                    captured_at = (
                        time.monotonic()
                        if person_captured_at is None else person_captured_at
                    )
                    inference_timing = {
                        "source_captured_at": captured_at,
                        "inference_started_at": captured_at,
                        "inference_completed_at": captured_at,
                    }
            else:
                inference_sequence = yolo_sequence
                person = None
            visual_override = self._current_visual_target_override()
            override_state = self._visual_target_override
            visual_override_updated_at = (
                override_state.get("updated_at") if visual_override is not None
                else None)
            inference_is_new = inference_sequence != last_inference_sequence
            override_is_new = (visual_override_updated_at
                               != last_visual_override_updated_at)
            if not inference_is_new and not override_is_new:
                if (missing_person_since is not None
                        and loop.time() - missing_person_since
                        >= person_timeout_seconds):
                    if self._follow_person_active:
                        if not pan_reset_for_current_absence:
                            pan_reset_for_current_absence = True
                            await self._reset_pan_for_follow()
                    else:
                        await self.stop(
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
            if inference_is_new:
                await self._publish_person_visual_observation(
                    person,
                    captured_at=inference_timing["source_captured_at"],
                )
            if person is None:
                self.stable_seeds.clear()
                if missing_person_since is None:
                    missing_person_since = loop.time()
                elif (loop.time() - missing_person_since
                      >= person_timeout_seconds):
                    if self._follow_person_active:
                        if not pan_reset_for_current_absence:
                            pan_reset_for_current_absence = True
                            await self._reset_pan_for_follow()
                    else:
                        await self.stop(
                            reason_code="PERSON_LOST",
                            status="failed",
                            outcome="target_lost_after_reacquisition",
                        )
                        return
                await asyncio.sleep(0.05)
                continue
            missing_person_since = None
            pan_reset_for_current_absence = False
            current_name = self.person_path[self.person_path_index]
            torso_target = torso_estimator.update(
                person,
                allow_bootstrap=(
                    prediction_attempted or current_name == "torso_center"
                ),
            )

            # Intermediate joints only guide the camera until a better point
            # on the remaining skeleton path is visible. Prefer the furthest
            # visible downstream point so we do not finish centering a joint
            # that is no longer needed to reveal the requested target.
            better_keypoint = None
            for candidate_index in range(
                len(self.person_path) - 1,
                self.person_path_index,
                -1,
            ):
                candidate_name = self.person_path[candidate_index]
                candidate_keypoint = person_keypoint(
                    person,
                    candidate_name,
                    torso_target=torso_target,
                )
                if candidate_keypoint is not None:
                    self.person_path_index = candidate_index
                    current_name = candidate_name
                    better_keypoint = candidate_keypoint
                    current_reached = False
                    predicted_target = None
                    prediction_attempted = False
                    print(f"\nTRACING TO: {current_name}\n")
                    break

            current_keypoint = better_keypoint
            if current_keypoint is None:
                current_keypoint = person_keypoint(
                    person,
                    current_name,
                    torso_target=torso_target,
                )
            using_prediction = False
            selected_target_name = current_name

            if not current_reached:
                if current_keypoint is None:
                    self.stable_seeds.clear()
                    await asyncio.sleep(0.05)
                    continue
                target_x = current_keypoint["normalized_x"]
                target_y = current_keypoint["normalized_y"]

            if self.person_path_index < len(self.person_path) - 1:
                if current_reached:
                    next_name = self.person_path[self.person_path_index + 1]
                    if (predicted_target is not None
                            and not prediction_attempted):
                        target_x, target_y = predicted_target
                        using_prediction = True
                        selected_target_name = f"prediction:{next_name}"
                    else:
                        self.stable_seeds.clear()
                        await asyncio.sleep(0.05)
                        continue

                else:
                    if (abs(target_x - 0.5) < self.stable_threshold
                            and abs(target_y - 0.5) < self.stable_threshold):
                        predicted_target = predict_next_keypoint(
                            person,
                            self.person_path,
                            self.person_path_index,
                        )
                        # A path that starts at a hip has no previous joint to
                        # extrapolate from. In that case the torso estimator's
                        # configured hip offset is the next step immediately.
                        prediction_attempted = predicted_target is None
                        current_reached = True

            if visual_override is not None:
                target_x, target_y = visual_override
                filtered_target = None
                filtered_target_name = None
            else:
                target_name = selected_target_name
                if filtered_target is None or filtered_target_name != target_name:
                    filtered_target = (target_x, target_y)
                    filtered_target_name = target_name
                else:
                    alpha = self.PERSON_TARGET_SMOOTHING_ALPHA
                    filtered_target = (
                        filtered_target[0] + alpha *
                        (target_x - filtered_target[0]),
                        filtered_target[1] + alpha *
                        (target_y - filtered_target[1]),
                    )
                target_x, target_y = filtered_target

            target_centered = (abs(target_x - 0.5) < self.stable_threshold
                               and abs(target_y - 0.5) < self.stable_threshold)

            delta_pan_angle, delta_tilt_angle = self.target_to_angles(
                target_x, target_y)

            command_created_at = time.monotonic()
            captured_at = inference_timing["source_captured_at"]
            inference_started_at = inference_timing["inference_started_at"]
            inference_completed_at = inference_timing["inference_completed_at"]
            tracking_timing = {
                "sent_at_unix_ns": time.time_ns(),
                "capture_to_send_ms":
                (command_created_at - captured_at) * 1000.0,
                "capture_wait_ms":
                (inference_started_at - captured_at) * 1000.0,
                "inference_ms":
                (inference_completed_at - inference_started_at) * 1000.0,
                "post_inference_ms":
                (command_created_at - inference_completed_at) * 1000.0,
            }

            tracking_sequence = (f"hand:{visual_override_updated_at}"
                                 if visual_override is not None else
                                 f"pose:{inference_sequence}")

            if not target_centered:
                await self.zmq_pub_socket.send_json({
                    "delta_pan_angle":
                    delta_pan_angle,
                    "delta_tilt_angle":
                    delta_tilt_angle,
                    "tracking_sequence":
                    tracking_sequence,
                    "tracking_timing":
                    tracking_timing,
                    "tracking_valid": True,
                })

            if using_prediction and visual_override is None:
                # A prediction is an image-space camera offset, not a point
                # attached to the person. Apply it once, then use the latest
                # reached keypoint until the intended keypoint becomes visible.
                prediction_attempted = True

            final_keypoint = self.person_path_index == len(
                self.person_path) - 1

            seed_target = None
            if visual_override is not None and target_centered:
                seed_target = "hand"
            elif (visual_override is None and final_keypoint
                  and not using_prediction):
                seed_target = self.target

            if seed_target != last_seed_target:
                last_seed_update_at = None
                last_seed_target = seed_target

            if seed_target is None:
                self.stable_seeds.clear()
                await asyncio.sleep(0.01)
                continue

            now = loop.time()
            seed_update_interval = 1.0 / self.PERSON_SEED_UPDATE_HZ

            if (last_seed_update_at is None or now - last_seed_update_at >= seed_update_interval):
                last_seed_update_at = now

                camera_state = robot_state["camera"]

                if visual_override is not None or self._person_tof_intersects_mask(person, camera_state):

                    stable_seed = self.stable_seeds.update(
                        camera_state,
                        target=seed_target,
                        session_id=self.action_id,
                    )
                    if self._follow_person_active:
                        if self._valid_tof_coordinate(camera_state):
                            await self._publish_lidar_seed({
                                "x": camera_state["object_x"],
                                "y": camera_state["object_y"],
                                "target": seed_target,
                                "session_id": self.action_id,
                                "captured_at": camera_state["timestamp"],
                            })
                    elif stable_seed is not None:
                        await self._publish_lidar_seed(stable_seed)
                else:
                    self.stable_seeds.clear()

            await asyncio.sleep(0.01)

    async def wait_until_finished(self):
        return await asyncio.shield(self.completion_future)

    async def stop(
        self,
        reason_code="USER_REQUESTED",
        status="cancelled",
        outcome="tracking_stopped",
        reset_camera=True,
        data=None,
    ):
        if not self.active:
            return

        tracking_task = self._tracking_task
        was_person_tracking = self.target in HUMAN_TRACKABLE_PARTS
        preserve_person_tracker = (was_person_tracking and status == "failed"
                                   and self._follow_person_active)
        effective_reset_camera = (reset_camera is not False
                                  and not preserve_person_tracker)

        await self.send_robot_command({
            "command": "stop_tracking",
            "action_id": self.action_id,
            "reset_camera": effective_reset_camera,
        })

        result = self.complete_tracking(
            status=status,
            outcome=outcome,
            reason_code=reason_code,
            data=data,
        )

        if (tracking_task is not None
                and tracking_task is not asyncio.current_task()
                and not tracking_task.done()):
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

        if target in HUMAN_TRACKABLE_PARTS:
            stop_target_tracking = getattr(
                self.person_tracking_service, "stop_target_tracking", None)
            if callable(stop_target_tracking):
                stop_target_tracking()

        self.active = False
        self._visual_target_override = None
        self.stable_seeds.clear()
        self.csrt_tracker.stop_tracking()
        self.action_id = None
        self.target = None
        self._tracking_task = None
        self._tracked_person_id = None

        if completion_future is not None and not completion_future.done():
            completion_future.set_result(result)

        return result

    def target_to_angles(self, x, y):
        center_x_error = x - 0.5
        center_y_error = y - 0.5

        tan_half_fov_h = math.tan(math.radians(HORIZONTAL_FOV_DEG / 2.0))
        tan_half_fov_v = math.tan(math.radians(VERTICAL_FOV_DEG / 2.0))

        pan_angle = math.degrees(
            math.atan((center_x_error * 2.0) * tan_half_fov_h))
        tilt_angle = math.degrees(
            math.atan((center_y_error * 2.0) * tan_half_fov_v))

        delta_pan = -pan_angle
        delta_tilt = tilt_angle

        return delta_pan, delta_tilt
