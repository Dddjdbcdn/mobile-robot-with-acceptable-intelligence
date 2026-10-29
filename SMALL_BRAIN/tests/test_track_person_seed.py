import asyncio
import time
import unittest
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from actions.track_action import TrackingService
from actions.tracking.stable_seed import StableTargetSeedTracker
from cognition.manager.world_state import robot_state


class PersonTofMaskTests(unittest.TestCase):

    def setUp(self):
        self.previous_camera_state = robot_state["camera"]
        self.previous_robot_pose = robot_state.get("pose")
        self.service = TrackingService.__new__(TrackingService)
        self.service.action_id = "test-session"
        self.service._last_person_lidar_seed_key = None
        self.service.person_tracker_pub_socket = SimpleNamespace(
            send_json=AsyncMock(),
        )
        self.service.camera = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(
                tracking_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            ),
        )
        self.person = {
            "bbox": {"x1": 10, "y1": 10, "x2": 90, "y2": 90},
            "keypoints": {},
        }

    def tearDown(self):
        robot_state["camera"] = self.previous_camera_state
        robot_state["pose"] = self.previous_robot_pose

    def set_camera_seed(self, timestamp, x=1.0, y=0.2, tof_range=1.1):
        robot_state["camera"] = {
            "timestamp": timestamp,
            "camera_tof_range": tof_range,
            "object_x": x,
            "object_y": y,
            "object_angle": 0.1,
            "object_map_x": None,
            "object_map_y": None,
        }

    def test_tof_point_inside_person_mask_is_accepted(self):
        self.set_camera_seed(time.monotonic())
        mask = np.ones((100, 100), dtype=np.uint8)

        with (
            patch("actions.track_action.pose_body_mask", return_value=mask),
            patch(
                "actions.track_action.project_tof_region",
                return_value=(None, (50, 50)),
            ),
        ):
            intersects = self.service._person_tof_intersects_mask(
                self.person,
                robot_state["camera"],
            )

        self.assertTrue(intersects)

    def test_tof_point_outside_person_mask_is_rejected(self):
        self.set_camera_seed(time.monotonic())
        mask = np.zeros((100, 100), dtype=np.uint8)

        with (
            patch("actions.track_action.pose_body_mask", return_value=mask),
            patch(
                "actions.track_action.project_tof_region",
                return_value=(None, (50, 50)),
            ),
        ):
            intersects = self.service._person_tof_intersects_mask(
                self.person,
                robot_state["camera"],
            )

        self.assertFalse(intersects)

    def test_stale_tof_point_is_rejected(self):
        self.set_camera_seed(
            time.monotonic()
            - StableTargetSeedTracker.INPUT_MAX_AGE_SECONDS
            - 0.01
        )
        mask = np.ones((100, 100), dtype=np.uint8)

        with (
            patch("actions.track_action.pose_body_mask", return_value=mask),
            patch(
                "actions.track_action.project_tof_region",
                return_value=(None, (50, 50)),
            ),
        ):
            intersects = self.service._person_tof_intersects_mask(
                self.person,
                robot_state["camera"],
            )

        self.assertFalse(intersects)

    def test_tof_seed_beyond_six_metres_is_rejected(self):
        self.set_camera_seed(
            time.monotonic(),
            x=6.01,
            y=0.0,
            tof_range=6.01,
        )

        self.assertFalse(
            self.service._valid_tof_coordinate(robot_state["camera"])
        )

    def test_tof_seed_at_six_metres_is_usable(self):
        self.set_camera_seed(
            time.monotonic(),
            x=6.0,
            y=0.0,
            tof_range=6.0,
        )

        self.assertTrue(
            self.service._valid_tof_coordinate(robot_state["camera"])
        )

    def test_missing_camera_frame_during_reconnect_is_rejected(self):
        self.set_camera_seed(time.monotonic())
        self.service.camera.snapshot = Mock(
            side_effect=RuntimeError("No camera frame is available."),
        )

        intersects = self.service._person_tof_intersects_mask(
            self.person,
            robot_state["camera"],
        )

        self.assertFalse(intersects)

    def test_visual_observation_converts_box_motion_to_camera_bearing(self):
        person = {
            "confidence": 0.9,
            "bbox": {
                "normalized_x1": 0.65,
                "normalized_y1": 0.1,
                "normalized_x2": 0.85,
                "normalized_y2": 0.9,
                "normalized_center_x": 0.75,
                "normalized_center_y": 0.5,
            },
        }

        robot_state["camera"] = {"pan_angle": 95.0}
        robot_state["pose"] = None
        observation = asyncio.run(
            self.service._publish_person_visual_observation(
                person,
                captured_at=123.0,
            )
        )

        expected_image_angle = math.atan(
            0.5 * math.tan(math.radians(85.0) * 0.5)
        )
        self.assertAlmostEqual(
            observation["base_bearing_rad"],
            -expected_image_angle,
        )
        self.assertEqual(observation["person_visible"], True)
        self.assertAlmostEqual(
            observation["horizontal_fov_rad"], math.radians(85.0)
        )

    def test_visual_observation_remains_in_base_frame_when_pose_is_available(self):
        person = {
            "confidence": 0.9,
            "bbox": {
                "normalized_x1": 0.4,
                "normalized_y1": 0.1,
                "normalized_x2": 0.6,
                "normalized_y2": 0.9,
                "normalized_center_x": 0.5,
                "normalized_center_y": 0.5,
            },
        }

        robot_state["camera"] = {"pan_angle": 110.0}
        robot_state["pose"] = {"frame_id": "map", "yaw": 0.4}
        observation = asyncio.run(
            self.service._publish_person_visual_observation(
                person,
                captured_at=124.0,
            )
        )

        self.assertAlmostEqual(
            observation["base_bearing_rad"],
            math.radians(15.0),
        )

    def test_visual_observation_reports_empty_camera_view(self):
        robot_state["camera"] = {"pan_angle": 110.0}

        observation = asyncio.run(
            self.service._publish_person_visual_observation(
                None,
                captured_at=125.0,
            )
        )

        self.assertFalse(observation["person_visible"])
        self.assertAlmostEqual(
            observation["camera_bearing_rad"], math.radians(15.0)
        )
        self.assertNotIn("base_bearing_rad", observation)

    def test_same_lidar_seed_is_published_once(self):
        seed = {
            "x": 1.0,
            "y": 0.2,
            "target": "person",
            "session_id": "test-session",
            "captured_at": 123.0,
        }

        first = asyncio.run(self.service._publish_lidar_seed(seed))
        second = asyncio.run(self.service._publish_lidar_seed(seed))

        self.assertTrue(first)
        self.assertFalse(second)
        socket = self.service.person_tracker_pub_socket
        socket.send_json.assert_awaited_once()
        payload = socket.send_json.await_args.args[0]
        self.assertEqual(payload["captured_at_monotonic"], 123.0)

    def test_lidar_seed_beyond_six_metres_is_not_published(self):
        seed = {
            "x": 6.01,
            "y": 0.0,
            "target": "person",
            "session_id": "test-session",
            "captured_at": 123.0,
        }

        published = asyncio.run(self.service._publish_lidar_seed(seed))

        self.assertFalse(published)
        self.service.person_tracker_pub_socket.send_json.assert_not_awaited()


class FollowTrackingStateTests(unittest.TestCase):

    def setUp(self):
        self.service = TrackingService.__new__(TrackingService)
        self.service._follow_person_active = False
        self.service._last_person_lidar_seed_key = (
            "session", "person", 123.0
        )

    def test_stopping_follow_clears_follow_state(self):
        self.service.set_follow_person_active(False)

        self.assertFalse(self.service._follow_person_active)
        self.assertIsNone(self.service._last_person_lidar_seed_key)

    def test_follow_state_controls_person_id_persistence(self):
        persistence = []
        self.service.person_tracking_service = SimpleNamespace(
            set_target_persistent=persistence.append,
        )

        self.service.set_follow_person_active(True)
        self.service.set_follow_person_active(False)

        self.assertEqual(persistence, [True, False])


class TrackActionDiagnosticTests(unittest.TestCase):

    def test_debug_snapshot_identifies_hand_seed_and_tof_state(self):
        service = TrackingService.__new__(TrackingService)
        service.active = True
        service.action_id = "track-session"
        service.target = "person"
        service._follow_person_active = False
        service._tracked_person_id = 7
        service.stable_threshold = 0.05
        service._visual_target_override = {
            "x": 0.51,
            "y": 0.49,
            "updated_at": time.monotonic(),
        }
        service.stable_seeds = StableTargetSeedTracker()
        camera = {
            "timestamp": time.monotonic(),
            "camera_tof_range": 1.2,
            "object_x": 1.0,
            "object_y": 0.1,
        }

        with patch(
            "actions.track_action.smallbrain_diagnostics.update"
        ) as update:
            service._write_track_debug(
                "seed_building",
                sequence=42,
                target_xy=(0.51, 0.49),
                centered=True,
                seed_target="hand",
                camera_state=camera,
            )
            event = update.call_args.args[1]

        self.assertEqual(event["pipeline"], "seed_building")
        self.assertEqual(event["seed"]["target"], "hand")
        self.assertTrue(event["tof"]["valid"])
        self.assertTrue(event["centered"])


class TrackingLoopSupervisionTests(unittest.IsolatedAsyncioTestCase):

    def make_service(self, send_robot_command=None):
        service = TrackingService.__new__(TrackingService)
        service.active = True
        service.action_id = "tracking-session"
        service.target = "person"
        service._tracking_task = None
        service._tracked_person_id = 1
        service._visual_target_override = None
        service._follow_person_active = False
        service.stable_seeds = StableTargetSeedTracker()
        service.completion_future = asyncio.get_running_loop().create_future()
        service.send_robot_command = (
            send_robot_command
            or AsyncMock(return_value={"status": "accepted"})
        )
        service.person_tracking_service = SimpleNamespace(
            stop_target_tracking=Mock(),
        )
        service.csrt_tracker = SimpleNamespace(stop_tracking=Mock())
        return service

    async def test_loop_failure_stops_and_completes_tracking(self):
        service = self.make_service()

        async def failing_loop():
            raise RuntimeError("camera unavailable")

        task = asyncio.create_task(
            service._supervise_tracking_loop(
                failing_loop,
                service.action_id,
            ),
        )
        service._tracking_task = task
        await task

        self.assertFalse(service.active)
        service.send_robot_command.assert_awaited_once()
        self.assertEqual(
            service.send_robot_command.await_args.args[0]["command"],
            "stop_tracking",
        )
        result = service.completion_future.result()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason_code, "TRACKING_LOOP_ERROR")
        self.assertEqual(
            result.data["error"],
            "RuntimeError: camera unavailable",
        )

    async def test_loop_failure_clears_state_when_stop_command_fails(self):
        command = AsyncMock(side_effect=TimeoutError("robot unavailable"))
        service = self.make_service(command)

        async def failing_loop():
            raise ValueError("bad tracking frame")

        task = asyncio.create_task(
            service._supervise_tracking_loop(
                failing_loop,
                service.action_id,
            ),
        )
        service._tracking_task = task
        await task

        self.assertFalse(service.active)
        result = service.completion_future.result()
        self.assertEqual(result.reason_code, "TRACKING_LOOP_ERROR")
        self.assertEqual(
            result.data["cleanup_error"],
            "TimeoutError: robot unavailable",
        )


class MissingKeypointTrackingTests(unittest.IsolatedAsyncioTestCase):

    @staticmethod
    def person(*, ankle_x, knee_x, hip_x=None):
        keypoints = {
            "left_ankle": {
                "normalized_x": ankle_x,
                "normalized_y": 0.5,
            },
            "left_knee": {
                "normalized_x": knee_x,
                "normalized_y": 0.5,
            },
        }
        if hip_x is not None:
            keypoints["left_hip"] = {
                "normalized_x": hip_x,
                "normalized_y": 0.5,
            }
        return {
            "class": "person",
            "confidence": 0.9,
            "bbox": {
                "normalized_x1": 0.1,
                "normalized_y1": 0.1,
                "normalized_x2": 0.9,
                "normalized_y2": 0.9,
            },
            "keypoints": keypoints,
        }

    async def test_does_not_fall_back_after_prediction_while_next_is_missing(
        self,
    ):
        service = TrackingService.__new__(TrackingService)
        frames = [
            self.person(ankle_x=0.4, knee_x=0.5),
            self.person(ankle_x=0.3, knee_x=0.4),
            self.person(ankle_x=0.2, knee_x=0.3),
            self.person(ankle_x=0.3, knee_x=0.4, hip_x=0.7),
        ]

        class YoloFrames:
            def __init__(self):
                self.index = 0

            def detection_snapshot_with_frame(self):
                index = min(self.index, len(frames) - 1)
                self.index += 1
                if self.index == len(frames):
                    service.active = False
                captured_at = time.monotonic()
                return (
                    self.index,
                    [frames[index]],
                    None,
                    {
                        "source_captured_at": captured_at,
                        "inference_started_at": captured_at,
                        "inference_completed_at": captured_at,
                    },
                )

        class CommandSocket:
            def __init__(self):
                self.payloads = []

            async def send_json(self, payload):
                self.payloads.append(payload)

        class PersonTracks:
            def tracking_snapshot_with_time(self):
                index = min(max(service.yolo.index - 1, 0), len(frames) - 1)
                return (
                    service.yolo.index,
                    (SimpleNamespace(
                        track_id=1,
                        visible=True,
                        person=frames[index],
                    ),),
                    time.monotonic(),
                )

        service.active = True
        service.target = "person"
        service.action_id = "track-person"
        service.person_path = ["left_ankle", "left_knee", "left_hip"]
        service.person_path_index = 1
        service.person_tracking_service = PersonTracks()
        service._tracked_person_id = 1
        service._visual_target_override = None
        service._follow_person_active = False
        service.person_tracker_pub_socket = None
        service.stable_seeds = StableTargetSeedTracker()
        service.stable_threshold = 0.05
        service.yolo = YoloFrames()
        service.zmq_pub_socket = CommandSocket()

        await asyncio.wait_for(service._person_tracking_loop(), timeout=1.0)

        payloads = service.zmq_pub_socket.payloads
        self.assertEqual(len(payloads), 2)
        self.assertLess(payloads[0]["delta_pan_angle"], 0.0)
        self.assertLess(payloads[1]["delta_pan_angle"], 0.0)
        self.assertEqual(service.person_path_index, 2)

    async def test_skips_to_furthest_visible_keypoint_before_current_is_centered(
        self,
    ):
        service = TrackingService.__new__(TrackingService)
        person = self.person(ankle_x=0.2, knee_x=0.3, hip_x=0.8)

        class YoloFrame:
            def __init__(self):
                self.sequence = 0

            def detection_snapshot_with_frame(self):
                self.sequence += 1
                service.active = False
                captured_at = time.monotonic()
                return (
                    self.sequence,
                    [person],
                    None,
                    {
                        "source_captured_at": captured_at,
                        "inference_started_at": captured_at,
                        "inference_completed_at": captured_at,
                    },
                )

        class PersonTracks:
            def tracking_snapshot_with_time(self):
                return (
                    service.yolo.sequence,
                    (SimpleNamespace(
                        track_id=1,
                        visible=True,
                        person=person,
                    ),),
                    time.monotonic(),
                )

        service.active = True
        service.target = "person"
        service.action_id = "track-person"
        service.person_path = ["left_ankle", "left_knee", "left_hip"]
        service.person_path_index = 0
        service.person_tracking_service = PersonTracks()
        service._tracked_person_id = 1
        service._visual_target_override = None
        service._follow_person_active = False
        service.person_tracker_pub_socket = None
        service.stable_seeds = StableTargetSeedTracker()
        service.stable_threshold = 0.05
        service.yolo = YoloFrame()
        service.zmq_pub_socket = SimpleNamespace(send_json=AsyncMock())

        await asyncio.wait_for(service._person_tracking_loop(), timeout=1.0)

        self.assertEqual(service.person_path_index, 2)
        payload = service.zmq_pub_socket.send_json.await_args.args[0]
        self.assertLess(payload["delta_pan_angle"], 0.0)

    async def test_path_starting_at_hip_advances_to_bootstrap_torso_offset(
        self,
    ):
        service = TrackingService.__new__(TrackingService)
        person = {
            "class": "person",
            "confidence": 0.9,
            "bbox": {
                "normalized_x1": 0.1,
                "normalized_y1": 0.1,
                "normalized_x2": 0.9,
                "normalized_y2": 0.9,
            },
            "keypoints": {
                "left_hip": {
                    "normalized_x": 0.5,
                    "normalized_y": 0.5,
                },
            },
        }

        class YoloFrames:
            def __init__(self):
                self.sequence = 0

            def detection_snapshot_with_frame(self):
                self.sequence += 1
                if self.sequence == 2:
                    service.active = False
                captured_at = time.monotonic()
                return (
                    self.sequence,
                    [person],
                    None,
                    {
                        "source_captured_at": captured_at,
                        "inference_started_at": captured_at,
                        "inference_completed_at": captured_at,
                    },
                )

        class PersonTracks:
            def tracking_snapshot_with_time(self):
                return (
                    service.yolo.sequence,
                    (SimpleNamespace(
                        track_id=1,
                        visible=True,
                        person=person,
                    ),),
                    time.monotonic(),
                )

        service.active = True
        service.target = "person"
        service.action_id = "track-person"
        service.person_path = ["left_hip", "torso_center"]
        service.person_path_index = 0
        service.person_tracking_service = PersonTracks()
        service._tracked_person_id = 1
        service._visual_target_override = None
        service._follow_person_active = False
        service.person_tracker_pub_socket = None
        service.stable_seeds = StableTargetSeedTracker()
        service.stable_threshold = 0.05
        service.yolo = YoloFrames()
        service.zmq_pub_socket = AsyncMock()

        await asyncio.wait_for(service._person_tracking_loop(), timeout=1.0)

        self.assertEqual(service.person_path_index, 1)

    async def test_final_keypoint_loss_restarts_from_best_visible_keypoint(
        self,
    ):
        service = TrackingService.__new__(TrackingService)
        person = {
            "class": "person",
            "confidence": 0.9,
            "bbox": {
                "normalized_x1": 0.1,
                "normalized_y1": 0.1,
                "normalized_x2": 0.9,
                "normalized_y2": 0.9,
            },
            "keypoints": {
                "left_shoulder": {
                    "normalized_x": 0.8,
                    "normalized_y": 0.4,
                },
                "left_ankle": {
                    "normalized_x": 0.2,
                    "normalized_y": 0.8,
                },
            },
        }

        class YoloFrame:
            def __init__(self):
                self.sequence = 0

            def detection_snapshot_with_frame(self):
                self.sequence += 1
                service.active = False
                captured_at = time.monotonic()
                return (
                    self.sequence,
                    [person],
                    None,
                    {
                        "source_captured_at": captured_at,
                        "inference_started_at": captured_at,
                        "inference_completed_at": captured_at,
                    },
                )

        class PersonTracks:
            def tracking_snapshot_with_time(self):
                return (
                    service.yolo.sequence,
                    (SimpleNamespace(
                        track_id=1,
                        visible=True,
                        person=person,
                    ),),
                    time.monotonic(),
                )

        service.active = True
        service.target = "person"
        service.action_id = "track-person"
        service.person_path = [
            "left_ankle", "left_knee", "left_hip", "torso_center",
        ]
        service.person_path_index = 3
        service.person_tracking_service = PersonTracks()
        service._tracked_person_id = 1
        service._visual_target_override = None
        service._follow_person_active = False
        service.person_tracker_pub_socket = None
        service.stable_seeds = StableTargetSeedTracker()
        service.stable_threshold = 0.05
        service.PERSON_KEYPOINT_LOSS_TOLERANCE_SECONDS = 0.0
        service.yolo = YoloFrame()
        service.zmq_pub_socket = SimpleNamespace(send_json=AsyncMock())

        await asyncio.wait_for(service._person_tracking_loop(), timeout=1.0)

        self.assertEqual(
            service.person_path,
            ["left_shoulder", "torso_center"],
        )
        self.assertEqual(service.person_path_index, 0)
        service.zmq_pub_socket.send_json.assert_awaited_once()

    async def test_final_keypoint_loss_does_not_restart_during_tolerance(self):
        service = TrackingService.__new__(TrackingService)
        person = self.person(ankle_x=0.2, knee_x=0.3)

        class YoloFrame:
            def __init__(self):
                self.sequence = 0

            def detection_snapshot_with_frame(self):
                self.sequence += 1
                service.active = False
                captured_at = time.monotonic()
                return (
                    self.sequence,
                    [person],
                    None,
                    {
                        "source_captured_at": captured_at,
                        "inference_started_at": captured_at,
                        "inference_completed_at": captured_at,
                    },
                )

        class PersonTracks:
            def tracking_snapshot_with_time(self):
                return (
                    service.yolo.sequence,
                    (SimpleNamespace(
                        track_id=1,
                        visible=True,
                        person=person,
                    ),),
                    time.monotonic(),
                )

        service.active = True
        service.target = "person"
        service.action_id = "track-person"
        service.person_path = [
            "left_ankle", "left_knee", "left_hip", "torso_center",
        ]
        service.person_path_index = 3
        service.person_tracking_service = PersonTracks()
        service._tracked_person_id = 1
        service._visual_target_override = None
        service._follow_person_active = False
        service.person_tracker_pub_socket = None
        service.stable_seeds = StableTargetSeedTracker()
        service.stable_threshold = 0.05
        service.yolo = YoloFrame()
        service.zmq_pub_socket = SimpleNamespace(send_json=AsyncMock())

        await asyncio.wait_for(service._person_tracking_loop(), timeout=1.0)

        self.assertEqual(service.person_path_index, 3)
        service.zmq_pub_socket.send_json.assert_not_awaited()

    async def test_prediction_timeout_restarts_from_best_visible_keypoint(
        self,
    ):
        service = TrackingService.__new__(TrackingService)
        person = self.person(ankle_x=0.4, knee_x=0.5)

        class YoloFrames:
            def __init__(self):
                self.sequence = 0

            def detection_snapshot_with_frame(self):
                self.sequence += 1
                if self.sequence == 3:
                    service.active = False
                captured_at = time.monotonic()
                return (
                    self.sequence,
                    [person],
                    None,
                    {
                        "source_captured_at": captured_at,
                        "inference_started_at": captured_at,
                        "inference_completed_at": captured_at,
                    },
                )

        class PersonTracks:
            def tracking_snapshot_with_time(self):
                return (
                    service.yolo.sequence,
                    (SimpleNamespace(
                        track_id=1,
                        visible=True,
                        person=person,
                    ),),
                    time.monotonic(),
                )

        service.active = True
        service.target = "person"
        service.action_id = "track-person"
        service.person_path = [
            "left_ankle", "left_knee", "left_hip", "torso_center",
        ]
        service.person_path_index = 1
        service.person_tracking_service = PersonTracks()
        service._tracked_person_id = 1
        service._visual_target_override = None
        service._follow_person_active = False
        service.person_tracker_pub_socket = None
        service.stable_seeds = StableTargetSeedTracker()
        service.stable_threshold = 0.05
        service.PERSON_KEYPOINT_LOSS_TOLERANCE_SECONDS = 0.0
        service.yolo = YoloFrames()
        service.zmq_pub_socket = SimpleNamespace(send_json=AsyncMock())

        await asyncio.wait_for(service._person_tracking_loop(), timeout=1.0)

        self.assertEqual(
            service.person_path,
            ["left_knee", "left_hip", "torso_center"],
        )
        self.assertEqual(service.person_path_index, 0)


if __name__ == "__main__":
    unittest.main()
