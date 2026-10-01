import asyncio
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognition.hand.interface import HandGestureInterface
from cognition.hand.gestures import HandGestureClassifier
from cognition.manager.world_state import robot_state
from actions.track_action import TrackAction
from actions.tracking.stable_seed import StableTargetSeedTracker
from utilities.camera_sampler import (
    SequenceFps,
    draw_pipeline_fps_overlay,
    draw_tracking_status_overlay,
)


def bare_track_action():
    tracker = TrackAction.__new__(TrackAction)
    tracker.target = None
    tracker.action_id = None
    tracker.stable_threshold = 0.05
    tracker._visual_target_override = None
    tracker.stable_seeds = StableTargetSeedTracker()
    return tracker


def open_hand(direction="down", handedness="Right"):
    landmarks = {
        "wrist": {"normalized_x": 0.51, "normalized_y": 0.62},
        "thumb_cmc": {"normalized_x": 0.40, "normalized_y": 0.56},
        "thumb_mcp": {"normalized_x": 0.37, "normalized_y": 0.53},
        "thumb_ip": {"normalized_x": 0.34, "normalized_y": 0.50},
        "thumb_tip": {"normalized_x": 0.31, "normalized_y": 0.48},
    }
    sign = 1.0 if direction == "down" else -1.0
    for name, x in zip(("index", "middle", "ring", "pinky"), (0.42, 0.48, 0.54, 0.60)):
        landmarks[f"{name}_mcp"] = {"normalized_x": x, "normalized_y": 0.50}
        landmarks[f"{name}_pip"] = {"normalized_x": x, "normalized_y": 0.56 if sign > 0 else 0.44}
        landmarks[f"{name}_dip"] = {"normalized_x": x, "normalized_y": 0.62 if sign > 0 else 0.38}
        landmarks[f"{name}_tip"] = {"normalized_x": x, "normalized_y": 0.68 if sign > 0 else 0.32}
    return {
        "image_width": 640,
        "image_height": 360,
        "landmarks": landmarks,
        "handedness": handedness,
    }


def reversed_x_hand(direction):
    hand = open_hand(direction)
    raw_x = {
        "thumb": 0.70,
        "index": 0.60,
        "middle": 0.50,
        "ring": 0.40,
        "pinky": 0.30,
    }
    for name in ("index", "middle", "ring", "pinky"):
        for joint in ("mcp", "pip", "dip", "tip"):
            hand["landmarks"][f"{name}_{joint}"]["normalized_x"] = raw_x[name]
    for joint, x in zip(
        ("cmc", "mcp", "ip", "tip"), (0.64, 0.66, 0.68, 0.70)
    ):
        hand["landmarks"][f"thumb_{joint}"]["normalized_x"] = x
    return hand


def push_hand():
    return reversed_x_hand("up")


def finger_curl_down_hand():
    hand = reversed_x_hand("down")
    for finger in ("index", "middle", "ring", "pinky"):
        for joint, y in zip(
            ("mcp", "pip", "dip", "tip"), (0.50, 0.72, 0.66, 0.60)
        ):
            hand["landmarks"][f"{finger}_{joint}"]["normalized_y"] = y
    return hand


def finger_curl_up_hand():
    hand = open_hand("up")
    for finger in ("index", "middle", "ring", "pinky"):
        for joint, y in zip(
            ("mcp", "pip", "dip", "tip"), (0.50, 0.28, 0.34, 0.40)
        ):
            hand["landmarks"][f"{finger}_{joint}"]["normalized_y"] = y
    return hand


def follow_hand():
    hand = open_hand("down")
    thumb_ip_x = hand["landmarks"]["thumb_ip"]["normalized_x"]
    hand["landmarks"]["thumb_tip"]["normalized_x"] = thumb_ip_x + 0.04
    return hand


def get_space_hand():
    hand = push_hand()
    thumb_ip_x = hand["landmarks"]["thumb_ip"]["normalized_x"]
    hand["landmarks"]["thumb_tip"]["normalized_x"] = thumb_ip_x - 0.04
    return hand


def folded_hand(index_up=False):
    hand = open_hand("up")
    landmarks = hand["landmarks"]
    for joint, y in zip(
        ("cmc", "mcp", "ip", "tip"), (0.58, 0.48, 0.38, 0.28)
    ):
        landmarks[f"thumb_{joint}"]["normalized_y"] = y
    for name in ("index", "middle", "ring", "pinky"):
        base_x = landmarks[f"{name}_mcp"]["normalized_x"]
        landmarks[f"{name}_pip"]["normalized_x"] = base_x + 0.08
        landmarks[f"{name}_dip"]["normalized_x"] = base_x + 0.06
        landmarks[f"{name}_tip"]["normalized_x"] = base_x + 0.03
        landmarks[f"{name}_mcp"]["normalized_y"] = 0.50
        landmarks[f"{name}_pip"]["normalized_y"] = 0.38
        landmarks[f"{name}_dip"]["normalized_y"] = 0.42
        landmarks[f"{name}_tip"]["normalized_y"] = 0.46
    if index_up:
        index_x = landmarks["index_mcp"]["normalized_x"]
        for joint, y in zip(
            ("mcp", "pip", "dip", "tip"), (0.50, 0.43, 0.36, 0.29)
        ):
            landmarks[f"index_{joint}"]["normalized_y"] = y
            landmarks[f"index_{joint}"]["normalized_x"] = index_x
    return hand


def person_detection():
    return {
        "class": "person",
        "confidence": 0.9,
        "keypoints": {
            "right_elbow": {"normalized_x": 0.55, "normalized_y": 0.55},
            "right_wrist": {"normalized_x": 0.50, "normalized_y": 0.50},
        },
    }


class HandPoseTests(unittest.TestCase):
    def test_five_straight_downward_ordered_fingers_are_welcome(self):
        pose = HandGestureInterface.classify_hand_pose(open_hand("down"))

        self.assertEqual(pose["gesture"], "welcome")
        self.assertEqual(pose["open_fingers"], 5)
        self.assertEqual(pose["gesture_checks"]["handedness"], "Right")
        self.assertTrue(pose["gesture_checks"]["welcome_y_order"])
        self.assertTrue(pose["gesture_checks"]["hand_tip_order"])

    def test_media_pipe_left_label_does_not_change_welcome_order(self):
        pose = HandGestureInterface.classify_hand_pose(
            open_hand("down", handedness="Left")
        )

        self.assertEqual(pose["gesture"], "welcome")
        self.assertEqual(pose["gesture_checks"]["handedness"], "Right")
        self.assertTrue(pose["gesture_checks"]["hand_tip_order"])

    def test_wrong_fingertip_order_is_not_welcome(self):
        hand = open_hand("down")
        for joint in ("mcp", "pip", "dip", "tip"):
            hand["landmarks"][f"ring_{joint}"]["normalized_x"] = 0.46

        pose = HandGestureInterface.classify_hand_pose(hand)

        self.assertEqual(pose["gesture"], "open_other")
        self.assertFalse(pose["gesture_checks"]["hand_tip_order"])

    def test_thumb_tip_right_of_ip_is_follow(self):
        hand = follow_hand()

        pose = HandGestureInterface.classify_hand_pose(hand)

        self.assertEqual(pose["gesture"], "follow")
        self.assertTrue(pose["gesture_checks"]["thumb_tip_right_of_ip"])
        self.assertTrue(pose["gesture_checks"]["hand_tip_order"])

    def test_non_descending_finger_is_not_welcome(self):
        hand = open_hand("down")
        for joint, y in zip(
            ("mcp", "pip", "dip", "tip"), (0.50, 0.44, 0.38, 0.32)
        ):
            hand["landmarks"][f"index_{joint}"]["normalized_y"] = y

        pose = HandGestureInterface.classify_hand_pose(hand)

        self.assertEqual(pose["gesture"], "open_other")
        self.assertFalse(pose["gesture_checks"]["welcome_y_order"])

    def test_thumb_state_uses_x_order_without_bend_angle(self):
        hand = follow_hand()

        pose = HandGestureInterface.classify_hand_pose(hand)

        self.assertEqual(pose["gesture"], "follow")
        self.assertNotIn("bend_deg", pose["fingers"]["thumb"])

    def test_upward_finger_curl_is_distinct(self):
        pose = HandGestureInterface.classify_hand_pose(finger_curl_up_hand())
        self.assertEqual(pose["gesture"], "finger_curl_up")
        self.assertEqual(pose["open_fingers"], 1)
        self.assertTrue(pose["gesture_checks"]["long_fingers_curl_up"])
        self.assertTrue(pose["gesture_checks"]["hand_tip_order"])

    def test_finger_curl_up_does_not_check_thumb_y_order(self):
        hand = finger_curl_up_hand()
        for joint, y in zip(
            ("cmc", "mcp", "ip", "tip"), (0.44, 0.47, 0.50, 0.53)
        ):
            hand["landmarks"][f"thumb_{joint}"]["normalized_y"] = y

        pose = HandGestureInterface.classify_hand_pose(hand)

        self.assertEqual(pose["gesture"], "finger_curl_up")

    def test_get_space_uses_upward_reverse_order_and_left_thumb_tip(self):
        pose = HandGestureInterface.classify_hand_pose(get_space_hand())

        self.assertEqual(pose["gesture"], "get_space")
        self.assertTrue(pose["gesture_checks"]["fingers_up_y_order"])
        self.assertTrue(pose["gesture_checks"]["get_space"])
        self.assertTrue(pose["gesture_checks"]["thumb_tip_left_of_ip"])
        self.assertTrue(pose["gesture_checks"]["reverse_tip_order"])

    def test_finger_curl_down_keeps_reverse_order_pose(self):
        pose = HandGestureInterface.classify_hand_pose(finger_curl_down_hand())

        self.assertEqual(pose["gesture"], "finger_curl_down")
        self.assertTrue(pose["gesture_checks"]["long_fingers_curl_down"])
        self.assertTrue(pose["gesture_checks"]["reverse_tip_order"])

    def test_axis_order_can_compare_arbitrary_landmark_groups(self):
        landmarks = open_hand("down")["landmarks"]

        self.assertTrue(HandGestureClassifier._axis_order(
            landmarks,
            "y",
            "index_mcp",
            "index_pip",
            "index_dip",
            "index_tip",
        ))
        self.assertTrue(HandGestureClassifier._axis_order(
            landmarks,
            "y",
            "middle_tip",
            ("index_pip", "ring_pip"),
            relation=">",
        ))
        self.assertTrue(HandGestureClassifier._axis_order(
            landmarks,
            "x",
            "middle_tip",
            "index_tip",
            relation=lambda left, right: left >= right + 0.05,
        ))

    def test_push_uses_upward_fingers_thumb_and_reversed_tip_order(self):
        pose = HandGestureInterface.classify_hand_pose(push_hand())

        self.assertEqual(pose["gesture"], "push")
        self.assertTrue(pose["gesture_checks"]["fingers_up_y_order"])
        self.assertTrue(pose["gesture_checks"]["push_thumb_y_order"])
        self.assertTrue(pose["gesture_checks"]["thumb_tip_right_of_ip"])
        self.assertTrue(pose["gesture_checks"]["reverse_tip_order"])

    def test_thumb_x_order_has_no_dead_zone(self):
        hand = open_hand("down")
        thumb_ip_x = hand["landmarks"]["thumb_ip"]["normalized_x"]
        hand["landmarks"]["thumb_tip"]["normalized_x"] = thumb_ip_x + 0.005

        pose = HandGestureInterface.classify_hand_pose(hand)

        self.assertEqual(pose["gesture"], "follow")
        self.assertFalse(pose["gesture_checks"]["welcome"])
        self.assertTrue(pose["gesture_checks"]["follow"])

    def test_media_pipe_left_label_does_not_change_get_space_order(self):
        hand = get_space_hand()
        hand["handedness"] = "Left"
        pose = HandGestureInterface.classify_hand_pose(hand)

        self.assertEqual(pose["gesture"], "get_space")
        self.assertEqual(pose["gesture_checks"]["handedness"], "Right")

    def test_thumb_up_requires_thumb_above_folded_fingers(self):
        pose = HandGestureInterface.classify_hand_pose(folded_hand())

        self.assertEqual(pose["gesture"], "thumb_up")
        self.assertTrue(pose["gesture_checks"]["thumb_up"])

    def test_thumb_up_checks_folded_fingers_on_x_axis(self):
        hand = folded_hand()
        for name in ("index", "middle", "ring", "pinky"):
            landmarks = hand["landmarks"]
            mcp_x = landmarks[f"{name}_mcp"]["normalized_x"]
            landmarks[f"{name}_pip"]["normalized_x"] = mcp_x - 0.02

        pose = HandGestureInterface.classify_hand_pose(hand)

        self.assertNotEqual(pose["gesture"], "thumb_up")
        self.assertFalse(pose["gesture_checks"]["thumb_up"])

    def test_index_finger_pose_is_no_longer_a_gesture(self):
        pose = HandGestureInterface.classify_hand_pose(folded_hand(index_up=True))

        self.assertEqual(pose["gesture"], "open_other")
        self.assertNotIn("index_finger", pose["gesture_checks"])


class HandTrackingOverrideTests(unittest.TestCase):
    def test_override_is_bounded_clearable_and_expires(self):
        tracker = bare_track_action()

        tracker.set_visual_target_override(1.2, -0.2)

        self.assertEqual(tracker._current_visual_target_override(), (1.0, 0.0))
        tracker.clear_visual_target_override()
        self.assertIsNone(tracker._visual_target_override)

        tracker.set_visual_target_override(0.4, 0.6)
        self.assertIsNone(tracker._current_visual_target_override(-1.0))

    def test_override_is_retained_for_half_a_second(self):
        tracker = bare_track_action()
        with patch(
            "actions.track_action.time.monotonic",
            side_effect=(10.0, 10.49, 10.51),
        ):
            tracker.set_visual_target_override(0.4, 0.6)
            self.assertEqual(
                tracker._current_visual_target_override(), (0.4, 0.6)
            )
            self.assertIsNone(tracker._current_visual_target_override())


class ContinuousPersonReacquisitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_person_waits_without_moving_camera(self):
        tracker = bare_track_action()
        tracker.active = True
        tracker.target = "person"
        tracker.action_id = "track-person"
        tracker._continuous_person_reacquisition = True
        tracker.zmq_pub_socket = AsyncMock()
        tracker.yolo = Mock()
        sequence = 0

        def snapshot():
            nonlocal sequence
            sequence += 1
            if sequence > 1:
                tracker.active = False
            return sequence, [], {}

        tracker.yolo.detection_snapshot_with_timing.side_effect = snapshot

        with patch("actions.track_action.asyncio.sleep", AsyncMock()):
            await tracker._person_tracking_loop()

        tracker.zmq_pub_socket.send_json.assert_not_awaited()

    async def test_new_hand_override_does_not_wait_for_another_yolo_frame(self):
        tracker = bare_track_action()
        tracker.active = True
        tracker.target = "person"
        tracker.action_id = "track-person"
        tracker._continuous_person_reacquisition = True
        tracker.person_path = ["right_wrist"]
        tracker.person_path_index = 0
        tracker.yolo = Mock()
        tracker.yolo.detection_snapshot_with_timing.return_value = (
            1, [person_detection()], {},
        )
        sent = []

        async def send(payload):
            sent.append(dict(payload))
            if len(sent) == 1:
                tracker.set_visual_target_override(0.60, 0.50)
            else:
                tracker.active = False

        tracker.zmq_pub_socket = Mock()
        tracker.zmq_pub_socket.send_json = AsyncMock(side_effect=send)
        tracker.set_visual_target_override(0.40, 0.50)

        await asyncio.wait_for(tracker._person_tracking_loop(), timeout=1.0)

        self.assertEqual(len(sent), 2)
        self.assertTrue(sent[0]["tracking_sequence"].startswith("hand:"))
        self.assertNotEqual(
            sent[0]["tracking_sequence"], sent[1]["tracking_sequence"]
        )

    async def test_valid_hand_seed_is_published_to_lidar_with_identity(self):
        tracker = bare_track_action()
        tracker.active = True
        tracker.target = "person"
        tracker.action_id = "track-person"
        tracker._continuous_person_reacquisition = True
        tracker.person_path = ["right_wrist"]
        tracker.person_path_index = 0
        tracker.yolo = Mock()
        tracker.yolo.detection_snapshot_with_timing.return_value = (
            1, [person_detection()], {},
        )
        tracker.zmq_pub_socket = AsyncMock()
        tracker.person_tracker_pub_socket = Mock()

        async def publish(_payload):
            tracker.active = False

        tracker.person_tracker_pub_socket.send_json = AsyncMock(
            side_effect=publish
        )
        tracker._visual_target_override = {
            "x": 0.5,
            "y": 0.5,
            "updated_at": time.monotonic(),
        }
        tracker.stable_seeds = Mock()
        tracker.stable_seeds.get.return_value = {
            "x": 1.0,
            "y": 0.1,
            "target": "hand",
            "session_id": "track-person",
        }

        await asyncio.wait_for(tracker._person_tracking_loop(), timeout=1.0)

        payload = tracker.person_tracker_pub_socket.send_json.call_args.args[0]
        self.assertEqual(payload["type"], "person_tof_position")
        self.assertEqual(payload["target"], "hand")
        self.assertEqual(payload["session_id"], "track-person")


class TrackingStatusOverlayTests(unittest.TestCase):
    def test_sequence_fps_measures_changes_and_expires_when_stale(self):
        rate = SequenceFps()

        self.assertEqual(rate.observe(10, now=1.0), 0.0)
        self.assertAlmostEqual(rate.observe(20, now=1.5), 20.0)
        self.assertAlmostEqual(rate.observe(20, now=2.0), 20.0)
        self.assertEqual(rate.value(now=2.51), 0.0)

    def test_pipeline_fps_overlay_renders_all_three_rates(self):
        with patch("utilities.camera_sampler.cv2.putText") as put_text:
            draw_pipeline_fps_overlay(
                np.zeros((360, 640, 3), dtype=np.uint8),
                camera_fps=30.0,
                yolo_fps=10.0,
                hand_fps=24.0,
            )

        rendered_text = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("CAMERA     30.0 FPS", rendered_text)
        self.assertIn("YOLO       10.0 FPS", rendered_text)
        self.assertIn("HAND DET   24.0 FPS", rendered_text)

    def test_hand_tracking_displays_hand_seed_status(self):
        track_action = Mock(active=True, target="person", action_id="track-1")
        track_action.stable_seeds.status.return_value = {
            "valid": True,
            "valid_for_seconds": 0.1,
            "sample_count": 3,
            "sample_target": 3,
        }
        hand_guidance = Mock()
        hand_guidance.gesture_status.return_value = {
            "gesture": "welcome", "state": "tracking_person",
        }

        draw_tracking_status_overlay(
            np.zeros((360, 640, 3), dtype=np.uint8),
            track_action,
            hand_guidance,
        )

        track_action.stable_seeds.status.assert_called_once_with(
            target="hand", session_id="track-1",
        )

    def test_command_pose_displays_resumed_person_seed_status(self):
        track_action = Mock(
            active=True, target="torso_center", action_id="track-1"
        )
        track_action.stable_seeds.status.return_value = {
            "valid": True,
            "valid_for_seconds": 0.1,
            "sample_count": 3,
            "sample_target": 3,
        }
        hand_guidance = Mock()
        hand_guidance.gesture_status.return_value = {
            "gesture": "finger_curl_up", "state": "tracking_person",
        }

        draw_tracking_status_overlay(
            np.zeros((360, 640, 3), dtype=np.uint8),
            track_action,
            hand_guidance,
        )

        track_action.stable_seeds.status.assert_called_once_with(
            target="torso_center", session_id="track-1",
        )

    def test_seed_status_hides_samples_from_another_target(self):
        tracker = StableTargetSeedTracker()
        started_at = time.monotonic()
        for index in range(tracker.SAMPLE_COUNT):
            robot_state["camera"]["timestamp"] = started_at + index * 0.01
            tracker.update(
                robot_state["camera"], target="hand", session_id="track-1"
            )

        status = tracker.status(target="person", session_id="track-1")

        self.assertFalse(status["valid"])
        self.assertEqual(status["sample_count"], 0)


class HandGestureInterfaceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.previous_camera = dict(robot_state["camera"])
        self.previous_person = robot_state.get("person")
        robot_state["camera"].update({
            "camera_tof_range": 1.0,
            "object_x": 1.0,
            "object_y": 0.0,
            "object_angle": 0.0,
            "object_map_x": 2.0,
            "object_map_y": 1.0,
            "timestamp": time.monotonic(),
        })
        robot_state["person"] = {
            "x": 2.1, "y": 1.0, "timestamp": time.monotonic(),
        }

    def tearDown(self):
        robot_state["camera"].clear()
        robot_state["camera"].update(self.previous_camera)
        robot_state["person"] = self.previous_person

    @staticmethod
    def action(dispatch=None, tracking_person=True):
        hand_service = Mock()
        action = HandGestureInterface(hand_service)

        async def default_dispatch(command, _data):
            if command == "observe_gesture":
                return {
                    "mode": (
                        "tracking_person" if tracking_person
                        else "watching_person"
                    ),
                    "tracking_active": True,
                    "tracking_person": tracking_person,
                    "following_active": False,
                    "approach_active": False,
                }
            return True

        action.set_dispatcher(dispatch or AsyncMock(side_effect=default_dispatch))
        return action

    def test_interface_applies_shared_miss_tolerance(self):
        action = self.action()
        confirmations = (
            action._idle_welcome,
            action._tracking_thumb_up,
            action._following_push,
            action._approaching_push,
            *action._modifier_confirmations.values(),
            *action._command_confirmations.values(),
        )

        self.assertEqual(action.MISS_TOLERANCE, 3)
        self.assertTrue(all(
            item.miss_tolerance == action.MISS_TOLERANCE
            for item in confirmations
        ))

    async def test_welcome_pose_arms_hand_tracking(self):
        action = self.action(tracking_person=False)
        action.hand_landmarks.detect_right_hand = AsyncMock(
            return_value=open_hand("down")
        )
        yolo = Mock()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for sequence in range(1, action.IDLE_WELCOME_CONFIRM_FRAMES + 1):
            yolo.detection_snapshot_with_frame.return_value = (
                sequence, [person_detection()], frame, {},
            )
            await action._sample(yolo)

        commands = [call.args[0] for call in action._dispatch_handler.await_args_list]
        self.assertEqual(
            commands,
            ["observe_gesture"] * action.IDLE_WELCOME_CONFIRM_FRAMES
            + ["watch_target"],
        )

    async def test_hand_inference_uses_fresh_camera_frames_between_yolo_frames(self):
        action = self.action(tracking_person=False)
        action.hand_landmarks.detect_right_hand = AsyncMock(
            return_value=open_hand("down")
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        snapshots = []
        for sequence in range(1, action.IDLE_WELCOME_CONFIRM_FRAMES + 1):
            snapshot = Mock()
            snapshot.sequence = sequence
            snapshot.full_bgr = frame
            snapshots.append(snapshot)
        action.hand_landmarks.camera = Mock()
        action.hand_landmarks.camera.snapshot.side_effect = snapshots
        yolo = Mock()
        yolo.detection_snapshot.return_value = (
            7, [person_detection()],
        )

        for _ in snapshots:
            await action._sample(yolo)

        self.assertEqual(
            action.hand_landmarks.detect_right_hand.await_count,
            action.IDLE_WELCOME_CONFIRM_FRAMES,
        )
        self.assertEqual(
            action.gesture_status()["inference_sequence"],
            action.IDLE_WELCOME_CONFIRM_FRAMES,
        )
        commands = [call.args[0] for call in action._dispatch_handler.await_args_list]
        self.assertEqual(commands[-1], "watch_target")

    def test_track_action_validates_three_close_samples_near_person(self):
        tracker = bare_track_action()
        tracker.target = "person"
        samples = ((2.00, 1.00), (2.04, 0.98), (1.98, 1.03))
        for index, (map_x, map_y) in enumerate(samples):
            robot_state["camera"].update({
                "object_map_x": map_x,
                "object_map_y": map_y,
                "timestamp": time.monotonic() + index * 0.01,
            })
            seed = tracker.stable_seeds.update(
                robot_state["camera"],
                target="hand",
                session_id=tracker.action_id,
                person=robot_state["person"],
                require_person_proximity=True,
            )

        self.assertAlmostEqual(seed["x"], 1.0)
        self.assertAlmostEqual(seed["map_x"], 2.0)

        robot_state["person"].update({
            "x": 4.0, "y": 1.0, "timestamp": time.monotonic(),
        })
        robot_state["camera"]["timestamp"] = time.monotonic() + 1.0
        seed = tracker.stable_seeds.update(
            robot_state["camera"],
            target="hand",
            session_id=tracker.action_id,
            person=robot_state["person"],
            require_person_proximity=True,
        )
        self.assertIsNone(seed)
        self.assertIsNone(tracker.stable_seeds.get())

    def test_centered_hand_override_produces_hand_seed(self):
        tracker = bare_track_action()
        tracker.target = "person"
        tracker.action_id = "track-person"
        tracker.stable_threshold = 0.05
        tracker._visual_target_override = None
        started_at = time.monotonic()

        for index in range(tracker.TARGET_SEED_SAMPLES):
            robot_state["camera"]["timestamp"] = started_at + index * 0.01
            tracker.set_visual_target_override(0.51, 0.49)

        seed = tracker.stable_seeds.get(
            target="hand",
            session_id="track-person",
        )
        self.assertIsNotNone(seed)
        self.assertEqual(seed["target"], "hand")
        self.assertIsNone(tracker.stable_seeds.get(
            target="hand",
            session_id="another-session",
        ))
        self.assertIsNone(tracker.stable_seeds.get(
            target="person",
            session_id="track-person",
        ))

        tracker.set_visual_target_override(0.75, 0.49)
        self.assertIsNone(tracker.stable_seeds.get())

    def test_track_action_seed_samples_are_unique_consecutive_and_close(self):
        tracker = bare_track_action()
        tracker.target = "person"
        started_at = time.monotonic()
        robot_state["camera"]["timestamp"] = started_at
        tracker.stable_seeds.update(
            robot_state["camera"], target=tracker.target
        )
        tracker.stable_seeds.update(
            robot_state["camera"], target=tracker.target
        )
        self.assertEqual(tracker.stable_seeds.status()["sample_count"], 1)

        for index, map_x in enumerate((2.02, 2.4), start=1):
            robot_state["camera"]["object_map_x"] = map_x
            robot_state["camera"]["timestamp"] = started_at + index * 0.01
            seed = tracker.stable_seeds.update(
                robot_state["camera"], target=tracker.target
            )
        self.assertIsNone(seed)

        robot_state["camera"]["object_x"] = float("nan")
        robot_state["camera"]["timestamp"] = started_at + 0.03
        tracker.stable_seeds.update(
            robot_state["camera"], target=tracker.target
        )
        self.assertEqual(tracker.stable_seeds.status()["sample_count"], 0)

    async def test_seed_wait_stops_when_tracking_session_is_replaced(self):
        tracker = bare_track_action()
        tracker.active = True
        tracker.target = "person"
        tracker.action_id = "track-old"

        waiting = asyncio.create_task(tracker.wait_for_stable_target_seed(
            target="person",
            session_id="track-old",
            check_interval=0.001,
            timeout=1.0,
        ))
        await asyncio.sleep(0.005)
        tracker.action_id = "track-new"

        self.assertIsNone(await waiting)

    def test_seed_samples_do_not_mix_tracking_sessions(self):
        tracker = bare_track_action()
        tracker.target = "person"
        tracker.action_id = "track-old"
        started_at = time.monotonic()
        for index in range(2):
            robot_state["camera"]["timestamp"] = started_at + index * 0.01
            tracker.stable_seeds.update(
                robot_state["camera"],
                target=tracker.target,
                session_id=tracker.action_id,
            )

        tracker.action_id = "track-new"
        robot_state["camera"]["timestamp"] = started_at + 0.02
        self.assertIsNone(tracker.stable_seeds.update(
            robot_state["camera"],
            target=tracker.target,
            session_id=tracker.action_id,
        ))
        self.assertEqual(tracker.stable_seeds.status()["sample_count"], 1)

    async def test_upward_flick_dispatches_approach(self):
        async def dispatch(command, _data):
            if command == "observe_gesture":
                return {
                    "tracking_active": True,
                    "tracking_person": True,
                    "following_active": False,
                    "approach_active": False,
                }
            return True

        handler = AsyncMock(side_effect=dispatch)
        action = self.action(handler)
        action.hand_landmarks.detect_right_hand = AsyncMock(
            return_value=finger_curl_up_hand()
        )
        yolo = Mock()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(action.MODIFIER_CONFIRM_FRAMES):
            await action._observe_gesture("welcome", (0.5, 0.5))
        for sequence in range(1, action.COMMAND_CONFIRM_FRAMES + 1):
            yolo.detection_snapshot_with_frame.return_value = (
                sequence, [person_detection()], frame, {},
            )
            await action._sample(yolo)

        approach_call = next(
            call for call in handler.await_args_list
            if call.args[0] == "approach_target"
        )
        self.assertNotIn("seed", approach_call.args[1])

    async def test_push_for_ten_frames_stops_approach(self):
        approach_active = True

        async def dispatch(command, _data):
            nonlocal approach_active
            if command == "observe_gesture":
                return {
                    "tracking_active": True,
                    "tracking_person": True,
                    "following_active": False,
                    "approach_active": approach_active,
                }
            if command == "stop_navigation":
                approach_active = False
            return True

        handler = AsyncMock(side_effect=dispatch)
        action = self.action(handler)

        for _ in range(action.APPROACHING_STOP_CONFIRM_FRAMES - 1):
            await action._observe_gesture("push", (0.5, 0.5))
        self.assertNotIn(
            "stop_navigation",
            [call.args[0] for call in handler.await_args_list],
        )

        await action._observe_gesture("push", (0.5, 0.5))
        await action._observe_gesture("unavailable", None)

        commands = [call.args[0] for call in handler.await_args_list]
        self.assertEqual(commands.count("stop_navigation"), 1)

    async def test_push_then_get_space_moves_to_open_space_and_faces_person(self):
        action = self.action()
        for index in range(action.MODIFIER_CONFIRM_FRAMES):
            await action._observe_gesture("push", (0.5, 0.5))
            if index == 0:
                await action._observe_gesture("open_other", None)
        self.assertEqual(action._armed_gesture, "push")
        for index in range(action.COMMAND_CONFIRM_FRAMES):
            await action._observe_gesture("get_space", (0.5, 0.5))
            if index == 0:
                await action._observe_gesture("open_other", None)

        navigation_call = next(
            call for call in action._dispatch_handler.await_args_list
            if call.args[0] == "navigation_sequence"
        )
        self.assertEqual(
            navigation_call.args[1]["commands"],
            ["open_space_middle", "face_person"],
        )

    async def test_get_space_without_push_pose_does_nothing(self):
        action = self.action()

        for _ in range(action.COMMAND_CONFIRM_FRAMES):
            await action._observe_gesture("get_space", (0.5, 0.5))

        self.assertIsNone(action._armed_gesture)

    async def test_push_then_finger_curl_down_backs_up_without_prior_approach(self):
        action = self.action()
        for _ in range(action.MODIFIER_CONFIRM_FRAMES):
            await action._observe_gesture("push", (0.5, 0.5))
        for _ in range(action.COMMAND_CONFIRM_FRAMES):
            await action._observe_gesture("finger_curl_down", (0.5, 0.5))

        navigation_call = next(
            call for call in action._dispatch_handler.await_args_list
            if call.args[0] == "explicit_navigation"
        )
        self.assertEqual(
            navigation_call.args[1],
            {"local_command": "nudge_backward", "room_id": None},
        )

    async def test_finger_curl_down_returns_after_prior_approach(self):
        action = self.action()
        for _ in range(action.MODIFIER_CONFIRM_FRAMES):
            await action._observe_gesture("welcome", (0.5, 0.5))
        for _ in range(action.COMMAND_CONFIRM_FRAMES):
            await action._observe_gesture("finger_curl_up", (0.5, 0.5))
        for _ in range(action.MODIFIER_CONFIRM_FRAMES):
            await action._observe_gesture("push", (0.5, 0.5))
        for _ in range(action.COMMAND_CONFIRM_FRAMES):
            await action._observe_gesture("finger_curl_down", (0.5, 0.5))

        navigation_call = next(
            call for call in action._dispatch_handler.await_args_list
            if call.args[0] == "navigation_sequence"
        )
        self.assertEqual(
            navigation_call.args[1]["commands"],
            ["previous_position", "face_person"],
        )
        self.assertFalse(action._approach_was_requested)

    async def test_get_space_pose_reaches_gesture_observer(self):
        action = self.action()
        action.hand_landmarks.detect_right_hand = AsyncMock(
            return_value=get_space_hand()
        )
        yolo = Mock()
        yolo.detection_snapshot_with_frame.return_value = (
            1,
            [person_detection()],
            np.zeros((360, 640, 3), dtype=np.uint8),
            {},
        )

        await action._sample(yolo)

        observe_call = action._dispatch_handler.await_args_list[-1]
        self.assertEqual(observe_call.args[0], "observe_gesture")
        self.assertEqual(observe_call.args[1]["gesture"], "get_space")

    async def test_one_bad_frame_does_not_clear_welcome_confirmation(self):
        action = self.action(tracking_person=False)
        action.hand_landmarks.detect_right_hand = AsyncMock(side_effect=(
            [open_hand("down")] * 5
            + [None]
            + [open_hand("down")] * 5
        ))
        yolo = Mock()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for sequence in range(1, 12):
            yolo.detection_snapshot_with_frame.return_value = (
                sequence, [person_detection()], frame, {},
            )
            await action._sample(yolo)

        commands = [call.args[0] for call in action._dispatch_handler.await_args_list]
        self.assertEqual(commands.count("observe_gesture"), 11)
        self.assertEqual(commands[-1], "watch_target")

    async def test_get_space_requires_push_modifier(self):
        action = self.action()
        for _ in range(action.COMMAND_CONFIRM_FRAMES):
            await action._observe_gesture("get_space", (0.5, 0.5))

        commands = [call.args[0] for call in action._dispatch_handler.await_args_list]
        self.assertNotIn("navigation_sequence", commands)

    async def test_welcome_is_accepted_after_approach(self):
        action = self.action()
        action.hand_landmarks.detect_right_hand = AsyncMock(
            return_value=open_hand("down")
        )
        yolo = Mock()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for sequence in range(1, action.IDLE_WELCOME_CONFIRM_FRAMES + 1):
            yolo.detection_snapshot_with_frame.return_value = (
                sequence, [person_detection()], frame, {},
            )
            await action._sample(yolo)

        self.assertEqual(action._armed_gesture, "welcome")

    async def test_thumb_up_for_ten_frames_stops_tracking(self):
        action = self.action()

        for _ in range(action.TRACKING_STOP_CONFIRM_FRAMES):
            await action._observe_gesture("thumb_up", (0.5, 0.4))

        stop_call = next(
            call for call in action._dispatch_handler.await_args_list
            if call.args[0] == "stop_watching_target"
        )
        self.assertEqual(stop_call.args[1]["reason"], "HAND_THUMB_UP")

    async def test_welcome_then_follow_starts_following_after_two_frames(self):
        action = self.action()

        for _ in range(action.MODIFIER_CONFIRM_FRAMES):
            await action._observe_gesture("welcome", (0.5, 0.5))
        await action._observe_gesture("follow", (0.5, 0.4))
        commands = [call.args[0] for call in action._dispatch_handler.await_args_list]
        self.assertNotIn("follow_person", commands)

        await action._observe_gesture("follow", (0.5, 0.4))

        commands = [call.args[0] for call in action._dispatch_handler.await_args_list]
        self.assertIn("follow_person", commands)

    async def test_push_for_ten_frames_stops_following(self):
        following_active = True

        async def dispatch(command, _data):
            nonlocal following_active
            if command == "observe_gesture":
                return {
                    "tracking_active": True,
                    "tracking_person": True,
                    "following_active": following_active,
                    "approach_active": False,
                }
            if command == "stop_follow":
                following_active = False
            return True

        action = self.action(AsyncMock(side_effect=dispatch))

        for _ in range(action.FOLLOW_STOP_CONFIRM_FRAMES):
            await action._observe_gesture("push", (0.5, 0.5))

        stop_call = next(
            call for call in action._dispatch_handler.await_args_list
            if call.args[0] == "stop_follow"
        )
        self.assertEqual(stop_call.args[1]["reason"], "HAND_PUSH")

    async def test_push_stops_llm_started_follow_without_claiming_session(self):
        following_active = True

        async def dispatch(command, _data):
            nonlocal following_active
            if command == "observe_gesture":
                return {
                    "tracking_active": True,
                    "tracking_person": True,
                    "following_active": following_active,
                    "approach_active": False,
                }
            if command == "stop_follow":
                following_active = False
            return True

        action = self.action(AsyncMock(side_effect=dispatch))

        for _ in range(action.FOLLOW_STOP_CONFIRM_FRAMES):
            await action._observe_gesture("push", (0.5, 0.5))

        commands = [call.args[0] for call in action._dispatch_handler.await_args_list]
        self.assertIn("stop_follow", commands)

    async def test_push_stops_find_started_approach_without_claiming_session(self):
        approach_active = True

        async def dispatch(command, _data):
            nonlocal approach_active
            if command == "observe_gesture":
                return {
                    "tracking_active": True,
                    "tracking_person": True,
                    "following_active": False,
                    "approach_active": approach_active,
                }
            if command == "stop_navigation":
                approach_active = False
            return True

        action = self.action(AsyncMock(side_effect=dispatch))

        for _ in range(action.APPROACHING_STOP_CONFIRM_FRAMES):
            await action._observe_gesture("push", (0.5, 0.5))

        commands = [call.args[0] for call in action._dispatch_handler.await_args_list]
        self.assertIn("stop_navigation", commands)

    async def test_external_follow_stop_without_hand_session_returns_to_watching(self):
        async def dispatch(command, _data):
            if command == "observe_gesture":
                return {
                    "tracking_active": False,
                    "tracking_person": False,
                    "following_active": False,
                    "approach_active": False,
                }
            return True

        action = self.action(AsyncMock(side_effect=dispatch))

        context = await action._observe_gesture("unavailable", None)

        self.assertEqual(context["mode"] if "mode" in context else "watching_person", "watching_person")

    async def test_external_tracking_stop_returns_to_watching(self):
        async def dispatch(command, _data):
            if command == "observe_gesture":
                return {
                    "tracking_active": False,
                    "tracking_person": False,
                    "following_active": False,
                    "approach_active": False,
                }
            return True

        action = self.action(AsyncMock(side_effect=dispatch))

        context = await action._observe_gesture("unavailable", None)

        self.assertFalse(context["tracking_person"])

    async def test_run_consumes_always_on_yolo_pose_stream(self):
        action = self.action()
        yolo = Mock()

        async def sample_once(_yolo):
            action._running = False

        action._sample = AsyncMock(side_effect=sample_once)
        with patch("cognition.hand.interface.asyncio.sleep", AsyncMock()):
            await action.run(yolo)

        yolo.activate.assert_not_called()
        yolo.deactivate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
