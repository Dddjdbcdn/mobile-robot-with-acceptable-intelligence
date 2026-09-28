import asyncio
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognition.hand_guided_navigation import HandGuidedNavigation
from cognition.state import robot_state
from actions.track_action import TrackAction


def open_hand(direction="down"):
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
    return {"image_width": 640, "image_height": 360, "landmarks": landmarks}


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
    def test_relaxed_downward_open_hand_is_welcome(self):
        pose = HandGuidedNavigation.classify_hand_pose(open_hand("down"))
        self.assertEqual(pose["gesture"], "welcome")
        self.assertEqual(pose["open_fingers"], 4)
        self.assertGreater(pose["finger_axis_y"], 0.0)

    def test_upward_finger_flick_is_distinct(self):
        pose = HandGuidedNavigation.classify_hand_pose(open_hand("up"))
        self.assertEqual(pose["gesture"], "fingers_up")
        self.assertEqual(pose["open_fingers"], 4)
        self.assertLess(pose["finger_axis_y"], 0.0)


class HandTrackingOverrideTests(unittest.TestCase):
    def test_override_is_bounded_owned_and_expires(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker._visual_target_override = None

        tracker.set_visual_target_override("hand", 1.2, -0.2)

        self.assertEqual(tracker._current_visual_target_override(), (1.0, 0.0))
        tracker.clear_visual_target_override("someone-else")
        self.assertIsNotNone(tracker._visual_target_override)
        tracker.clear_visual_target_override("hand")
        self.assertIsNone(tracker._visual_target_override)

        tracker.set_visual_target_override("hand", 0.4, 0.6)
        self.assertIsNone(tracker._current_visual_target_override(-1.0))


class HandGuidedNavigationTests(unittest.IsolatedAsyncioTestCase):
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
    def action(send_robot_command=None):
        hand_service = Mock()
        track_action = Mock(active=True, target="person")
        track_action.set_visual_target_override = Mock()
        track_action.clear_visual_target_override = Mock()
        track_action.get_stable_target_seed = Mock(return_value=None)
        return HandGuidedNavigation(
            hand_service,
            track_action,
            send_robot_command or AsyncMock(return_value={"status": "accepted"}),
        )

    async def test_welcome_pose_arms_hand_tracking(self):
        action = self.action()
        action.hand_landmarks.detect_right_hand = AsyncMock(
            return_value=open_hand("down")
        )
        yolo = Mock()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for sequence in range(1, action.WELCOME_CONFIRM_FRAMES + 1):
            yolo.detection_snapshot_with_frame.return_value = (
                sequence, [person_detection()], frame, {},
            )
            await action._sample(yolo)

        self.assertTrue(action.active)
        self.assertEqual(action.state, "tracking_hand")
        action.track_action.set_visual_target_override.assert_called()

    def test_track_action_validates_three_close_samples_near_person(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker.target = "person"
        samples = ((2.00, 1.00), (2.04, 0.98), (1.98, 1.03))
        for index, (map_x, map_y) in enumerate(samples):
            robot_state["camera"].update({
                "object_map_x": map_x,
                "object_map_y": map_y,
                "timestamp": time.monotonic() + index * 0.01,
            })
            seed = tracker.update_stable_target_seed(
                "hand_guided_navigation",
                target="hand",
                require_person_proximity=True,
            )

        self.assertAlmostEqual(seed["x"], 1.0)
        self.assertAlmostEqual(seed["map_x"], 2.0)
        self.assertEqual(seed["owner"], "hand_guided_navigation")

        robot_state["person"].update({
            "x": 4.0, "y": 1.0, "timestamp": time.monotonic(),
        })
        robot_state["camera"]["timestamp"] = time.monotonic() + 1.0
        seed = tracker.update_stable_target_seed(
            "hand_guided_navigation",
            target="hand",
            require_person_proximity=True,
        )
        self.assertIsNone(seed)
        self.assertIsNone(
            tracker.get_stable_target_seed("hand_guided_navigation")
        )

    def test_centered_hand_override_produces_owned_stable_seed(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker.target = "person"
        tracker.action_id = "track-person"
        tracker.stable_threshold = 0.05
        tracker._visual_target_override = None
        started_at = time.monotonic()

        for index in range(tracker.TARGET_SEED_SAMPLES):
            robot_state["camera"]["timestamp"] = started_at + index * 0.01
            tracker.set_visual_target_override(
                "hand_guided_navigation", 0.51, 0.49
            )

        seed = tracker.get_stable_target_seed(
            "hand_guided_navigation",
            target="hand",
            session_id="track-person",
        )
        self.assertIsNotNone(seed)
        self.assertEqual(seed["target"], "hand")
        self.assertIsNone(tracker.get_stable_target_seed(
            "hand_guided_navigation",
            target="hand",
            session_id="another-session",
        ))
        self.assertIsNone(tracker.get_stable_target_seed(
            "hand_guided_navigation",
            target="person",
            session_id="track-person",
        ))

        tracker.set_visual_target_override(
            "hand_guided_navigation", 0.75, 0.49
        )
        self.assertIsNone(
            tracker.get_stable_target_seed("hand_guided_navigation")
        )

    def test_track_action_seed_samples_are_unique_consecutive_and_close(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker.target = "person"
        started_at = time.monotonic()
        robot_state["camera"]["timestamp"] = started_at
        tracker.update_stable_target_seed("test")
        tracker.update_stable_target_seed("test")
        self.assertEqual(tracker._seed_tracker().sample_count("test"), 1)

        for index, map_x in enumerate((2.02, 2.4), start=1):
            robot_state["camera"]["object_map_x"] = map_x
            robot_state["camera"]["timestamp"] = started_at + index * 0.01
            seed = tracker.update_stable_target_seed("test")
        self.assertIsNone(seed)

        robot_state["camera"]["object_x"] = float("nan")
        robot_state["camera"]["timestamp"] = started_at + 0.03
        tracker.update_stable_target_seed("test")
        self.assertEqual(tracker._seed_tracker().sample_count("test"), 0)

    async def test_seed_wait_stops_when_tracking_session_is_replaced(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker.active = True
        tracker.target = "person"
        tracker.action_id = "track-old"

        waiting = asyncio.create_task(tracker.wait_for_stable_target_seed(
            tracker.TRACKED_TARGET_SEED_OWNER,
            target="person",
            session_id="track-old",
            check_interval=0.001,
            timeout=1.0,
        ))
        await asyncio.sleep(0.005)
        tracker.action_id = "track-new"

        self.assertIsNone(await waiting)

    def test_seed_samples_do_not_mix_tracking_sessions(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker.target = "person"
        tracker.action_id = "track-old"
        started_at = time.monotonic()
        for index in range(2):
            robot_state["camera"]["timestamp"] = started_at + index * 0.01
            tracker.update_stable_target_seed("test")

        tracker.action_id = "track-new"
        robot_state["camera"]["timestamp"] = started_at + 0.02
        self.assertIsNone(tracker.update_stable_target_seed("test"))
        self.assertEqual(tracker._seed_tracker().sample_count("test"), 1)

    async def test_upward_flick_dispatches_approach_to_stable_seed(self):
        commands = []
        action = None

        async def send(payload):
            commands.append(dict(payload))
            asyncio.get_running_loop().call_soon(
                action.handle_navigation_event,
                {
                    "event": "navigation",
                    "action_id": payload["action_id"],
                    "status": "Goal Reached",
                },
            )
            return {"status": "accepted"}

        action = self.action(send)
        action.active = True
        action.state = "tracking_hand"
        action.action_id = "hand-test"
        action.track_action.get_stable_target_seed.return_value = {
            "captured_at": time.monotonic(),
            "validated_at": time.monotonic(),
            "owner": "hand_guided_navigation", "target": "hand",
            "x": 1.0, "y": 0.1, "angle": 0.1,
            "tof_range": 1.0, "map_x": 2.0, "map_y": 1.0,
        }
        action.hand_landmarks.detect_right_hand = AsyncMock(
            return_value=open_hand("up")
        )
        yolo = Mock()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for sequence in range(1, action.UP_FLICK_CONFIRM_FRAMES + 1):
            yolo.detection_snapshot_with_frame.return_value = (
                sequence, [person_detection()], frame, {},
            )
            await action._sample(yolo)

        task = action._navigation_task
        self.assertIsNotNone(task)
        await task

        self.assertEqual(commands[0]["command"], "navigate_to_approach")
        self.assertEqual(commands[0]["standoff_m"], action.APPROACH_STANDOFF_M)
        self.assertEqual(action.state, "awaiting_push")

    async def test_upward_hand_and_tof_shrink_returns_then_faces_person(self):
        commands = []
        action = None

        async def send(payload):
            commands.append(dict(payload))
            asyncio.get_running_loop().call_soon(
                action.handle_navigation_event,
                {
                    "event": "navigation",
                    "action_id": payload["action_id"],
                    "status": "Goal Reached",
                },
            )
            return {"status": "accepted"}

        action = self.action(send)
        action.active = True
        action.state = "awaiting_push"
        action.action_id = "hand-test"
        for _ in range(action.PUSH_BASELINE_SAMPLES):
            robot_state["camera"].update({
                "camera_tof_range": 0.40,
                "timestamp": time.monotonic(),
            })
            action._observe_push("fingers_up")
        for _ in range(action.PUSH_CONFIRM_FRAMES):
            robot_state["camera"].update({
                "camera_tof_range": 0.32,
                "timestamp": time.monotonic(),
            })
            action._observe_push("fingers_up")

        task = action._navigation_task
        self.assertIsNotNone(task)
        await task

        local_commands = [
            item.get("local_command") for item in commands
            if item["command"] == "navigate_local"
        ]
        self.assertEqual(local_commands, ["previous_position", "face_person"])
        self.assertEqual(action.state, "watching_person")
        self.assertFalse(action.active)


if __name__ == "__main__":
    unittest.main()
