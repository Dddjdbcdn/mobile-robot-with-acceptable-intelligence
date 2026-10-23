import asyncio
import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from actions.action_result import ActionResult
from cognition.manager.cognition_manager import CognitionManager
from cognition.sequence.follow_person_executor import FollowPersonExecutor


class FollowLifecycleTests(unittest.IsolatedAsyncioTestCase):

    async def test_acquisition_delegates_to_find_executor_and_waits_for_seed(self):
        find_target = SimpleNamespace(
            start=AsyncMock(return_value=ActionResult(
                "follow-1:acquire-person",
                "find_target",
                "succeeded",
                target="person",
                outcome="person_acquired",
            )),
        )
        track_action = SimpleNamespace(
            session_id="follow-1:acquire-person:direct_person",
            wait_for_stable_target_seed=AsyncMock(return_value={
                "target": "person",
                "x": 1.0,
                "y": 0.0,
            }),
        )
        executor = FollowPersonExecutor(find_target, track_action, AsyncMock())
        executor.action_id = "follow-1"

        result = await executor._acquire_target()

        self.assertEqual(result.status, "succeeded")
        find_target.start.assert_awaited_once_with(
            target="person",
            action_id="follow-1:acquire-person",
            target_kind="person",
            approach=False,
        )
        track_action.wait_for_stable_target_seed.assert_awaited_once_with(
            target="person",
            session_id="follow-1:acquire-person:direct_person",
            timeout=10.0,
        )

    async def test_accepted_follow_is_event_driven_after_setup(self):
        class TrackAction:
            def __init__(self):
                self.follow_states = []

            def set_follow_person_active(self, active):
                self.follow_states.append(active)

        commands = []

        async def send_robot_command(command):
            commands.append(command)
            return {"status": "accepted"}

        track_action = TrackAction()
        executor = FollowPersonExecutor(None, track_action, send_robot_command)
        executor._acquire_target = AsyncMock(return_value=ActionResult(
            "follow-1", "follow_person", "succeeded", target="person",
            outcome="person_acquired",
        ))

        await executor.start("person", "follow-1")
        self.assertEqual(track_action.follow_states, [True])
        await executor._runner

        self.assertTrue(executor.active)
        self.assertFalse(executor.completion_future.done())
        self.assertEqual(track_action.follow_states, [True])
        self.assertEqual(commands, [{
            "command": "follow_action",
            "action_id": "follow-1",
        }])

        handled = await executor.handle_navigation_event({
            "event": "navigation",
            "action_id": "follow-1",
            "status": "Failed or Canceled (Status 6)",
            "data": {"nav2_error_code": 202},
        })
        result = await executor.wait_until_finished()

        self.assertTrue(handled)
        self.assertFalse(executor.active)
        self.assertEqual(result.reason_code, "FOLLOW_NAVIGATION_ENDED")
        self.assertEqual(result.data["navigation"]["nav2_error_code"], 202)
        self.assertEqual(track_action.follow_states, [True, False])

    async def test_lidar_lost_event_stops_follow(self):
        track_action = SimpleNamespace(
            set_follow_person_active=Mock(),
        )
        commands = []

        async def send_robot_command(command):
            commands.append(command)
            return {"status": "accepted"}

        executor = FollowPersonExecutor(None, track_action, send_robot_command)
        executor.active = True
        executor.action_id = "follow-1"
        executor.target = "person"
        executor.completion_future = (
            asyncio.get_running_loop().create_future()
        )

        handled = await executor.handle_navigation_event({
            "event": "person_tracker",
            "status": "lost",
            "action_id": "follow-1",
        })
        result = await executor.wait_until_finished()

        self.assertTrue(handled)
        self.assertEqual(result.reason_code, "LIDAR_TRACK_LOST")
        self.assertEqual(commands, [{
            "command": "stop_follow_action",
            "action_id": "follow-1",
        }])

    async def test_visual_recovery_progress_is_owned_by_big_brain(self):
        commands = []

        async def send_robot_command(command):
            commands.append(command)
            return {"status": "accepted"}

        executor = FollowPersonExecutor(
            None,
            SimpleNamespace(set_follow_person_active=Mock()),
            send_robot_command,
        )
        executor.active = True
        executor.action_id = "follow-1"

        handled = await executor.handle_navigation_event({
            "event": "person_tracker",
            "status": "visual_recovery",
            "action_id": "follow-1",
            "data": {
                "visual_search_direction": -1,
                "last_visual_base_bearing_rad": -0.3,
                "last_visual_camera_bearing_rad": 0.2,
            },
        })

        self.assertFalse(handled)
        self.assertEqual(commands, [])


class FollowHandOverrideTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.commands = []

        async def send_robot_command(command):
            self.commands.append(command)
            return {"status": "accepted"}

        self.executor = FollowPersonExecutor(None, None, send_robot_command)
        self.executor.active = True
        self.executor.action_id = "follow-1"

    async def test_valid_map_seed_updates_and_then_clears_override(self):
        accepted = await self.executor.update_hand_target_override({
            "target": "hand",
            "map_x": 1.25,
            "map_y": -0.5,
        })
        cleared = await self.executor.clear_hand_target_override()

        self.assertTrue(accepted)
        self.assertTrue(cleared)
        self.assertEqual(self.commands, [
            {
                "command": "set_follow_target_override",
                "action_id": "follow-1",
                "target": "hand",
                "override_type": "welcome",
                "frame_id": "map",
                "x": 1.25,
                "y": -0.5,
            },
            {
                "command": "clear_follow_target_override",
                "action_id": "follow-1",
            },
        ])

    async def test_seed_without_map_position_does_not_override_following(self):
        accepted = await self.executor.update_hand_target_override({
            "target": "hand",
            "x": 0.4,
            "y": 0.1,
        })

        self.assertFalse(accepted)
        self.assertEqual(self.commands, [])

    async def test_clear_is_idempotent(self):
        await self.executor.update_hand_target_override({
            "map_x": 1.0,
            "map_y": 2.0,
        })
        await self.executor.clear_hand_target_override()
        await self.executor.clear_hand_target_override()

        clear_commands = [
            command for command in self.commands
            if command["command"] == "clear_follow_target_override"
        ]
        self.assertEqual(len(clear_commands), 1)


class FollowHandGestureRoutingTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        seed = {"target": "hand", "map_x": 1.0, "map_y": 2.0}

        class StableSeeds:
            @staticmethod
            def get(**_kwargs):
                return dict(seed)

        class TrackAction:
            action_id = "track-1"
            stable_seeds = StableSeeds()

            def __init__(self):
                self.override = None
                self.clear_count = 0

            def set_visual_target_override(self, x, y):
                self.override = (x, y)

            def clear_visual_target_override(self):
                self.override = None
                self.clear_count += 1

        class FollowExecutor:
            active = True

            def __init__(self):
                self.updated_seed = None
                self.clear_count = 0

            async def update_hand_target_override(
                self, updated_seed, gesture="welcome"
            ):
                self.updated_seed = dict(updated_seed)
                self.updated_gesture = gesture
                return True

            async def clear_hand_target_override(self):
                self.clear_count += 1
                return True

        self.manager = CognitionManager.__new__(CognitionManager)
        self.manager.track_action = TrackAction()
        self.manager.follow_person_executor = FollowExecutor()
        self.manager.action_state = SimpleNamespace(
            person_tracking=True,
            gesture_context={"mode": "following"},
        )
        self.manager.navigation_action = SimpleNamespace(
            active=False,
            mode=None,
        )
        self.manager._hand_location = None
        self.manager._hand_approach_started_at = None

    async def test_welcome_routes_valid_hand_seed_to_follow_override(self):
        await self.manager.handle_hand_command(
            "observe_gesture",
            {"gesture": "welcome", "palm_center": (0.4, 0.6)},
        )

        self.assertEqual(self.manager.track_action.override, (0.4, 0.6))
        self.assertEqual(
            self.manager.follow_person_executor.updated_seed["target"],
            "hand",
        )
        self.assertEqual(
            self.manager.follow_person_executor.updated_gesture,
            "welcome",
        )

    async def test_push_routes_hand_seed_as_retreat_override(self):
        await self.manager.handle_hand_command(
            "observe_gesture",
            {"gesture": "push", "palm_center": (0.4, 0.6)},
        )

        self.assertEqual(
            self.manager.follow_person_executor.updated_gesture,
            "push",
        )

    async def test_non_welcome_immediately_restores_lidar_following(self):
        await self.manager.handle_hand_command(
            "observe_gesture",
            {"gesture": "unavailable", "palm_center": None},
        )

        self.assertEqual(self.manager.track_action.clear_count, 1)
        self.assertEqual(
            self.manager.follow_person_executor.clear_count,
            1,
        )


if __name__ == "__main__":
    unittest.main()
