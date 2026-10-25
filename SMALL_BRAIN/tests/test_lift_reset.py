import asyncio
from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from cognition.manager.cognition_manager import CognitionManager, ToolRequest


class FakeAction:
    def __init__(self, active=True):
        self.active = active
        self.reasons = []

    async def stop(self, reason):
        self.reasons.append(reason)
        self.active = False


class FakeTrackAction(FakeAction):
    def __init__(self, active=True):
        super().__init__(active)
        self.override_cleared = 0

    def clear_visual_target_override(self):
        self.override_cleared += 1


class LiftResetTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_reset_does_not_enable_lift_lock(self):
        manager = object.__new__(CognitionManager)
        manager.events = asyncio.Queue()
        manager._lift_reset_active = False

        await manager.handle_tool_msg({
            "event": "robot_reset",
            "reason_code": "MANUAL_RESET",
        })

        reset = manager.events.get_nowait()
        self.assertEqual(reset.function_name, "reset_robot")
        self.assertEqual(reset.arguments["reason"], "MANUAL_RESET")
        self.assertFalse(manager._lift_reset_active)

    async def test_reset_event_is_latched_until_lift_returns_to_normal(self):
        manager = object.__new__(CognitionManager)
        manager.events = asyncio.Queue()
        manager._lift_reset_active = False
        manager._last_lift_state = None
        manager._pending_world_state = None
        manager._world_state_queued = False

        await manager.handle_tool_msg({
            "event": "robot_reset",
            "reason_code": "ROBOT_LIFTED",
        })
        reset = manager.events.get_nowait()
        self.assertEqual(reset.function_name, "reset_robot")
        self.assertTrue(reset.silent)
        self.assertTrue(manager._lift_reset_active)

        # A state packet sent just before the lift-state callback must not
        # prematurely unlock SMALL_BRAIN or enqueue a duplicate reset.
        await manager.publish_world_state({"lift_state": "normal"})
        self.assertTrue(manager._lift_reset_active)
        await manager.publish_world_state({"lift_state": "lifted"})
        self.assertTrue(manager._lift_reset_active)

        await manager.publish_world_state({"lift_state": "normal"})
        self.assertFalse(manager._lift_reset_active)

    async def test_full_stop_cancels_tasks_and_every_action(self):
        manager = object.__new__(CognitionManager)
        manager.follow_person_executor = FakeAction()
        manager.find_target_executor = FakeAction()
        manager.navigation_action = FakeAction()
        manager.map_navigation_action = FakeAction()
        manager.search_action = FakeAction()
        manager.track_action = FakeTrackAction()
        manager.hand_gesture_interface = SimpleNamespace(stop=AsyncMock())
        manager.current_action_task = asyncio.create_task(asyncio.sleep(60))
        manager._autonomy_task = asyncio.create_task(asyncio.sleep(60))
        manager._hand_location = {"x": 1.0}
        manager._hand_approach_started_at = 1.0
        manager._tof_history = deque([0.5])
        manager._proximity_candidate_count = 2
        manager._startup_person_acquisition_pending = True

        request = ToolRequest(
            function_name="reset_robot",
            arguments={"reason": "ROBOT_LIFTED"},
            call_id="reset",
            action_id="reset",
            response_metadata={"silent": True, "robot_reset": True},
        )
        await manager.execute_stop(request)

        for action in (
            manager.follow_person_executor,
            manager.find_target_executor,
            manager.navigation_action,
            manager.map_navigation_action,
            manager.search_action,
            manager.track_action,
        ):
            self.assertFalse(action.active)
            self.assertEqual(action.reasons, ["ROBOT_LIFTED"])
        self.assertTrue(manager.current_action_task.cancelled())
        self.assertTrue(manager._autonomy_task.cancelled())
        manager.hand_gesture_interface.stop.assert_awaited_once_with(
            "ROBOT_LIFTED"
        )
        self.assertEqual(manager.track_action.override_cleared, 1)
        self.assertIsNone(manager._hand_location)
        self.assertIsNone(manager._hand_approach_started_at)
        self.assertEqual(list(manager._tof_history), [])
        self.assertEqual(manager._proximity_candidate_count, 0)
        self.assertFalse(manager._startup_person_acquisition_pending)

    async def test_hand_push_stops_motion_but_preserves_perception(self):
        manager = object.__new__(CognitionManager)
        manager.follow_person_executor = FakeAction()
        manager.find_target_executor = FakeAction()
        manager.navigation_action = FakeAction()
        manager.map_navigation_action = FakeAction()
        manager.search_action = FakeAction()
        manager.track_action = FakeTrackAction()

        request = ToolRequest(
            function_name="stop_movement",
            arguments={"reason": "HAND_PUSH"},
            call_id="push",
            action_id="push",
            response_metadata={"silent": True},
        )
        await manager.execute_stop(request)

        for action in (
            manager.follow_person_executor,
            manager.find_target_executor,
            manager.navigation_action,
            manager.map_navigation_action,
        ):
            self.assertFalse(action.active)
            self.assertEqual(action.reasons, ["HAND_PUSH"])
        self.assertTrue(manager.search_action.active)
        self.assertTrue(manager.track_action.active)
        self.assertEqual(manager.search_action.reasons, [])
        self.assertEqual(manager.track_action.reasons, [])


if __name__ == "__main__":
    unittest.main()
