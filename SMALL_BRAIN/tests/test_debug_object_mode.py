import asyncio
from types import SimpleNamespace
import unittest

from cognition.manager.cognition_manager import CognitionManager


class FakeYolo:
    def __init__(self):
        self.debug_object_mode = False

    def set_debug_object_mode(self, enabled):
        self.debug_object_mode = bool(enabled)


class DebugObjectModeTests(unittest.TestCase):
    def make_manager(self, *, action_running=False, tracking=False):
        manager = CognitionManager.__new__(CognitionManager)
        manager.yolo = FakeYolo()
        manager.current_action_task = object() if action_running else None
        manager._autonomy_task = None
        manager.track_action = SimpleNamespace(active=tracking)
        manager.action_state = SimpleNamespace(
            snapshot=lambda _task: {"active_actions": []}
        )
        return manager

    def test_idle_button_toggles_object_mode(self):
        manager = self.make_manager()

        self.assertTrue(manager.toggle_debug_object_mode())
        self.assertTrue(manager.yolo.debug_object_mode)
        self.assertTrue(manager.toggle_debug_object_mode())
        self.assertFalse(manager.yolo.debug_object_mode)

    def test_button_is_rejected_during_action_or_tracking(self):
        for options in ({"action_running": True}, {"tracking": True}):
            with self.subTest(options=options):
                manager = self.make_manager(**options)
                self.assertFalse(manager.toggle_debug_object_mode())
                self.assertFalse(manager.yolo.debug_object_mode)


class DebugObjectModeActionStartTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_start_restores_pose_mode(self):
        manager = CognitionManager.__new__(CognitionManager)
        manager.yolo = FakeYolo()
        manager.yolo.set_debug_object_mode(True)
        manager.PREEMPTIVE_ACTIONS = set()

        result = await manager.execute_request(SimpleNamespace(
            function_name="unsupported-test-action",
            arguments={},
            action_id="action-1",
        ))

        self.assertFalse(manager.yolo.debug_object_mode)
        self.assertEqual(result.reason_code, "UNKNOWN_ACTION")


if __name__ == "__main__":
    unittest.main()
