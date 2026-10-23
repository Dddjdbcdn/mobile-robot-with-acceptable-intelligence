import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock


SMALL_BRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(SMALL_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(SMALL_BRAIN_ROOT))

from actions.move_camera_action import MoveCameraAction
from actions.track_action import TrackingService
from cognition.manager.world_state import robot_state


class ReadyCamera:
    def wait_for_frame_captured_after(self, moved_at, timeout):
        return True


class MoveCameraResetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.previous_camera_state = dict(robot_state["camera"])

    def tearDown(self):
        robot_state["camera"] = self.previous_camera_state

    async def test_reset_pan_preserves_tilt_without_stopping_tracking(self):
        commands = []

        async def send_robot_command(command):
            commands.append(command)
            return {"status": "accepted"}

        tracking = type("Tracking", (), {})()
        tracking.active = True
        tracking.stop = AsyncMock()
        robot_state["camera"] = {
            "timestamp": 1.0,
            "pan_angle": 130.0,
            "tilt_angle": 62.0,
        }
        movement = MoveCameraAction(
            camera=ReadyCamera(),
            send_robot_command=send_robot_command,
            tracking=tracking,
            settle_seconds=0.0,
        )

        result = await movement.reset_pan(
            action_id="follow-1:reset-pan",
            stop_tracking=False,
        )

        self.assertEqual(result.status, "succeeded")
        tracking.stop.assert_not_awaited()
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["command"], "move_camera")
        self.assertEqual(commands[0]["delta_pan_angle"], -35.0)
        self.assertEqual(commands[0]["delta_tilt_angle"], 0.0)

    async def test_tracking_follow_reset_uses_non_preempting_camera_move(self):
        movement = type("Movement", (), {})()
        movement.reset_pan = AsyncMock(return_value=None)
        tracking = TrackingService.__new__(TrackingService)
        tracking.action_id = "follow-1:track"
        tracking.move_camera_action = movement

        await tracking._reset_pan_for_follow()

        movement.reset_pan.assert_awaited_once_with(
            action_id="follow-1:track:person-absent-reset-pan",
            stop_tracking=False,
        )


if __name__ == "__main__":
    unittest.main()
