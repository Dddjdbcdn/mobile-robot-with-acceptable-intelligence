from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actions.action_result import ActionResult
from actions.see_action import SeeAction
from cognition.cognition_manager import CognitionManager, ToolRequest
from cognition.state import robot_state


def inactive_action(**attributes):
    return SimpleNamespace(
        active=False,
        action_id=None,
        target=None,
        **attributes,
    )


class CognitionCameraArbitrationTests(unittest.IsolatedAsyncioTestCase):
    def make_manager(self):
        manager = CognitionManager.__new__(CognitionManager)
        manager.follow_executor = inactive_action(
            start=AsyncMock(), stop=AsyncMock()
        )
        manager.goal_executor = inactive_action(
            action_type="find_object", start=AsyncMock(), stop=AsyncMock()
        )
        manager.navigate_action = inactive_action(
            start=AsyncMock(), stop=AsyncMock()
        )
        manager.search_action = inactive_action()
        manager.track_action = inactive_action(
            start_tracking=AsyncMock(), stop_tracking=AsyncMock()
        )
        manager.approach_action = inactive_action()
        manager.astra_action = None
        manager.move_action = inactive_action()
        manager.see_action = inactive_action(
            see=AsyncMock(), move_to_region=AsyncMock()
        )
        manager._settled = False
        manager._autonomy_task = None
        manager.response_manager = SimpleNamespace(
            send_function_output=AsyncMock(),
            create_voice_response=AsyncMock(),
        )
        return manager

    @staticmethod
    def request(name, arguments=None):
        return ToolRequest(
            function_name=name,
            arguments=arguments or {},
            call_id="call-1",
            action_id="action-1",
            response_metadata={},
        )

    async def test_see_stops_passive_watch_without_resetting_camera(self):
        manager = self.make_manager()
        manager.track_action.active = True

        async def stop_tracking(**_kwargs):
            manager.track_action.active = False

        manager.track_action.stop_tracking.side_effect = stop_tracking
        expected = ActionResult(
            "action-1", "see_action", "succeeded", outcome="image_context_added"
        )
        manager.see_action.see.return_value = expected

        result = await manager.execute_request(self.request(
            "see_action", {"query": "What is here?", "region": None}
        ))

        self.assertIs(result, expected)
        manager.track_action.stop_tracking.assert_awaited_once_with(
            reason_code="REPLACED",
            status="cancelled",
            outcome="replaced_by_see_action",
            reset_camera=False,
        )
        manager.see_action.see.assert_awaited_once()

    async def test_see_is_rejected_while_search_owns_camera(self):
        manager = self.make_manager()
        manager.search_action.active = True

        result = await manager.execute_request(self.request(
            "see_action", {"query": "What is here?", "region": "left"}
        ))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason_code, "CAMERA_IN_USE")
        self.assertEqual(result.data["camera_owner"], "search_action")
        manager.track_action.stop_tracking.assert_not_awaited()
        manager.see_action.see.assert_not_awaited()

    async def test_find_replaces_passive_watch_before_busy_check(self):
        manager = self.make_manager()
        manager.track_action.active = True

        async def stop_tracking(**_kwargs):
            manager.track_action.active = False

        manager.track_action.stop_tracking.side_effect = stop_tracking
        expected = ActionResult("action-1", "find_object", "running")
        manager.goal_executor.start.return_value = expected

        result = await manager.execute_request(self.request(
            "find_object", {"target": "bottle"}
        ))

        self.assertIs(result, expected)
        manager.goal_executor.start.assert_awaited_once_with(
            goal="find_object", target="bottle", action_id="action-1"
        )

    async def test_stop_tracking_tool_stops_lingering_watch(self):
        manager = self.make_manager()
        manager.track_action.active = True
        manager.track_action.stop_tracking.return_value = ActionResult(
            "watch-1", "track_action", "cancelled", outcome="tracking_stopped"
        )

        await manager.execute_stop(self.request("stop_tracking"))

        manager.track_action.stop_tracking.assert_awaited_once_with()
        manager.response_manager.send_function_output.assert_awaited_once()
        manager.response_manager.create_voice_response.assert_awaited_once()

    async def test_stop_tracking_does_not_interrupt_find_owned_tracking(self):
        manager = self.make_manager()
        manager.goal_executor.active = True
        manager.track_action.active = True
        manager.track_action.target = "bottle"

        await manager.execute_stop(self.request("stop_tracking"))

        manager.track_action.stop_tracking.assert_not_awaited()
        envelope = manager.response_manager.send_function_output.await_args.args[1]
        self.assertEqual(envelope["result"]["reason_code"], "CAMERA_IN_USE")
        self.assertEqual(envelope["result"]["action_type"], "stop_tracking")
        self.assertEqual(
            envelope["decision_state"]["active_actions"][-1]["action_type"],
            "find_object",
        )


class SeeActionCameraMovementTests(unittest.IsolatedAsyncioTestCase):
    async def test_see_action_owns_semantic_camera_movement(self):
        previous_camera_state = dict(robot_state["camera"])
        robot_state["camera"].update({
            "timestamp": 1.0,
            "pan_angle": 95.0,
            "tilt_angle": 90.0,
        })
        send_robot_command = AsyncMock(return_value={"status": "accepted"})
        action = SeeAction(
            ws=AsyncMock(),
            camera=SimpleNamespace(
                wait_for_frame_captured_after=lambda _captured_at, _timeout: True,
            ),
            send_robot_command=send_robot_command,
            settle_seconds=0.0,
        )

        try:
            with patch(
                "actions.see_action.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep:
                result = await action.move_to_region("upper-left", "see-1:move")
        finally:
            robot_state["camera"].clear()
            robot_state["camera"].update(previous_camera_state)

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.action_type, "see_action")
        self.assertEqual(result.data["region"], "upper_left")
        self.assertEqual(sleep.await_args_list, [call(0.0)])
        send_robot_command.assert_awaited_once()
        command = send_robot_command.await_args.args[0]
        self.assertEqual(command["command"], "move_camera")
        self.assertEqual(command["action_id"], "see-1:move")


if __name__ == "__main__":
    unittest.main()
