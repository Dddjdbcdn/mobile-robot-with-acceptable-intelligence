from pathlib import Path
from collections import deque
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch


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
        manager.follow_person_executor = inactive_action(
            start=AsyncMock(), stop=AsyncMock()
        )
        manager.find_target_executor = inactive_action(
            action_type="find_target", start=AsyncMock(), stop=AsyncMock()
        )
        manager.explicit_navigation_action = inactive_action(
            start=AsyncMock(), stop=AsyncMock()
        )
        manager.map_navigation_action = inactive_action()
        manager.hand_guided_navigation = inactive_action(
            stop=AsyncMock(), handle_navigation_event=Mock()
        )
        manager.search_action = inactive_action()
        manager.track_action = inactive_action(
            start_tracking=AsyncMock(), stop_tracking=AsyncMock()
        )
        manager.approach_action = inactive_action()
        manager.see_action = inactive_action(
            see=AsyncMock(), move_to_region=AsyncMock()
        )
        manager._autonomy_task = None
        manager._acquire_person_after_startup = True
        manager._startup_person_acquisition_pending = True
        manager._last_proximity_reaction = 0.0
        manager.proximity_cooldown = 8.0
        manager._tof_history = deque(maxlen=5)
        manager._proximity_candidate_count = 0
        manager.world_state = {}
        manager.current_action_task = None
        manager.action_results = []
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
        expected = ActionResult("action-1", "find_target", "running")
        manager.find_target_executor.start.return_value = expected

        result = await manager.execute_request(self.request(
            "find_target", {"target": "bottle", "target_kind": "object"}
        ))

        self.assertIs(result, expected)
        manager.find_target_executor.start.assert_awaited_once_with(
            target="bottle", action_id="action-1", target_kind="object"
        )

    async def test_follow_reuses_passive_person_tracking(self):
        manager = self.make_manager()
        manager.track_action.active = True
        manager.track_action.target = "person"
        expected = ActionResult("action-1", "follow_person", "running")
        manager.follow_person_executor.start.return_value = expected

        result = await manager.execute_request(self.request(
            "follow_person", {"target": "person"}
        ))

        self.assertIs(result, expected)
        manager.track_action.stop_tracking.assert_not_awaited()
        manager.follow_person_executor.start.assert_awaited_once_with(
            target="person", action_id="action-1"
        )

    async def test_automatic_hand_guidance_blocks_other_camera_motion(self):
        manager = self.make_manager()
        manager.hand_guided_navigation.active = True
        manager.hand_guided_navigation.action_id = "automatic-hand"

        result = await manager.execute_request(self.request(
            "explicit_navigation",
            {"local_command": "rotate_left", "room_id": None},
        ))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason_code, "CAMERA_IN_USE")
        self.assertEqual(result.data["camera_owner"], "hand_guided_navigation")
        manager.track_action.stop_tracking.assert_not_awaited()
        manager.explicit_navigation_action.start.assert_not_awaited()

    async def test_find_target_rejects_person_kind(self):
        manager = self.make_manager()

        result = await manager.execute_request(self.request(
            "find_target", {"target": "person", "target_kind": "person"}
        ))

        self.assertEqual(result.status, "failed")
        self.assertEqual(
            result.reason_code, "PERSON_TARGET_REQUIRES_PERSON_ACTION"
        )
        manager.find_target_executor.start.assert_not_awaited()

    async def test_explicit_navigation_routes_to_one_shot_action(self):
        manager = self.make_manager()
        expected = ActionResult("action-1", "explicit_navigation", "running")
        manager.explicit_navigation_action.start.return_value = expected

        result = await manager.execute_request(self.request(
            "explicit_navigation",
            {"local_command": "rotate_left", "room_id": None},
        ))

        self.assertIs(result, expected)
        manager.explicit_navigation_action.start.assert_awaited_once_with(
            "rotate_left", "action-1"
        )
        manager.find_target_executor.start.assert_not_awaited()

    async def test_explicit_navigation_replaces_passive_tracking(self):
        manager = self.make_manager()
        manager.track_action.active = True
        manager.track_action.target = "person"

        async def stop_tracking(**_kwargs):
            manager.track_action.active = False

        manager.track_action.stop_tracking.side_effect = stop_tracking
        expected = ActionResult("action-1", "explicit_navigation", "running")
        manager.explicit_navigation_action.start.return_value = expected

        result = await manager.execute_request(self.request(
            "explicit_navigation",
            {"local_command": "rotate_left", "room_id": None},
        ))

        self.assertIs(result, expected)
        manager.track_action.stop_tracking.assert_awaited_once_with(
            reason_code="REPLACED",
            status="cancelled",
            outcome="replaced_by_explicit_navigation",
            reset_camera=True,
        )
        manager.explicit_navigation_action.start.assert_awaited_once_with(
            "rotate_left", "action-1"
        )

    async def test_watch_target_runs_find_without_approaching(self):
        manager = self.make_manager()
        expected = ActionResult("action-1", "watch_target", "running")
        manager.find_target_executor.start.return_value = expected

        result = await manager.execute_request(self.request(
            "watch_target",
            {"target": "person", "target_kind": "person"},
        ))

        self.assertIs(result, expected)
        manager.find_target_executor.start.assert_awaited_once_with(
            target="person",
            action_id="action-1",
            target_kind="person",
            approach=False,
            continuous_person_reacquisition=True,
            action_type="watch_target",
        )
        manager.explicit_navigation_action.start.assert_not_awaited()

    async def test_stop_watching_stops_only_passive_target_tracking(self):
        manager = self.make_manager()
        manager.track_action.active = True

        async def stop_tracking(**_kwargs):
            manager.track_action.active = False
            return ActionResult(
                "passive-look", "track_action", "cancelled",
                outcome="tracking_stopped",
            )

        manager.track_action.stop_tracking.side_effect = stop_tracking

        await manager.execute_stop(self.request("stop_watching_target"))

        manager.track_action.stop_tracking.assert_awaited_once_with()
        output = manager.response_manager.send_function_output.await_args.args[1]
        self.assertEqual(output["result"]["command"], "stop_watching_target")
        self.assertEqual(output["result"]["status"], "completed")
        manager.response_manager.create_voice_response.assert_awaited_once_with()

    async def test_stop_watching_cancels_find_based_watch_acquisition(self):
        manager = self.make_manager()
        manager.find_target_executor.active = True
        manager.find_target_executor.action_type = "watch_target"
        manager.find_target_executor.stop.return_value = ActionResult(
            "action-1", "watch_target", "cancelled", outcome="goal_cancelled"
        )

        await manager.execute_stop(self.request("stop_watching_target"))

        manager.find_target_executor.stop.assert_awaited_once_with()
        manager.track_action.stop_tracking.assert_not_awaited()
        output = manager.response_manager.send_function_output.await_args.args[1]
        self.assertEqual(output["result"]["status"], "completed")

    async def test_stop_watching_does_not_interrupt_action_owned_tracking(self):
        manager = self.make_manager()
        manager.find_target_executor.active = True
        manager.track_action.active = True

        await manager.execute_stop(self.request("stop_watching_target"))

        manager.track_action.stop_tracking.assert_not_awaited()
        output = manager.response_manager.send_function_output.await_args.args[1]
        self.assertEqual(output["result"]["status"], "failed")
        self.assertEqual(output["result"]["reason_code"], "CAMERA_IN_USE")
        self.assertEqual(output["result"]["camera_owner"], "find_target")
        manager.response_manager.create_voice_response.assert_awaited_once_with()

    async def test_first_world_state_starts_person_acquisition_without_approach(self):
        manager = self.make_manager()
        manager.find_target_executor.start.return_value = ActionResult(
            "startup-person", "find_target", "succeeded", target="person"
        )

        await manager._handle_world_state({"camera_tof_range": 2.0})
        task = manager._autonomy_task
        await task

        manager.find_target_executor.start.assert_awaited_once()
        arguments = manager.find_target_executor.start.await_args.kwargs
        self.assertEqual(arguments["target"], "person")
        self.assertEqual(arguments["target_kind"], "person")
        self.assertFalse(arguments["approach"])
        self.assertTrue(arguments["continuous_person_reacquisition"])

    async def test_proximity_retreat_only_moves_backward(self):
        manager = self.make_manager()
        manager._startup_person_acquisition_pending = False
        manager.explicit_navigation_action.start.return_value = ActionResult(
            "retreat", "explicit_navigation", "succeeded", target="nudge_backward"
        )

        await manager._run_proximity_retreat()

        manager.explicit_navigation_action.start.assert_awaited_once()
        command = manager.explicit_navigation_action.start.await_args.args
        self.assertEqual(command[0], "nudge_backward")
        manager.see_action.see.assert_not_awaited()
        manager.response_manager.create_voice_response.assert_not_awaited()


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
