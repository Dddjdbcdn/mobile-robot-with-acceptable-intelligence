import asyncio
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
from cognition.manager.cognition_manager import CognitionManager, ToolRequest
from cognition.manager.action_reporting import action_envelope, compact_result
from cognition.manager.action_state import ActionState
from cognition.manager.world_state import robot_state


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
            start=AsyncMock(),
            stop=AsyncMock(),
            wait_until_finished=AsyncMock(),
        )
        manager.map_navigation_action = inactive_action()
        manager.hand_gesture_interface = inactive_action(
            stop=AsyncMock()
        )
        manager.search_action = inactive_action()
        manager.track_action = inactive_action(
            start_tracking=AsyncMock(),
            stop_tracking=AsyncMock(),
            wait_until_finished=AsyncMock(),
            set_visual_target_override=Mock(),
            clear_visual_target_override=Mock(),
            stable_seeds=Mock(),
        )
        manager.approach_action = inactive_action(
            start=AsyncMock(),
            wait_until_finished=AsyncMock(),
            stop_approaching=AsyncMock(),
        )
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
        manager.current_action = None
        manager.action_results = []
        manager.events = asyncio.Queue()
        manager._hand_location = None
        manager._hand_approach_started_at = None
        manager.response_manager = SimpleNamespace(
            send_function_output=AsyncMock(),
            send_system_context=AsyncMock(),
            create_voice_response=AsyncMock(),
        )
        manager.action_state = ActionState(
            search=manager.search_action,
            tracking=manager.track_action,
            approach=manager.approach_action,
            explicit_navigation=manager.explicit_navigation_action,
            find_target=manager.find_target_executor,
            follow_person=manager.follow_person_executor,
        )
        return manager

    def test_manager_wires_hand_gesture_dispatcher(self):
        seeded = self.make_manager()
        hand_executor = inactive_action(
            stop=AsyncMock(),
            set_dispatcher=Mock(),
        )

        manager = CognitionManager(
            approach_action=seeded.approach_action,
            search_action=seeded.search_action,
            see_action=seeded.see_action,
            track_action=seeded.track_action,
            explicit_navigation_action=seeded.explicit_navigation_action,
            map_navigation_action=seeded.map_navigation_action,
            hand_gesture_interface=hand_executor,
            find_target_executor=seeded.find_target_executor,
            follow_person_executor=seeded.follow_person_executor,
            response_manager=seeded.response_manager,
        )

        callback = hand_executor.set_dispatcher.call_args.args[0]
        self.assertIs(callback.__self__, manager)
        self.assertIs(
            callback.__func__,
            CognitionManager.handle_hand_command,
        )

    def test_global_action_state_drives_gesture_mode(self):
        manager = self.make_manager()

        self.assertEqual(
            manager.action_state.gesture_context["mode"],
            "watching_person",
        )

        manager.track_action.active = True
        manager.track_action.target = "person"
        self.assertEqual(
            manager.action_state.gesture_context["mode"],
            "tracking_person",
        )

        manager.follow_person_executor.active = True
        self.assertEqual(
            manager.action_state.gesture_context["mode"],
            "following",
        )

    def test_action_reporting_is_independent_of_manager(self):
        result = ActionResult(
            "find-1", "find_target", "failed", target="bottle",
            reason_code="OBJECT_DETECTION_FAILED", retryable=True,
        )
        state = {"active_actions": [], "autonomy_active": False}

        self.assertEqual(compact_result(result), {
            "action_type": "find_target",
            "status": "failed",
            "target": "bottle",
            "reason_code": "OBJECT_DETECTION_FAILED",
            "retryable": True,
        })
        envelope = action_envelope("command_result", result, state)
        self.assertEqual(envelope["event_type"], "command_result")
        self.assertNotIn("decision_state", envelope)

    async def test_hand_dispatch_resolves_tracking_target_and_seed(self):
        manager = self.make_manager()
        seed = {"x": 1.0, "y": 0.1, "angle": 0.1, "tof_range": 1.0}
        manager.track_action.action_id = "track-1"
        manager.track_action.active = True
        manager.track_action.target = "person"
        manager.track_action.stable_seeds.get.side_effect = (None, seed)

        context = await manager.handle_hand_command(
            "observe_gesture",
            {"gesture": "welcome", "palm_center": (0.5, 0.4)},
        )

        manager.track_action.set_visual_target_override.assert_called_once_with(
            0.5, 0.4
        )
        self.assertNotIn("stable_seed", context)
        self.assertEqual(manager._hand_location, seed)

    async def test_hand_dropout_does_not_immediately_clear_camera_override(self):
        manager = self.make_manager()
        manager.track_action.active = True
        manager.track_action.target = "person"
        manager.track_action.stable_seeds.get.return_value = None

        await manager.handle_hand_command(
            "observe_gesture",
            {"gesture": "unavailable", "palm_center": None},
        )

        manager.track_action.clear_visual_target_override.assert_not_called()

    async def test_hand_approach_uses_silent_tool_request(self):
        manager = self.make_manager()
        seed = {"x": 1.0, "y": 0.1, "angle": 0.1, "tof_range": 1.0}
        manager.approach_action.start.return_value = ActionResult(
            "hand-1", "approach_action", "running"
        )
        manager._hand_location = seed

        queued = await manager.handle_hand_command(
            "approach_target", {"target": "hand"}
        )
        request = await manager.events.get()
        result = await manager.execute_request(request)

        self.assertTrue(queued)
        self.assertTrue(request.silent)
        self.assertEqual(request.function_name, "approach_target")
        self.assertEqual(request.arguments["location"], seed)
        self.assertEqual(result.status, "running")
        approach_call = manager.approach_action.start.await_args
        self.assertEqual(approach_call.kwargs["location"], seed)
        self.assertTrue(approach_call.kwargs["action_id"].startswith("hand-"))
        self.assertEqual(approach_call.kwargs["standoff_m"], 0.05)
        self.assertIsNotNone(manager._hand_approach_started_at)

    async def test_hand_dispatch_stops_active_approach(self):
        manager = self.make_manager()
        manager.approach_action.active = True

        queued = await manager.handle_hand_command(
            "stop_navigation",
            {"reason": "HAND_PUSH"},
        )
        request = await manager.events.get()
        await manager.handle_tool_request(request)

        self.assertTrue(queued)
        self.assertEqual(request.function_name, "stop_navigation")
        manager.approach_action.stop_approaching.assert_awaited_once_with(
            "HAND_PUSH"
        )
        manager.response_manager.send_function_output.assert_not_awaited()

    async def test_hand_dispatch_resolves_return_sequence(self):
        manager = self.make_manager()
        manager.track_action.active = True

        async def stop_tracking(*_args, **_kwargs):
            manager.track_action.active = False

        manager.track_action.stop_tracking.side_effect = stop_tracking
        manager.explicit_navigation_action.start.side_effect = (
            ActionResult("hand-1:return", "explicit_navigation", "running"),
            ActionResult("hand-1:face-person", "explicit_navigation", "running"),
        )
        manager.explicit_navigation_action.wait_until_finished.side_effect = (
            ActionResult("hand-1:return", "explicit_navigation", "succeeded"),
            ActionResult("hand-1:face-person", "explicit_navigation", "succeeded"),
        )

        queued = await manager.handle_hand_command(
            "navigation_sequence",
            {"commands": ["previous_position", "face_person"]},
        )
        request = await manager.events.get()
        result = await manager.execute_request(request)

        self.assertTrue(queued)
        self.assertEqual(request.function_name, "navigation_sequence")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(
            [item.args[0] for item in
             manager.explicit_navigation_action.start.await_args_list],
            ["previous_position", "face_person"],
        )
        manager.track_action.stop_tracking.assert_not_awaited()
        self.assertTrue(manager.track_action.active)

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

    async def test_hand_interface_does_not_claim_camera_ownership(self):
        manager = self.make_manager()

        result = await manager.execute_request(self.request(
            "explicit_navigation",
            {"local_command": "rotate_left", "room_id": None},
        ))

        manager.explicit_navigation_action.start.assert_awaited_once()

    async def test_gesture_request_uses_normal_handler_without_llm_response(self):
        manager = self.make_manager()
        manager.explicit_navigation_action.start.return_value = ActionResult(
            "hand-move", "explicit_navigation", "succeeded",
            outcome="navigation_reached",
        )

        await manager.handle_hand_command(
            "explicit_navigation",
            {"local_command": "nudge_backward", "room_id": None},
        )
        request = await manager.events.get()

        self.assertIsInstance(request, ToolRequest)
        self.assertEqual(request.function_name, "explicit_navigation")
        self.assertTrue(request.silent)

        await manager.handle_tool_request(request)
        await manager.current_action_task
        event = await manager.events.get()
        await manager.handle_action_executed(event)

        manager.explicit_navigation_action.start.assert_awaited_once_with(
            "nudge_backward", request.action_id
        )
        manager.response_manager.send_function_output.assert_not_awaited()
        manager.response_manager.create_voice_response.assert_not_awaited()

    async def test_hand_stop_follow_keeps_existing_tracking(self):
        manager = self.make_manager()
        manager._hand_location = {"x": 1.0}
        manager.follow_person_executor.active = True
        manager.track_action.active = True

        queued = await manager.handle_hand_command(
            "stop_follow",
            {"reason": "HAND_PUSH"},
        )
        request = await manager.events.get()
        await manager.handle_tool_request(request)

        self.assertTrue(queued)
        self.assertEqual(request.function_name, "stop_follow")
        manager.follow_person_executor.stop.assert_awaited_once_with(
            "HAND_PUSH"
        )
        manager.track_action.stop_tracking.assert_not_awaited()
        manager.track_action.start_tracking.assert_not_awaited()

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

    async def test_explicit_navigation_keeps_passive_tracking(self):
        manager = self.make_manager()
        manager.track_action.active = True
        manager.track_action.target = "person"

        expected = ActionResult("action-1", "explicit_navigation", "running")
        manager.explicit_navigation_action.start.return_value = expected

        result = await manager.execute_request(self.request(
            "explicit_navigation",
            {"local_command": "rotate_left", "room_id": None},
        ))

        self.assertIs(result, expected)
        manager.track_action.stop_tracking.assert_not_awaited()
        self.assertTrue(manager.track_action.active)
        manager.explicit_navigation_action.start.assert_awaited_once_with(
            "rotate_left", "action-1"
        )

    async def test_approach_keeps_passive_tracking(self):
        manager = self.make_manager()
        manager.track_action.active = True
        manager.track_action.target = "person"
        location = {"x": 1.0, "y": 0.0, "angle": 0.0, "tof_range": 1.5}
        expected = ActionResult("action-1", "approach_action", "running")
        manager.approach_action.start.return_value = expected

        result = await manager.execute_request(self.request(
            "approach_target",
            {"target": "hand", "location": location},
        ))

        self.assertIs(result, expected)
        manager.track_action.stop_tracking.assert_not_awaited()
        self.assertTrue(manager.track_action.active)
        manager.approach_action.start.assert_awaited_once_with(
            location=location,
            action_id="action-1",
            target="hand",
            standoff_m=None,
        )

    def test_navigation_does_not_take_camera_from_tracking(self):
        manager = self.make_manager()
        manager.track_action.active = True
        manager.track_action.target = "person"
        manager.explicit_navigation_action.active = True

        self.assertEqual(
            manager.action_state.camera_owner,
            "target_tracking",
        )

    async def test_watch_target_uses_direct_tracker_without_dino(self):
        manager = self.make_manager()
        manager.track_action.start_tracking.return_value = ActionResult(
            "action-1", "track_action", "running", target="person",
            outcome="tracking",
        )

        result = await manager.execute_request(self.request(
            "watch_target",
            {"target": "person", "target_kind": "person"},
        ))

        self.assertEqual(result.action_type, "watch_target")
        self.assertEqual(result.status, "running")
        self.assertEqual(result.data["watch_source"], "direct_tracker")
        manager.track_action.start_tracking.assert_awaited_once_with(
            target="person",
            action_id="action-1",
            allow_grounding_dino=False,
            continuous_person_reacquisition=True,
        )
        manager.find_target_executor.start.assert_not_awaited()

    async def test_watch_target_falls_back_to_find_without_approaching(self):
        manager = self.make_manager()
        manager.track_action.start_tracking.return_value = ActionResult(
            "action-1", "track_action", "failed", target="person",
            outcome="target_not_detected",
            reason_code="PERSON_DETECTION_FAILED",
        )
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

    async def test_direct_watch_uses_tracker_lifecycle(self):
        manager = self.make_manager()
        request = self.request(
            "watch_target",
            {"target": "person", "target_kind": "person"},
        )
        manager.current_action = request
        manager.track_action.wait_until_finished.return_value = ActionResult(
            "action-1", "track_action", "failed", target="person",
            outcome="person_lost", reason_code="PERSON_LOST",
        )
        started = ActionResult(
            "action-1", "watch_target", "running", target="person",
            outcome="tracking", data={"watch_source": "direct_tracker"},
        )

        await manager.handle_action_executed(SimpleNamespace(
            request=request,
            result=started,
            error=None,
        ))
        lifecycle = await asyncio.wait_for(manager.events.get(), timeout=1.0)

        manager.track_action.wait_until_finished.assert_awaited_once_with()
        await manager.handle_lifecycle_finished(lifecycle)
        self.assertEqual(manager.action_results[-1].action_type, "watch_target")
        self.assertEqual(manager.action_results[-1].reason_code, "PERSON_LOST")

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

    async def test_stop_navigation_cancels_approach_and_keeps_hand_tracking(self):
        manager = self.make_manager()
        manager.approach_action.active = True

        await manager.execute_stop(self.request("stop_navigation"))

        manager.approach_action.stop_approaching.assert_awaited_once_with()
        output = manager.response_manager.send_function_output.await_args.args[1]
        self.assertEqual(output["result"]["status"], "completed")

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
