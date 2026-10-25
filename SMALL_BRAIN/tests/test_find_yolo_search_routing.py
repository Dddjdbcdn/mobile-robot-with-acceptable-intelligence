import asyncio
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from actions.action_result import ActionResult
from cognition.sequence.find_target_executor import FindTargetExecutor


def search_action(result):
    return SimpleNamespace(
        active=False,
        target=None,
        search_mode="specific",
        last_observation_frame=None,
        last_coverage_frames=[],
        start_searching=AsyncMock(return_value=result),
        wait_until_finished=AsyncMock(),
        stop=AsyncMock(),
    )


def succeeded_camera_move():
    return ActionResult(
        "camera",
        "move_camera_action",
        "succeeded",
        outcome="camera_positioned",
    )


class FindYoloSearchRoutingTests(unittest.IsolatedAsyncioTestCase):
    def make_executor(self, regular_result, yolo_result, map_result=None):
        regular = search_action(regular_result)
        yolo = search_action(yolo_result)
        map_navigation = SimpleNamespace(
            active=False,
            start_find_target_step=AsyncMock(return_value=map_result),
            wait_until_finished=AsyncMock(),
            stop=AsyncMock(),
        )
        inactive = SimpleNamespace(active=False)
        executor = FindTargetExecutor(
            search_action=regular,
            yolo_search_action=yolo,
            track_action=inactive,
            navigation_action=inactive,
            see_action=inactive,
            map_navigation_action=map_navigation,
            send_map_overlay=AsyncMock(),
            move_camera_action=SimpleNamespace(
                move_to_region=AsyncMock(
                    return_value=succeeded_camera_move()
                ),
            ),
        )
        executor.action_id = "find-1"
        executor.target_kind = "object"
        executor._search_started_at = time.monotonic()
        return executor, regular, yolo, map_navigation

    async def test_supported_object_uses_yolo_search(self):
        found = ActionResult(
            "find-1:scan",
            "yolo_search_action",
            "succeeded",
            target="bottle",
            outcome="found",
        )
        executor, regular, yolo, _ = self.make_executor(found, found)
        executor.target = "water bottle"

        result = await executor._search("scan", initial_view_only=False)

        self.assertIs(result, found)
        yolo.start_searching.assert_awaited_once()
        regular.start_searching.assert_not_awaited()
        self.assertIs(executor._last_search_action, yolo)

    async def test_new_exported_object_uses_yolo_search(self):
        found = ActionResult(
            "find-1:scan",
            "yolo_search_action",
            "succeeded",
            target="elevator",
            outcome="found",
        )
        executor, regular, yolo, _ = self.make_executor(found, found)
        executor.target = "elevator"

        result = await executor._search("scan", initial_view_only=False)

        self.assertIs(result, found)
        yolo.start_searching.assert_awaited_once()
        regular.start_searching.assert_not_awaited()

    async def test_unsupported_object_uses_regular_search(self):
        found = ActionResult(
            "find-1:scan",
            "search_action",
            "succeeded",
            target="fire extinguisher",
            outcome="found",
        )
        executor, regular, yolo, _ = self.make_executor(found, found)
        executor.target = "fire extinguisher"

        result = await executor._search("scan", initial_view_only=False)

        self.assertIs(result, found)
        regular.start_searching.assert_awaited_once()
        yolo.start_searching.assert_not_awaited()
        self.assertIs(executor._last_search_action, regular)

    async def test_yolo_miss_continues_to_map_exploration(self):
        missed = ActionResult(
            "find-1:initial_center_scan",
            "yolo_search_action",
            "failed",
            target="bottle",
            outcome="not_found",
            reason_code="TARGET_NOT_VISIBLE",
        )
        no_candidates = ActionResult(
            "find-1:explore_1",
            "map_navigation_action",
            "failed",
            target="bottle",
            outcome="no_candidates",
            reason_code="NO_NAVIGATION_CANDIDATES",
        )
        executor, regular, yolo, map_navigation = self.make_executor(
            missed,
            missed,
            no_candidates,
        )
        executor.target = "bottle"

        result = await executor._search_environment()

        self.assertEqual(result.reason_code, "OBJECT_SEARCH_EXHAUSTED")
        yolo.start_searching.assert_awaited_once()
        regular.start_searching.assert_not_awaited()
        self.assertEqual(
            map_navigation.start_find_target_step.await_args.kwargs["mode"],
            "exploration",
        )

    async def test_stop_cancels_active_yolo_search(self):
        cancelled = ActionResult(
            "find-1:scan",
            "yolo_search_action",
            "cancelled",
            target="bottle",
            outcome="search_cancelled",
        )
        executor, regular, yolo, _ = self.make_executor(cancelled, cancelled)
        executor.active = True
        executor.target = "bottle"
        executor.yolo_search_action.active = True
        executor.completion_future = asyncio.get_running_loop().create_future()

        await executor.stop()

        yolo.stop.assert_awaited_once_with("USER_REQUESTED")
        regular.stop.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
