import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np

from actions.action_result import ActionResult
from actions.track_action import TrackingService
from actions.tracking.target_catalog import (
    YOLO_OBJECT_CLASSES,
    normalize_object_target,
)
from cognition.sequence.find_target_executor import FindTargetExecutor


def object_detection(x1, y1, x2, y2, confidence=0.9):
    return {
        "class": "bottle",
        "class_id": 8,
        "confidence": confidence,
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
    }


class YoloFrames:
    def __init__(self, observations):
        self.observations = list(observations)
        self.index = -1
        self.object_mode = False

    @asynccontextmanager
    async def object_detection_mode(self):
        self.object_mode = True
        try:
            yield
        finally:
            self.object_mode = False

    def detection_snapshot(self):
        return 10, []

    async def wait_for_inference_after(self, _sequence, timeout):
        del timeout
        self.index += 1
        await asyncio.sleep(0)
        return self.index < len(self.observations)

    def detection_snapshot_with_tracking_frame(self):
        detections, frame, camera_sequence = self.observations[self.index]
        return (
            11 + self.index,
            detections,
            frame,
            {"camera_sequence": camera_sequence},
        )


class YoloObjectAcquisitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_alias_normalizes_into_the_yolo_object_set(self):
        self.assertEqual(normalize_object_target("water bottles"), "bottle")
        self.assertIn(normalize_object_target("water bottles"), YOLO_OBJECT_CLASSES)

    async def test_new_object_aliases_normalize_into_the_yolo_object_set(self):
        aliases = {
            "whiteboard": "board",
            "vehicles": "car",
            "elevator doors": "elevator",
            "trees": "tree",
        }
        for alias, expected in aliases.items():
            with self.subTest(alias=alias):
                normalized = normalize_object_target(alias)
                self.assertEqual(normalized, expected)
                self.assertIn(normalized, YOLO_OBJECT_CLASSES)

    async def test_first_fresh_detection_initializes_csrt_from_exact_frame(self):
        frame = np.full((360, 640, 3), 1, dtype=np.uint8)
        yolo = YoloFrames([
            ([object_detection(100, 80, 180, 200)], frame, 101),
        ])
        service = TrackingService.__new__(TrackingService)
        service.yolo = yolo
        service.csrt_tracker = Mock()

        result = await service._detect_yolo_object("bottle")

        self.assertEqual(result, ("yoloe-11m", 0.9))
        self.assertFalse(yolo.object_mode)
        call = service.csrt_tracker.begin_tracking.call_args.kwargs
        self.assertEqual(call["detection_sequence"], 101)
        np.testing.assert_array_equal(call["initialization_frame"], frame)
        self.assertEqual(call["bbox_xywh"], (100, 80, 80, 120))
        self.assertEqual(call["target"], "bottle")

    async def test_fresh_frame_without_target_does_not_initialize_csrt(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        yolo = YoloFrames([
            ([], frame, 1),
        ])
        service = TrackingService.__new__(TrackingService)
        service.yolo = yolo
        service.csrt_tracker = Mock()

        result = await service._detect_yolo_object("bottle")

        self.assertIsNone(result)
        self.assertFalse(yolo.object_mode)
        service.csrt_tracker.begin_tracking.assert_not_called()


class FindObjectFastPathTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def make_executor(track_result):
        track_action = SimpleNamespace(
            start_tracking=AsyncMock(return_value=track_result),
        )
        inactive_action = SimpleNamespace(active=False)
        executor = FindTargetExecutor(
            search_action=inactive_action,
            track_action=track_action,
            navigation_action=inactive_action,
            see_action=inactive_action,
            map_navigation_action=inactive_action,
            send_map_overlay=AsyncMock(),
        )
        return executor, track_action

    async def test_yolo_object_checks_current_view_before_search(self):
        executor, track_action = self.make_executor(ActionResult(
            "object-1:direct_object",
            "track_action",
            "running",
            target="bottle",
            outcome="tracking",
            data={"detection_source": "yoloe-11m"},
        ))
        executor._search_environment = AsyncMock()

        await executor.start(
            target="water bottle",
            action_id="object-1",
            target_kind="object",
            approach=False,
        )
        result = await executor.wait_until_finished()

        self.assertEqual(result.status, "succeeded")
        track_action.start_tracking.assert_awaited_once_with(
            target="water bottle",
            action_id="object-1:direct_object",
            allow_grounding_dino=False,
        )
        executor._search_environment.assert_not_awaited()

    async def test_yolo_current_view_miss_falls_back_to_search(self):
        executor, _ = self.make_executor(ActionResult(
            "object-2:direct_object",
            "track_action",
            "failed",
            target="bottle",
            outcome="target_not_detected",
            reason_code="OBJECT_DETECTION_FAILED",
        ))
        executor._search_environment = AsyncMock(return_value=ActionResult(
            "object-2:search",
            "search_action",
            "succeeded",
            target="bottle",
            outcome="target_tracking_started",
        ))

        await executor.start(
            target="bottle",
            action_id="object-2",
            target_kind="object",
            approach=False,
        )
        result = await executor.wait_until_finished()

        self.assertEqual(result.status, "succeeded")
        executor._search_environment.assert_awaited_once_with()

    async def test_non_yolo_object_skips_fast_track(self):
        executor, track_action = self.make_executor(Mock())
        executor._search_environment = AsyncMock(return_value=ActionResult(
            "object-3:search",
            "search_action",
            "succeeded",
            target="fire extinguisher",
            outcome="target_tracking_started",
        ))

        await executor.start(
            target="fire extinguisher",
            action_id="object-3",
            target_kind="object",
            approach=False,
        )
        result = await executor.wait_until_finished()

        self.assertEqual(result.status, "succeeded")
        track_action.start_tracking.assert_not_awaited()
        executor._search_environment.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
