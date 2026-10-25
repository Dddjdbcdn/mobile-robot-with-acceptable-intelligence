import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import numpy as np

from actions.action_result import ActionResult
from actions.yolo_search_action import YoloSearchAction


def detection(class_name="bottle", confidence=0.9, center=(0.5, 0.5)):
    center_x, center_y = center
    return {
        "class": class_name,
        "class_id": 8,
        "confidence": confidence,
        "bbox": {
            "x1": 100.0,
            "y1": 80.0,
            "x2": 180.0,
            "y2": 200.0,
            "normalized_center_x": center_x,
            "normalized_center_y": center_y,
        },
    }


class FakeYolo:
    def __init__(self, detections_per_frame):
        self.frames = list(detections_per_frame)
        self.index = -1
        self.sequence = 0
        self.object_mode = False
        self.inference_count = 0

    @asynccontextmanager
    async def object_detection_mode(self):
        self.object_mode = True
        try:
            yield
        finally:
            self.object_mode = False

    def detection_snapshot(self):
        return self.sequence, []

    async def wait_for_inference_after(self, sequence, timeout):
        del sequence, timeout
        await asyncio.sleep(0)
        self.index += 1
        if self.index >= len(self.frames):
            return False
        if self.frames[self.index] is None:
            return False
        self.sequence += 1
        self.inference_count += 1
        return True

    def detection_snapshot_with_tracking_frame(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        return (
            self.sequence,
            self.frames[self.index],
            frame,
            {"camera_sequence": 100 + self.sequence},
        )


class BlockingYolo(FakeYolo):
    def __init__(self):
        super().__init__([])
        self.waiting = asyncio.Event()

    async def wait_for_inference_after(self, sequence, timeout):
        del sequence, timeout
        self.waiting.set()
        await asyncio.Event().wait()


def successful_camera_move():
    return ActionResult(
        action_id="camera",
        action_type="move_camera",
        status="succeeded",
        outcome="camera_positioned",
    )


class YoloSearchActionTests(unittest.IsolatedAsyncioTestCase):
    def make_action(self, yolo):
        move_camera = SimpleNamespace(
            move_to_angles=AsyncMock(return_value=successful_camera_move()),
        )
        action = YoloSearchAction(
            ws=None,
            send_robot_command=AsyncMock(),
            camera=SimpleNamespace(),
            move_camera_action=move_camera,
            yolo=yolo,
        )
        return action, move_camera

    async def test_rejects_target_outside_yolo_classes(self):
        yolo = FakeYolo([])
        action, _ = self.make_action(yolo)

        result = await action.start_searching(
            target="fire extinguisher",
            action_id="search-1",
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason_code, "YOLO_TARGET_UNSUPPORTED")
        self.assertFalse(action.active)
        self.assertFalse(yolo.object_mode)

    async def test_finds_alias_in_first_fresh_object_inference(self):
        yolo = FakeYolo([[detection()]])
        action, _ = self.make_action(yolo)

        started = await action.start_searching(
            target="water bottle",
            action_id="search-2",
            initial_view_only=True,
        )
        result = await action.wait_until_finished()

        self.assertEqual(started.status, "running")
        self.assertEqual(started.target, "bottle")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.outcome, "found")
        self.assertEqual(result.data["detection_source"], "yoloe-11m")
        self.assertEqual(result.data["detection"]["class"], "bottle")
        self.assertEqual(yolo.inference_count, 1)
        self.assertFalse(yolo.object_mode)
        self.assertEqual(action.last_found_target, "bottle")
        self.assertEqual(action.last_observation_frame["camera_sequence"], 101)

    async def test_initial_view_miss_returns_target_not_visible(self):
        yolo = FakeYolo([[]])
        action, move_camera = self.make_action(yolo)

        await action.start_searching(
            target="bottle",
            action_id="search-3",
            initial_view_only=True,
        )
        result = await action.wait_until_finished()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.outcome, "not_found")
        self.assertEqual(result.reason_code, "TARGET_NOT_VISIBLE")
        self.assertEqual(result.data["searched_frames"], 1)
        self.assertFalse(yolo.object_mode)
        move_camera.move_to_angles.assert_not_awaited()

    async def test_waits_for_later_inference_at_same_view(self):
        yolo = FakeYolo([[], [detection()]])
        action, move_camera = self.make_action(yolo)

        await action.start_searching(
            target="bottle",
            action_id="search-later-frame",
            initial_view_only=True,
        )
        result = await action.wait_until_finished()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(yolo.inference_count, 2)
        self.assertEqual(result.data["searched_frames"], 1)
        self.assertEqual(result.data["detection"]["class"], "bottle")
        self.assertEqual(move_camera.move_to_angles.await_count, 1)

    async def test_sweeps_after_current_view_miss(self):
        yolo = FakeYolo([[], None, [detection(center=(0.4, 0.5))]])
        action, move_camera = self.make_action(yolo)

        await action.start_searching(
            target="bottle",
            action_id="search-4",
            effort="center",
            sweep_directions=("left",),
        )
        result = await action.wait_until_finished()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(yolo.inference_count, 2)
        self.assertEqual(move_camera.move_to_angles.await_count, 2)
        first_move = move_camera.move_to_angles.await_args_list[0]
        self.assertEqual(first_move.args[:2], (155.0, 90.0))
        self.assertFalse(yolo.object_mode)

    async def test_stop_cancels_inference_wait_and_restores_pose_mode(self):
        yolo = BlockingYolo()
        action, _ = self.make_action(yolo)

        await action.start_searching(
            target="bottle",
            action_id="search-5",
            initial_view_only=True,
        )
        await asyncio.wait_for(yolo.waiting.wait(), timeout=0.2)
        result = await action.stop()

        self.assertEqual(result.status, "cancelled")
        self.assertFalse(action.active)
        self.assertFalse(yolo.object_mode)


if __name__ == "__main__":
    unittest.main()
