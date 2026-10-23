import asyncio
import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock


SMALL_BRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(SMALL_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(SMALL_BRAIN_ROOT))

from actions.action_result import ActionResult
from actions.search_action import SearchAction
from cognition.manager.cognition_manager import CognitionManager, Shutdown, ToolRequest


class BlockingCameraMovement:
    def __init__(self):
        self.calls = []
        self.first_move_started = asyncio.Event()
        self.first_move_cancelled = False

    async def move_to_angles(
        self,
        target_pan,
        target_tilt,
        *,
        action_id,
        target=None,
        current_pan=None,
        current_tilt=None,
    ):
        self.calls.append((target_pan, target_tilt))
        if len(self.calls) == 1:
            self.first_move_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.first_move_cancelled = True
                raise
        return ActionResult(
            action_id=action_id,
            action_type="move_camera",
            status="succeeded",
            outcome="camera_positioned",
        )


class SnapshotCamera:
    def jpeg_bytes_snapshot(self, *args, **kwargs):
        return b"jpeg"


def yolo_person(confidence=0.9):
    return {
        "class": "person",
        "confidence": confidence,
        "bbox": {
            "normalized_x1": 0.3,
            "normalized_y1": 0.1,
            "normalized_x2": 0.7,
            "normalized_y2": 0.9,
            "normalized_center_x": 0.5,
            "normalized_center_y": 0.5,
        },
        "keypoints": {"nose": {"normalized_x": 0.5, "normalized_y": 0.2}},
    }


class ControlledYolo:
    def __init__(self):
        self.sequence = 0
        self.detections = []

    def detection_snapshot(self):
        return self.sequence, list(self.detections)

    def publish(self, *detections):
        self.sequence += 1
        self.detections = list(detections)


class SearchCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_cancels_assessment_before_recentering(self):
        camera_movement = BlockingCameraMovement()
        search = SearchAction(
            ws=None,
            send_robot_command=None,
            camera=None,
            move_camera_action=camera_movement,
        )
        search.active = True
        search.action_id = "find-1"
        search.target = "bottle"
        search.completion_future = asyncio.get_running_loop().create_future()

        assessment_task = search.create_assessment_task(
            search.move_camera_angles(35.0, 90.0),
            name="assess_batch_search",
        )
        await camera_movement.first_move_started.wait()

        result = await search.stop_searching("USER_REQUESTED")

        self.assertTrue(assessment_task.cancelled())
        self.assertTrue(camera_movement.first_move_cancelled)
        self.assertEqual(len(camera_movement.calls), 2)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.reason_code, "USER_REQUESTED")
        self.assertFalse(search.active)

    async def test_person_search_uses_raw_yolo_detection(self):
        yolo = ControlledYolo()
        search = SearchAction(
            ws=type("WS", (), {"send": AsyncMock()})(),
            send_robot_command=None,
            camera=SnapshotCamera(),
            yolo=yolo,
        )

        started = await search.start_searching(
            target="person",
            action_id="find-person",
            initial_view_only=True,
            person_search=True,
        )
        self.assertEqual(started.status, "running")

        self.assertTrue(search.active)

        yolo.publish(yolo_person())
        result = await asyncio.wait_for(search.wait_until_finished(), 1.0)

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.outcome, "found")
        self.assertEqual(
            result.data["detection_source"],
            "yolo",
        )
        self.assertEqual(result.data["person_detection"], yolo_person())
        self.assertFalse(search.active)

    async def test_person_search_ignores_low_confidence_yolo_detection(self):
        yolo = ControlledYolo()
        search = SearchAction(
            ws=type("WS", (), {"send": AsyncMock()})(),
            send_robot_command=None,
            camera=SnapshotCamera(),
            yolo=yolo,
        )
        await search.start_searching(
            target="person",
            action_id="find-person-reset",
            initial_view_only=True,
            person_search=True,
        )

        yolo.publish(yolo_person(confidence=0.5))
        await asyncio.sleep(0.1)
        self.assertTrue(search.active)

        yolo.publish(yolo_person())
        result = await asyncio.wait_for(search.wait_until_finished(), 1.0)
        self.assertEqual(result.status, "succeeded")


class CognitionLoopResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_exception_does_not_terminate_loop(self):
        manager = object.__new__(CognitionManager)
        manager.events = asyncio.Queue()
        manager._running = False

        async def fail_request(_request):
            raise RuntimeError("expected failure")

        manager.handle_tool_request = fail_request
        await manager.events.put(ToolRequest(
            function_name="stop_goal",
            arguments={},
            call_id="call-1",
            action_id="action-1",
            response_metadata={},
        ))
        await manager.events.put(Shutdown())

        await asyncio.wait_for(manager.cognition_loop(), timeout=1.0)

        self.assertFalse(manager._running)


if __name__ == "__main__":
    unittest.main()
