import json
from pathlib import Path
import sys
import unittest


SMALL_BRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(SMALL_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(SMALL_BRAIN_ROOT))

from actions.see_action import SeeAction


class FakeWebSocket:
    def __init__(self):
        self.events = []

    async def send(self, payload):
        self.events.append(json.loads(payload))


class FakeSnapshot:
    sequence = 42
    captured_at = 3.5


class FakeCamera:
    def jpeg_bytes_snapshot(self, quality, tracking_bgr, save_path):
        return b"\xff\xd8fake-jpeg\xff\xd9"

    def snapshot(self):
        return FakeSnapshot()


class FailingCamera(FakeCamera):
    def jpeg_bytes_snapshot(self, quality, tracking_bgr, save_path):
        raise RuntimeError("camera offline")


class SeeActionTests(unittest.IsolatedAsyncioTestCase):
    async def test_adds_current_frame_to_visual_context(self):
        websocket = FakeWebSocket()
        action = SeeAction(websocket, FakeCamera())

        result = await action.start_seeing("What is in front of me?")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.msg["result"], "image_context_added")
        self.assertEqual(result.msg["frame_sequence"], 42)
        self.assertEqual(websocket.events[0]["type"], "conversation.item.create")

    async def test_rejects_an_empty_query_without_capturing(self):
        websocket = FakeWebSocket()
        action = SeeAction(websocket, FakeCamera())

        result = await action.start_seeing("  ")

        self.assertEqual(result.status, "failure")
        self.assertEqual(result.msg, "vision_query_is_required")
        self.assertEqual(websocket.events, [])

    async def test_converts_camera_exception_to_failure_result(self):
        action = SeeAction(FakeWebSocket(), FailingCamera())

        result = await action.start_seeing("What is visible?")

        self.assertEqual(result.status, "failure")
        self.assertEqual(result.msg["result"], "vision_capture_failed")
        self.assertFalse(action.active)


if __name__ == "__main__":
    unittest.main()
