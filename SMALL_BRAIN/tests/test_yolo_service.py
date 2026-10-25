import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from services.vision.yolo_service import YoloService


class YoloServiceTest(unittest.TestCase):
    def make_service(self):
        pose_model = Mock()
        object_model = Mock()
        with patch(
            "services.vision.yolo_service.YOLO",
            side_effect=(pose_model, object_model),
        ) as yolo:
            service = YoloService(camera=SimpleNamespace(latest=None))
        self.assertEqual(yolo.call_count, 2)
        self.assertEqual(yolo.call_args_list[0].kwargs["task"], "pose")
        self.assertEqual(yolo.call_args_list[1].kwargs["task"], "detect")
        return service, pose_model, object_model

    def test_warms_both_models_before_starting_worker(self):
        service, pose_model, object_model = self.make_service()
        service.inference_thread = Mock()

        service.start_background()

        self.assertTrue(service.running)
        self.assertEqual(pose_model.predict.call_count, 3)
        self.assertEqual(object_model.predict.call_count, 3)
        for model in (pose_model, object_model):
            for call in model.predict.call_args_list:
                self.assertEqual(call.kwargs["device"], "intel:gpu")
                self.assertEqual(call.kwargs["source"].shape, (320, 640, 3))
        service.inference_thread.start.assert_called_once_with()

    def test_object_mode_invalidates_the_previous_snapshot(self):
        service, _, _ = self.make_service()
        service.detections = [{"class": "person"}]
        service._inference_full_frame_bgr = np.ones((2, 2, 3))
        service._inference_mode = "pose"

        service.object_mode = True

        self.assertTrue(service.object_mode)
        self.assertEqual(service.detections, [])
        self.assertIsNone(service._inference_full_frame_bgr)
        self.assertIsNone(service._inference_mode)

    def test_debug_reset_does_not_interrupt_action_owned_object_mode(self):
        service, _, _ = self.make_service()
        service.object_mode = True

        service.set_debug_object_mode(False)

        self.assertTrue(service.object_mode)
        self.assertFalse(service.debug_object_mode)

    def test_mode_change_during_inference_discards_old_mode_result(self):
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        latest = SimpleNamespace(
            sequence=42,
            tracking_bgr=frame,
            full_bgr=frame,
            captured_at=1.0,
        )
        service, pose_model, object_model = self.make_service()
        service.camera = SimpleNamespace(latest=latest)
        service.fps = 1000.0
        service.parse_pose = Mock(return_value=[{"class": "person"}])
        service.parse_objects = Mock(return_value=[{"class": "bottle"}])
        pose_result = Mock()

        def switch_to_object_mode(**_kwargs):
            service.object_mode = True
            return [pose_result]

        pose_model.predict.side_effect = switch_to_object_mode
        object_model.predict.return_value = [Mock()]
        service.running = True
        worker = threading.Thread(target=service.timer_callback)
        worker.start()
        deadline = time.monotonic() + 1.0
        while service._inference_sequence < 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        service.running = False
        worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertGreaterEqual(service._inference_sequence, 1)
        self.assertEqual(service.detections, [{"class": "bottle"}])
        self.assertEqual(service._inference_mode, "object")
        self.assertEqual(service._inference_camera_sequence, 42)
        service.parse_pose.assert_called_once_with(pose_result)

    def test_parses_household_objects_in_shared_detection_shape(self):
        service, _, _ = self.make_service()
        result = SimpleNamespace(
            orig_shape=(100, 200),
            names={8: "bottle", 16: "book"},
            boxes=(
                SimpleNamespace(
                    conf=np.array([0.876]),
                    cls=np.array([8]),
                    xyxy=np.array([[20.0, 10.0, 100.0, 70.0]]),
                ),
                SimpleNamespace(
                    conf=np.array([0.1]),
                    cls=np.array([16]),
                    xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]),
                ),
            ),
        )

        detections = service.parse_objects(result)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "bottle")
        self.assertEqual(detections[0]["class_id"], 8)
        self.assertEqual(detections[0]["confidence"], 0.876)
        self.assertEqual(detections[0]["bbox"]["normalized_center_x"], 0.3)
        self.assertEqual(detections[0]["bbox"]["normalized_center_y"], 0.4)


class YoloObjectModeContextTest(unittest.IsolatedAsyncioTestCase):
    async def test_context_restores_pose_mode_after_cancellation(self):
        with patch(
            "services.vision.yolo_service.YOLO",
            side_effect=(Mock(), Mock()),
        ):
            service = YoloService(camera=SimpleNamespace(latest=None))

        async def use_object_mode():
            async with service.object_detection_mode():
                self.assertTrue(service.object_mode)
                raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await use_object_mode()
        self.assertFalse(service.object_mode)


if __name__ == "__main__":
    unittest.main()
