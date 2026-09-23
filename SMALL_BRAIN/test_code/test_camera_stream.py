from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import camera_stream as stream_module
from services.camera_stream import CameraStream


class UsbCameraControlTests(unittest.TestCase):
    def test_environment_values_are_parsed_and_manual_modes_are_inferred(self):
        controls = CameraStream.usb_controls_from_env({
            "USB_CAMERA_EXPOSURE_TIME_ABSOLUTE": "300",
            "USB_CAMERA_WHITE_BALANCE_TEMPERATURE": "5000",
            "USB_CAMERA_EXPOSURE_DYNAMIC_FRAMERATE": "true",
            "USB_CAMERA_POWER_LINE_FREQUENCY": "60hz",
            "USB_CAMERA_BACKLIGHT_COMPENSATION": "20",
        })

        self.assertEqual(controls, {
            "exposure_time_absolute": 300,
            "white_balance_temperature": 5000,
            "exposure_dynamic_framerate": 1,
            "power_line_frequency": 2,
            "backlight_compensation": 20,
            "auto_exposure": 1,
            "white_balance_automatic": 0,
        })

    def test_explicit_automatic_modes_are_preserved(self):
        controls = CameraStream.usb_controls_from_env({
            "USB_CAMERA_AUTO_EXPOSURE": "auto",
            "USB_CAMERA_WHITE_BALANCE_AUTOMATIC": "yes",
        })

        self.assertEqual(controls, {
            "auto_exposure": 3,
            "white_balance_automatic": 1,
        })

    def test_native_controls_are_applied_in_dependency_order(self):
        stream = CameraStream.__new__(CameraStream)
        stream.usb_device = "/dev/video7"
        stream.usb_controls = {
            "brightness": 0,
            "exposure_time_absolute": 300,
            "auto_exposure": 1,
        }
        stream.applied_usb_controls = {}

        completed = subprocess.CompletedProcess([], 0, "", "")
        with patch(
            "services.camera_stream.shutil.which",
            return_value="/usr/bin/v4l2-ctl",
        ), patch(
            "services.camera_stream.subprocess.run",
            return_value=completed,
        ) as run:
            stream._apply_usb_controls()

        self.assertEqual(run.call_args_list, [
            call(
                [
                    "/usr/bin/v4l2-ctl", "-d", "/dev/video7",
                    "--set-ctrl", "auto_exposure=1",
                ],
                capture_output=True,
                text=True,
                check=False,
            ),
            call(
                [
                    "/usr/bin/v4l2-ctl", "-d", "/dev/video7",
                    "--set-ctrl", "exposure_time_absolute=300",
                ],
                capture_output=True,
                text=True,
                check=False,
            ),
            call(
                [
                    "/usr/bin/v4l2-ctl", "-d", "/dev/video7",
                    "--set-ctrl", "brightness=0",
                ],
                capture_output=True,
                text=True,
                check=False,
            ),
        ])
        self.assertEqual(stream.applied_usb_controls, {
            "auto_exposure": 1,
            "exposure_time_absolute": 300,
            "brightness": 0,
        })


class UsbCameraRecoveryTests(unittest.TestCase):
    def test_open_uses_stable_device_path(self):
        device = "/dev/v4l/by-id/test-camera-video-index0"
        stream = CameraStream(usb_device=device)
        camera = MagicMock()
        camera.isOpened.return_value = True

        with patch(
            "services.camera_stream.cv2.VideoCapture",
            return_value=camera,
        ) as video_capture:
            opened = stream._open_usb_camera()

        self.assertIs(opened, camera)
        video_capture.assert_called_once_with(
            device, stream_module.cv2.CAP_V4L2
        )
        self.assertEqual(camera.set.call_count, 4)
        stream._zmq_context.term()

    def test_capture_loop_reopens_after_repeated_read_failures(self):
        device = "/dev/v4l/by-id/test-camera-video-index0"
        stream = CameraStream(usb_device=device)
        stream.USB_RECONNECT_DELAY = 0
        stream.USB_READ_FAILURE_LIMIT = 2

        failed_camera = MagicMock()
        failed_camera.isOpened.return_value = True
        failed_camera.read.return_value = (False, None)

        recovered_camera = MagicMock()
        recovered_camera.isOpened.return_value = True
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        def recovered_read():
            stream.stop_event.set()
            return True, frame

        recovered_camera.read.side_effect = recovered_read

        with patch(
            "services.camera_stream.cv2.VideoCapture",
            side_effect=[failed_camera, recovered_camera],
        ) as video_capture:
            stream._capture_loop()

        self.assertEqual(video_capture.call_count, 2)
        failed_camera.release.assert_called_once()
        recovered_camera.release.assert_called_once()
        self.assertEqual(stream.sequence, 1)
        stream._zmq_context.term()


    def test_wait_for_frame_captured_after_ignores_pre_move_frame(self):
        stream = CameraStream()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stream._record_frame("usb", frame)
        move_completed_at = time.monotonic()

        def publish_new_frame():
            time.sleep(0.02)
            stream._record_frame("usb", frame)

        publisher = threading.Thread(target=publish_new_frame)
        publisher.start()
        try:
            self.assertTrue(stream.wait_for_frame_captured_after(
                move_completed_at,
                timeout=0.5,
            ))
        finally:
            publisher.join()
            stream._zmq_context.term()

    def test_wait_for_frame_captured_after_times_out_on_stale_frame(self):
        stream = CameraStream()
        stream._record_frame(
            "usb",
            np.zeros((4, 4, 3), dtype=np.uint8),
        )

        try:
            self.assertFalse(stream.wait_for_frame_captured_after(
                time.monotonic(),
                timeout=0.01,
            ))
        finally:
            stream._zmq_context.term()


if __name__ == "__main__":
    unittest.main()
