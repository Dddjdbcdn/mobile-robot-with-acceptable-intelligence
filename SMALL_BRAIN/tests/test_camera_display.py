import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from services.display import camera_display


def person(x1, x2, nose_x):
    return {
        "class": "person",
        "confidence": 0.9,
        "bbox": {"x1": x1, "y1": 20, "x2": x2, "y2": 300},
        "keypoints": {
            "nose": {"x": nose_x, "y": 80, "confidence": 0.9},
        },
    }


class CameraDisplayTests(unittest.TestCase):
    def test_draws_pose_only_for_person_expected_by_tracker(self):
        detections = [person(300, 500, 400), person(100, 220, 200)]
        yolo = SimpleNamespace(detections=detections)
        tracker = SimpleNamespace(
            expected_person_detection=lambda candidates: candidates[1]
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with patch.object(camera_display, "_draw_pose_body_mask") as draw_mask:
            camera_display.draw_yolo_overlay(frame, yolo, tracker)

        draw_mask.assert_called_once_with(
            frame,
            {"nose": (440, 80)},
            (420, 20, 540, 300),
        )


if __name__ == "__main__":
    unittest.main()
