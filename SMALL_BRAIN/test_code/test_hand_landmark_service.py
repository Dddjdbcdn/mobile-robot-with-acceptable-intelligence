from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.hand_landmark_service import HandLandmarkService


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Category:
    def __init__(self, category_name="Right", score=0.95):
        self.category_name = category_name
        self.score = score


class FakeLandmarker:
    def __init__(self):
        self.timestamps = []
        self.image_shapes = []

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        self.image_shapes.append(image.shape)
        points = [Point(0.5, 0.5) for _ in range(21)]
        points[6] = Point(0.45, 0.45)
        points[7] = Point(0.40, 0.60)
        points[8] = Point(0.35, 0.75)
        return type("Result", (), {
            "hand_landmarks": [points],
            "handedness": [[Category()]],
        })()


class RetryLandmarker(FakeLandmarker):
    def detect_for_video(self, image, timestamp_ms):
        if not self.timestamps:
            self.timestamps.append(timestamp_ms)
            self.image_shapes.append(image.shape)
            return type("Result", (), {
                "hand_landmarks": [],
                "handedness": [],
            })()
        return super().detect_for_video(image, timestamp_ms)


def person_detection():
    return {
        "keypoints": {
            "right_elbow": {"normalized_x": 0.40, "normalized_y": 0.60},
            "right_wrist": {"normalized_x": 0.50, "normalized_y": 0.40},
        }
    }


class HandLandmarkServiceTests(unittest.TestCase):
    def test_video_mode_timestamps_are_strictly_increasing(self):
        landmarker = FakeLandmarker()
        service = HandLandmarkService(
            camera=None,
            landmarker=landmarker,
            image_factory=lambda image: image,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        first = service._detect(frame, person_detection())
        second = service._detect(frame, person_detection())

        self.assertLess(
            first["mediapipe_timestamp_ms"],
            second["mediapipe_timestamp_ms"],
        )
        self.assertEqual(
            landmarker.timestamps,
            [
                first["mediapipe_timestamp_ms"],
                second["mediapipe_timestamp_ms"],
            ],
        )

    def test_runs_landmarker_on_right_hand_crop(self):
        landmarker = FakeLandmarker()
        service = HandLandmarkService(
            camera=None,
            landmarker=landmarker,
            image_factory=lambda image: image,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        result = service._detect(frame, person_detection())

        self.assertEqual(len(landmarker.image_shapes), 1)
        crop_height, crop_width, channels = landmarker.image_shapes[0]
        self.assertEqual(crop_height, crop_width)
        self.assertEqual(channels, 3)
        self.assertLess(crop_width, frame.shape[1])
        self.assertEqual(result["image_width"], 640)
        self.assertEqual(result["image_height"], 360)
        self.assertIn("index_pip", result["landmarks"])
        self.assertIn("index_dip", result["landmarks"])
        self.assertIn("index_tip", result["landmarks"])
        self.assertLess(
            result["landmarks"]["index_tip"]["normalized_x"],
            result["landmarks"]["index_pip"]["normalized_x"],
        )

    def test_low_right_arm_confidence_uses_full_frame(self):
        landmarker = FakeLandmarker()
        service = HandLandmarkService(
            camera=None,
            landmarker=landmarker,
            image_factory=lambda image: image,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        person = person_detection()
        person["keypoint_confidences"] = {
            "right_wrist": 0.49,
            "right_elbow": 0.90,
        }

        result = service._detect(frame, person)

        self.assertEqual(landmarker.image_shapes, [(360, 640, 3)])
        self.assertEqual(result["inference_scope"], "full_frame")
        self.assertEqual(result["crop_box"], [0, 0, 640, 360])

    def test_right_arm_confidence_at_threshold_uses_crop(self):
        landmarker = FakeLandmarker()
        service = HandLandmarkService(
            camera=None,
            landmarker=landmarker,
            image_factory=lambda image: image,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        person = person_detection()
        person["keypoint_confidences"] = {
            "right_wrist": 0.50,
            "right_elbow": 0.50,
        }

        result = service._detect(frame, person)

        self.assertEqual(result["inference_scope"], "crop")
        self.assertLess(landmarker.image_shapes[0][1], frame.shape[1])

    def test_missing_right_elbow_and_wrist_uses_full_frame(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )

        result = service._detect(
            np.zeros((360, 640, 3), dtype=np.uint8), {"keypoints": {}}
        )

        self.assertEqual(result["inference_scope"], "full_frame")
        self.assertEqual(result["crop_box"], [0, 0, 640, 360])

    def test_missed_hand_expands_the_next_frame_crop(self):
        landmarker = RetryLandmarker()
        service = HandLandmarkService(
            camera=None,
            landmarker=landmarker,
            image_factory=lambda image: image,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        first = service._detect(frame, person_detection())
        result = service._detect(frame, person_detection())

        self.assertIsNone(first)
        self.assertIsNotNone(result)
        self.assertEqual(len(landmarker.image_shapes), 2)
        self.assertLess(
            landmarker.image_shapes[0][0], landmarker.image_shapes[1][0]
        )

    def test_crop_center_smooths_yolo_keypoint_jitter(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        first = service._detect(frame, person_detection())
        shifted_person = person_detection()
        for point in shifted_person["keypoints"].values():
            point["normalized_x"] += 0.10
        second = service._detect(frame, shifted_person)

        first_center = (first["crop_box"][0] + first["crop_box"][2]) / 2
        second_center = (second["crop_box"][0] + second["crop_box"][2]) / 2
        self.assertGreater(second_center, first_center)
        self.assertLess(second_center - first_center, 640 * 0.10)

    def test_returns_all_21_hand_landmarks(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = service._detect(frame, person_detection())

        self.assertEqual(len(result["landmarks"]), 21)
        self.assertIn("thumb_tip", result["landmarks"])
        self.assertIn("middle_tip", result["landmarks"])
        self.assertIn("ring_tip", result["landmarks"])
        self.assertIn("pinky_tip", result["landmarks"])
        self.assertEqual(result["handedness"], "Right")
        self.assertAlmostEqual(result["handedness_score"], 0.95)

    def test_saves_annotated_crop_for_debugging(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            debug_path = Path(directory) / "hand.jpg"
            result = service._detect(
                frame, person_detection(), debug_path=debug_path
            )

            self.assertTrue(debug_path.is_file())
            self.assertEqual(result["debug_image_path"], str(debug_path))

if __name__ == "__main__":
    unittest.main()
