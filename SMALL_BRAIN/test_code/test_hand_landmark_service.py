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


class FakeLandmarker:
    def __init__(self):
        self.timestamps = []

    def detect_for_video(self, _image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        points = [Point(0.5, 0.5) for _ in range(21)]
        points[6] = Point(0.45, 0.45)
        points[7] = Point(0.40, 0.60)
        points[8] = Point(0.35, 0.75)
        return type("Result", (), {"hand_landmarks": [points]})()


class HandLandmarkServiceTests(unittest.TestCase):
    def test_video_mode_timestamps_are_strictly_increasing(self):
        landmarker = FakeLandmarker()
        service = HandLandmarkService(
            camera=None,
            landmarker=landmarker,
            image_factory=lambda image: image,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        person = {
            "keypoints": {
                "right_elbow": {"x": 400.0, "y": 400.0},
                "right_wrist": {"x": 600.0, "y": 300.0},
            }
        }

        first = service._detect(frame, person)
        second = service._detect(frame, person)

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

    def test_maps_crop_landmarks_back_to_full_frame(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        person = {
            "keypoints": {
                "right_elbow": {"x": 260.0, "y": 180.0},
                "right_wrist": {"x": 300.0, "y": 140.0},
            }
        }

        result = service._detect(frame, person)

        self.assertEqual(result["image_width"], 640)
        self.assertEqual(result["image_height"], 360)
        self.assertIn("index_pip", result["landmarks"])
        self.assertIn("index_dip", result["landmarks"])
        self.assertIn("index_tip", result["landmarks"])
        self.assertLess(
            result["landmarks"]["index_tip"]["normalized_x"],
            result["landmarks"]["index_pip"]["normalized_x"],
        )

    def test_requires_yolo_right_arm_for_the_crop(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.assertIsNone(service._detect(frame, {"keypoints": {}}))

    def test_full_resolution_crop_uses_normalized_pose_coordinates(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        person = {
            "keypoints": {
                "right_elbow": {
                    "x": 260.0, "y": 180.0,
                    "normalized_x": 260.0 / 640.0,
                    "normalized_y": 180.0 / 360.0,
                },
                "right_wrist": {
                    "x": 300.0, "y": 140.0,
                    "normalized_x": 300.0 / 640.0,
                    "normalized_y": 140.0 / 360.0,
                },
            }
        }

        crop_box = service._crop_box(frame.shape, person)
        result = service._detect(frame, person)

        self.assertEqual(result["image_width"], 1280)
        self.assertEqual(result["image_height"], 720)
        self.assertGreater(crop_box[2] - crop_box[0], 96)
        self.assertLess(crop_box[0], 600)
        self.assertGreater(crop_box[2], 600)

    def test_crop_geometry_scales_with_forearm_length(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        first_person = {
            "keypoints": {
                "right_elbow": {
                    "normalized_x": 0.40, "normalized_y": 0.55,
                },
                "right_wrist": {
                    "normalized_x": 0.48, "normalized_y": 0.38,
                },
            }
        }
        moved_person = {
            "keypoints": {
                "right_elbow": {
                    "normalized_x": 0.38, "normalized_y": 0.60,
                },
                "right_wrist": {
                    "normalized_x": 0.56, "normalized_y": 0.30,
                },
            }
        }

        first_box = service._crop_box(frame.shape, first_person)
        moved_box = service._crop_box(frame.shape, moved_person)

        self.assertEqual(
            first_box[2] - first_box[0], first_box[3] - first_box[1]
        )
        self.assertEqual(
            moved_box[2] - moved_box[0], moved_box[3] - moved_box[1]
        )
        self.assertGreater(
            moved_box[2] - moved_box[0], first_box[2] - first_box[0]
        )

    def test_returns_all_21_hand_landmarks(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        person = {
            "keypoints": {
                "right_elbow": {"x": 400.0, "y": 400.0},
                "right_wrist": {"x": 600.0, "y": 300.0},
            }
        }

        result = service._detect(frame, person)

        self.assertEqual(len(result["landmarks"]), 21)
        self.assertIn("thumb_tip", result["landmarks"])
        self.assertIn("middle_tip", result["landmarks"])
        self.assertIn("ring_tip", result["landmarks"])
        self.assertIn("pinky_tip", result["landmarks"])

    def test_dynamic_crop_contains_wrist_and_forward_hand_region(self):
        frame_shape = (720, 1280, 3)
        elbow = (400.0, 400.0)
        wrist = (600.0, 300.0)
        person = {
            "keypoints": {
                "right_elbow": {"x": elbow[0], "y": elbow[1]},
                "right_wrist": {"x": wrist[0], "y": wrist[1]},
            }
        }

        box = HandLandmarkService._crop_box(frame_shape, person)
        forward_hand = (
            wrist[0] + (wrist[0] - elbow[0]),
            wrist[1] + (wrist[1] - elbow[1]),
        )

        for x, y in (wrist, forward_hand):
            self.assertLessEqual(box[0], x)
            self.assertLess(x, box[2])
            self.assertLessEqual(box[1], y)
            self.assertLess(y, box[3])

    def test_edge_crop_keeps_constant_square_dimensions(self):
        frame_shape = (360, 640, 3)
        person = {
            "keypoints": {
                "right_elbow": {"x": 30.0, "y": 170.0},
                "right_wrist": {"x": 5.0, "y": 120.0},
            }
        }

        box = HandLandmarkService._crop_box(frame_shape, person)

        self.assertEqual(box[0], 0)
        self.assertEqual(box[2] - box[0], box[3] - box[1])

    def test_saves_annotated_hand_crop_for_debugging(self):
        service = HandLandmarkService(
            camera=None,
            landmarker=FakeLandmarker(),
            image_factory=lambda image: image,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        person = {
            "keypoints": {
                "right_elbow": {"x": 260.0, "y": 180.0},
                "right_wrist": {"x": 300.0, "y": 140.0},
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            debug_path = Path(directory) / "hand.jpg"
            result = service._detect(frame, person, debug_path=debug_path)

            self.assertTrue(debug_path.is_file())
            self.assertEqual(result["debug_image_path"], str(debug_path))

if __name__ == "__main__":
    unittest.main()
