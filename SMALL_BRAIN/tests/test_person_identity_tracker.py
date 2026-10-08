import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from actions.tracking.person_identity_tracker import (
    EnrollmentCandidate,
    PersonIdentityTracker,
)


def person(x1, y1, x2, y2, embedding, confidence=0.9):
    keypoints = {
        name: {
            "normalized_x": (x1 + x2) * 0.5,
            "normalized_y": (y1 + y2) * 0.5,
            "confidence": 0.9,
        }
        for name in (
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "right_wrist",
        )
    }
    return {
        "class": "person",
        "confidence": confidence,
        "bbox": {
            "normalized_x1": x1,
            "normalized_y1": y1,
            "normalized_x2": x2,
            "normalized_y2": y2,
        },
        "keypoints": keypoints,
        "_test_embedding": embedding,
    }


class FakeReIDService:
    model_name = "fake-reid"

    async def extract_embedding_async(self, frame_bgr, bbox, padding=0.0):
        del frame_bgr, padding
        return np.asarray(bbox["_test_embedding"], dtype=np.float32)


class PersonIdentityTrackerTests(unittest.IsolatedAsyncioTestCase):

    def make_tracker(self, directory, **options):
        root = Path(directory)
        return PersonIdentityTracker(
            FakeReIDService(),
            profile_path=root / "profile.json",
            artifact_directory=root / "images",
            **options,
        )

    async def test_keeps_current_geometric_behavior_without_running_reid(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.make_tracker(directory)
            initial = person(0.10, 0.10, 0.35, 0.90, [1, 0])
            continued = person(0.12, 0.10, 0.37, 0.90, [1, 0])
            stranger = person(0.65, 0.10, 0.90, 0.90, [0, 1])
            tracker.start(initial)

            decision = await tracker.observe(
                sequence=1,
                frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
                detections=[stranger, continued],
            )

            self.assertIs(decision.person, continued)
            self.assertEqual(decision.reason, "geometric_continuity")
            self.assertTrue(decision.allow_lidar_seed)

    async def test_reid_recovers_only_after_geometry_fails_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.make_tracker(
                directory,
                recovery_interval_seconds=0.0,
                recovery_confirmations=2,
            )
            tracker.embedding_record.add("front", [1, 0], quality=1.0)
            tracker.start(person(0.05, 0.10, 0.25, 0.90, [1, 0]))
            target = person(0.65, 0.10, 0.85, 0.90, [1, 0])
            stranger = person(0.35, 0.10, 0.55, 0.90, [0, 1])
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            pending = await tracker.observe(
                sequence=2,
                frame_bgr=frame,
                detections=[stranger, target],
            )
            recovered = await tracker.observe(
                sequence=3,
                frame_bgr=frame,
                detections=[stranger, target],
            )

            self.assertIsNone(pending.person)
            self.assertEqual(pending.reason, "reid_confirmation_pending")
            self.assertIs(recovered.person, target)
            self.assertEqual(recovered.reason, "reid_recovered")
            self.assertEqual(recovered.identity, "verified")

    async def test_enrollment_saves_quality_crops_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.make_tracker(directory, samples_per_view=2)
            tracked = person(0.15, 0.10, 0.85, 0.90, [1, 0])
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            for row in range(120):
                frame[row, :, :] = 255 if row % 2 else 0
            self.assertGreater(
                cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                              cv2.CV_64F).var(),
                tracker.MIN_BLUR_VARIANCE,
            )
            quality = tracker.candidate_quality(frame,
                                                tracked,
                                                enrollment=True)
            self.assertIsNotNone(quality)
            candidates = [
                EnrollmentCandidate(frame, tracked, quality, float(index + 1))
                for index in range(3)
            ]

            result = await tracker.enroll(candidates)

            self.assertEqual(result["view"], "front")
            self.assertEqual(result["accepted"], 2)
            self.assertTrue(Path(result["profile_path"]).is_file())
            samples = tracker.embedding_record.samples("front")
            self.assertEqual(len(samples), 2)
            self.assertTrue(
                all(Path(item.image_path).is_file() for item in samples))
            self.assertEqual(tracker.next_enrollment_view(), "side")


if __name__ == "__main__":
    unittest.main()
