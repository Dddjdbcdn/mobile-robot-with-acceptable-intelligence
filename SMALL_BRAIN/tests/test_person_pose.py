import unittest

from actions.tracking.person_pose import (
    TorsoTargetEstimator,
    find_best_person_path,
    person_keypoint,
    predict_next_keypoint,
)


class PersonPoseTest(unittest.TestCase):
    def test_torso_center_requires_a_shoulder_and_a_hip(self):
        person = {
            "bbox": {
                "normalized_center_x": 0.8,
                "normalized_center_y": 0.7,
            },
            "keypoints": {
                "left_hip": {"normalized_x": 0.4, "normalized_y": 0.6},
            },
        }

        self.assertIsNone(person_keypoint(person, "torso_center"))

    def test_torso_center_uses_visible_shoulder_and_hip(self):
        person = {
            "keypoints": {
                "left_shoulder": {
                    "normalized_x": 0.4,
                    "normalized_y": 0.2,
                },
                "left_hip": {"normalized_x": 0.6, "normalized_y": 0.6},
            },
        }

        self.assertEqual(
            person_keypoint(person, "torso_center"),
            {"normalized_x": 0.5, "normalized_y": 0.4},
        )

    def test_feet_path_predicts_beyond_hip_when_torso_is_not_visible(self):
        path = find_best_person_path(
            {
                "keypoints": {
                    "left_ankle": {
                        "normalized_x": 0.4,
                        "normalized_y": 0.9,
                    },
                },
            },
            "person",
        )
        person = {
            "keypoints": {
                "left_ankle": {"normalized_x": 0.4, "normalized_y": 0.9},
                "left_knee": {"normalized_x": 0.4, "normalized_y": 0.7},
                "left_hip": {"normalized_x": 0.4, "normalized_y": 0.5},
            },
        }

        self.assertEqual(
            path,
            ["left_ankle", "left_knee", "left_hip", "torso_center"],
        )
        predicted = predict_next_keypoint(person, path, 2)
        self.assertAlmostEqual(predicted[0], 0.4)
        self.assertAlmostEqual(predicted[1], 0.3)
        self.assertIsNone(person_keypoint(person, "torso_center"))

    def test_torso_estimator_waits_for_prediction_before_bootstrap(self):
        estimator = TorsoTargetEstimator(bootstrap_offset_y=-0.25)
        person = {
            "bbox": {
                "normalized_y1": 0.1,
                "normalized_y2": 0.9,
            },
            "keypoints": {
                "left_hip": {
                    "normalized_x": 0.4,
                    "normalized_y": 0.7,
                },
                "right_hip": {
                    "normalized_x": 0.6,
                    "normalized_y": 0.7,
                },
            },
        }

        self.assertIsNone(estimator.update(person, allow_bootstrap=False))
        target = estimator.update(person)

        self.assertAlmostEqual(target["normalized_x"], 0.5)
        self.assertAlmostEqual(target["normalized_y"], 0.5)
        self.assertEqual(target["source"], "bootstrap_offset")

    def test_torso_estimator_calibrates_then_survives_shoulder_loss(self):
        estimator = TorsoTargetEstimator()
        observed = {
            "bbox": {
                "normalized_y1": 0.1,
                "normalized_y2": 0.9,
            },
            "keypoints": {
                "left_shoulder": {
                    "normalized_x": 0.4,
                    "normalized_y": 0.3,
                },
                "right_shoulder": {
                    "normalized_x": 0.6,
                    "normalized_y": 0.3,
                },
                "left_hip": {
                    "normalized_x": 0.4,
                    "normalized_y": 0.7,
                },
                "right_hip": {
                    "normalized_x": 0.6,
                    "normalized_y": 0.7,
                },
            },
        }
        calibrated = estimator.update(observed)
        hips_only = {
            "bbox": observed["bbox"],
            "keypoints": {
                "left_hip": {
                    "normalized_x": 0.5,
                    "normalized_y": 0.7,
                },
                "right_hip": {
                    "normalized_x": 0.7,
                    "normalized_y": 0.7,
                },
            },
        }

        estimated = estimator.update(hips_only)

        self.assertAlmostEqual(calibrated["normalized_x"], 0.5)
        self.assertAlmostEqual(calibrated["normalized_y"], 0.5)
        self.assertAlmostEqual(estimated["normalized_x"], 0.6)
        self.assertAlmostEqual(estimated["normalized_y"], 0.5)
        self.assertEqual(estimated["source"], "calibrated_offset")

    def test_torso_estimator_uses_calibrated_shoulder_when_hips_disappear(self):
        estimator = TorsoTargetEstimator()
        observed = {
            "bbox": {
                "normalized_y1": 0.1,
                "normalized_y2": 0.9,
            },
            "keypoints": {
                "left_shoulder": {
                    "normalized_x": 0.4,
                    "normalized_y": 0.3,
                },
                "left_hip": {
                    "normalized_x": 0.4,
                    "normalized_y": 0.7,
                },
            },
        }
        estimator.update(observed)
        shoulder_only = {
            "bbox": observed["bbox"],
            "keypoints": {
                "left_shoulder": {
                    "normalized_x": 0.5,
                    "normalized_y": 0.3,
                },
            },
        }

        estimated = estimator.update(shoulder_only)

        self.assertAlmostEqual(estimated["normalized_x"], 0.5)
        self.assertAlmostEqual(estimated["normalized_y"], 0.5)
        self.assertEqual(estimated["source"], "calibrated_offset")

    def test_torso_estimator_refreshes_offsets_from_each_visible_center(self):
        estimator = TorsoTargetEstimator()
        first_pose = {
            "bbox": {
                "normalized_y1": 0.1,
                "normalized_y2": 0.9,
            },
            "keypoints": {
                "left_shoulder": {
                    "normalized_x": 0.5,
                    "normalized_y": 0.3,
                },
                "left_hip": {
                    "normalized_x": 0.5,
                    "normalized_y": 0.7,
                },
            },
        }
        changed_pose = {
            "bbox": first_pose["bbox"],
            "keypoints": {
                "left_shoulder": {
                    "normalized_x": 0.5,
                    "normalized_y": 0.1,
                },
                "left_hip": {
                    "normalized_x": 0.5,
                    "normalized_y": 0.7,
                },
            },
        }
        hip_only = {
            "bbox": first_pose["bbox"],
            "keypoints": {
                "left_hip": {
                    "normalized_x": 0.5,
                    "normalized_y": 0.7,
                },
            },
        }

        estimator.update(first_pose)
        estimator.update(changed_pose)
        estimated = estimator.update(hip_only)

        self.assertAlmostEqual(estimated["normalized_x"], 0.5)
        self.assertAlmostEqual(estimated["normalized_y"], 0.4)
        self.assertEqual(estimated["source"], "calibrated_offset")


if __name__ == "__main__":
    unittest.main()
