import unittest

from actions.tracking.person_pose import (
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


if __name__ == "__main__":
    unittest.main()
