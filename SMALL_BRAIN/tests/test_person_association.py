import unittest

from actions.tracking.person_association import (
    select_tracked_person,
    update_tracked_bbox,
)


def person(bbox, confidence):
    x1, y1, x2, y2 = bbox
    return {
        "class": "person",
        "confidence": confidence,
        "bbox": {
            "normalized_x1": x1,
            "normalized_y1": y1,
            "normalized_x2": x2,
            "normalized_y2": y2,
        },
        "keypoints": {"nose": {"normalized_x": 0.5, "normalized_y": 0.2}},
    }


class PersonAssociationTests(unittest.TestCase):
    def test_prefers_previous_target_over_higher_confidence_person(self):
        previous = person((0.10, 0.05, 0.45, 0.95), 0.8)["bbox"]
        target = person((0.12, 0.05, 0.47, 0.95), 0.7)
        stranger = person((0.60, 0.05, 0.95, 0.95), 0.99)

        selected = select_tracked_person([stranger, target], previous)

        self.assertIs(selected, target)

    def test_prefers_duplicate_aligned_with_previous_box(self):
        previous = person((0.15, 0.05, 0.85, 0.95), 0.8)["bbox"]
        aligned = person((0.16, 0.05, 0.84, 0.95), 0.65)
        partial = person((0.35, 0.20, 0.75, 0.80), 0.98)

        selected = select_tracked_person([partial, aligned], previous)

        self.assertIs(selected, aligned)

    def test_returns_none_instead_of_switching_to_unmatched_person(self):
        previous = person((0.05, 0.05, 0.35, 0.95), 0.8)["bbox"]
        stranger = person((0.65, 0.05, 0.95, 0.95), 0.99)

        selected = select_tracked_person([stranger], previous)

        self.assertIsNone(selected)

    def test_bbox_state_moves_gradually(self):
        previous = person((0.10, 0.10, 0.50, 0.90), 0.8)["bbox"]
        current = person((0.20, 0.20, 0.60, 1.00), 0.8)

        updated = update_tracked_bbox(previous, current, alpha=0.25)

        self.assertAlmostEqual(updated["normalized_x1"], 0.125)
        self.assertAlmostEqual(updated["normalized_y1"], 0.125)
        self.assertAlmostEqual(updated["normalized_x2"], 0.525)
        self.assertAlmostEqual(updated["normalized_y2"], 0.925)


if __name__ == "__main__":
    unittest.main()
