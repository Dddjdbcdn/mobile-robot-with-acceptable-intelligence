import copy
import unittest

from cognition.hand.gestures import HandGestureClassifier


def point(x, y):
    return {
        "x": float(x),
        "y": float(y),
        "normalized_x": float(x) / 1000.0,
        "normalized_y": float(y) / 1000.0,
    }


def ok_hand():
    return {
        "image_width": 1000,
        "image_height": 1000,
        "landmarks": {
            "wrist": point(500, 800),
            "thumb_cmc": point(350, 620),
            "thumb_mcp": point(390, 550),
            "thumb_ip": point(450, 480),
            "thumb_tip": point(500, 435),
            "index_mcp": point(400, 600),
            "index_pip": point(390, 480),
            "index_dip": point(485, 440),
            "index_tip": point(500, 430),
            "middle_mcp": point(470, 600),
            "middle_pip": point(470, 480),
            "middle_dip": point(470, 380),
            "middle_tip": point(470, 280),
            "ring_mcp": point(540, 600),
            "ring_pip": point(540, 490),
            "ring_dip": point(540, 400),
            "ring_tip": point(540, 310),
            "pinky_mcp": point(600, 600),
            "pinky_pip": point(600, 510),
            "pinky_dip": point(600, 435),
            "pinky_tip": point(600, 360),
        },
    }


class HandGestureClassifierTests(unittest.TestCase):
    def test_recognizes_ok_pose(self):
        result = HandGestureClassifier.classify(ok_hand())

        self.assertEqual(result["gesture"], "ok")
        self.assertTrue(result["gesture_checks"]["ok_tips_touching"])
        self.assertLess(
            result["gesture_checks"]["ok_tip_gap_distance"],
            result["gesture_checks"]["ok_index_tip_to_pip_distance"],
        )
        self.assertTrue(result["gesture_checks"]["ok_index_bent"])
        self.assertGreater(
            result["gesture_checks"]["ok_index_pip_bend_degrees"],
            45.0,
        )
        self.assertTrue(
            result["gesture_checks"]["ok_three_fingers_up"]
        )

    def test_rejects_pose_when_fingertips_do_not_touch(self):
        hand = ok_hand()
        hand["landmarks"]["thumb_tip"] = point(400, 520)

        result = HandGestureClassifier.classify(hand)

        self.assertNotEqual(result["gesture"], "ok")
        self.assertFalse(result["gesture_checks"]["ok_tips_touching"])

    def test_accepts_relaxed_fingers_when_their_y_order_points_up(self):
        hand = ok_hand()
        hand["landmarks"]["middle_pip"] = point(430, 500)
        hand["landmarks"]["middle_dip"] = point(500, 400)
        hand["landmarks"]["middle_tip"] = point(450, 300)

        result = HandGestureClassifier.classify(hand)

        self.assertFalse(
            HandGestureClassifier._finger_is_straight(
                hand,
                hand["landmarks"],
                "middle",
            )
        )
        self.assertEqual(result["gesture"], "ok")
        self.assertTrue(result["gesture_checks"]["ok_three_fingers_up"])

    def test_rejects_pinch_with_folded_remaining_fingers(self):
        hand = copy.deepcopy(ok_hand())
        for offset, finger in enumerate(("middle", "ring", "pinky")):
            x = 470 + offset * 65
            hand["landmarks"][f"{finger}_pip"] = point(x, 520)
            hand["landmarks"][f"{finger}_dip"] = point(x + 45, 570)
            hand["landmarks"][f"{finger}_tip"] = point(x, 610)

        result = HandGestureClassifier.classify(hand)

        self.assertNotEqual(result["gesture"], "ok")
        self.assertFalse(
            result["gesture_checks"]["ok_three_fingers_up"]
        )

    def test_rejects_ok_pose_when_index_pip_bend_is_only_45_degrees(self):
        hand = ok_hand()
        hand["landmarks"]["index_mcp"] = point(400, 600)
        hand["landmarks"]["index_pip"] = point(400, 500)
        hand["landmarks"]["index_dip"] = point(500, 400)

        result = HandGestureClassifier.classify(hand)

        self.assertNotEqual(result["gesture"], "ok")
        self.assertFalse(result["gesture_checks"]["ok_index_bent"])
        self.assertAlmostEqual(
            result["gesture_checks"]["ok_index_pip_bend_degrees"],
            45.0,
        )


if __name__ == "__main__":
    unittest.main()
