import unittest

from cognition.hand.interface import HandGestureInterface


def person(wrist_x, confidence=0.8, wrist_confidence=0.8):
    return {
        "class": "person",
        "confidence": confidence,
        "bbox": {},
        "keypoints": {
            "right_wrist": {
                "normalized_x": wrist_x,
                "normalized_y": 0.5,
                "confidence": wrist_confidence,
            },
        },
        "keypoint_confidences": {
            "right_wrist": wrist_confidence,
        },
    }


class WelcomePersonBindingTests(unittest.TestCase):
    def test_selects_box_whose_wrist_is_closest_to_confirmed_palm(self):
        high_confidence_duplicate = person(0.62, confidence=0.95)
        palm_aligned = person(0.51, confidence=0.70)

        selected = HandGestureInterface._welcome_person_candidate(
            [high_confidence_duplicate, palm_aligned],
            (0.50, 0.50),
        )

        self.assertIs(selected, palm_aligned)

    def test_rejects_low_confidence_person(self):
        low_confidence = person(0.50, confidence=0.40)

        selected = HandGestureInterface._welcome_person_candidate(
            [low_confidence],
            (0.50, 0.50),
        )

        self.assertIsNone(selected)


class WelcomePersonDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_welcome_passes_exact_person_to_tracking(self):
        interface = HandGestureInterface(hand_landmarks=None)
        target = person(0.50)
        calls = []

        async def dispatch(command, data):
            calls.append((command, data))
            if command == "observe_gesture":
                return {"tracking_person": False, "movement_active": False}
            return True

        interface.set_dispatcher(dispatch)
        for _ in range(interface.IDLE_WELCOME_CONFIRM_FRAMES):
            await interface._observe_gesture(
                "welcome",
                (0.50, 0.50),
                [target],
            )

        watch_calls = [item for item in calls if item[0] == "watch_target"]
        self.assertEqual(len(watch_calls), 1)
        self.assertIs(watch_calls[0][1]["initial_person"], target)


if __name__ == "__main__":
    unittest.main()
