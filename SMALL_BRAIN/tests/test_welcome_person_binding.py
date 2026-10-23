import unittest

from cognition.hand.interface import HandGestureInterface


def person(x1=0.25, y1=0.2, x2=0.75, y2=0.9, confidence=0.8):
    return {
        "class": "person",
        "confidence": confidence,
        "bbox": {
            "normalized_x1": x1,
            "normalized_y1": y1,
            "normalized_x2": x2,
            "normalized_y2": y2,
        },
    }


class WelcomePersonBindingTests(unittest.TestCase):

    def test_selects_visible_person_box_containing_confirmed_palm(self):
        outside = person(0.55, 0.2, 0.9, 0.9, confidence=0.95)
        containing = person(0.2, 0.2, 0.6, 0.9, confidence=0.70)

        selected = HandGestureInterface._welcome_person_candidate(
            [outside, containing],
            (0.50, 0.50),
        )

        self.assertIs(selected, containing)

    def test_does_not_require_wrist_keypoint(self):
        visible_person = person()

        selected = HandGestureInterface._welcome_person_candidate(
            [visible_person],
            (0.50, 0.50),
        )

        self.assertIs(selected, visible_person)

    def test_rejects_low_confidence_person(self):
        low_confidence = person(confidence=0.40)

        selected = HandGestureInterface._welcome_person_candidate(
            [low_confidence],
            (0.50, 0.50),
        )

        self.assertIsNone(selected)


class WelcomePersonDispatchTests(unittest.IsolatedAsyncioTestCase):

    async def test_confirmed_welcome_passes_exact_person_to_tracking(self):
        interface = HandGestureInterface(hand_landmarks=None)
        target = person()
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

    async def test_push_does_not_stop_an_active_follow(self):
        interface = HandGestureInterface(hand_landmarks=None)
        calls = []

        async def dispatch(command, data):
            calls.append((command, data))
            if command == "observe_gesture":
                return {
                    "tracking_person": True,
                    "movement_active": True,
                    "mode": "following",
                }
            return True

        interface.set_dispatcher(dispatch)
        for _ in range(interface.MOVEMENT_STOP_CONFIRM_FRAMES + 1):
            await interface._observe_gesture("push", (0.50, 0.50), [])

        stop_calls = [item for item in calls if item[0] == "stop_movement"]
        self.assertEqual(stop_calls, [])

    async def test_confirmed_thumb_up_stops_an_active_follow(self):
        interface = HandGestureInterface(hand_landmarks=None)
        calls = []

        async def dispatch(command, data):
            calls.append((command, data))
            if command == "observe_gesture":
                return {
                    "tracking_person": True,
                    "movement_active": True,
                    "mode": "following",
                }
            return True

        interface.set_dispatcher(dispatch)
        for _ in range(interface.TRACKING_STOP_CONFIRM_FRAMES):
            await interface._observe_gesture("thumb_up", (0.50, 0.50), [])

        stop_calls = [item for item in calls if item[0] == "stop_follow"]
        self.assertEqual(len(stop_calls), 1)
        self.assertEqual(stop_calls[0][1]["reason"], "HAND_THUMB_UP")


if __name__ == "__main__":
    unittest.main()
