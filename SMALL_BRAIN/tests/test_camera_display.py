import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from services.display import camera_display
from services.vision.person_tracking_service import PersonTrackSnapshot


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
    def test_debug_board_toggle_is_clickable_and_shows_state(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with patch.object(camera_display.cv2, "putText") as put_text:
            bounds = camera_display.draw_debug_board_toggle(frame, False)

        left, top, right, bottom = bounds
        self.assertIn("DEBUG OFF", put_text.call_args.args[1])
        self.assertTrue(camera_display.debug_board_toggle_clicked(
            camera_display.cv2.EVENT_LBUTTONUP,
            (left + right) // 2,
            (top + bottom) // 2,
            bounds,
        ))
        self.assertFalse(camera_display.debug_board_toggle_clicked(
            camera_display.cv2.EVENT_LBUTTONUP,
            0,
            0,
            bounds,
        ))

    def test_scene_id_is_attached_to_existing_yolo_label(self):
        detection = person(100, 300, 200)
        track = PersonTrackSnapshot(
            track_id=7,
            state="confirmed",
            visible=True,
            person=detection,
            bbox=(0.1, 0.1, 0.3, 0.9),
            association_confidence=0.91,
            appearance_samples=4,
            age_seconds=2.0,
            missing_seconds=0.0,
            appearance_match_source="G3",
            reid_similarity=0.84,
            association_cost=0.19,
            geometry_cost=0.24,
            appearance_cost=0.15,
            pose_cost=0.20,
        )
        yolo = SimpleNamespace(detections=[detection])
        tracker = SimpleNamespace(tracked_person_id=None)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with (
            patch.object(camera_display.cv2, "putText") as put_text,
            patch.object(camera_display.cv2, "rectangle") as rectangle,
        ):
            camera_display.draw_yolo_overlay(
                frame,
                yolo,
                tracker,
                [track],
            )

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertTrue(any("ID 7" in label for label in labels))
        self.assertTrue(any("G3:0.84" in label for label in labels))
        self.assertIn("C0.19 G0.24 A0.15 P0.20", labels)
        outline_boxes = [
            call for call in rectangle.call_args_list
            if len(call.args) >= 5 and call.args[4] == 2
        ]
        self.assertEqual(len(outline_boxes), 1)

    def test_raw_person_is_drawn_without_tentative_id(self):
        detection = person(100, 300, 200)
        tentative = PersonTrackSnapshot(
            track_id=7,
            state="tentative",
            visible=True,
            person=detection,
            bbox=(0.1, 0.1, 0.3, 0.9),
            association_confidence=0.91,
            appearance_samples=1,
            age_seconds=0.2,
            missing_seconds=0.0,
        )
        yolo = SimpleNamespace(detections=[detection])
        tracker = SimpleNamespace(tracked_person_id=None)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with (
            patch.object(camera_display.cv2, "rectangle") as rectangle,
            patch.object(camera_display.cv2, "putText") as put_text,
        ):
            camera_display.draw_yolo_overlay(
                frame,
                yolo,
                tracker,
                [tentative],
            )

        outline_colors = [
            call.args[3]
            for call in rectangle.call_args_list
            if len(call.args) >= 5 and call.args[4] == 2
        ]
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(outline_colors, [(0, 0, 255)])
        self.assertFalse(any("ID 7" in label for label in labels))

    def test_geometry_only_cost_label_marks_missing_reid_and_pose_fallback(
            self):
        detection = person(100, 300, 200)
        track = PersonTrackSnapshot(
            track_id=3,
            state="confirmed",
            visible=True,
            person=detection,
            bbox=(0.1, 0.1, 0.3, 0.9),
            association_confidence=0.88,
            appearance_samples=2,
            age_seconds=1.0,
            missing_seconds=0.0,
            association_cost=0.12,
            geometry_cost=0.07,
            appearance_cost=None,
            pose_cost=0.50,
            pose_cost_is_fallback=True,
        )
        yolo = SimpleNamespace(detections=[detection])
        tracker = SimpleNamespace(tracked_person_id=None)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with patch.object(camera_display.cv2, "putText") as put_text:
            camera_display.draw_yolo_overlay(
                frame,
                yolo,
                tracker,
                [track],
            )

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("C0.12 G0.07 A-- P0.50*", labels)

    def test_raw_yolo_people_are_not_marked_as_the_tracked_person(self):
        detections = [person(300, 500, 400), person(100, 220, 200)]
        yolo = SimpleNamespace(detections=detections)
        tracker = SimpleNamespace(tracked_person_id=9)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with (
            patch.object(camera_display, "_draw_pose_body_mask") as draw_mask,
            patch.object(camera_display.cv2, "rectangle") as rectangle,
        ):
            camera_display.draw_yolo_overlay(frame, yolo, tracker)

        draw_mask.assert_not_called()
        outline_colors = [
            call.args[3]
            for call in rectangle.call_args_list
            if len(call.args) >= 5 and call.args[4] == 2
        ]
        self.assertEqual(outline_colors, [(0, 0, 255), (0, 0, 255)])

    def test_persistent_track_id_controls_blue_target_box(self):
        first = person(300, 500, 400)
        second = person(100, 220, 200)
        tracks = [
            PersonTrackSnapshot(
                track_id=4,
                state="confirmed",
                visible=True,
                person=first,
                bbox=(0.47, 0.06, 0.78, 0.83),
                association_confidence=0.9,
                appearance_samples=2,
                age_seconds=1.0,
                missing_seconds=0.0,
            ),
            PersonTrackSnapshot(
                track_id=9,
                state="confirmed",
                visible=True,
                person=second,
                bbox=(0.16, 0.06, 0.34, 0.83),
                association_confidence=0.9,
                appearance_samples=2,
                age_seconds=1.0,
                missing_seconds=0.0,
            ),
        ]
        yolo = SimpleNamespace(detections=[first, second])
        tracker = SimpleNamespace(
            tracked_person_id=9,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with patch.object(camera_display.cv2, "rectangle") as rectangle:
            camera_display.draw_yolo_overlay(frame, yolo, tracker, tracks)

        outline_colors = [
            call.args[3]
            for call in rectangle.call_args_list
            if len(call.args) >= 5 and call.args[4] == 2
        ]
        self.assertEqual(outline_colors, [(0, 0, 255), (255, 0, 0)])


if __name__ == "__main__":
    unittest.main()
