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
    def test_person_status_shows_id_gallery_size_and_update_age(self):
        track = PersonTrackSnapshot(
            track_id=14,
            state="lost",
            visible=False,
            person=None,
            bbox=(0.1, 0.1, 0.3, 0.9),
            association_confidence=0.0,
            appearance_samples=4,
            age_seconds=8.0,
            missing_seconds=1.2,
            gallery_update_age_seconds=2.6,
        )

        value, color = camera_display.person_tracking_status_display([track])

        self.assertEqual(
            value,
            "ID 14  |  4 IMAGES  |  2.6s",
        )
        self.assertEqual(color, (0, 210, 255))

    def test_person_status_shows_empty_state(self):
        value, color = camera_display.person_tracking_status_display(())

        self.assertEqual(
            value,
            "ID --  |  0 IMAGES  |  --",
        )
        self.assertEqual(color, (170, 170, 170))

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

    def test_audio_controls_are_independently_clickable(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with patch.object(camera_display.cv2, "putText") as put_text:
            bounds = camera_display.draw_audio_controls(frame, False, True)

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("MUTE", labels)
        self.assertNotIn("MUTE [M]", labels)
        self.assertIn("UNCOVER EAR [E]", labels)
        for name in ("speaker", "ear"):
            left, top, right, bottom = bounds[name]
            self.assertTrue(camera_display.control_button_clicked(
                camera_display.cv2.EVENT_LBUTTONUP,
                (left + right) // 2,
                (top + bottom) // 2,
                bounds[name],
            ))

        with patch.object(camera_display.cv2, "putText"):
            debug_bounds = camera_display.draw_debug_board_toggle(
                frame, False, bounds["speaker"]
            )
        self.assertEqual(debug_bounds[0], bounds["speaker"][0])
        self.assertLess(debug_bounds[3], bounds["speaker"][1])

    def test_vision_mode_control_shows_mode_and_lock_state(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with patch.object(camera_display.cv2, "putText") as put_text:
            bounds = camera_display.draw_vision_mode_control(
                frame, object_mode=True, enabled=True
            )
        self.assertIn("YOLO OBJECT [O]", put_text.call_args.args[1])
        left, top, right, bottom = bounds
        self.assertTrue(camera_display.control_button_clicked(
            camera_display.cv2.EVENT_LBUTTONUP,
            (left + right) // 2,
            (top + bottom) // 2,
            bounds,
        ))

        with patch.object(camera_display.cv2, "putText") as put_text:
            camera_display.draw_vision_mode_control(
                frame, object_mode=False, enabled=False
            )
        self.assertIn("YOLO POSE LOCKED", put_text.call_args.args[1])

    def test_debug_board_no_longer_shows_lift_row(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        previous_health = camera_display.robot_state.get("stm32_health")
        camera_display.robot_state["stm32_health"] = {
            "message": "DEGRADED failing=tof_sensor reason=no_valid_targets rc=0",
            "age_seconds": 0.4,
        }
        track_action = SimpleNamespace(
            target=None,
            action_id=None,
            active=False,
            stable_seeds=SimpleNamespace(
                status=lambda **_kwargs: {
                    "valid": False,
                    "sample_count": 0,
                    "sample_target": 3,
                }
            ),
        )
        hand = SimpleNamespace(gesture_status=lambda: {})

        try:
            with patch.object(camera_display.cv2, "putText") as put_text:
                camera_display.draw_tracking_status_overlay(
                    frame, track_action, hand
                )
        finally:
            camera_display.robot_state["stm32_health"] = previous_health

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertNotIn("LIFT", labels)
        self.assertNotIn("STM32", labels)

    def test_reset_overlay_shows_recovery_and_brief_ready_state(self):
        state = camera_display.ResetOverlayState(ready_seconds=3.0)

        self.assertEqual(state.visible_state("lifted", 10.0), "lifted")
        self.assertEqual(
            state.visible_state("resetting_lift", 11.0), "resetting_lift"
        )
        self.assertEqual(state.visible_state("ready", 12.0), "ready")
        self.assertIsNone(state.visible_state("ready", 15.1))

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with patch.object(camera_display.cv2, "putText") as put_text:
            camera_display.draw_reset_state_overlay(frame, "lifted")
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("ROBOT LIFTED", labels)
        self.assertIn("PLACE ROBOT DOWN", labels)

    def test_debug_board_shows_compact_navigation_status(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        previous_navigation = camera_display.robot_state.get("navigation")
        camera_display.robot_state["navigation"] = {
            "state": "navigating",
            "distance_remaining": 1.64,
            "recoveries": 1,
        }
        track_action = SimpleNamespace(
            target=None,
            action_id=None,
            active=False,
            stable_seeds=SimpleNamespace(
                status=lambda **_kwargs: {
                    "valid": False,
                    "sample_count": 0,
                    "sample_target": 3,
                }
            ),
        )
        hand = SimpleNamespace(gesture_status=lambda: {})

        try:
            with patch.object(camera_display.cv2, "putText") as put_text:
                camera_display.draw_tracking_status_overlay(
                    frame, track_action, hand
                )
        finally:
            camera_display.robot_state["navigation"] = previous_navigation

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("NAV2", labels)
        self.assertIn("MOVING  |  1.6m  |  R1", labels)

    def test_navigation_failure_retains_nav2_error_code(self):
        value, color = camera_display.navigation_status_display({
            "state": "failed",
            "error_code": 201,
        })

        self.assertEqual(value, "FAILED  |  CODE 201")
        self.assertEqual(color, (80, 80, 255))

    def test_debug_board_shows_bracketed_voice_state(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        track_action = SimpleNamespace(
            target=None,
            action_id=None,
            active=False,
            stable_seeds=SimpleNamespace(
                status=lambda **_kwargs: {
                    "valid": False,
                    "sample_count": 0,
                    "sample_target": 3,
                }
            ),
        )
        hand = SimpleNamespace(gesture_status=lambda: {})

        with patch.object(camera_display.cv2, "putText") as put_text:
            camera_display.draw_tracking_status_overlay(
                frame, track_action, hand, voice_state="listening"
            )

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("VOICE", labels)
        self.assertIn("[LISTENING]", labels)
        self.assertEqual(
            camera_display.voice_status_display("talking"),
            ("[TALKING]", (70, 240, 70)),
        )

    def test_left_debug_board_width_adapts_to_its_longest_value(self):
        short_rows = [("NAV2", "IDLE", (170, 170, 170))]
        long_rows = [(
            "NAV2",
            "MOVING  |  12.4m  |  R3",
            (70, 240, 70),
        )]

        short_width = camera_display.tracking_panel_width(
            short_rows, frame_width=1280, scale=1.0
        )
        long_width = camera_display.tracking_panel_width(
            long_rows, frame_width=1280, scale=1.0
        )

        self.assertLess(short_width, long_width)
        self.assertLess(long_width, 760)
        self.assertLessEqual(long_width, 1280 - 24)

    def test_right_panel_shows_wifi_and_compact_stm32_health(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        stm32_status, stm32_detail, stm32_color = (
            camera_display.stm32_health_display({
                "message": (
                    "DEGRADED failing=tof_sensor "
                    "reason=no_valid_targets rc=0"
                ),
                "age_seconds": 0.4,
            })
        )

        with patch.object(camera_display.cv2, "putText") as put_text:
            camera_display.draw_pipeline_fps_overlay(
                frame,
                30.0,
                12.0,
                8.0,
                42.0,
                "VIETTEL 1671 1",
                True,
                stm32_status,
                stm32_detail,
                stm32_color,
            )

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("WIFI       VIETTEL 1671 1", labels)
        self.assertIn("STM32      DEGRADED", labels)
        self.assertIn("TOF SENSOR  |  NO VALID TARGETS", labels)
        stm32_call = next(
            call for call in put_text.call_args_list
            if call.args[1] == "STM32      DEGRADED"
        )
        detail_call = next(
            call for call in put_text.call_args_list
            if call.args[1] == "TOF SENSOR  |  NO VALID TARGETS"
        )
        self.assertGreater(detail_call.args[2][1], stm32_call.args[2][1])
        self.assertLess(detail_call.args[4], stm32_call.args[4])

    def test_compact_stm32_health_retains_nonzero_error_code(self):
        self.assertEqual(
            camera_display.compact_stm32_health_message(
                "FAIL failing=micro_ros reason=agent_ping rc=4"
            ),
            ("FAIL", "MICRO ROS  |  AGENT PING  |  RC 4"),
        )

    def test_stm32_diagnostic_is_truncated_inside_right_panel(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        long_detail = "VERY LONG SENSOR NAME  |  " + "FAILURE " * 20

        with patch.object(camera_display.cv2, "putText") as put_text:
            camera_display.draw_pipeline_fps_overlay(
                frame, 30.0, 12.0, 8.0, 42.0,
                stm32_status="DEGRADED",
                stm32_detail=long_detail,
            )

        detail_call = put_text.call_args_list[-1]
        rendered = detail_call.args[1]
        origin_x = detail_call.args[2][0]
        rendered_width = camera_display.cv2.getTextSize(
            rendered,
            camera_display.cv2.FONT_HERSHEY_SIMPLEX,
            detail_call.args[4],
            detail_call.args[6],
        )[0][0]
        self.assertTrue(rendered.endswith("..."))
        self.assertLessEqual(origin_x + rendered_width, frame.shape[1] - 4)

    def test_wifi_status_reads_active_network_and_caches_it(self):
        output = "wifi:connected:Lab Network\nethernet:unavailable:\n"
        with patch.object(
            camera_display.subprocess,
            "run",
            return_value=SimpleNamespace(stdout=output),
        ) as run:
            status = camera_display.WifiStatus(sample_interval_s=2.0)
            self.assertEqual(status.sample(now=10.0), "Lab Network")
            self.assertEqual(status.sample(now=11.0), "Lab Network")
        run.assert_called_once()

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
