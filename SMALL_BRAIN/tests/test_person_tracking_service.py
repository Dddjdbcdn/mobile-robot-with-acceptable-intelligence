import asyncio
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from services.vision.person_tracking_service import (
    PersonTrackingConfig,
    PersonTrackingService,
)


def person(
    name,
    center_x,
    embedding,
    *,
    width=0.18,
    confidence=0.95,
):
    keypoints = {
        key: {
            "normalized_x": center_x + offset,
            "normalized_y": y,
            "confidence": 0.9,
        }
        for key, offset, y in (
            ("nose", 0.0, 0.20),
            ("left_shoulder", -0.03, 0.35),
            ("right_shoulder", 0.03, 0.35),
            ("left_hip", -0.02, 0.60),
            ("right_hip", 0.02, 0.60),
            ("left_ankle", -0.02, 0.90),
            ("right_ankle", 0.02, 0.90),
        )
    }
    return {
        "class": "person",
        "confidence": confidence,
        "name": name,
        "bbox": {
            "normalized_x1": center_x - width * 0.5,
            "normalized_y1": 0.10,
            "normalized_x2": center_x + width * 0.5,
            "normalized_y2": 0.95,
            "normalized_center_x": center_x,
            "normalized_center_y": 0.525,
        },
        "keypoints": keypoints,
        "_embedding": embedding,
    }


def partial_person(name, center_x, *, confidence=0.95):
    detection = person(name, center_x, [1, 0, 0], confidence=confidence)
    detection["keypoints"] = {
        key: detection["keypoints"][key]
        for key in ("left_ankle", "right_ankle")
    }
    return detection


class FakeReIDService:
    def __init__(self):
        self.calls = []

    async def extract_embedding_async(self, frame_bgr, detection, padding=0.0):
        del frame_bgr, padding
        self.calls.append(detection["name"])
        vector = np.asarray(detection["_embedding"], dtype=np.float32)
        return vector / np.linalg.norm(vector)


class PersonTrackingServiceTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def frame():
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def make_service(self, *, config=None, tracking_active=True, **options):
        options.setdefault("embedding_interval_frames", 1)
        options.setdefault("confirmation_hits", 1)
        options.setdefault("save_gallery_debug_images", False)
        service = PersonTrackingService(
            yolo=None,
            reid_service=FakeReIDService(),
            config=config,
            **options,
        )
        if tracking_active:
            service.start_target_tracking()
        return service

    async def test_active_update_prints_tracking_decision_scores_and_gates(self):
        service = self.make_service()
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            service._debug_log_path = Path(directory) / "person_tracking.log"
            with redirect_stdout(output):
                await service.update(
                    1,
                    self.frame(),
                    [person("target", 0.5, [1, 0, 0])],
                    captured_at=10.0,
                )
            saved_trace = service._debug_log_path.read_text(encoding="utf-8")

        debug_line = output.getvalue()
        self.assertIn("[PERSON_TRACK] seq=1 state=confirmed mode=solo", debug_line)
        self.assertIn("geom=--", debug_line)
        self.assertIn("reid=--", debug_line)
        self.assertIn("gallery_action=updated", debug_line)
        self.assertIn("decision=acquisition_confirmed_geometry", debug_line)
        self.assertIn(debug_line.strip(), saved_trace)

    async def test_idle_service_does_not_create_person_identity(self):
        service = self.make_service(tracking_active=False)

        for sequence in range(1, 7):
            tracks = await service.update(
                sequence,
                self.frame(),
                [person("bystander", 0.5, [1, 0, 0])],
                captured_at=5.0 + sequence * 0.1,
            )

        self.assertEqual(tracks, ())
        self.assertEqual(service.scene_mode, "idle")
        self.assertIsNone(service._target())
        self.assertEqual(service.reid_service.calls, [])

    async def test_initial_scene_uses_center_box_not_highest_confidence(self):
        service = self.make_service()
        centered = person(
            "centered", 0.50, [1, 0, 0], width=0.30, confidence=0.75)
        high_confidence = person(
            "high_confidence", 0.82, [0, 1, 0], confidence=0.99)

        tracks = await service.update(
            1, self.frame(), [high_confidence, centered], captured_at=10.0)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].person["name"], "centered")
        self.assertEqual(service.reid_service.calls, ["centered"])

    async def test_confirmed_target_uses_geometry_not_detection_confidence(self):
        service = self.make_service()
        initial = await service.update(
            1, self.frame(),
            [person("target", 0.30, [1, 0, 0])], captured_at=20.0)
        track_id = initial[0].track_id

        updated = await service.update(
            2,
            self.frame(),
            [
                person("target", 0.32, [1, 0, 0], confidence=0.70),
                partial_person("glitch", 0.42, confidence=0.99),
            ],
            captured_at=20.1,
        )

        self.assertEqual(updated[0].track_id, track_id)
        self.assertEqual(updated[0].person["name"], "target")
        self.assertTrue(updated[0].visible)

    async def test_low_quality_solo_observation_keeps_target_without_reid(self):
        service = self.make_service()
        initial = await service.update(
            1, self.frame(),
            [person("target", 0.4, [1, 0, 0])], captured_at=30.0)
        calls = len(service.reid_service.calls)

        updated = await service.update(
            2, self.frame(),
            [partial_person("legs", 0.75)], captured_at=30.1)

        self.assertEqual(updated[0].track_id, initial[0].track_id)
        self.assertEqual(updated[0].person["name"], "legs")
        self.assertEqual(len(service.reid_service.calls), calls)

    async def test_separated_high_quality_people_trigger_multi_reid(self):
        service = self.make_service()
        await service.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=40.0)
        service.reid_service.calls.clear()

        tracks = await service.update(
            2, self.frame(), [
                person("stranger", 0.22, [0, 1, 0], confidence=0.99),
                person("target", 0.80, [1, 0, 0], confidence=0.80),
            ], captured_at=40.1)

        self.assertEqual(service.scene_mode, "multi")
        self.assertEqual(service.reid_service.calls, ["stranger", "target"])
        self.assertEqual(tracks[0].person["name"], "target")
        self.assertGreater(tracks[0].reid_similarity, 0.99)

    async def test_overlapping_boxes_remain_solo_and_geometry_wins(self):
        service = self.make_service()
        await service.update(
            1, self.frame(),
            [person("target", 0.50, [1, 0, 0], width=0.45)],
            captured_at=50.0)
        service.reid_service.calls.clear()
        outer = person("outer", 0.50, [1, 0, 0], width=0.50, confidence=0.91)
        inner = person("inner", 0.52, [0, 1, 0], width=0.18, confidence=0.99)

        tracks = await service.update(
            2, self.frame(), [outer, inner], captured_at=50.1)

        self.assertEqual(service.scene_mode, "solo")
        self.assertEqual(tracks[0].person["name"], "outer")
        self.assertEqual(service.reid_service.calls, ["outer"])

    async def test_clear_target_trusts_separated_yolo_person_without_joints(self):
        service = self.make_service()
        await service.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=60.0)
        sparse = partial_person("sparse", 0.80, confidence=0.99)
        target = person("target", 0.20, [1, 0, 0], confidence=0.80)

        tracks = await service.update(
            2, self.frame(), [target, sparse], captured_at=60.1)

        self.assertEqual(service.scene_mode, "multi")
        self.assertEqual(tracks[0].person["name"], "target")

    async def test_separated_multi_scene_updates_gallery_support(self):
        service = self.make_service(gallery_size=4)
        initial = await service.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=70.0)
        sample_count = initial[0].appearance_samples
        internal = service._target()
        support = [(sample.last_supported_sequence, sample.support_count)
                   for sample in internal.gallery]

        tracks = await service.update(
            2, self.frame(), [
                person("target", 0.20, [0.98, 0.2, 0]),
                person("stranger", 0.80, [0, 1, 0]),
            ], captured_at=70.1)

        self.assertEqual(tracks[0].appearance_samples, sample_count)
        updated_support = [
            (sample.last_supported_sequence, sample.support_count)
            for sample in internal.gallery
        ]
        self.assertNotEqual(updated_support, support)

    async def test_clear_separated_multi_winner_updates_gallery_immediately(self):
        service = self.make_service(gallery_size=4)
        await service.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=72.0)
        target_view = [0.78, np.sqrt(1.0 - 0.78**2), 0.0]

        tracks = await service.update(
            2,
            self.frame(),
            [
                person("target", 0.20, target_view),
                person("stranger", 0.80, [0, 0, 1]),
            ],
            captured_at=72.2,
        )

        self.assertEqual(tracks[0].appearance_samples, 2)

    async def test_multi_gallery_reuses_normal_gallery_admission_logic(self):
        service = self.make_service(gallery_size=4)
        await service.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=74.0)
        weak_view = [0.70, np.sqrt(1.0 - 0.70**2), 0.0]

        tracks = ()
        for sequence in range(2, 7):
            tracks = await service.update(
                sequence,
                self.frame(),
                [
                    person("target", 0.20, weak_view),
                    person("stranger", 0.80, [0, 0, 1]),
                ],
                captured_at=74.0 + sequence * 0.1,
            )

        self.assertTrue(tracks[0].visible)
        self.assertEqual(tracks[0].appearance_samples, 2)

    async def test_overlapping_target_box_never_updates_multi_gallery(self):
        service = self.make_service(gallery_size=4)
        await service.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=74.8)
        target_view = [0.78, np.sqrt(1.0 - 0.78**2), 0.0]

        tracks = ()
        for sequence in range(2, 7):
            tracks = await service.update(
                sequence,
                self.frame(),
                [
                    person("target", 0.20, target_view, width=0.18),
                    person("target_duplicate", 0.21, target_view, width=0.12),
                    person("stranger", 0.80, [0, 0, 1]),
                ],
                captured_at=74.8 + sequence * 0.1,
            )

        self.assertTrue(tracks[0].visible)
        self.assertEqual(tracks[0].appearance_samples, 1)

    async def test_duplicate_target_boxes_do_not_defeat_multi_reid_margin(self):
        service = self.make_service()
        initial = await service.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=75.0)
        target_id = initial[0].track_id
        full = person("target_full", 0.20, [1, 0, 0], confidence=0.94)
        duplicate = person(
            "target_duplicate", 0.21, [0.999, 0.02, 0],
            width=0.12,
            confidence=0.98,
        )
        stranger = person("stranger", 0.80, [0, 1, 0])

        tracks = await service.update(
            2,
            self.frame(),
            [duplicate, stranger, full],
            captured_at=75.1,
        )

        self.assertTrue(tracks[0].visible)
        self.assertEqual(tracks[0].track_id, target_id)
        self.assertIn(
            tracks[0].person["name"],
            {"target_full", "target_duplicate"},
        )

    async def test_ambiguous_multi_reid_keeps_geometry_target(self):
        service = self.make_service()
        initial = await service.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=80.0)

        ambiguous = await service.update(
            2, self.frame(), [
                person("candidate_a", 0.20, [1, 0, 0]),
                person("candidate_b", 0.80, [0.999, 0.04, 0]),
            ], captured_at=80.1)

        self.assertEqual(ambiguous[0].track_id, initial[0].track_id)
        self.assertTrue(ambiguous[0].visible)
        self.assertEqual(ambiguous[0].person["name"], "candidate_a")

    async def test_lost_target_reacquires_same_id_after_three_reid_wins(self):
        service = self.make_service(lost_timeout_seconds=3.0)
        initial = await service.update(
            1, self.frame(),
            [person("target", 0.15, [1, 0, 0])], captured_at=90.0)
        track_id = initial[0].track_id
        await service.update(2, self.frame(), [], captured_at=90.1)

        first = await service.update(
            3, self.frame(),
            [person("target", 0.85, [1, 0, 0])], captured_at=90.2)
        second = await service.update(
            4, self.frame(),
            [person("target", 0.82, [1, 0, 0])], captured_at=90.3)
        third = await service.update(
            5, self.frame(),
            [person("target", 0.80, [1, 0, 0])], captured_at=90.4)

        self.assertFalse(first[0].visible)
        self.assertFalse(second[0].visible)
        self.assertTrue(third[0].visible)
        self.assertEqual(third[0].track_id, track_id)

    async def test_recovery_uses_separate_single_and_multi_person_thresholds(self):
        borderline_embedding = [0.68, 0.7332121, 0]

        single = self.make_service()
        await single.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=95.0)
        await single.update(2, self.frame(), [], captured_at=95.1)
        for sequence in range(3, 6):
            single_tracks = await single.update(
                sequence,
                self.frame(),
                [person("target", 0.75, borderline_embedding)],
                captured_at=95.0 + sequence * 0.1,
            )
        self.assertTrue(single_tracks[0].visible)

        multi = self.make_service()
        await multi.update(
            1, self.frame(),
            [person("target", 0.20, [1, 0, 0])], captured_at=96.0)
        await multi.update(2, self.frame(), [], captured_at=96.1)
        for sequence in range(3, 6):
            multi_tracks = await multi.update(
                sequence,
                self.frame(),
                [
                    person("target", 0.25, borderline_embedding),
                    person("stranger", 0.80, [0, 0, 1]),
                ],
                captured_at=96.0 + sequence * 0.1,
            )
        self.assertFalse(multi_tracks[0].visible)

    async def test_solo_reid_debug_score_does_not_control_tracking(self):
        service = self.make_service()
        with tempfile.TemporaryDirectory() as directory:
            service._debug_log_path = Path(directory) / "person_tracking.log"
            await service.update(
                1, self.frame(),
                [person("target", 0.50, [1, 0, 0])], captured_at=97.0)
            tracks = await service.update(
                2, self.frame(),
                [person("target", 0.51, [0.8, 0.6, 0])],
                captured_at=97.1,
            )
            trace = service._debug_log_path.read_text(encoding="utf-8")

        self.assertTrue(tracks[0].visible)
        self.assertIn("seq=2 state=confirmed mode=solo", trace)
        self.assertIn("reid=0.800 reid_min=--", trace)
        self.assertIn("reid_role=solo_debug", trace)
        self.assertIn("decision=solo_geometry", trace)

    async def test_recovery_rejects_stranger_even_inside_recent_geometry(self):
        service = self.make_service()
        await service.update(
            1, self.frame(),
            [person("target", 0.40, [1, 0, 0])], captured_at=100.0)
        await service.update(2, self.frame(), [], captured_at=100.1)

        tracks = await service.update(
            3, self.frame(),
            [person("stranger", 0.41, [0, 1, 0])], captured_at=100.2)

        self.assertFalse(tracks[0].visible)

    async def test_recent_geometry_recovers_when_partial_crop_has_no_embedding(self):
        config = PersonTrackingConfig(
            confirmation_hits=1,
            reacquisition_confirmation_frames=2,
            save_gallery_debug_images=False,
        )
        service = self.make_service(config=config)
        initial = await service.update(
            1, self.frame(),
            [person("target", 0.40, [1, 0, 0])], captured_at=110.0)
        await service.update(2, self.frame(), [], captured_at=110.1)
        await service.update(
            3, self.frame(),
            [partial_person("legs", 0.42, confidence=0.30)],
            captured_at=110.2)
        recovered = await service.update(
            4, self.frame(),
            [partial_person("legs", 0.43, confidence=0.30)],
            captured_at=110.3)

        self.assertTrue(recovered[0].visible)
        self.assertEqual(recovered[0].track_id, initial[0].track_id)

    async def test_stale_geometry_cannot_recover_without_reid(self):
        config = PersonTrackingConfig(
            confirmation_hits=1,
            geometry_memory_seconds=0.5,
            lost_timeout_seconds=3.0,
            save_gallery_debug_images=False,
        )
        service = self.make_service(config=config)
        await service.update(
            1, self.frame(),
            [person("target", 0.40, [1, 0, 0])], captured_at=120.0)
        await service.update(2, self.frame(), [], captured_at=120.1)
        tracks = await service.update(
            3, self.frame(),
            [partial_person("legs", 0.40, confidence=0.30)],
            captured_at=121.0)

        self.assertFalse(tracks[0].visible)

    async def test_target_requires_uninterrupted_confirmation_hits(self):
        service = self.make_service(confirmation_hits=3)
        for sequence in (1, 2):
            tracks = await service.update(
                sequence, self.frame(),
                [person("target", 0.3, [1, 0, 0])],
                captured_at=130.0 + sequence * 0.1)
            self.assertEqual(tracks, ())
        confirmed = await service.update(
            3, self.frame(),
            [person("target", 0.32, [0, 1, 0])], captured_at=130.3)

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].track_id, 1)

    async def test_acquisition_never_extracts_embedding_before_confirmation(self):
        service = self.make_service(confirmation_hits=3)

        for sequence, center_x in ((1, 0.40), (2, 0.41)):
            tracks = await service.update(
                sequence,
                self.frame(),
                [person("target", center_x, [1, 0, 0])],
                captured_at=135.0 + sequence * 0.1,
            )
            self.assertEqual(tracks, ())
            self.assertEqual(service.scene_mode, "acquiring")
            self.assertEqual(service.reid_service.calls, [])

        confirmed = await service.update(
            3,
            self.frame(),
            [person("target", 0.42, [1, 0, 0])],
            captured_at=135.3,
        )

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(service.reid_service.calls, ["target"])

    async def test_acquisition_does_not_jump_to_a_distant_person(self):
        service = self.make_service(confirmation_hits=2)
        await service.update(
            1,
            self.frame(),
            [
                person("selected", 0.45, [1, 0, 0], confidence=0.70),
                person("other", 0.82, [0, 1, 0], confidence=0.99),
            ],
            captured_at=135.5,
        )

        tracks = await service.update(
            2,
            self.frame(),
            [person("other", 0.82, [0, 1, 0], confidence=0.99)],
            captured_at=135.6,
        )

        self.assertEqual(tracks, ())
        self.assertEqual(service.scene_mode, "acquiring")
        self.assertEqual(service._target().person["name"], "selected")
        self.assertFalse(service._target().visible)
        self.assertEqual(service.reid_service.calls, [])

    async def test_partial_person_can_confirm_without_learning_gallery(self):
        service = self.make_service(confirmation_hits=2)

        tentative = await service.update(
            1,
            self.frame(),
            [partial_person("fragment", 0.50)],
            captured_at=136.0,
        )
        self.assertEqual(tentative, ())
        self.assertEqual(service.scene_mode, "acquiring")
        self.assertEqual(service.reid_service.calls, [])

        confirmed = await service.update(
            2,
            self.frame(),
            [partial_person("legs", 0.51)],
            captured_at=136.1,
        )
        self.assertEqual(len(confirmed), 1)
        self.assertTrue(confirmed[0].visible)
        self.assertEqual(confirmed[0].appearance_samples, 0)
        self.assertEqual(service.reid_service.calls, [])

        clear = await service.update(
            3,
            self.frame(),
            [person("target", 0.52, [1, 0, 0])],
            captured_at=136.2,
        )
        self.assertEqual(clear[0].appearance_samples, 1)
        self.assertEqual(service.reid_service.calls, ["target"])

    async def test_unclear_target_requires_other_person_to_pass_joint_gate(self):
        service = self.make_service()
        await service.update(
            1,
            self.frame(),
            [person("target", 0.20, [1, 0, 0])],
            captured_at=137.0,
        )
        calls = len(service.reid_service.calls)

        tracks = await service.update(
            2,
            self.frame(),
            [
                partial_person("target_partial", 0.21, confidence=0.80),
                partial_person("other_partial", 0.80, confidence=0.99),
            ],
            captured_at=137.1,
        )

        self.assertEqual(service.scene_mode, "solo")
        self.assertEqual(tracks[0].person["name"], "target_partial")
        self.assertEqual(len(service.reid_service.calls), calls)

    async def test_crossing_keeps_multi_mode_and_freezes_gallery(self):
        service = self.make_service()
        await service.update(
            1,
            self.frame(),
            [person("target", 0.20, [1, 0, 0])],
            captured_at=138.0,
        )
        await service.update(
            2,
            self.frame(),
            [
                person("target", 0.30, [1, 0, 0]),
                person("other", 0.75, [0, 1, 0]),
            ],
            captured_at=138.1,
        )
        internal = service._target()
        support_before = [sample.support_count for sample in internal.gallery]

        crossing = await service.update(
            3,
            self.frame(),
            [
                person("target_crossing", 0.48, [1, 0, 0], width=0.32),
                person("other_crossing", 0.52, [0, 1, 0], width=0.32),
            ],
            captured_at=138.2,
        )

        self.assertEqual(service.scene_mode, "multi")
        self.assertEqual(crossing[0].person["name"], "target_crossing")
        self.assertEqual(
            [sample.support_count for sample in internal.gallery],
            support_before,
        )

        merged = await service.update(
            4,
            self.frame(),
            [person("target_merged", 0.50, [1, 0, 0], width=0.40)],
            captured_at=138.3,
        )
        self.assertEqual(service.scene_mode, "multi")
        self.assertTrue(merged[0].visible)

    async def test_explicit_handoff_restarts_acquisition_on_that_person(self):
        service = self.make_service(confirmation_hits=2)
        centered = person("centered", 0.50, [1, 0, 0])
        selected = person("selected", 0.82, [0, 1, 0])
        await service.update(
            1,
            self.frame(),
            [centered, selected],
            captured_at=139.0,
        )

        waiting = asyncio.create_task(
            service.wait_for_confirmed_track(selected, timeout=0.5))
        await asyncio.sleep(0)
        await service.update(
            2,
            self.frame(),
            [
                person("centered", 0.50, [1, 0, 0]),
                person("selected", 0.81, [0, 1, 0]),
            ],
            captured_at=139.1,
        )
        await service.update(
            3,
            self.frame(),
            [
                person("centered", 0.50, [1, 0, 0]),
                person("selected", 0.80, [0, 1, 0]),
            ],
            captured_at=139.2,
        )

        track = await waiting
        self.assertIsNotNone(track)
        self.assertEqual(track.person["name"], "selected")
        self.assertAlmostEqual(
            track.person["bbox"]["normalized_center_x"], 0.80)

    async def test_gallery_is_bounded_in_clean_solo_scene(self):
        service = self.make_service(gallery_size=2)
        embeddings = (
            [1.0, 0.0, 0.0],
            [0.65, 0.76, 0.0],
            [0.65, 0.0, 0.76],
        )
        tracks = ()
        for sequence, embedding in enumerate(embeddings, start=1):
            tracks = await service.update(
                sequence, self.frame(),
                [person("target", 0.3, embedding)],
                captured_at=140.0 + sequence * 0.1)

        self.assertEqual(tracks[0].appearance_samples, 2)

    async def test_target_memory_expires_after_timeout(self):
        service = self.make_service(lost_timeout_seconds=1.0)
        await service.update(
            1, self.frame(),
            [person("target", 0.3, [1, 0, 0])], captured_at=150.0)
        await service.update(2, self.frame(), [], captured_at=150.2)
        expired = await service.update(3, self.frame(), [], captured_at=151.3)

        self.assertEqual(expired, ())

    async def test_snapshot_reports_gallery_update_age(self):
        service = self.make_service()
        initial = await service.update(
            1, self.frame(),
            [person("target", 0.3, [1, 0, 0])], captured_at=152.0)
        self.assertEqual(initial[0].gallery_update_age_seconds, 0.0)

        lost = await service.update(
            2, self.frame(), [], captured_at=152.4)
        self.assertAlmostEqual(lost[0].gallery_update_age_seconds, 0.4)

    async def test_follow_session_retains_target_past_timeout(self):
        service = self.make_service(lost_timeout_seconds=1.0)
        await service.update(
            1, self.frame(),
            [person("target", 0.3, [1, 0, 0])], captured_at=155.0)
        service.set_target_persistent(True)

        await service.update(2, self.frame(), [], captured_at=155.2)
        retained = await service.update(
            3, self.frame(), [], captured_at=156.3)

        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].state, "lost")
        self.assertFalse(retained[0].visible)

        service.set_target_persistent(False)
        expired = await service.update(
            4, self.frame(), [], captured_at=156.4)

        self.assertEqual(expired, ())

    async def test_wait_for_confirmed_track_binds_handoff_to_target(self):
        service = self.make_service(confirmation_hits=2)
        handoff = person("target", 0.3, [1, 0, 0])
        await service.update(1, self.frame(), [handoff], captured_at=160.0)
        waiting = asyncio.create_task(
            service.wait_for_confirmed_track(handoff, timeout=0.2))
        await asyncio.sleep(0)
        await service.update(
            2, self.frame(),
            [person("target", 0.32, [0, 1, 0])], captured_at=160.1)

        track = await waiting
        self.assertIsNotNone(track)
        self.assertEqual(track.track_id, 1)
        self.assertAlmostEqual(
            track.person["bbox"]["normalized_center_x"], 0.32)

    async def test_gallery_survives_expiry_until_new_gallery_is_active(self):
        with tempfile.TemporaryDirectory() as directory:
            config = PersonTrackingConfig(
                embedding_interval_frames=1,
                confirmation_hits=1,
                lost_timeout_seconds=0.5,
                save_gallery_debug_images=True,
                gallery_debug_directory=directory,
            )
            service = self.make_service(
                config=config, save_gallery_debug_images=True)
            await service.update(
                1, self.frame(),
                [person("target", 0.3, [1, 0, 0])], captured_at=170.0)
            path = Path(directory) / "id_0001_gallery.jpg"
            self.assertTrue(path.is_file())

            await service.update(2, self.frame(), [], captured_at=170.1)
            self.assertTrue(path.is_file())
            await service.update(3, self.frame(), [], captured_at=170.7)
            self.assertTrue(path.is_file())
            retired = service._retired_gallery_track
            self.assertIsNotNone(retired)
            self.assertEqual(len(retired.gallery), 1)

            # A confirmed partial detection does not have a valid ReID crop,
            # so it must not retire the previous gallery yet.
            await service.update(
                4, self.frame(),
                [partial_person("partial", 0.4)], captured_at=170.8)
            self.assertTrue(path.is_file())
            self.assertIs(service._retired_gallery_track, retired)

            await service.update(
                5, self.frame(),
                [person("replacement", 0.4, [0, 1, 0])],
                captured_at=170.9,
            )
            self.assertIsNone(service._retired_gallery_track)
            self.assertFalse(path.exists())
            self.assertTrue(
                (Path(directory) / "id_0002_gallery.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
