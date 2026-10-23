import math
import time
import unittest

from geometry_msgs.msg import TransformStamped
import numpy as np

from robot.person_pose.lidar_clusters import Cluster
from robot.person_pose.person_tracker_state import (
    PersonTrackerState,
    TrackerSettings,
)
from robot.person_pose.stationary_leg_anchors import (
    StationaryLegAnchor,
    StationaryLegAnchorMap,
)
from robot.person_pose.yolo_correction import (
    CameraView,
    VisualConstraint,
    VisualContext,
    VisualObservation,
    YoloCorrection,
    YoloSettings,
)


class VisualLidarAssociationTests(unittest.TestCase):
    @staticmethod
    def cluster(x, y, width=0.16):
        half = width * 0.5
        points = np.array([
            [x, y - half],
            [x, y],
            [x, y + half],
        ])
        return Cluster(points, np.array([x, y]), width)

    def test_begin_scan_clears_anchor_classification_ids(self):
        tracker = PersonTrackerState(TrackerSettings())
        tracker.stationary_cluster_ids = {1}
        tracker.near_anchor_cluster_ids = {2}
        tracker.clear_cluster_ids = {3}

        tracker._begin_scan([self.cluster(1.0, 0.0)])

        self.assertEqual(tracker.stationary_cluster_ids, set())
        self.assertEqual(tracker.near_anchor_cluster_ids, set())
        self.assertEqual(tracker.clear_cluster_ids, set())

    @staticmethod
    def transform(yaw=0.0, x=0.0, y=0.0):
        transform = TransformStamped()
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation.z = math.sin(yaw * 0.5)
        transform.transform.rotation.w = math.cos(yaw * 0.5)
        return transform

    @staticmethod
    def visual_context(
        visual,
        camera,
        base_bearing=0.0,
        camera_bearing=0.0,
    ):
        observation = VisualObservation(
            base_bearing_rad=(
                base_bearing if camera.person_visible else None
            ),
            camera_bearing_rad=camera_bearing,
            horizontal_fov_rad=camera.half_fov * 2.0,
            person_visible=camera.person_visible,
            captured_at=0.0,
            received_at=0.0,
            session_id="follow",
        )
        return VisualContext(observation, camera, visual)

    def seed_tracker(
        self,
        seed=np.array([1.0, 0.0]),
        settings=None,
    ):
        tracker = PersonTrackerState(settings or TrackerSettings())
        tracker.activate()
        tracker.accept_tof(seed)
        tracker.update([], 10.0, seed)
        return tracker

    def acquire_single(self, settings=None):
        tracker = self.seed_tracker(settings=settings)
        leg = self.cluster(1.0, 0.0)
        tracker.update([leg], 10.1, None)
        self.assertEqual(tracker.track.state, "tracking")
        return tracker

    def test_yolo_correction_uses_latest_observed_bearing(self):
        correction = YoloCorrection(YoloSettings(), "base_footprint")
        captured_at = time.monotonic() - 0.2
        first = {
            "type": "person_visual_observation",
            "base_bearing_rad": 0.0,
            "camera_bearing_rad": 0.0,
            "horizontal_fov_rad": math.radians(85.0),
            "person_visible": True,
            "captured_at_monotonic": captured_at,
            "session_id": "follow",
        }
        second = {
            **first,
            "base_bearing_rad": 0.1,
            "captured_at_monotonic": captured_at + 0.1,
        }

        correction.accept(first)
        correction.accept(second)
        context = correction.context(
            lambda _frame, _captured_at: self.transform()
        )

        self.assertAlmostEqual(context.constraint.bearing, 0.1)
        self.assertAlmostEqual(
            context.constraint.gate,
            math.radians(25.0),
        )
        self.assertAlmostEqual(context.observation.base_bearing_rad, 0.1)

    def test_observed_bearing_uses_capture_time_robot_rotation(self):
        correction = YoloCorrection(YoloSettings(), "base_footprint")
        captured_at = time.monotonic() - 0.2
        first = {
            "type": "person_visual_observation",
            "base_bearing_rad": 0.0,
            "camera_bearing_rad": 0.0,
            "horizontal_fov_rad": math.radians(85.0),
            "person_visible": True,
            "captured_at_monotonic": captured_at,
            "session_id": "follow",
        }
        second = {
            **first,
            "base_bearing_rad": -0.1,
            "captured_at_monotonic": captured_at + 0.1,
        }
        transforms = {
            first["captured_at_monotonic"]: self.transform(yaw=0.0),
            second["captured_at_monotonic"]: self.transform(yaw=0.1),
        }

        correction.accept(first)
        correction.accept(second)
        context = correction.context(
            lambda _frame, stamp: transforms[stamp]
        )

        self.assertAlmostEqual(context.constraint.bearing, 0.0)

    def test_empty_camera_frame_clears_bearing_and_preserves_fov(self):
        correction = YoloCorrection(YoloSettings(), "base_footprint")
        correction.accept({
            "type": "person_visual_observation",
            "camera_bearing_rad": 0.2,
            "horizontal_fov_rad": math.radians(85.0),
            "person_visible": False,
            "captured_at_monotonic": time.monotonic(),
            "session_id": "follow",
        })

        context = correction.context(
            lambda _frame, _captured_at: self.transform(yaw=0.1)
        )

        self.assertIsNone(context.constraint)
        self.assertFalse(context.camera_view.person_visible)
        self.assertAlmostEqual(context.camera_view.bearing, 0.3)

    def test_yolo_reset_discards_visual_history(self):
        correction = YoloCorrection(YoloSettings(), "base_footprint")
        correction.accept({
            "type": "person_visual_observation",
            "base_bearing_rad": 0.0,
            "camera_bearing_rad": 0.0,
            "horizontal_fov_rad": math.radians(85.0),
            "person_visible": True,
            "captured_at_monotonic": time.monotonic(),
            "session_id": "follow",
        })

        correction.reset()

        self.assertIsNone(correction.snapshot())

    def test_visual_gate_discards_candidates_after_persistence_delay(self):
        tracker = self.acquire_single(TrackerSettings(
            visual_gate_confirm_seconds=1.0,
            anchor_clear_bonus_m=0.0,
            leg_pair_bonus_m=0.0,
        ))
        visual = VisualConstraint(
            origin=np.zeros(2), bearing=0.0, gate=0.02
        )
        camera = CameraView(
            origin=np.zeros(2),
            bearing=0.0,
            half_fov=0.8,
            person_visible=True,
        )
        tracker.update(
            [], 10.2, None, self.visual_context(visual, camera),
            monotonic_now=20.0,
        )
        outside = self.cluster(1.0, 0.05)
        inside = self.cluster(1.1, 0.0)

        tracker.update(
            [outside, inside], 10.3, None,
            self.visual_context(visual, camera),
            monotonic_now=21.1,
        )

        self.assertEqual(tracker.selected_clusters, [inside])

    def test_missing_yolo_outside_camera_fov_does_not_start_recovery(self):
        tracker = self.acquire_single()
        camera = CameraView(
            origin=np.zeros(2),
            bearing=math.pi,
            half_fov=0.4,
            person_visible=False,
        )

        tracker.update(
            [], 10.2, None, self.visual_context(None, camera),
            monotonic_now=20.0,
        )
        tracker.update(
            [], 10.3, None, self.visual_context(None, camera),
            monotonic_now=21.2,
        )

        self.assertNotEqual(tracker.published_state, "visual_recovery")

    def test_missing_yolo_inside_camera_fov_starts_delayed_recovery(self):
        tracker = self.acquire_single(TrackerSettings(
            visual_missing_in_fov_timeout_seconds=5.0,
        ))
        camera = CameraView(
            origin=np.zeros(2),
            bearing=0.0,
            half_fov=0.4,
            person_visible=False,
        )

        first = tracker.update(
            [], 10.2, None, self.visual_context(None, camera),
            monotonic_now=20.0,
        )
        before_timeout = tracker.update(
            [], 10.3, None, self.visual_context(None, camera),
            monotonic_now=24.9,
        )
        after_timeout = tracker.update(
            [], 10.4, None, self.visual_context(None, camera),
            monotonic_now=25.1,
        )

        self.assertNotIn("visual_recovery_started", first)
        self.assertNotIn("visual_recovery_started", before_timeout)
        self.assertIn("visual_recovery_started", after_timeout)
        self.assertEqual(tracker.published_state, "visual_recovery")

    def test_visual_return_waits_until_centered_then_requests_fresh_seed(self):
        tracker = self.acquire_single(TrackerSettings(
            visual_missing_in_fov_timeout_seconds=1.0,
            visual_recovery_confirm_seconds=1.0,
        ))
        missing = CameraView(
            origin=np.zeros(2), bearing=0.0, half_fov=0.4,
            person_visible=False,
        )
        tracker.update(
            [], 10.2, None, self.visual_context(None, missing),
            monotonic_now=20.0,
        )
        tracker.update(
            [], 10.3, None, self.visual_context(None, missing),
            monotonic_now=21.1,
        )
        visual = VisualConstraint(
            origin=np.zeros(2), bearing=0.0, gate=0.1
        )
        visible = CameraView(
            origin=np.zeros(2), bearing=0.0, half_fov=0.4,
            person_visible=True,
        )

        found = tracker.update(
            [self.cluster(1.1, 0.0)],
            10.4,
            None,
            self.visual_context(visual, visible, base_bearing=0.3),
            monotonic_now=21.2,
        )
        centered = tracker.update(
            [self.cluster(1.1, 0.0)],
            10.5,
            None,
            self.visual_context(visual, visible, base_bearing=0.05),
            monotonic_now=21.5,
        )

        self.assertIn("visual_found", found)
        self.assertEqual(tracker.published_state, "validating")
        self.assertIn("visual_centered", centered)
        self.assertEqual(tracker.selected_clusters, [])

        validated = tracker.update(
            [self.cluster(1.1, 0.0)],
            10.6,
            None,
            self.visual_context(visual, visible, base_bearing=0.04),
            monotonic_now=22.3,
        )

        self.assertEqual(tracker.published_state, "awaiting_seed")
        self.assertIn("visual_validated", validated)

        stale_seed = tracker.accept_tof(np.array([3.0, 0.0]), 22.2)
        fresh_seed = tracker.accept_tof(np.array([3.0, 0.0]), 22.4)
        initialized = tracker.update([], 10.7, np.array([3.0, 0.0]))

        self.assertEqual(stale_seed, "deferred")
        self.assertEqual(fresh_seed, "reseed")
        self.assertIn("initialized", initialized)
        np.testing.assert_allclose(tracker.track.position, [3.0, 0.0])

    def test_visual_recovery_timeout_declares_track_lost(self):
        tracker = self.acquire_single(TrackerSettings(
            visual_missing_in_fov_timeout_seconds=1.0,
            visual_recovery_timeout_seconds=8.0,
        ))
        missing = CameraView(
            origin=np.zeros(2), bearing=0.0, half_fov=0.4,
            person_visible=False,
        )
        tracker.update(
            [], 10.2, None, self.visual_context(None, missing),
            monotonic_now=20.0,
        )
        tracker.update(
            [], 10.3, None, self.visual_context(None, missing),
            monotonic_now=21.1,
        )

        events = tracker.update(
            [], 10.4, None, None, monotonic_now=29.2
        )

        self.assertIn("visual_recovery_failed", events)
        self.assertEqual(tracker.published_state, "lost")

    def test_tof_seed_immediately_creates_publishable_pose_without_legs(self):
        tracker = PersonTrackerState(TrackerSettings())
        tracker.activate()
        seed = np.array([1.0, 0.0])
        tracker.accept_tof(seed)

        events = tracker.update(
            [self.cluster(1.0, 0.0, width=0.60)],
            10.0,
            seed,
        )

        self.assertEqual(events, ["initialized"])
        self.assertEqual(tracker.track.state, "seeded")
        self.assertEqual(tracker.track.observation_kind, "seed")
        np.testing.assert_allclose(tracker.track.position, seed)
        self.assertEqual(tracker.leg_clusters, [])

    def test_seed_acquires_single_leg_inside_motion_gate(self):
        tracker = self.seed_tracker()
        leg = self.cluster(1.1, 0.0)

        events = tracker.update([leg], 10.1, None)

        self.assertEqual(events, ["leg_acquired"])
        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [leg])
        np.testing.assert_allclose(tracker.track.position, leg.center)

    def test_pair_midpoint_is_preferred_over_individual_legs(self):
        tracker = self.seed_tracker(np.array([0.5, 0.0]))
        left = self.cluster(0.5, -0.15)
        right = self.cluster(0.5, 0.15)

        tracker.update(
            [left, right],
            10.1,
            None,
        )

        self.assertEqual(tracker.track.observation_kind, "pair")
        self.assertEqual(tracker.selected_clusters, [left, right])
        np.testing.assert_allclose(
            tracker.track.position, np.array([0.5, 0.0])
        )

    def test_pair_can_switch_to_one_clean_leg(self):
        tracker = self.seed_tracker()
        left = self.cluster(1.0, -0.10)
        right = self.cluster(1.0, 0.10)
        tracker.update([left, right], 10.1, None)
        moved_leg = self.cluster(1.15, 0.10)

        tracker.update([moved_leg], 10.2, None)

        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [moved_leg])
        np.testing.assert_allclose(tracker.track.position, moved_leg.center)

    def test_normal_tracking_switches_from_single_back_to_pair(self):
        tracker = self.acquire_single()
        left = self.cluster(1.0, -0.10)
        right = self.cluster(1.0, 0.10)

        tracker.update([left, right], 10.2, None)

        self.assertEqual(tracker.track.observation_kind, "pair")
        self.assertEqual(tracker.selected_clusters, [left, right])
        np.testing.assert_allclose(
            tracker.track.position, np.array([1.0, 0.0])
        )

    def test_clean_pair_wins_tie_with_persisting_single(self):
        tracker = self.acquire_single()
        persisting = self.cluster(1.0, 0.0)
        new_leg = self.cluster(1.0, 0.20)

        tracker.update([persisting, new_leg], 10.2, None)

        self.assertEqual(tracker.track.observation_kind, "pair")
        self.assertEqual(tracker.selected_clusters, [persisting, new_leg])

    def test_false_pair_with_persisting_leg_uses_midpoint_distance(self):
        tracker = self.acquire_single()
        persisting = self.cluster(1.0, 0.0)
        obstacle = self.cluster(1.60, 0.0)

        tracker.update([persisting, obstacle], 10.2, None)

        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [persisting])
        np.testing.assert_allclose(tracker.track.position, persisting.center)

    def test_pair_bonus_does_not_override_much_closer_single(self):
        tracker = self.acquire_single(TrackerSettings(
            leg_motion_radius=0.50,
            min_leg_pair_distance=0.27,
            anchor_clear_bonus_m=0.0,
            leg_pair_bonus_m=0.20,
        ))
        closer_single = self.cluster(1.0, 0.0)
        pair_left = self.cluster(1.22, -0.14)
        pair_right = self.cluster(1.22, 0.14)

        tracker.update([closer_single, pair_left, pair_right], 10.2, None)

        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [closer_single])

    def test_pair_bonus_overrides_small_motion_disadvantage(self):
        tracker = self.acquire_single(TrackerSettings(
            leg_motion_radius=0.50,
            min_leg_pair_distance=0.20,
            anchor_clear_bonus_m=0.0,
            leg_pair_bonus_m=0.20,
        ))
        closer_single = self.cluster(1.0, 0.0)
        pair_left = self.cluster(1.10, -0.11)
        pair_right = self.cluster(1.10, 0.11)

        tracker.update([closer_single, pair_left, pair_right], 10.2, None)

        self.assertEqual(tracker.track.observation_kind, "pair")
        self.assertEqual(tracker.selected_clusters, [pair_left, pair_right])

    def test_pair_bonus_is_reduced_when_multiple_pairs_are_eligible(self):
        tracker = self.acquire_single(TrackerSettings(
            leg_motion_radius=0.50,
            min_leg_pair_distance=0.20,
            anchor_clear_bonus_m=0.0,
            leg_pair_bonus_m=0.20,
            leg_pair_bonus_m_cluster=0.10,
        ))
        closer_single = self.cluster(1.0, 0.0)
        first_pair_left = self.cluster(1.10, -0.11)
        first_pair_right = self.cluster(1.10, 0.11)
        second_pair_left = self.cluster(1.20, -0.11)
        second_pair_right = self.cluster(1.20, 0.11)

        tracker.update([
            closer_single,
            first_pair_left,
            first_pair_right,
            second_pair_left,
            second_pair_right,
        ], 10.2, None)

        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [closer_single])

    def test_cluster_pair_bonus_can_overcome_small_motion_disadvantage(self):
        tracker = self.acquire_single(TrackerSettings(
            leg_motion_radius=0.50,
            min_leg_pair_distance=0.20,
            anchor_clear_bonus_m=0.0,
            leg_pair_bonus_m=0.20,
            leg_pair_bonus_m_cluster=0.10,
        ))
        closer_single = self.cluster(1.0, 0.0)
        preferred_left = self.cluster(1.08, -0.11)
        preferred_right = self.cluster(1.08, 0.11)
        other_left = self.cluster(1.30, -0.11)
        other_right = self.cluster(1.30, 0.11)

        tracker.update([
            closer_single,
            preferred_left,
            preferred_right,
            other_left,
            other_right,
        ], 10.2, None)

        self.assertEqual(tracker.track.observation_kind, "pair")
        self.assertEqual(
            tracker.selected_clusters,
            [preferred_left, preferred_right],
        )

    def test_pair_outside_motion_gate_does_not_disable_pair_bonus(self):
        tracker = self.acquire_single(TrackerSettings(
            leg_motion_radius=0.50,
            min_leg_pair_distance=0.20,
            anchor_clear_bonus_m=0.0,
            leg_pair_bonus_m=0.20,
        ))
        closer_single = self.cluster(1.0, 0.0)
        eligible_left = self.cluster(1.10, -0.11)
        eligible_right = self.cluster(1.10, 0.11)
        outside_left = self.cluster(2.00, -0.11)
        outside_right = self.cluster(2.00, 0.11)

        tracker.update([
            closer_single,
            eligible_left,
            eligible_right,
            outside_left,
            outside_right,
        ], 10.2, None)

        self.assertEqual(tracker.track.observation_kind, "pair")
        self.assertEqual(
            tracker.selected_clusters,
            [eligible_left, eligible_right],
        )

    def test_pair_with_anchor_owned_component_falls_back_to_single(self):
        tracker = self.acquire_single()
        persisting = self.cluster(1.0, 0.0)
        uncertain_leg = self.cluster(1.0, 0.20)
        anchor_observation = self.cluster(1.0, 0.18)
        tracker.anchor_map.enabled = True
        tracker.anchor_map.anchors = [StationaryLegAnchor(
            position=anchor_observation.center.copy(),
            first_seen_at=9.0,
            last_seen_at=10.1,
            observations=tracker.settings.stationary_confirm_scans,
        )]

        tracker.update(
            [persisting, uncertain_leg, anchor_observation],
            10.2,
            None,
        )

        self.assertIn(id(uncertain_leg), tracker.stationary_cluster_ids)
        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [persisting])

    def test_split_returns_near_one_anchor_are_classified_by_distance(self):
        tracker = PersonTrackerState(TrackerSettings())
        owned_first = self.cluster(1.05, 0.0)
        owned_second = self.cluster(1.14, 0.0)
        near = self.cluster(1.18, 0.0)
        clear = self.cluster(1.23, 0.0)
        tracker._begin_scan([
            owned_first,
            owned_second,
            near,
            clear,
        ])
        tracker.anchor_map.enabled = True
        tracker.anchor_map.anchors = [StationaryLegAnchor(
            position=np.array([1.0, 0.0]),
            first_seen_at=9.0,
            last_seen_at=10.1,
            observations=tracker.settings.stationary_confirm_scans,
        )]

        tracker._prepare_cluster_anchor_relations()

        self.assertEqual(
            tracker.stationary_cluster_ids,
            {id(owned_first), id(owned_second)},
        )
        self.assertEqual(tracker.near_anchor_cluster_ids, {id(near)})
        self.assertEqual(tracker.clear_cluster_ids, {id(clear)})

    def test_near_anchor_single_is_not_eligible_for_recovery(self):
        tracker = self.acquire_single()
        recovery_x = 1.0 + tracker.settings.leg_motion_radius + 0.05
        near_anchor = self.cluster(recovery_x, 0.0)
        anchor_observation = self.cluster(recovery_x, 0.18)
        tracker.anchor_map.enabled = True
        tracker.anchor_map.anchors = [StationaryLegAnchor(
            position=anchor_observation.center.copy(),
            first_seen_at=9.0,
            last_seen_at=10.1,
            observations=tracker.settings.stationary_confirm_scans,
        )]

        for stamp in (10.2, 10.3, 10.4):
            tracker.update(
                [near_anchor, anchor_observation],
                stamp,
                None,
            )

        self.assertNotIn(id(near_anchor), tracker.clear_cluster_ids)
        self.assertIn(
            id(near_anchor), tracker.near_anchor_cluster_ids
        )
        self.assertEqual(tracker.selected_clusters, [])
        np.testing.assert_allclose(tracker.track.position, [1.0, 0.0])

    def test_pair_motion_gate_uses_midpoint_not_individual_leg_distance(self):
        tracker = self.seed_tracker(
            settings=TrackerSettings(leg_motion_radius=0.20),
        )
        old_left = self.cluster(1.0, -0.29)
        old_right = self.cluster(1.0, 0.29)
        tracker.update([old_left, old_right], 10.1, None)
        new_left = self.cluster(1.15, -0.29)
        new_right = self.cluster(1.15, 0.29)

        tracker.update([new_left, new_right], 10.2, None)

        self.assertFalse(tracker.search_expanded)
        self.assertEqual(tracker.track.observation_kind, "pair")
        self.assertEqual(tracker.selected_clusters, [new_left, new_right])
        np.testing.assert_allclose(
            tracker.track.position, np.array([1.15, 0.0])
        )

    def test_pair_to_single_transition_uses_single_centroid(self):
        tracker = self.seed_tracker(
            settings=TrackerSettings(leg_motion_radius=0.35),
        )
        left = self.cluster(1.0, -0.29)
        right = self.cluster(1.0, 0.29)
        tracker.update([left, right], 10.1, None)
        moved_right = self.cluster(1.15, 0.29)

        tracker.update([moved_right], 10.2, None)

        self.assertFalse(tracker.search_expanded)
        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [moved_right])

    def test_single_does_not_form_pair_with_distant_clutter(self):
        tracker = self.acquire_single()
        near = self.cluster(1.25, 0.0)
        far = self.cluster(1.25, 0.90)

        tracker.update([near, far], 10.2, None)

        self.assertFalse(tracker.search_expanded)
        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [near])
        np.testing.assert_allclose(tracker.track.position, near.center)

    def test_pair_builds_an_averaged_stationary_anchor(self):
        tracker = self.seed_tracker()
        left = self.cluster(1.0, -0.10)
        right = self.cluster(1.0, 0.10)

        for stamp, obstacle_x in (
            (10.1, 1.30),
            (10.2, 1.32),
            (10.3, 1.28),
        ):
            obstacle = self.cluster(obstacle_x, 0.40)
            tracker.update([left, right, obstacle], stamp, None)

        self.assertTrue(tracker.anchor_map.enabled)
        self.assertEqual(len(tracker.anchor_map.anchors), 1)
        anchor = tracker.anchor_map.anchors[0]
        self.assertEqual(anchor.observations, 3)
        np.testing.assert_allclose(anchor.position, np.array([1.30, 0.40]))

    def test_confirmed_stationary_leg_is_suppressed(self):
        tracker = self.seed_tracker()
        left = self.cluster(1.0, -0.10)
        right = self.cluster(1.0, 0.10)
        for stamp in (10.1, 10.2, 10.3):
            tracker.update(
                [left, right, self.cluster(1.30, 0.35)],
                stamp,
                None,
            )
        original = tracker.track.position.copy()

        tracker.update([self.cluster(1.31, 0.35)], 10.4, None)

        self.assertEqual(tracker.track.state, "tracking")
        self.assertEqual(tracker.selected_clusters, [])
        np.testing.assert_allclose(tracker.track.position, original)

    def test_leg_can_be_selected_after_leaving_stationary_anchor(self):
        tracker = self.seed_tracker()
        left = self.cluster(1.0, -0.10)
        right = self.cluster(1.0, 0.10)
        for stamp in (10.1, 10.2, 10.3):
            tracker.update(
                [left, right, self.cluster(1.30, 0.35)],
                stamp,
                None,
            )
        separated = self.cluster(1.18, 0.12)

        tracker.update([separated], 10.4, None)

        self.assertEqual(tracker.track.state, "tracking")
        self.assertEqual(tracker.selected_clusters, [separated])
        np.testing.assert_allclose(tracker.track.position, separated.center)

    def test_nearby_stationary_anchor_freezes_instead_of_following_jitter(self):
        settings = TrackerSettings()
        anchors = StationaryLegAnchorMap(settings)
        anchor_x = 1.0 + settings.person_filter_radius - 0.01
        anchor = StationaryLegAnchor(
            position=np.array([anchor_x, 0.0]),
            first_seen_at=10.0,
            last_seen_at=10.0,
            observations=settings.stationary_confirm_scans,
        )
        anchors.enabled = True
        anchors.anchors = [anchor]
        original = anchor.position.copy()

        anchors.prepare(10.1, np.array([1.0, 0.0]))
        anchors.update(
            [self.cluster(anchor_x, 0.02)],
            [],
            10.1,
            np.array([1.0, 0.0]),
            allow_new=False,
        )

        self.assertTrue(anchor.frozen)
        self.assertEqual(anchor.observations, settings.stationary_confirm_scans)
        np.testing.assert_allclose(anchor.position, original)

    def test_person_uncertainty_prevents_new_anchor_creation(self):
        settings = TrackerSettings()
        anchors = StationaryLegAnchorMap(settings)
        person = np.array([1.0, 0.0])

        anchors.update(
            [self.cluster(1.0 + settings.person_filter_radius - 0.01, 0.0)],
            [],
            10.0,
            person,
            allow_new=True,
        )

        self.assertEqual(anchors.anchors, [])

    def test_fresh_single_is_trusted_for_anchor_protection(self):
        tracker = self.acquire_single()

        np.testing.assert_allclose(
            tracker._trusted_person_position(),
            tracker.track.position,
        )

        tracker.track.last_hit_at = (
            time.monotonic() - tracker.settings.recovery_delay_seconds - 0.1
        )

        self.assertIsNone(tracker._trusted_person_position())

    def test_person_leg_near_anchor_is_not_rejected_with_anchor_observation(self):
        tracker = self.acquire_single(TrackerSettings(
            leg_motion_radius=0.50,
        ))
        anchor = StationaryLegAnchor(
            position=np.array([1.30, 0.0]),
            first_seen_at=9.0,
            last_seen_at=10.1,
            observations=tracker.settings.stationary_confirm_scans,
        )
        tracker.anchor_map.enabled = True
        tracker.anchor_map.anchors = [anchor]
        obstacle = self.cluster(1.30, 0.0)
        person_leg = self.cluster(1.48, 0.0)

        tracker.update([obstacle, person_leg], 10.2, None)

        self.assertEqual(tracker.selected_clusters, [person_leg])
        np.testing.assert_allclose(tracker.track.position, person_leg.center)

    def test_unseen_stationary_anchor_expires_after_timeout(self):
        settings = TrackerSettings(stationary_timeout_seconds=0.5)
        tracker = PersonTrackerState(settings)
        tracker.activate()
        seed = np.array([1.0, 0.0])
        tracker.accept_tof(seed)
        tracker.update([], 10.0, seed)
        left = self.cluster(1.0, -0.10)
        right = self.cluster(1.0, 0.10)
        far_obstacle = self.cluster(2.0, 0.0)
        for stamp in (10.1, 10.2, 10.3):
            tracker.update(
                [left, right, far_obstacle],
                stamp,
                None,
            )
        self.assertEqual(len(tracker.anchor_map.anchors), 1)

        tracker.update([left, right], 10.9, None)

        self.assertEqual(tracker.anchor_map.anchors, [])

    def test_wide_merged_cluster_holds_last_leg_pose(self):
        tracker = self.acquire_single()
        original = tracker.track.position.copy()
        wide = self.cluster(1.05, 0.0, width=0.60)

        tracker.update([wide], 10.2, None)

        self.assertEqual(tracker.track.state, "tracking")
        self.assertEqual(tracker.selected_clusters, [])
        np.testing.assert_allclose(tracker.track.position, original)

    def test_clean_leg_reappears_after_recovery(self):
        tracker = self.acquire_single()
        tracker.track.last_hit_at = time.monotonic() - 1.0
        tracker.update([], 10.2, None)
        self.assertEqual(tracker.track.state, "recovering")
        reappeared = self.cluster(1.2, 0.0)

        tracker.update([reappeared], 10.3, None)

        self.assertEqual(tracker.track.state, "tracking")
        self.assertEqual(tracker.selected_clusters, [reappeared])

    def test_missing_leg_enters_recovery_after_one_second(self):
        tracker = self.acquire_single()
        tracker.track.last_hit_at = time.monotonic() - 0.9

        tracker.update([], 10.2, None)
        self.assertEqual(tracker.track.state, "tracking")

        tracker.track.last_hit_at = time.monotonic() - 1.1
        tracker.update([], 10.3, None)
        self.assertEqual(tracker.track.state, "recovering")

    def test_recovery_search_expands_after_time_delay(self):
        tracker = self.acquire_single(TrackerSettings(
            prediction_fallback_delay_seconds=0.10,
            recovery_expand_delay_seconds=0.20,
        ))
        outside = self.cluster(
            1.0 + tracker.settings.leg_search_radius + 0.01,
            0.0,
        )

        tracker.update([outside], 10.2, None)
        self.assertEqual(tracker.selected_clusters, [])
        self.assertFalse(tracker.search_expanded)

        tracker.update([outside], 10.31, None)
        self.assertEqual(tracker.selected_clusters, [])
        self.assertEqual(tracker.search_phase, "fallback")

        tracker.update([outside], 10.51, None)

        self.assertEqual(tracker.selected_clusters, [outside])
        self.assertTrue(tracker.search_expanded)
        self.assertEqual(
            tracker.active_search_radius,
            tracker.settings.recovery_leg_search_radius,
        )
        np.testing.assert_allclose(
            tracker.active_search_center, [1.0, 0.0]
        )
        np.testing.assert_allclose(tracker.track.position, outside.center)

    def test_prediction_reacquisition_clears_miss_timeout(self):
        tracker = self.acquire_single()
        local = self.cluster(1.10, 0.0)

        tracker.update([], 10.2, None)
        self.assertIsNotNone(tracker.prediction_miss_started_at)

        tracker.update([local], 10.3, None)

        self.assertEqual(tracker.selected_clusters, [local])
        self.assertIsNone(tracker.prediction_miss_started_at)
        self.assertFalse(tracker.fallback_active)

    def test_local_leg_reports_normal_search_radius(self):
        tracker = self.acquire_single()
        local = self.cluster(1.10, 0.0)

        tracker.update([local], 10.2, None)

        self.assertFalse(tracker.search_expanded)
        self.assertEqual(
            tracker.active_search_radius,
            tracker.settings.leg_search_radius,
        )

    def test_velocity_prediction_is_preferred_over_previous_pose(self):
        settings = TrackerSettings(
            min_leg_pair_distance=0.30,
            leg_motion_radius=0.15,
            velocity_alpha=1.0,
            velocity_beta=1.0,
        )
        tracker = self.seed_tracker(np.array([0.0, 0.0]), settings)
        tracker.update([self.cluster(0.10, 0.0)], 10.1, None)
        previous = self.cluster(0.10, 0.0)
        predicted = self.cluster(0.20, 0.0)

        tracker.update([previous, predicted], 10.2, None)

        self.assertEqual(tracker.selected_clusters, [predicted])
        np.testing.assert_allclose(tracker.active_search_center, [0.20, 0.0])
        self.assertEqual(tracker.search_phase, "prediction")
        self.assertIsNone(tracker.active_fallback_center)

    def test_direction_bias_allows_fast_forward_motion_over_lateral_jump(self):
        settings = TrackerSettings(
            min_leg_pair_distance=0.50,
            leg_motion_radius=0.35,
            motion_direction_min_speed=0.15,
            lateral_motion_penalty_gain=1.5,
        )
        tracker = self.acquire_single(settings)
        tracker.track.velocity = np.array([0.20, 0.0])
        tracker.track.prediction_stamp_seconds = 10.1
        fast_forward = self.cluster(1.30, 0.0)
        closer_lateral = self.cluster(1.02, 0.12)

        tracker.update([fast_forward, closer_lateral], 10.2, None)

        self.assertEqual(tracker.selected_clusters, [fast_forward])

    def test_direction_bias_penalizes_reverse_motion(self):
        settings = TrackerSettings(
            min_leg_pair_distance=0.50,
            motion_direction_min_speed=0.15,
            reverse_motion_penalty_gain=1.5,
        )
        tracker = self.acquire_single(settings)
        tracker.track.velocity = np.array([0.20, 0.0])
        tracker.track.prediction_stamp_seconds = 10.1
        forward = self.cluster(1.08, 0.0)
        closer_reverse = self.cluster(0.98, 0.0)

        tracker.update([forward, closer_reverse], 10.2, None)

        self.assertEqual(tracker.selected_clusters, [forward])

    def test_direction_bias_is_off_below_minimum_speed(self):
        settings = TrackerSettings(
            min_leg_pair_distance=0.50,
            motion_direction_min_speed=0.15,
            lateral_motion_penalty_gain=1.5,
        )
        tracker = self.acquire_single(settings)
        tracker.track.velocity = np.array([0.10, 0.0])
        tracker.track.prediction_stamp_seconds = 10.1
        farther_forward = self.cluster(1.08, 0.0)
        closer_lateral = self.cluster(1.01, 0.04)

        tracker.update([farther_forward, closer_lateral], 10.2, None)

        self.assertEqual(tracker.selected_clusters, [closer_lateral])

    def test_previous_pose_gate_recovers_from_wrong_prediction(self):
        settings = TrackerSettings(
            leg_motion_radius=0.15,
            velocity_alpha=1.0,
            velocity_beta=1.0,
        )
        tracker = self.seed_tracker(np.array([0.0, 0.0]), settings)
        tracker.update([self.cluster(0.10, 0.0)], 10.1, None)
        abrupt_stop = self.cluster(0.00, 0.0)

        tracker.update([abrupt_stop], 10.2, None)
        self.assertEqual(tracker.selected_clusters, [])
        fallback_stamp = (
            10.2 + settings.prediction_fallback_delay_seconds + 0.01
        )
        tracker.update([abrupt_stop], fallback_stamp, None)

        self.assertEqual(tracker.selected_clusters, [abrupt_stop])
        self.assertFalse(tracker.search_expanded)
        self.assertEqual(tracker.search_phase, "fallback")
        np.testing.assert_allclose(tracker.track.velocity, [0.0, 0.0])

    def test_prediction_shift_is_clamped(self):
        settings = TrackerSettings(prediction_max_shift_m=0.20)
        tracker = self.seed_tracker(np.array([0.0, 0.0]), settings)
        tracker.track.filtered_position = np.array([1.0, 2.0])
        tracker.track.velocity = np.array([10.0, 0.0])
        tracker.track.prediction_stamp_seconds = 10.0

        prediction = tracker.track.update_prediction(10.5, settings)

        np.testing.assert_allclose(prediction, [1.20, 2.0])

    def test_expanded_recovery_is_centered_on_previous_pose(self):
        settings = TrackerSettings(
            velocity_alpha=1.0,
            velocity_beta=1.0,
            prediction_fallback_delay_seconds=0.10,
            recovery_expand_delay_seconds=0.10,
        )
        tracker = self.seed_tracker(np.array([0.0, 0.0]), settings)
        tracker.update([self.cluster(0.10, 0.0)], 10.1, None)

        tracker.update([], 10.2, None)
        tracker.update([], 10.41, None)

        self.assertTrue(tracker.search_expanded)
        np.testing.assert_allclose(tracker.active_search_center, [0.10, 0.0])
        self.assertIsNone(tracker.active_fallback_center)

    def test_velocity_decays_while_prediction_is_missing(self):
        settings = TrackerSettings(
            leg_motion_radius=0.15,
            velocity_alpha=1.0,
            velocity_beta=1.0,
            prediction_fallback_delay_seconds=1.0,
            velocity_decay_time_seconds=0.20,
        )
        tracker = self.seed_tracker(np.array([0.0, 0.0]), settings)
        tracker.update([self.cluster(0.10, 0.0)], 10.1, None)
        initial_speed = float(np.linalg.norm(tracker.track.velocity))

        tracker.update([], 10.2, None)
        tracker.update([], 10.4, None)

        self.assertGreater(initial_speed, 0.0)
        self.assertLess(
            float(np.linalg.norm(tracker.track.velocity)),
            initial_speed,
        )

    def test_recovery_search_prefers_single_when_pairs_are_ambiguous(self):
        tracker = self.acquire_single(TrackerSettings(
            prediction_fallback_delay_seconds=0.10,
            recovery_expand_delay_seconds=0.10,
        ))
        person_leg = self.cluster(
            1.0 + tracker.settings.leg_search_radius + 0.01,
            0.0,
        )
        clutter_left = self.cluster(1.80, -0.10)
        clutter_right = self.cluster(1.80, 0.10)

        tracker.update(
            [person_leg, clutter_left, clutter_right],
            10.2,
            None,
        )
        tracker.update(
            [person_leg, clutter_left, clutter_right],
            10.3,
            None,
        )
        tracker.update(
            [person_leg, clutter_left, clutter_right],
            10.4,
            None,
        )
        tracker.update(
            [person_leg, clutter_left, clutter_right],
            10.51,
            None,
        )

        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [person_leg])

    def test_recovery_prefers_single_beyond_pair_bonus(self):
        tracker = self.acquire_single(TrackerSettings(
            prediction_fallback_delay_seconds=0.10,
            recovery_expand_delay_seconds=0.10,
        ))
        closer_single = self.cluster(
            1.0 + tracker.settings.leg_motion_radius + 0.01,
            0.0,
        )
        pair_first = self.cluster(1.0, -0.65)
        pair_second = self.cluster(1.0, -0.85)

        for stamp in (10.2, 10.31, 10.41):
            tracker.update(
                [closer_single, pair_first, pair_second],
                stamp,
                None,
            )

        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [closer_single])

    def test_normal_search_prefers_single_when_pairs_are_ambiguous(self):
        tracker = self.acquire_single()
        person_leg = self.cluster(1.05, 0.0)
        clutter_left = self.cluster(1.35, -0.10)
        clutter_right = self.cluster(1.35, 0.10)

        tracker.update(
            [person_leg, clutter_left, clutter_right],
            10.2,
            None,
        )

        self.assertEqual(tracker.track.observation_kind, "single")
        self.assertEqual(tracker.selected_clusters, [person_leg])

    def test_leg_outside_recovery_search_radius_is_not_selected(self):
        tracker = self.acquire_single()
        outside = self.cluster(
            1.0 + tracker.settings.recovery_leg_search_radius + 0.01,
            0.0,
        )

        tracker.update([outside], 10.2, None)

        self.assertEqual(tracker.track.state, "tracking")
        self.assertEqual(tracker.selected_clusters, [])
        np.testing.assert_allclose(
            tracker.track.position, np.array([1.0, 0.0])
        )

    def test_seed_acquisition_does_not_use_recovery_search_radius(self):
        tracker = self.seed_tracker()
        outside = self.cluster(
            1.0 + tracker.settings.leg_search_radius + 0.01,
            0.0,
        )

        tracker.update([outside], 10.1, None)

        self.assertEqual(tracker.track.state, "seeded")
        self.assertEqual(tracker.selected_clusters, [])
        np.testing.assert_allclose(
            tracker.track.position, np.array([1.0, 0.0])
        )

    def test_leg_motion_radius_is_independent_of_pair_distance(self):
        settings = TrackerSettings(
            leg_pair_distance=0.60,
            leg_motion_radius=0.42,
        )

        self.assertAlmostEqual(settings.leg_search_radius, 0.42)

    def test_cluster_narrower_than_minimum_leg_width_is_ignored(self):
        tracker = self.seed_tracker()
        noise = self.cluster(
            1.0,
            0.0,
            width=tracker.settings.min_leg_width * 0.5,
        )

        tracker.update([noise], 10.1, None)

        self.assertEqual(tracker.leg_clusters, [])
        self.assertEqual(tracker.track.state, "seeded")

    def test_tracker_reset_discards_seed_and_leg_state(self):
        tracker = self.acquire_single()

        tracker.reset()

        self.assertIsNone(tracker.track)
        self.assertIsNone(tracker.initialization_seed)
        self.assertEqual(tracker.acquisition_state, "disabled")
        self.assertEqual(tracker.leg_clusters, [])
        self.assertEqual(tracker.selected_clusters, [])
        self.assertEqual(tracker.anchor_map.anchors, [])
        self.assertFalse(tracker.anchor_map.enabled)


if __name__ == "__main__":
    unittest.main()
