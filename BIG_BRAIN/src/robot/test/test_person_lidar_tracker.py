import math
import threading
from unittest.mock import Mock

import numpy as np
from sensor_msgs.msg import LaserScan

from robot.person_pose.person_lidar_tracker import (
    Cluster,
    Detection,
    PersonTrack,
    TrackerSettings,
    cluster_scan,
    person_detections,
)


def test_cluster_scan_splits_invalid_gap():
    scan = LaserScan()
    scan.angle_min = -0.1
    scan.angle_increment = 0.02
    scan.range_min = 0.1
    scan.range_max = 10.0
    scan.ranges = [2.0, 2.0, 2.0, math.inf, 2.5, 2.5, 2.5]

    clusters = cluster_scan(scan, base_break_distance=0.10, min_points=3)

    assert [len(cluster) for cluster in clusters] == [3, 3]


def test_scan_processing_stays_dormant_until_first_seed():
    from robot.person_pose.person_lidar_tracker import PersonLidarTracker

    tracker = PersonLidarTracker.__new__(PersonLidarTracker)
    tracker.seed_lock = threading.Lock()
    tracker.latest_seed = None
    tracker.enabled = False
    tracker._publish_disabled = Mock()
    tracker._lookup_transform = Mock()
    scan = LaserScan()

    tracker.scan_callback(scan)

    tracker._publish_disabled.assert_called_once_with(scan.header.stamp)
    tracker._lookup_transform.assert_not_called()


def test_person_detections_prefers_leg_pair_midpoint():
    left = np.array([[1.95, 0.12], [2.0, 0.15], [2.05, 0.17]])
    right = np.array([[1.95, -0.17], [2.0, -0.15], [2.05, -0.12]])
    clusters = [
        Cluster(left, np.median(left, axis=0), 0.12),
        Cluster(right, np.median(right, axis=0), 0.12),
    ]

    detections = person_detections(
        clusters,
        reference=np.array([2.0, 0.0]),
        search_radius=0.6,
        single_leg_width=0.35,
        min_pair_distance=0.10,
        pair_distance=0.75,
        partial_radius=0.3,
        min_points=3,
    )

    pairs = [detection for detection in detections if detection.kind == "pair"]
    assert len(pairs) == 1
    assert np.allclose(pairs[0].position, [2.0, 0.0], atol=0.03)


def test_track_starts_stationary_then_learns_motion():
    track = PersonTrack(
        position=np.array([2.0, 0.0]),
        velocity=np.zeros(2),
        stamp_seconds=1.0,
        last_hit_at=0.0,
    )

    stationary_prediction, _ = track.predict(1.1)
    assert np.allclose(stationary_prediction, [2.0, 0.0])

    track.update(
        np.array([2.1, 0.0]),
        1.1,
        position_gain=0.55,
        velocity_gain=0.12,
        max_speed=2.5,
    )
    moving_prediction, _ = track.predict(1.2)
    assert moving_prediction[0] > track.position[0]


def test_conflicting_seed_starts_handoff_without_dropping_active_track():
    from robot.person_pose.person_lidar_tracker import PersonLidarTracker

    tracker = PersonLidarTracker.__new__(PersonLidarTracker)
    tracker.settings = TrackerSettings(
        seed_correction_radius=0.7,
        seed_correction_gain=0.12,
        seed_override_count=3,
    )
    tracker.get_logger = Mock(return_value=Mock())
    tracker.track = PersonTrack(
        position=np.array([0.0, 0.0]),
        velocity=np.zeros(2),
        stamp_seconds=1.0,
        last_hit_at=1.0,
    )
    tracker.pending_position = None
    tracker.pending_hits = 0
    tracker.last_detections = []
    tracker.last_seed_position = None
    tracker.last_seed_at = None
    tracker.seed_mismatch_hits = 0
    tracker.handoff_position = None
    tracker.handoff_velocity = np.zeros(2)
    tracker.handoff_stamp_seconds = None
    tracker.handoff_hits = 0
    tracker.handoff_started_at = None
    tracker.acquisition_state = "tracking"
    tracker.acquisition_started_at = None

    tracker._accept_trusted_seed(np.array([2.0, 0.0]))
    tracker._accept_trusted_seed(np.array([2.1, 0.0]))
    assert tracker.track is not None

    tracker._accept_trusted_seed(np.array([2.2, 0.0]))

    assert tracker.track is not None
    assert np.allclose(tracker.track.position, [0.0, 0.0])
    assert tracker.handoff_started_at is not None
    assert tracker.handoff_hits == 0
    assert np.allclose(tracker.last_seed_position, [2.2, 0.0])


def test_seed_handoff_switches_only_after_lidar_confirmation():
    from robot.person_pose.person_lidar_tracker import PersonLidarTracker

    tracker = PersonLidarTracker.__new__(PersonLidarTracker)
    tracker.settings = TrackerSettings(
        confirmation_scans=3,
        seed_correction_radius=0.7,
        seed_override_count=3,
    )
    tracker.get_logger = Mock(return_value=Mock())
    old_track = PersonTrack(
        position=np.array([0.0, 0.0]),
        velocity=np.zeros(2),
        stamp_seconds=1.0,
        last_hit_at=1.0,
    )
    tracker.track = old_track
    tracker.last_seed_position = None
    tracker.last_seed_at = None
    tracker.seed_mismatch_hits = 0
    tracker.handoff_position = None
    tracker.handoff_velocity = np.zeros(2)
    tracker.handoff_stamp_seconds = None
    tracker.handoff_hits = 0
    tracker.handoff_started_at = None
    tracker.acquisition_state = "tracking"
    tracker.acquisition_started_at = None

    for x in (2.0, 2.1, 2.2):
        tracker._accept_trusted_seed(np.array([x, 0.0]))

    points = np.array([[1.95, 0.0], [2.0, 0.0], [2.05, 0.0]])
    cluster = Cluster(points, np.median(points, axis=0), 0.10)

    assert not tracker._update_seed_handoff([cluster], 1.1)
    assert tracker.track is old_track
    assert not tracker._update_seed_handoff([cluster], 1.2)
    assert tracker.track is old_track
    assert tracker._update_seed_handoff([cluster], 1.3)

    assert tracker.track is not old_track
    assert np.allclose(tracker.track.position, [2.0, 0.0])
    assert tracker.track.state == "tracking"
    assert tracker.handoff_started_at is None


def test_trusted_seed_biases_candidate_choice_without_becoming_pose():
    from robot.person_pose.person_lidar_tracker import PersonLidarTracker

    chair = Detection(np.array([0.0, 0.0]), "partial")
    person = Detection(np.array([0.25, 0.0]), "pair")

    selected = PersonLidarTracker._best_detection(
        [chair, person],
        reference=np.array([0.0, 0.0]),
        trusted_seed=np.array([0.25, 0.0]),
        seed_weight=0.5,
    )

    assert selected is person


def test_pending_seed_activates_lidar_processing_once():
    from robot.person_pose.person_lidar_tracker import PersonLidarTracker

    tracker = PersonLidarTracker.__new__(PersonLidarTracker)
    tracker.seed_lock = threading.Lock()
    tracker.enabled = False
    tracker.latest_seed = {"x": 1.0, "y": 2.0}
    tracker.acquisition_state = "disabled"
    tracker.get_logger = Mock(return_value=Mock())

    tracker._activate_from_pending_seed()

    assert tracker.enabled
    assert tracker.acquisition_state == "waiting"
    tracker.get_logger().info.assert_called_once()


def test_enabled_lidar_ignores_later_activation_attempts():
    from robot.person_pose.person_lidar_tracker import PersonLidarTracker

    tracker = PersonLidarTracker.__new__(PersonLidarTracker)
    tracker.seed_lock = threading.Lock()
    tracker.enabled = True
    tracker.acquisition_state = "tracking"
    tracker.latest_seed = {"x": 3.0, "y": 4.0}
    tracker.get_logger = Mock(return_value=Mock())

    tracker._activate_from_pending_seed()

    assert tracker.enabled
    assert tracker.acquisition_state == "tracking"
    assert tracker.latest_seed == {"x": 3.0, "y": 4.0}
    tracker.get_logger().info.assert_not_called()
