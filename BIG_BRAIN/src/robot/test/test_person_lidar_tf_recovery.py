import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

from builtin_interfaces.msg import Time
import numpy as np
from sensor_msgs.msg import LaserScan

from robot.person_pose.lidar_clusters import Cluster, Detection
from robot.person_pose.person_lidar_tracker import PersonLidarTracker
from robot.person_pose.person_tracker_renderer import PersonTrackerRenderer
from robot.person_pose.yolo_correction import CameraView, VisualConstraint


class PersonLidarTfRecoveryTests(unittest.TestCase):

    def test_velocity_arrow_appears_only_above_direction_threshold(self):
        track = SimpleNamespace(
            position=np.array([1.0, 2.0]),
            filtered_position=np.array([0.9, 2.0]),
            velocity=np.array([0.10, 0.0]),
            state="tracking",
            observation_kind="single",
        )
        state = SimpleNamespace(
            track=track,
            settings=SimpleNamespace(motion_direction_min_speed=0.15),
            leg_clusters=[],
            selected_clusters=[],
            anchor_map=SimpleNamespace(anchors=[]),
        )
        renderer = PersonTrackerRenderer("odom")

        below = renderer.markers(Time(sec=10), [], state, None, None)
        self.assertNotIn(
            "person_velocity",
            {marker.ns for marker in below.markers},
        )

        track.velocity = np.array([0.20, 0.0])
        above = renderer.markers(Time(sec=11), [], state, None, None)
        arrow = next(
            marker for marker in above.markers
            if marker.ns == "person_velocity"
        )

        self.assertEqual(arrow.type, arrow.ARROW)
        self.assertEqual(len(arrow.points), 2)
        self.assertAlmostEqual(arrow.points[0].x, 1.0)
        self.assertGreater(arrow.points[1].x, arrow.points[0].x)

    def test_markers_separate_raw_candidate_and_selected_leg_points(self):
        leg = Cluster(
            points=np.array([[1.0, -0.05], [1.0, 0.05]]),
            center=np.array([1.0, 0.0]),
            width=0.1,
        )
        wide = Cluster(
            points=np.array([[2.0, -0.3], [2.0, 0.3]]),
            center=np.array([2.0, 0.0]),
            width=0.6,
        )
        state = SimpleNamespace(
            track=SimpleNamespace(
                position=np.array([1.0, 0.0]),
                state="tracking",
                observation_kind="single",
            ),
            leg_clusters=[leg],
            selected_clusters=[leg],
            last_detections=[],
            active_search_radius=1.0,
            active_search_center=np.array([1.0, 0.0]),
            search_expanded=True,
            anchor_map=SimpleNamespace(
                anchors=[SimpleNamespace(
                    position=np.array([1.4, 0.2]),
                    frozen=True,
                )],
            ),
        )
        renderer = PersonTrackerRenderer("odom")

        result = renderer.markers(
            Time(sec=10), [leg, wide], state, None, None
        )

        by_namespace = {
            marker.ns: marker for marker in result.markers if marker.ns
        }
        self.assertEqual(len(by_namespace["lidar_clusters"].points), 4)
        self.assertEqual(len(by_namespace["leg_clusters"].points), 2)
        self.assertEqual(
            len(by_namespace["selected_leg_clusters"].points), 2
        )
        self.assertGreater(
            by_namespace["selected_leg_clusters"].color.g,
            by_namespace["selected_leg_clusters"].color.r,
        )
        self.assertNotIn("leg_width_labels", by_namespace)
        self.assertNotIn("leg_pair_lines", by_namespace)
        self.assertNotIn("leg_pair_distance_labels", by_namespace)
        self.assertNotIn("tracked_person", by_namespace)
        self.assertNotIn("stationary_leg_anchors", by_namespace)
        self.assertEqual(
            len(by_namespace["frozen_stationary_anchors"].points), 1
        )
        self.assertGreater(
            by_namespace["frozen_stationary_anchors"].color.b,
            by_namespace["frozen_stationary_anchors"].color.r,
        )
        search = by_namespace["person_search_radius"]
        self.assertEqual(search.type, search.LINE_STRIP)
        self.assertEqual(len(search.points), 73)
        self.assertAlmostEqual(search.points[0].x, 2.0)
        self.assertGreater(search.color.r, search.color.g)

        person_result = renderer.person_markers(Time(sec=10), state)
        person_by_namespace = {
            marker.ns: marker
            for marker in person_result.markers if marker.ns
        }
        self.assertEqual(person_by_namespace["tracked_person"].type, 3)
        self.assertIn("search: expanded 1.00 m", person_by_namespace[
            "tracked_person_label"
        ].text)

    def test_markers_explain_leg_relationships_near_confirmed_anchor(self):
        owned = Cluster(
            points=np.array([[1.00, 0.00]]),
            center=np.array([1.00, 0.00]),
            width=0.08,
        )
        near = Cluster(
            points=np.array([[1.04, 0.00]]),
            center=np.array([1.04, 0.00]),
            width=0.08,
        )
        clear = Cluster(
            points=np.array([[1.30, 0.00]]),
            center=np.array([1.30, 0.00]),
            width=0.08,
        )
        anchor = SimpleNamespace(position=np.array([1.01, 0.00]))
        state = SimpleNamespace(
            track=None,
            leg_clusters=[owned, near, clear],
            selected_clusters=[],
            stationary_cluster_ids={id(owned)},
            near_anchor_cluster_ids={id(near)},
            clear_cluster_ids={id(clear)},
            settings=SimpleNamespace(
                stationary_match_radius=0.06,
                near_anchor_radius=0.12,
            ),
            anchor_map=SimpleNamespace(
                anchors=[anchor],
                confirmed=[anchor],
            ),
        )

        result = PersonTrackerRenderer("odom").markers(
            Time(sec=10), [owned, near, clear], state, None, None
        )
        by_namespace = {
            marker.ns: marker for marker in result.markers if marker.ns
        }

        self.assertEqual(
            len(by_namespace["anchor_owned_leg_clusters"].points), 1
        )
        self.assertEqual(
            len(by_namespace["near_anchor_leg_clusters"].points), 1
        )
        self.assertEqual(
            len(by_namespace["clear_leg_clusters"].points), 1
        )
        self.assertEqual(
            len(by_namespace["anchor_owned_relations"].points), 2
        )
        self.assertEqual(
            len(by_namespace["near_anchor_relations"].points), 2
        )
        self.assertEqual(
            len(by_namespace["stationary_anchor_match_radius"].points),
            72,
        )
        self.assertEqual(
            len(by_namespace["stationary_anchor_near_radius"].points),
            72,
        )
        self.assertEqual(
            len(by_namespace["frozen_stationary_anchors"].points), 0
        )

    def test_unclassified_cluster_is_not_rendered_as_near_anchor(self):
        cluster = Cluster(
            points=np.array([[3.00, 0.00]]),
            center=np.array([3.00, 0.00]),
            width=0.08,
        )
        anchor = SimpleNamespace(position=np.array([1.00, 0.00]))
        state = SimpleNamespace(
            track=None,
            leg_clusters=[cluster],
            selected_clusters=[],
            stationary_cluster_ids=set(),
            near_anchor_cluster_ids=set(),
            clear_cluster_ids=set(),
            settings=SimpleNamespace(stationary_match_radius=0.06),
            anchor_map=SimpleNamespace(
                anchors=[anchor],
                confirmed=[anchor],
            ),
        )

        result = PersonTrackerRenderer("odom").markers(
            Time(sec=10), [cluster], state, None, None
        )
        by_namespace = {
            marker.ns: marker for marker in result.markers if marker.ns
        }

        self.assertEqual(
            len(by_namespace["near_anchor_leg_clusters"].points), 0
        )
        self.assertNotIn("near_anchor_relations", by_namespace)

    def test_markers_hide_leg_pair_distance_debug(self):
        left = Cluster(
            points=np.array([[1.0, -0.15]]),
            center=np.array([1.0, -0.15]),
            width=0.08,
        )
        right = Cluster(
            points=np.array([[1.0, 0.15]]),
            center=np.array([1.0, 0.15]),
            width=0.09,
        )
        pair = Detection(
            position=np.array([1.0, 0.0]),
            kind="pair",
            clusters=(left, right),
        )
        state = SimpleNamespace(
            track=None,
            leg_clusters=[left, right],
            selected_clusters=[left, right],
            last_detections=[pair],
            anchor_map=SimpleNamespace(anchors=[]),
        )
        renderer = PersonTrackerRenderer("odom")

        result = renderer.markers(
            Time(sec=10), [left, right], state, None, None
        )

        namespaces = {marker.ns for marker in result.markers}
        self.assertNotIn("leg_pair_lines", namespaces)
        self.assertNotIn("leg_pair_distance_labels", namespaces)

    def test_camera_fov_is_diagnostic_only(self):
        state = SimpleNamespace(
            track=None,
            leg_clusters=[],
            selected_clusters=[],
            anchor_map=SimpleNamespace(anchors=[]),
        )
        camera_view = CameraView(
            origin=np.zeros(2),
            bearing=0.0,
            half_fov=np.deg2rad(40.0),
            person_visible=False,
        )

        result = PersonTrackerRenderer("odom", 2.0).markers(
            Time(sec=10), [], state, None, camera_view
        )

        fov = next(marker for marker in result.markers
                   if marker.ns == "camera_fov")
        self.assertEqual(fov.type, fov.TRIANGLE_LIST)
        self.assertEqual(len(fov.points), 72)
        self.assertEqual(
            (fov.scale.x, fov.scale.y, fov.scale.z),
            (1.0, 1.0, 1.0),
        )
        self.assertGreater(fov.color.b, fov.color.r)
        self.assertAlmostEqual(max(point.x for point in fov.points), 2.0)

    def test_person_bearing_fov_matches_visual_constraint_gate(self):
        state = SimpleNamespace(
            track=None,
            leg_clusters=[],
            selected_clusters=[],
            anchor_map=SimpleNamespace(anchors=[]),
        )
        visual_constraint = VisualConstraint(
            origin=np.zeros(2),
            bearing=0.0,
            gate=np.deg2rad(45.0),
        )

        result = PersonTrackerRenderer("odom", 2.0).markers(
            Time(sec=10), [], state, None, None, visual_constraint
        )

        fov = next(marker for marker in result.markers
                   if marker.ns == "person_bearing_fov")
        edge_angles = [
            np.arctan2(point.y, point.x)
            for point in fov.points
            if not np.isclose(np.hypot(point.x, point.y), 0.0)
        ]
        self.assertEqual(fov.type, fov.TRIANGLE_LIST)
        self.assertAlmostEqual(
            max(edge_angles) - min(edge_angles),
            np.deg2rad(90.0),
        )
        self.assertEqual(
            (fov.scale.x, fov.scale.y, fov.scale.z),
            (1.0, 1.0, 1.0),
        )

    def test_tof_seed_is_retried_after_transform_miss(self):
        seed = {
            "frame_id": "base_footprint",
            "x": 1.0,
            "y": 0.0,
            "target": "person",
            "session_id": "follow-1",
            "captured_at_monotonic": time.monotonic(),
        }
        state = SimpleNamespace(
            acquisition_state="tracking",
            update=Mock(return_value=[]),
            accept_tof=Mock(return_value=None),
        )
        tracker = SimpleNamespace(
            state=state,
            settings=SimpleNamespace(
                cluster_break_distance=0.1,
                min_cluster_points=2,
                seed_max_age_seconds=1.0,
            ),
            yolo=SimpleNamespace(
                context=Mock(return_value=None),
            ),
            latest_tof_seed=seed,
            processed_tof_seed=None,
            processed_tof_seed_key=None,
            tof_position=None,
            tof_position_at=None,
            _lookup_transform=Mock(return_value=object()),
            _lookup_transform_at_monotonic=Mock(return_value=None),
            _publish_filtered_scan=Mock(),
            _publish_tracking=Mock(),
            get_logger=Mock(return_value=Mock()),
        )
        scan = LaserScan()
        scan.header.frame_id = "lidar"
        scan.header.stamp = Time(sec=10)

        with patch(
            "robot.person_pose.person_lidar_tracker.clusters_from_scan",
            return_value=[],
        ):
            PersonLidarTracker.scan_callback(tracker, scan)

        self.assertIsNone(tracker.processed_tof_seed)
        state.accept_tof.assert_not_called()

        tracker._lookup_transform_at_monotonic.return_value = object()
        with (
            patch(
                "robot.person_pose.person_lidar_tracker.clusters_from_scan",
                return_value=[],
            ),
            patch(
                "robot.person_pose.person_lidar_tracker.transform_points",
                return_value=np.array([[1.0, 0.0]]),
            ),
        ):
            PersonLidarTracker.scan_callback(tracker, scan)

        self.assertIs(tracker.processed_tof_seed, seed)
        state.accept_tof.assert_called_once()
        self.assertIsNotNone(state.update.call_args.args[2])

        with patch(
            "robot.person_pose.person_lidar_tracker.clusters_from_scan",
            return_value=[],
        ):
            PersonLidarTracker.scan_callback(tracker, scan)

        state.accept_tof.assert_called_once()
        self.assertIsNone(state.update.call_args.args[2])
        self.assertIsNotNone(tracker._publish_tracking.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
