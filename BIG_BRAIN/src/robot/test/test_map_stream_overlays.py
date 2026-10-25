import math
from types import SimpleNamespace
import threading
import time
import unittest

from robot.map.map_stream import MapImageStream


def record(**values):
    return SimpleNamespace(**values)


class MapStreamOverlayTest(unittest.TestCase):
    def make_stream(self):
        stream = MapImageStream.__new__(MapImageStream)
        stream.state_lock = threading.Lock()
        stream.scan_msg = None
        stream.footprint_msg = None
        stream.plan_msg = None
        stream.navigation_goal = None
        stream.reset_state = None
        stream._overlay_received_at_unix_ns = {}
        stream._transform_points = (
            lambda points, _source, _target: list(points)
        )
        return stream

    def test_reset_state_is_retained_for_map_metadata(self):
        stream = self.make_stream()

        stream._reset_state(record(data="resetting_manual"))

        self.assertEqual("resetting_manual", stream.reset_state)

    def test_goal_replaces_stale_plan_and_clear_removes_both(self):
        stream = self.make_stream()
        stream.plan_msg = object()
        stream._overlay_received_at_unix_ns["plan"] = time.time_ns()

        stream.set_navigation_goal({
            "x": 1.0, "y": 2.0, "yaw": 0.4, "frame_id": "map",
        })

        self.assertEqual("map", stream.navigation_goal["frame_id"])
        self.assertIsNone(stream.plan_msg)
        stream.plan_msg = object()
        stream.clear_navigation_goal()
        self.assertIsNone(stream.navigation_goal)
        self.assertIsNone(stream.plan_msg)

    def test_snapshot_bounds_lidar_and_plan_point_counts(self):
        stream = self.make_stream()
        now = time.time_ns()
        stream._overlay_received_at_unix_ns = {
            "lidar": now,
            "footprint": now,
            "plan": now,
        }
        stream.scan_msg = record(
            ranges=[1.0] * 720,
            range_min=0.1,
            range_max=5.0,
            angle_min=-math.pi,
            angle_increment=2 * math.pi / 720,
            header=record(frame_id="laser"),
        )
        stream.footprint_msg = record(
            polygon=record(points=[
                record(x=-0.1, y=-0.1), record(x=0.1, y=-0.1),
                record(x=0.1, y=0.1), record(x=-0.1, y=0.1),
            ]),
            header=record(frame_id="map"),
        )
        stream.plan_msg = record(
            poses=[record(pose=record(position=record(x=i / 10, y=0.0)))
                   for i in range(500)],
            header=record(frame_id="map"),
        )
        stream.navigation_goal = {
            "x": 1.0, "y": 0.0, "yaw": 0.0, "frame_id": "map",
        }

        overlays = stream._diagnostic_overlays("map")

        self.assertLessEqual(len(overlays["lidar"]), 180)
        self.assertLessEqual(len(overlays["plan"]), 200)
        self.assertEqual(4, len(overlays["footprint"]))
        self.assertEqual(1.0, overlays["goal"]["x"])

    def test_stale_scan_and_footprint_are_hidden(self):
        stream = self.make_stream()
        stream.scan_msg = record(ranges=[])
        stream.footprint_msg = record(polygon=record(points=[]))

        overlays = stream._diagnostic_overlays("map")

        self.assertEqual([], overlays["lidar"])
        self.assertEqual([], overlays["footprint"])

    def test_crop_size_accepts_variable_metric_squares(self):
        self.assertIsNone(MapImageStream._normalize_crop_size(None))
        self.assertEqual(3.0, MapImageStream._normalize_crop_size(3))
        self.assertEqual(4.0, MapImageStream._normalize_crop_size("4"))
        self.assertEqual(5.0, MapImageStream._normalize_crop_size(5.0))

        with self.assertRaisesRegex(ValueError, "between 1 and 20"):
            MapImageStream._normalize_crop_size(0)


if __name__ == "__main__":
    unittest.main()
