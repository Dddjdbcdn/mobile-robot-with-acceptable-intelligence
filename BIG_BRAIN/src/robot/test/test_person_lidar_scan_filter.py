import math
import unittest

import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan

from robot.person_pose.lidar_clusters import filter_scan_around_person


def make_scan(ranges):
    scan = LaserScan()
    scan.angle_min = -0.2
    scan.angle_increment = 0.2
    scan.range_min = 0.1
    scan.range_max = 10.0
    scan.ranges = list(ranges)
    scan.intensities = [1.0] * len(ranges)
    return scan


def identity_transform():
    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    return transform


class PersonLidarScanFilterTests(unittest.TestCase):
    def test_masks_only_beams_near_person(self):
        scan = make_scan([2.0, 2.0, 2.0])

        filtered, count = filter_scan_around_person(
            scan,
            identity_transform(),
            np.array([2.0, 0.0]),
            radius=0.15,
        )

        self.assertEqual(count, 1)
        self.assertTrue(math.isnan(filtered.ranges[1]))
        self.assertEqual(filtered.ranges[0], 2.0)
        self.assertEqual(filtered.ranges[2], 2.0)
        self.assertEqual(list(scan.ranges), [2.0, 2.0, 2.0])
        self.assertEqual(list(filtered.intensities), list(scan.intensities))

    def test_uses_scan_to_tracking_frame_transform(self):
        scan = make_scan([1.0])
        transform = identity_transform()
        transform.transform.translation.x = 1.0

        filtered, count = filter_scan_around_person(
            scan,
            transform,
            np.array([1.0 + math.cos(-0.2), math.sin(-0.2)]),
            radius=0.01,
        )

        self.assertEqual(count, 1)
        self.assertTrue(math.isnan(filtered.ranges[0]))

    def test_preserves_scan_when_no_return_is_near_person(self):
        scan = make_scan([1.0, math.inf, 3.0])

        filtered, count = filter_scan_around_person(
            scan,
            identity_transform(),
            np.array([5.0, 5.0]),
            radius=0.45,
        )

        self.assertEqual(count, 0)
        self.assertEqual(filtered.ranges[0], 1.0)
        self.assertTrue(math.isinf(filtered.ranges[1]))
        self.assertEqual(filtered.ranges[2], 3.0)


if __name__ == "__main__":
    unittest.main()
