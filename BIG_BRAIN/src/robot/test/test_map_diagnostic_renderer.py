from types import SimpleNamespace
import unittest

import numpy as np

from robot.map.map_renderer import MapRenderer


def record(**values):
    return SimpleNamespace(**values)


def grid(values, width=8, height=6, resolution=0.1):
    return record(
        data=np.asarray(values, dtype=np.int8),
        info=record(
            width=width,
            height=height,
            resolution=resolution,
            origin=record(
                position=record(x=0.0, y=0.0),
                orientation=record(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
    )


class MapDiagnosticRendererTest(unittest.TestCase):
    def setUp(self):
        self.renderer = MapRenderer()
        # Keep the robot marker outside the tiny synthetic grid so palette
        # assertions inspect only OccupancyGrid rendering.
        self.pose = {"x": 10.0, "y": 10.0, "yaw": 0.0}

    def test_slam_unknown_free_and_occupied_cells_have_distinct_colors(self):
        values = np.zeros(48, dtype=np.int8)
        values[0] = -1
        values[1] = 100
        image = self.renderer.render_grid_diagnostic(
            grid(values), self.pose, "slam"
        )

        self.assertEqual((6, 8, 3), image.shape)
        self.assertTupleEqual((112, 112, 112), tuple(image[5, 0]))
        self.assertTupleEqual((24, 24, 24), tuple(image[5, 1]))
        self.assertTupleEqual((246, 246, 246), tuple(image[5, 2]))

    def test_costmap_marks_lethal_cells_red(self):
        values = np.zeros(48, dtype=np.int8)
        values[0] = 100
        image = self.renderer.render_grid_diagnostic(
            grid(values), self.pose, "local_costmap"
        )

        self.assertTupleEqual((45, 45, 210), tuple(image[5, 0]))

    def test_rejects_malformed_grid(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.renderer.render_grid_diagnostic(
                grid([0, 1], width=8, height=6), self.pose, "slam"
            )

    def test_draws_lidar_plan_footprint_and_goal_overlays(self):
        values = np.zeros(400, dtype=np.int8)
        image = self.renderer.render_grid_diagnostic(
            grid(values, width=20, height=20),
            self.pose,
            "slam",
            overlays={
                "lidar": [(1.5, 1.5)],
                "plan": [(0.1, 0.1), (0.8, 0.1)],
                "footprint": [(0.3, 1.0), (0.7, 1.0),
                              (0.7, 1.4), (0.3, 1.4)],
                "goal": {"x": 1.5, "y": 0.5, "yaw": 0.0},
            },
        )

        lidar_pixel = image[4, 15]
        self.assertLess(lidar_pixel[0], 50)
        self.assertGreater(lidar_pixel[1], 170)
        self.assertGreater(lidar_pixel[2], 240)
        goal_pixel = image[14, 15]
        self.assertGreater(goal_pixel[0], 180)
        self.assertLess(goal_pixel[1], 100)
        self.assertGreater(goal_pixel[2], 180)
        self.assertTrue(np.any(np.all(image == (220, 145, 30), axis=2)))
        self.assertTrue(np.any(np.all(image == (30, 210, 80), axis=2)))

    def test_opposite_viewer_rotation_flips_the_complete_image(self):
        image = np.asarray([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ], dtype=np.uint8)

        rotated = self.renderer.rotate_180(image)

        np.testing.assert_array_equal(
            rotated,
            np.asarray([
                [[10, 11, 12], [7, 8, 9]],
                [[4, 5, 6], [1, 2, 3]],
            ], dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()
