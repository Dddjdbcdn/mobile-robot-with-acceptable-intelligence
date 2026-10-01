"""Shared geometric helpers for map planners."""

import math

import numpy as np


class GeometryMixin:
    @staticmethod
    def _point_on_ray(origin, yaw, distance):
        return (
            float(origin["x"]) + distance * math.cos(yaw),
            float(origin["y"]) + distance * math.sin(yaw),
        )

    def _ray_cells(self, grid, origin, yaw, distances):
        xs, ys = self._point_on_ray(origin, yaw, distances)
        cols, rows = grid.cells(xs, ys)
        return xs, ys, cols, rows

    def _local_goal(self, x, y, yaw, pose):
        return {
            "x": float(x), "y": float(y), "angle": self._angle(float(yaw)),
            "frame_id": str(pose.get("frame_id") or "map"),
        }

    def _person_facing_yaw(self, x, y, person_pose):
        """Return the yaw from a destination to a usable lidar person pose."""
        if not isinstance(person_pose, dict):
            return None
        try:
            person_x = float(person_pose["x"])
            person_y = float(person_pose["y"])
            age = float(person_pose.get("age_seconds", 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (person_x, person_y, age)):
            return None
        if age < 0.0 or age > self.cfg.local_person_pose_max_age_seconds:
            return None
        dx, dy = person_x - float(x), person_y - float(y)
        if math.hypot(dx, dy) <= 1e-6:
            return None
        return math.atan2(dy, dx)

    @staticmethod
    def _angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _valid_endpoint(grid, reachable, x, y):
        col, row = grid.cells(x, y)
        return bool(
            grid.inside(col, row) and reachable[int(row), int(col)]
        )

    def _known_free(self, grid):
        return (grid.data >= 0) & (grid.data < self.cfg.occupied)

    def _furthest_reachable_on_ray(
        self, grid, reachable, origin, yaw, max_distance=None
    ):
        """Return the center of the last reachable cell before a ray is blocked."""
        if max_distance is None:
            height, width = grid.data.shape
            max_distance = math.hypot(width, height) * grid.resolution
        else:
            max_distance = max(0.0, float(max_distance))
        step = max(grid.resolution / 2.0, 1e-6)
        last_cell = None

        for distance in np.arange(0.0, max_distance + step, step):
            x, y = self._point_on_ray(origin, yaw, float(distance))
            col, row = (int(value) for value in grid.cells(x, y))
            if not grid.inside(col, row) or not reachable[row, col]:
                break
            last_cell = (col, row)

        if last_cell is None:
            return None
        x, y = grid.world(*last_cell)
        return float(x), float(y), last_cell
