"""Obstacle-aware target approach planning."""

import math

import cv2
import numpy as np


class ApproachNavigationMixin:
    def _approach_clearance(self, prepared):
        """Return obstacle clearance in metres for approach-ray checks."""
        grid = prepared["grid"]
        obstacles = grid.data >= self.cfg.occupied
        costs = prepared.get("costs")
        if costs is None and prepared.get("costmap") is not None:
            costs = self._costs_on_map(grid, prepared["costmap"])
        if costs is not None:
            obstacles |= np.asarray(costs) >= self.cfg.approach_cost_threshold

        minimum_cells = max(1, self.cfg.approach_obstacle_min_cells)
        if minimum_cells > 1 and np.any(obstacles):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                obstacles.astype(np.uint8), connectivity=8
            )
            keep = np.zeros(count, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_cells
            obstacles = keep[labels]

        return cv2.distanceTransform(
            (~obstacles).astype(np.uint8), cv2.DIST_L2, 5
        ) * grid.resolution

    def _first_approach_obstacle_distance(
        self, prepared, origin, yaw, max_distance
    ):
        grid = prepared["grid"]
        sample_count = max(
            1, int(math.ceil(max_distance / (grid.resolution / 2.0)))
        )
        distances = np.linspace(0.0, max_distance, sample_count + 1)[1:]
        _, _, cols, rows = self._ray_cells(grid, origin, yaw, distances)
        valid = np.flatnonzero(grid.inside(cols, rows))
        if not valid.size:
            return None

        clearance = self._approach_clearance(prepared)
        hits = valid[
            clearance[
                np.asarray(rows[valid], dtype=int),
                np.asarray(cols[valid], dtype=int),
            ] <= self.cfg.approach_ray_radius_m
        ]
        return float(distances[int(hits[0])]) if hits.size else None

    def _nearest_reachable_distance(
        self, grid, reachable, origin, yaw, max_distance
    ):
        sample_count = max(
            1, int(math.ceil(max_distance / (grid.resolution / 2.0)))
        )
        distances = np.linspace(max_distance, 0.0, sample_count + 1)
        _, _, cols, rows = self._ray_cells(grid, origin, yaw, distances)
        valid = np.flatnonzero(grid.inside(cols, rows))
        if valid.size:
            valid = valid[reachable[rows[valid], cols[valid]]]
        return float(distances[int(valid[0])]) if valid.size else None

    def resolve_approach_destination(
        self, prepared, pose, target_x, target_y, standoff_m=None
    ):
        """Return the first safe approach pose on an inflated target ray."""
        grid = prepared["grid"]
        reachable = prepared["reachable"]
        target_x = float(target_x)
        target_y = float(target_y)
        target_distance = math.hypot(target_x, target_y)
        if not math.isfinite(target_distance) or target_distance <= 0.0:
            raise ValueError("approach target must be away from the robot")

        local_bearing = math.atan2(target_y, target_x)
        map_yaw = self._angle(float(pose["yaw"]) + local_bearing)
        hit_distance = self._first_approach_obstacle_distance(
            prepared, pose, map_yaw, target_distance
        )

        if hit_distance is None:
            distance_limit = target_distance
            resolution = "standoff_from_target"
        else:
            distance_limit = hit_distance
            resolution = "standoff_from_obstacle"

        requested_standoff = (
            self.cfg.approach_obstacle_standoff_m
            if standoff_m is None else float(standoff_m)
        )
        if not math.isfinite(requested_standoff) or requested_standoff < 0.0:
            raise ValueError("approach standoff must be a non-negative finite number")
        resolved_distance = max(0.0, distance_limit - requested_standoff)

        destination_x, destination_y = self._point_on_ray(
            pose, map_yaw, resolved_distance
        )

        # If the desired cell is unreachable, walk backward toward the robot.
        # Geometric samples stay on the ray instead of shifting sideways.
        moved_to_reachable = not self._valid_endpoint(
            grid, reachable, destination_x, destination_y
        )
        if moved_to_reachable:
            reachable_distance = self._nearest_reachable_distance(
                grid, reachable, pose, map_yaw, resolved_distance
            )
            if reachable_distance is None:
                raise ValueError("no reachable approach pose on target ray")

            resolved_distance = reachable_distance
            destination_x, destination_y = self._point_on_ray(
                pose, map_yaw, resolved_distance
            )

        return {
            "x": float(destination_x),
            "y": float(destination_y),
            "angle": float(map_yaw),
            "frame_id": str(prepared.get("frame_id") or pose["frame_id"]),
            "resolution": resolution,
            "was_clamped": hit_distance is not None,
            "moved_to_reachable": moved_to_reachable,
            "target_distance_m": float(target_distance),
            "obstacle_distance_m": hit_distance,
            "resolved_distance_m": float(resolved_distance),
            "standoff_m": requested_standoff,
        }

