"""Costmap projection, reachability, and frontier analysis."""

import math

import cv2
import numpy as np

from .grid import Grid


class MapAnalysisMixin:
    def prepare(self, map_msg, cost_msg, pose):
        grid = Grid(map_msg)
        costmap = Grid(cost_msg)
        costs = self._costs_on_map(grid, costmap)
        reachable = self._reachable(grid, costs, pose, self.cfg.clearance)
        frontiers, frontier_mask = self._frontiers(grid, costs, pose)
        for index, frontier in enumerate(frontiers, 1):
            frontier["id"] = f"F{index}"
        return {
            "frame_id": pose["frame_id"],
            "grid": grid,
            "costmap": costmap,
            "costs": costs,
            "reachable": reachable,
            "frontiers": frontiers,
            "frontier_mask": frontier_mask,
        }

    def _costs_on_map(self, grid, costmap):
        rows, cols = np.indices(grid.data.shape)
        return costmap.sample(*grid.world(cols, rows))

    def _robot_cell(self, grid, pose):
        col, row = grid.cells(pose["x"], pose["y"])
        return int(col), int(row)

    @staticmethod
    def _connected(mask, robot_col, robot_row):
        """Return the component nearest the robot without trusting one cell."""
        valid_rows, valid_cols = np.nonzero(mask)
        if not valid_rows.size:
            return np.zeros_like(mask, bool)

        distance_sq = (
            (valid_rows - int(robot_row)) ** 2
            + (valid_cols - int(robot_col)) ** 2
        )
        nearest = int(np.argmin(distance_sq))
        seed_row = int(valid_rows[nearest])
        seed_col = int(valid_cols[nearest])
        _, labels = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=4
        )
        return labels == labels[seed_row, seed_col]

    def _reachable(self, grid, costs, pose, clearance=0.0):
        traversable = (
            (grid.data >= 0)
            & (grid.data < self.cfg.occupied)
            & (costs >= 0)
            & (costs < self.cfg.cost_limit)
        )
        robot_col, robot_row = self._robot_cell(grid, pose)
        connected = self._connected(traversable, robot_col, robot_row)
        if clearance <= 0 or not connected.any():
            return connected

        distance = cv2.distanceTransform(
            traversable.astype(np.uint8), cv2.DIST_L2, 5
        ) * grid.resolution
        safe = distance >= clearance

        # The robot may legitimately occupy a cell rejected by endpoint
        # clearance. Preserve a small traversable egress around its pose so
        # rays can reach safe cells instead of failing at distance zero.
        rows, cols = np.indices(traversable.shape)
        egress_radius = clearance + grid.resolution
        egress = (
            ((rows - robot_row) * grid.resolution) ** 2
            + ((cols - robot_col) * grid.resolution) ** 2
            <= egress_radius ** 2
        )
        return connected & (safe | egress)

    def _ray_gain(self, grid, row, col):
        visible = set()
        max_cells = self.cfg.frontier_gain_radius / grid.resolution

        distances = np.arange(0.5, max_cells + 0.5, 0.5)
        for angle in np.linspace(
            0, 2 * math.pi, self.cfg.frontier_rays, endpoint=False
        ):
            cols = np.rint(col + math.cos(angle) * distances).astype(int)
            rows = np.rint(row + math.sin(angle) * distances).astype(int)

            # Half-cell samples can round to the same cell consecutively.
            keep = np.ones(len(rows), dtype=bool)
            keep[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
            rows, cols = rows[keep], cols[keep]

            inside = (
                (rows >= 0) & (rows < grid.data.shape[0])
                & (cols >= 0) & (cols < grid.data.shape[1])
            )
            if not np.all(inside):
                stop = int(np.argmax(~inside))
                rows, cols = rows[:stop], cols[:stop]
            if not len(rows):
                continue

            values = grid.data[rows, cols]
            occupied = np.flatnonzero(values >= self.cfg.occupied)
            if len(occupied):
                stop = int(occupied[0])
                rows, cols, values = rows[:stop], cols[:stop], values[:stop]
            visible.update(zip(rows[values < 0].tolist(), cols[values < 0].tolist()))

        return len(visible) * grid.resolution ** 2

    def _frontier_samples(self, rows, cols, max_samples):
        points = np.column_stack((rows, cols)).astype(float)

        center = points.mean(axis=0)
        first = int(np.argmin(np.sum((points - center) ** 2, axis=1)))

        selected = [first]

        while len(selected) < min(max_samples, len(points)):
            chosen = points[selected]

            # For every point, find its distance to the nearest selected point.
            distances = np.min(
                np.sum(
                    (points[:, None, :] - chosen[None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )

            distances[selected] = -1
            selected.append(int(np.argmax(distances)))

        return np.asarray(selected, dtype=int)

    def _useful_unknown(self, grid):
        unknown = grid.data < 0
        count, labels = cv2.connectedComponents(
            unknown.astype(np.uint8),
            connectivity=4,
        )

        useful = unknown.copy()
        h, w = unknown.shape

        for label in range(1, count):
            component = labels == label
            rows, cols = np.nonzero(component)

            # If this unknown region touches the map boundary,
            # it is almost certainly part of the real unexplored world.
            touches_border = (
                (rows == 0).any()
                or (rows == h - 1).any()
                or (cols == 0).any()
                or (cols == w - 1).any()
            )

            area = len(rows) * grid.resolution ** 2

            # Small fully-enclosed unknown islands are map holes, not exploration goals.
            if not touches_border and area < self.cfg.frontier_hole_min_area:
                useful[component] = False

        return useful

    def _frontiers(self, grid, costs, pose):
        unknown = self._useful_unknown(grid)
        reachable = self._reachable(grid, costs, pose)

        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
        unknown_neighbors = cv2.filter2D(
            unknown.astype(np.uint8), -1, kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        raw = reachable & (unknown_neighbors > 0)

        count, labels = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)
        min_cells = max(1, math.ceil(self.cfg.frontier_min_length / grid.resolution))
        frontiers = []

        for label in range(1, count):
            rows, cols = np.nonzero(labels == label)
            if len(rows) < min_cells:
                continue

            samples = self._frontier_samples(
                rows, cols, self.cfg.frontier_samples_per_component
            )

            best = None
            for i in samples:
                row, col = int(rows[i]), int(cols[i])
                gain = self._ray_gain(grid, row, col)
                if best is None or gain > best[0]:
                    best = gain, row, col

            gain, row, col = best
            if gain < self.cfg.frontier_min_gain:
                continue

            x, y = grid.world(col, row)

            r0, r1 = max(0, row - 1), min(grid.data.shape[0], row + 2)
            c0, c1 = max(0, col - 1), min(grid.data.shape[1], col + 2)
            ur, uc = np.nonzero(unknown[r0:r1, c0:c1])
            unknown_x, unknown_y = grid.world(uc.mean() + c0, ur.mean() + r0)

            frontiers.append({
                "kind": "frontier",
                "x": float(x),
                "y": float(y),
                "yaw": math.atan2(float(unknown_y - y), float(unknown_x - x)),
                "information_gain_m2": round(float(gain), 3),
                "unknown_neighbor_count": int(unknown_neighbors[row, col]),
                "group_size_cells": int(len(rows)),
                "cell": [col, row],
                "_component_label": int(label),
            })

        # Greedy non-maximum suppression: when two frontier goals are too
        # close, retain the one expected to reveal the most unknown area.
        ranked = sorted(
            frontiers,
            key=lambda f: (
                -f["information_gain_m2"],
                -f["unknown_neighbor_count"],
                -f["group_size_cells"],
                f["y"],
                f["x"],
            ),
        )
        selected = []
        for frontier in ranked:
            if any(
                math.hypot(
                    frontier["x"] - other["x"],
                    frontier["y"] - other["y"],
                ) < self.cfg.frontier_min_separation
                for other in selected
            ):
                continue
            selected.append(frontier)

        accepted = np.zeros_like(raw)
        for frontier in selected:
            accepted[labels == frontier.pop("_component_label")] = True

        return selected, accepted

