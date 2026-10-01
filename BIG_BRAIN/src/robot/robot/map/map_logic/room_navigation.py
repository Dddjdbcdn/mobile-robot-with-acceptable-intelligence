"""Geometric room-core and room-exit navigation."""

from collections import deque
import math

import cv2
import numpy as np


class RoomNavigationMixin:
    def resolve_open_space_middle_step(
        self, prepared, pose, state=None, *, person_pose=None
    ):
        """Move toward a mapped room core, exploring briefly if none exists."""
        grid = prepared["grid"]
        reachable = prepared["reachable"]
        state = dict(state) if isinstance(state, dict) else {}
        state.setdefault("recovery_points", [])
        state.setdefault("recovery_moves", 0)
        state.setdefault("passes", 0)

        labels, component_areas = self._exit_open_components(grid, reachable)
        core_labels = [
            label for label, area in component_areas.items()
            if area >= self.cfg.exit_component_min_area_m2
        ]
        eligible_labels = np.where(np.isin(labels, core_labels), labels, 0)
        core_label = self._nearest_component_label(
            grid, eligible_labels, pose["x"], pose["y"],
            self.cfg.exit_component_lookup_radius_m,
        )
        if core_label == 0:
            recovery = self._exit_map_recovery_step(
                prepared, pose, state, [], "no_room_core"
            )
            if recovery is not None:
                return recovery
            raise ValueError(
                "no sufficiently open room core after bounded map recovery"
            )

        rows, cols = np.nonzero(labels == core_label)
        classification_free = (grid.data >= 0) & ~self._exit_wall_mask(grid)
        clearance = cv2.distanceTransform(
            classification_free.astype(np.uint8), cv2.DIST_L2, 5
        ) * grid.resolution
        xs, ys = grid.world(cols, rows)
        distance_sq = (xs - pose["x"]) ** 2 + (ys - pose["y"]) ** 2
        index = int(np.lexsort((distance_sq, -clearance[rows, cols]))[0])
        x, y = float(xs[index]), float(ys[index])
        travel_yaw = math.atan2(y - pose["y"], x - pose["x"])
        person_yaw = self._person_facing_yaw(x, y, person_pose)
        state["passes"] = int(state.get("passes", 0)) + 1
        return {
            "complete": False,
            "phase": "move_to_room_core",
            "state": state,
            "destination": self._local_goal(
                x, y, travel_yaw if person_yaw is None else person_yaw, pose
            ),
        }

    def resolve_exit_room_step(self, prepared, pose, state=None):
        """Explore the starting chamber, then cross its narrow boundary."""
        grid = prepared["grid"]
        reachable = prepared["reachable"]
        state = dict(state) if isinstance(state, dict) else {}
        if not isinstance(state.get("origin"), dict):
            state.update({
                "origin": {"x": float(pose["x"]), "y": float(pose["y"])},
                "visited_frontiers": [],
                "recovery_points": [],
                "recovery_moves": 0,
                "passes": 0,
            })
        visited = [
            point for point in state.get("visited_frontiers") or []
            if isinstance(point, list) and len(point) == 2
        ]
        labels, component_areas = self._exit_open_components(grid, reachable)
        origin_label = self._nearest_component_label(
            grid, labels, state["origin"]["x"], state["origin"]["y"],
            self.cfg.exit_component_lookup_radius_m,
        )
        # Before a core has ever been established, safe recovery motion may
        # expose one near the new pose but outside the original lookup radius.
        if origin_label == 0 and int(state.get("recovery_moves", 0)):
            origin_label = self._nearest_component_label(
                grid, labels, pose["x"], pose["y"],
                self.cfg.exit_component_lookup_radius_m,
            )
            if origin_label:
                state["origin"] = {
                    "x": float(pose["x"]), "y": float(pose["y"])
                }
        if origin_label == 0:
            recovery = self._exit_map_recovery_step(
                prepared, pose, state, visited, "no_room_core"
            )
            if recovery is not None:
                return recovery
            raise ValueError(
                "starting room has no sufficiently open interior after "
                "bounded map recovery"
            )

        robot_col, robot_row = self._robot_cell(grid, pose)
        current_label = (
            int(labels[robot_row, robot_col])
            if grid.inside(robot_col, robot_row) else 0
        )
        if current_label and current_label != origin_label:
            return {"complete": True, "phase": "outside", "state": state}

        other_labels = [
            label for label, area in component_areas.items()
            if label != origin_label
            and area >= self.cfg.exit_component_min_area_m2
        ]
        # Frontiers are generated from connected traversable cells without the
        # endpoint-clearance erosion used by `prepared["reachable"]`. Propagate
        # room ownership over that same mask so valid boundary cells are not
        # left unlabeled merely because they lie close to unknown space.
        frontier_reachable = reachable
        if prepared.get("costs") is not None:
            frontier_reachable = self._reachable(
                grid, prepared["costs"], pose
            )
        room_labels = self._exit_reachable_room_labels(
            frontier_reachable, labels, [origin_label, *other_labels]
        )
        room_frontiers = []
        for frontier in prepared.get("frontiers") or []:
            frontier_col, frontier_row = (
                int(value)
                for value in grid.cells(frontier["x"], frontier["y"])
            )
            frontier_label = (
                int(room_labels[frontier_row, frontier_col])
                if grid.inside(frontier_col, frontier_row) else 0
            )
            already_visited = any(
                math.hypot(frontier["x"] - point[0], frontier["y"] - point[1])
                < self.cfg.exit_frontier_revisit_m
                for point in visited
            )
            if frontier_label == origin_label and not already_visited:
                room_frontiers.append(frontier)

        if room_frontiers:
            def frontier_rank(item):
                distance = math.hypot(
                    item["x"] - pose["x"], item["y"] - pose["y"]
                )
                gain = float(item.get("information_gain_m2") or 0.0)
                utility = gain - (
                    self.cfg.exit_frontier_travel_penalty_m2_per_m * distance
                )
                return -utility, distance, -gain

            frontier = min(room_frontiers, key=frontier_rank)
            visited.append([float(frontier["x"]), float(frontier["y"])])
            state.update(
                visited_frontiers=visited,
                passes=int(state.get("passes", 0)) + 1,
            )
            destination = self._local_goal(
                frontier["x"], frontier["y"], frontier["yaw"], pose
            )
            return {
                "complete": False,
                "phase": "explore_room_frontier",
                "state": state,
                "destination": destination,
            }

        if not other_labels:
            recovery = self._exit_map_recovery_step(
                prepared, pose, state, visited, "no_exit_evidence"
            )
            if recovery is not None:
                return recovery
            raise ValueError(
                "room frontiers and bounded map recovery are exhausted but "
                "no geometric exit is mapped"
            )

        mask = np.isin(labels, other_labels)
        rows, cols = np.nonzero(mask)
        xs, ys = grid.world(cols, rows)
        index = int(np.argmin((xs - pose["x"]) ** 2 + (ys - pose["y"]) ** 2))
        x, y = float(xs[index]), float(ys[index])
        state["passes"] = int(state.get("passes", 0)) + 1
        return {
            "complete": False,
            "phase": "cross_room_boundary",
            "state": state,
            "destination": self._local_goal(
                x, y, math.atan2(y - pose["y"], x - pose["x"]), pose
            ),
        }

    def _exit_map_recovery_step(
        self, prepared, pose, state, visited_frontiers, reason
    ):
        """Choose one short, known-safe move to improve a noisy room map."""
        recovery_moves = int(state.get("recovery_moves", 0))
        if recovery_moves >= self.cfg.exit_recovery_max_moves:
            return None

        grid = prepared["grid"]
        reachable = np.asarray(prepared["reachable"], dtype=bool)
        rows, cols = np.nonzero(reachable)
        if not rows.size:
            return None
        xs, ys = grid.world(cols, rows)
        travel = np.hypot(xs - pose["x"], ys - pose["y"])
        eligible = (
            (travel >= self.cfg.exit_recovery_min_move_m)
            & (travel <= self.cfg.exit_recovery_max_move_m)
        )

        recovery_points = [
            point for point in state.get("recovery_points") or []
            if isinstance(point, list) and len(point) == 2
        ]
        for point in recovery_points:
            eligible &= (
                np.hypot(xs - point[0], ys - point[1])
                >= self.cfg.exit_recovery_revisit_m
            )
        if not np.any(eligible):
            return None

        rows, cols = rows[eligible], cols[eligible]
        xs, ys, travel = xs[eligible], ys[eligible], travel[eligible]
        known_free = (grid.data >= 0) & (grid.data < self.cfg.occupied)
        clearance = cv2.distanceTransform(
            known_free.astype(np.uint8), cv2.DIST_L2, 5
        ) * grid.resolution
        endpoint_clearance = clearance[rows, cols]

        unvisited_frontiers = [
            frontier for frontier in prepared.get("frontiers") or []
            if not any(
                math.hypot(
                    frontier["x"] - point[0], frontier["y"] - point[1]
                ) < self.cfg.exit_frontier_revisit_m
                for point in visited_frontiers
            )
        ]
        if unvisited_frontiers:
            frontier_distance_sq = np.min(
                np.stack([
                    (xs - frontier["x"]) ** 2
                    + (ys - frontier["y"]) ** 2
                    for frontier in unvisited_frontiers
                ]),
                axis=0,
            )
            index = int(np.lexsort(
                (-endpoint_clearance, frontier_distance_sq)
            )[0])
        else:
            # Prefer the safest endpoint, using useful motion as the tie-break.
            index = int(np.lexsort((-travel, -endpoint_clearance))[0])

        x, y = float(xs[index]), float(ys[index])
        recovery_points.append([x, y])
        state.update(
            recovery_points=recovery_points,
            recovery_moves=recovery_moves + 1,
            passes=int(state.get("passes", 0)) + 1,
        )
        return {
            "complete": False,
            "phase": "stabilize_room_map",
            "state": state,
            "recovery_reason": reason,
            "destination": self._local_goal(
                x, y, math.atan2(y - pose["y"], x - pose["x"]), pose
            ),
        }

    def _exit_wall_mask(self, grid):
        """Keep long occupied components as walls, ignoring small furniture."""
        occupied = grid.data >= self.cfg.occupied
        count, labels = cv2.connectedComponents(
            occupied.astype(np.uint8), connectivity=8
        )
        keep = np.zeros(count, dtype=bool)
        minimum_length = max(0.0, self.cfg.exit_wall_min_length_m)

        for label in range(1, count):
            rows, cols = np.nonzero(labels == label)
            if not rows.size:
                continue
            if rows.size == 1:
                length = grid.resolution
            else:
                points = np.column_stack((cols, rows)).astype(np.float32)
                centered = points - points.mean(axis=0)
                _, _, axes = np.linalg.svd(centered, full_matrices=False)
                projection = centered @ axes[0]
                length = (
                    float(projection.max() - projection.min()) + 1.0
                ) * grid.resolution
            keep[label] = length >= minimum_length

        return keep[labels]

    def _exit_open_components(self, grid, reachable):
        # Unknown space and structural walls bound rooms. Small occupied
        # components remain excluded by `reachable`, but do not create the
        # large clearance halo that can split one room around furniture.
        classification_free = (grid.data >= 0) & ~self._exit_wall_mask(grid)
        clearance = cv2.distanceTransform(
            classification_free.astype(np.uint8), cv2.DIST_L2, 5
        ) * grid.resolution
        open_space = reachable & (clearance >= self.cfg.exit_open_clearance_m)
        count, labels = cv2.connectedComponents(
            open_space.astype(np.uint8), connectivity=8
        )
        areas = {
            label: float(np.count_nonzero(labels == label) * grid.resolution ** 2)
            for label in range(1, count)
        }
        return labels, areas

    @staticmethod
    def _exit_reachable_room_labels(reachable, open_labels, room_labels):
        """Assign reachable floor to its nearest room core by path distance.

        Expansion is constrained to reachable cells, so labels cannot pass
        through walls. When multiple meaningful room cores exist, their waves
        compete and split connecting doorway/corridor space between them.
        """
        reachable = np.asarray(reachable, dtype=bool)
        open_labels = np.asarray(open_labels)
        allowed = np.asarray(room_labels, dtype=open_labels.dtype)
        owners = np.where(
            reachable & np.isin(open_labels, allowed), open_labels, 0
        ).astype(np.int32)
        queue = deque(
            (int(row), int(col))
            for row, col in zip(*np.nonzero(owners))
        )
        height, width = owners.shape
        neighbors = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
        while queue:
            row, col = queue.popleft()
            owner = owners[row, col]
            for row_offset, col_offset in neighbors:
                next_row = row + row_offset
                next_col = col + col_offset
                if not (
                    0 <= next_row < height and 0 <= next_col < width
                    and reachable[next_row, next_col]
                    and owners[next_row, next_col] == 0
                ):
                    continue
                owners[next_row, next_col] = owner
                queue.append((next_row, next_col))
        return owners

    @staticmethod
    def _nearest_component_label(grid, labels, x, y, max_distance_m):
        col, row = (int(value) for value in grid.cells(x, y))
        if grid.inside(col, row) and labels[row, col]:
            return int(labels[row, col])
        radius = max(1, math.ceil(max_distance_m / grid.resolution))
        r0, r1 = max(0, row - radius), min(labels.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(labels.shape[1], col + radius + 1)
        local_rows, local_cols = np.nonzero(labels[r0:r1, c0:c1])
        if not len(local_rows):
            return 0
        distances = (local_rows + r0 - row) ** 2 + (local_cols + c0 - col) ** 2
        index = int(np.argmin(distances))
        return int(labels[local_rows[index] + r0, local_cols[index] + c0])
