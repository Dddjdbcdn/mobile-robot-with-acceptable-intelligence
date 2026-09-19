"""Map analysis, candidate generation, coverage sampling, and search ranking."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

@dataclass
class LogicConfig:
    # Occupancy / costmap
    occupied: int = 50
    cost_limit: int = 99
    clearance: float = 0.10

    # Local navigation samples
    sample_radius: float = 3.0
    sample_step: float = 0.5
    sample_distances: tuple = (0.5, 1.0, 2.0, 3.0)
    sample_angle_deg: float = 45.0
    include_rotations: bool = True

    # Frontiers
    frontier_hole_min_area: float = 0.5
    frontier_min_length: float = 0.30
    frontier_gain_radius: float = 1.0
    frontier_min_gain: float = 0.25
    frontier_min_separation: float = 0.75
    frontier_rays: int = 120

    # Doors / rooms
    enable_doors: bool = False
    enable_rooms: bool = False
    door_min_width: float = 0.5
    door_max_width: float = 1.5
    door_probe: float = 0.5
    door_min_widening: float = 0.1
    door_wall_min_length: float = 0.3
    door_angles: int = 12
    door_min_separation: float = 0.5
    room_min_area: float = 1.0

    # Search views
    camera_center_pan_deg: float = 95.0
    camera_horizontal_fov_deg: float = 60.0
    camera_fov_max_range_m: float = 2.0
    context_radius_samples: int = 3
    context_bearing_fractions: tuple = (-0.75, 0.0, 0.75)
    context_backup_distance_m: float = 0.5
    exploration_step: float = 0.5
    exploration_headings: int = 8
    exploration_shortlist: int = 16
    exploration_frontier_distance_floor_m: float = 0.25


def yaw_of(q):
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z),
    )


class Grid:
    def __init__(self, msg):
        self.resolution = float(msg.info.resolution)
        self.data = np.asarray(msg.data, np.int16).reshape(
            msg.info.height, msg.info.width
        )
        self.origin = msg.info.origin.position
        self.origin_yaw = yaw_of(msg.info.origin.orientation)
        self.c = math.cos(self.origin_yaw)
        self.s = math.sin(self.origin_yaw)

    def world(self, col, row):
        x = (np.asarray(col) + 0.5) * self.resolution
        y = (np.asarray(row) + 0.5) * self.resolution
        return (
            self.origin.x + self.c * x - self.s * y,
            self.origin.y + self.s * x + self.c * y,
        )

    def cells(self, x, y):
        dx = np.asarray(x) - self.origin.x
        dy = np.asarray(y) - self.origin.y
        col = np.floor((self.c * dx + self.s * dy) / self.resolution).astype(int)
        row = np.floor((-self.s * dx + self.c * dy) / self.resolution).astype(int)
        return col, row

    def inside(self, col, row):
        h, w = self.data.shape
        return (col >= 0) & (row >= 0) & (col < w) & (row < h)

    def sample(self, x, y):
        col, row = self.cells(x, y)
        h, w = self.data.shape
        values = self.data[np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)]
        return np.where(self.inside(col, row), values, -1)

    def world_bounds(self):
        h, w = self.data.shape
        corners = [(0, 0), (w, 0), (0, h), (w, h)]
        points = []
        for col, row in corners:
            x = self.origin.x + self.c * col * self.resolution - self.s * row * self.resolution
            y = self.origin.y + self.s * col * self.resolution + self.c * row * self.resolution
            points.append((x, y))
        xs, ys = zip(*points)
        return {"xmin": min(xs), "xmax": max(xs), "ymin": min(ys), "ymax": max(ys)}

class MapLogic:
    def __init__(self, config=None):
        self.cfg = config if config is not None else LogicConfig()

    def prepare(self, map_msg, cost_msg, pose):
        grid = Grid(map_msg)
        costmap = Grid(cost_msg)
        costs = self._costs_on_map(grid, costmap)
        reachable = self._reachable(grid, costs, pose, self.cfg.clearance)
        frontiers, frontier_mask = self._frontiers(grid, costs, pose)
        if self.cfg.enable_doors:
            doors, door_candidate_mask = self._doors(grid, costs, pose)
        else:
            doors = []
            door_candidate_mask = np.zeros_like(grid.data, dtype=bool)
        if self.cfg.enable_rooms:
            rooms, current_room_mask = self._rooms(grid, pose, doors)
        else:
            rooms = []
            current_room_mask = np.zeros_like(grid.data, dtype=bool)
        for index, frontier in enumerate(frontiers, 1):
            frontier["id"] = f"F{index}"
        for index, door in enumerate(doors, 1):
            door["id"] = f"D{index}"
        return {
            "frame_id": pose["frame_id"],
            "grid": grid,
            "costmap": costmap,
            "reachable": reachable,
            "frontiers": frontiers,
            "frontier_mask": frontier_mask,
            "doors": doors,
            "door_candidate_mask": door_candidate_mask,
            "rooms": rooms,
            "current_room_mask": current_room_mask,
        }


    def live_camera_fov(self, pose, camera_pan_angle):
        if camera_pan_angle is None:
            return None
        pan = float(camera_pan_angle)
        return {
            "pan_angle_deg": pan,
            "horizontal_fov_deg": self.cfg.camera_horizontal_fov_deg,
            "max_range_m": self.cfg.camera_fov_max_range_m,
            "center_yaw_rad": float(
                pose["yaw"] + math.radians(
                    pan - self.cfg.camera_center_pan_deg
                )
            ),
        }

    def observation_fov(self, search_overlay):
        """Return the map-space FOV belonging to the selected clue frame."""
        if not isinstance(search_overlay, dict):
            return None
        observation = search_overlay.get("observation")
        if not isinstance(observation, dict):
            return None
        pose = observation.get("robot_pose")
        if not isinstance(pose, dict):
            return None
        try:
            normalized = {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "yaw": float(pose["yaw"]),
            }
            pan = float(observation.get(
                "pan_angle", self.cfg.camera_center_pan_deg
            ))
            horizontal_fov = float(search_overlay.get(
                "camera_horizontal_fov_deg",
                self.cfg.camera_horizontal_fov_deg,
            ))
            max_range = float(search_overlay.get(
                "camera_reliable_range_m",
                self.cfg.camera_fov_max_range_m,
            ))
        except (KeyError, TypeError, ValueError):
            return None
        return {
            "pose": normalized,
            "pan_angle_deg": pan,
            "horizontal_fov_deg": horizontal_fov,
            "max_range_m": max_range,
            "center_yaw_rad": float(
                normalized["yaw"] + math.radians(
                    pan - self.cfg.camera_center_pan_deg
                )
            ),
        }

    def plan_candidates(
        self, prepared, pose, search_overlay=None, *, coverage=None
    ):
        if not isinstance(search_overlay, dict):
            return []

        overlay = search_overlay
        mode = str(overlay.get("mode") or "goal")

        if mode == "context":
            return self._context_candidates(prepared, pose, overlay)
        elif mode == "exploration":
            if coverage is None:
                coverage = self.coverage_counts(prepared["grid"], overlay) > 0
            winner = self._exploration_candidate(
                prepared, pose, overlay, coverage
            )
            return [winner] if winner else []
        elif mode == "goal":
            candidates = self._local_candidates(
                prepared["grid"], prepared["reachable"], pose
            ) + [dict(item) for item in prepared["frontiers"]]
            return [self._describe(item, pose) for item in candidates]
        else:
            return []

    @staticmethod
    def _describe(candidate, pose):
        candidate["frame_id"] = pose["frame_id"]
        candidate["distance_m"] = float(math.hypot(
            candidate["x"] - pose["x"], candidate["y"] - pose["y"]
        ))
        return candidate

    @staticmethod
    def _angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _valid_endpoint(grid, reachable, x, y):
        col, row = grid.cells(x, y)
        return bool(
            grid.inside(col, row) and reachable[int(row), int(col)]
        )

    def _furthest_reachable_on_ray(self, grid, reachable, origin, yaw):
        """Return the center of the last reachable cell before a ray is blocked."""
        height, width = grid.data.shape
        max_distance = math.hypot(width, height) * grid.resolution
        step = max(grid.resolution / 2.0, 1e-6)
        last_cell = None

        for distance in np.arange(0.0, max_distance + step, step):
            x = origin["x"] + float(distance) * math.cos(yaw)
            y = origin["y"] + float(distance) * math.sin(yaw)
            col, row = (int(value) for value in grid.cells(x, y))
            if not grid.inside(col, row) or not reachable[row, col]:
                break
            last_cell = (col, row)

        if last_cell is None:
            return None
        x, y = grid.world(*last_cell)
        return float(x), float(y), last_cell

    def _context_candidates(self, prepared, pose, overlay):
        """Offer close clue views, aligned frontiers, and long center travel."""
        observation = self.observation_fov(overlay)
        if observation is None:
            return []

        grid, reachable = prepared["grid"], prepared["reachable"]
        origin = observation["pose"]
        center = observation["center_yaw_rad"]
        half_fov = math.radians(observation["horizontal_fov_deg"] / 2.0)
        max_range = observation["max_range_m"]
        target = (
            origin["x"] + max_range * math.cos(center),
            origin["y"] + max_range * math.sin(center),
        )

        radii = np.linspace(
            0.0,
            max_range,
            max(1, int(self.cfg.context_radius_samples)),
        )
        bearings = (
            center
            + np.asarray(self.cfg.context_bearing_fractions, dtype=float)
            * half_fov
        )
        candidates, used_cells = [], set()
        for radius in radii:
            for bearing in bearings:
                x = origin["x"] + float(radius) * math.cos(bearing)
                y = origin["y"] + float(radius) * math.sin(bearing)
                cell = tuple(int(value) for value in grid.cells(x, y))
                if cell in used_cells or not self._valid_endpoint(
                    grid, reachable, x, y
                ):
                    continue
                candidate = self._describe({
                    "id": f"C{len(candidates) + 1}",
                    "kind": "context",
                    "x": float(x),
                    "y": float(y),
                    "yaw": math.atan2(target[1] - y, target[0] - x),
                    "selection_reason": "inside_observation_fov",
                }, pose)
                candidates.append(candidate)
                used_cells.add(cell)

        candidates.sort(key=lambda item: (
            abs(self._angle(item["yaw"] - center)), item["distance_m"]
        ))

        # The reliable camera range limits close clue-view samples, but it
        # should not prevent a deliberate long move in the clue direction.
        # Replace any ordinary sample in the same cell with a clearly labeled
        # endpoint at the last reachable cell before the center ray is blocked.
        farthest = self._furthest_reachable_on_ray(
            grid, reachable, origin, center
        )
        if farthest is not None:
            x, y, cell = farthest

            if math.hypot(x - origin["x"], y - origin["y"]) >= observation["max_range_m"]:
                candidates.append(self._describe({
                    "id": "CF",
                    "kind": "context",
                    "x": x,
                    "y": y,
                    "yaw": float(center),
                    "selection_reason": "furthest_reachable_on_center_ray",
                }, pose))

        # A frontier remains relevant when it is farther away than the camera's
        # reliable range. Context mode therefore filters frontiers by the
        # observation's angular cone, rather than by max_range.
        for frontier in prepared.get("frontiers", []):
            dx = float(frontier["x"]) - origin["x"]
            dy = float(frontier["y"]) - origin["y"]
            if math.hypot(dx, dy) <= 1e-9:
                continue
            bearing = math.atan2(dy, dx)
            if abs(self._angle(bearing - center)) > half_fov:
                continue
            # Frontier extraction already guarantees a connected, free,
            # cost-valid endpoint. Do not recheck it against the clearance-
            # eroded local-candidate mask: frontier cells border unknown space
            # by definition and that erosion would remove most valid goals.
            candidate = self._describe(dict(frontier), pose)
            candidate["selection_reason"] = "frontier_inside_observation_fov"
            candidates.append(candidate)

        back_distance = cfg.context_backup_distance_m,

        x = pose["x"] - back_distance * math.cos(center)
        y = pose["y"] - back_distance * math.sin(center)
        if self._valid_endpoint(grid, reachable, x, y):
            backup = self._describe({
                "id": "CB",
                "kind": "context",
                "x": float(x),
                "y": float(y),
                "yaw": math.atan2(target[1] - y, target[0] - x),
                "selection_reason": "reverse_for_wider_view",
            }, pose)
            candidates.append(backup)
        return candidates

    def _exploration_candidate(self, prepared, pose, overlay, coverage):
        """Pick the sampled reachable pose that reveals most uncovered cells."""
        grid, reachable = prepared["grid"], prepared["reachable"]
        known_free = (grid.data >= 0) & (grid.data < self.cfg.occupied)
        unseen = known_free & ~coverage
        rows, cols = self._exploration_samples(
            grid, reachable, unseen, coverage, pose
        )
        candidates = []

        for index, (row, col) in enumerate(zip(rows, cols), 1):
            x, y = grid.world(col, row)
            at_robot = math.hypot(x - pose["x"], y - pose["y"]) < grid.resolution
            candidate = self._describe({
                "id": f"E{index}",
                "kind": "rotation" if at_robot else "exploration",
                "x": float(x),
                "y": float(y),
                "yaw": float(pose["yaw"]),
            }, pose)
            yaw, gain, fraction = self._best_unseen_view(
                grid, candidate, unseen, known_free, overlay
            )
            if gain == 0:
                continue
            candidate.update({
                "yaw": yaw,
                "uncovered_cell_count": gain,
                "uncovered_ahead_fraction": round(fraction, 4),
                "ranking_score": gain,
                "selection_reason": "largest_unseen_view",
            })
            candidates.append(candidate)

        if candidates:
            return max(candidates, key=lambda item: (
                item["uncovered_cell_count"], -item["distance_m"]
            ))

        frontiers = [
            self._describe(dict(item), pose)
            for item in prepared["frontiers"]
        ]
        if not frontiers:
            return None
        distance_floor = max(
            1e-6, self.cfg.exploration_frontier_distance_floor_m
        )
        winner = max(frontiers, key=lambda item: (
            item.get("information_gain_m2", 0.0)
            / max(distance_floor, item["distance_m"])
        ))
        winner["selection_reason"] = "coverage_complete_frontier"
        winner["ranking_score"] = round(
            winner.get("information_gain_m2", 0.0)
            / max(distance_floor, winner["distance_m"]), 4
        )
        return winner

    def _exploration_samples(self, grid, reachable, unseen, seen, pose):
        """Shortlist seen reachable cells near dense unseen areas."""
        radius = max(
            1, round(self.cfg.camera_fov_max_range_m / grid.resolution)
        )
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disk = (x * x + y * y <= radius * radius).astype(np.float32)
        density = cv2.filter2D(
            unseen.astype(np.float32), -1, disk,
            borderType=cv2.BORDER_CONSTANT,
        )

        stride = max(1, round(self.cfg.exploration_step / grid.resolution))
        sampled = np.zeros_like(reachable)
        sampled[::stride, ::stride] = True
        robot_col, robot_row = self._robot_cell(grid, pose)
        if grid.inside(robot_col, robot_row):
            sampled[robot_row, robot_col] = True

        eligible = reachable & seen
        if grid.inside(robot_col, robot_row):
            eligible[robot_row, robot_col] = reachable[robot_row, robot_col]

        rows, cols = np.nonzero(eligible & sampled)
        if not len(rows):
            return rows, cols
        distance = (rows - robot_row) ** 2 + (cols - robot_col) ** 2
        order = np.lexsort((distance, -density[rows, cols]))
        order = order[:self.cfg.exploration_shortlist]
        return rows[order], cols[order]

    def _best_unseen_view(self, grid, pose, unseen, known_free, overlay):
        """Choose the heading whose camera cone contains most unseen cells."""
        fov = float(overlay.get(
            "camera_horizontal_fov_deg", self.cfg.camera_horizontal_fov_deg
        ))
        max_range = float(overlay.get(
            "camera_reliable_range_m", self.cfg.camera_fov_max_range_m
        ))
        best = (float(pose["yaw"]), 0, 0.0)
        headings = np.linspace(
            0.0, 2 * math.pi, self.cfg.exploration_headings,
            endpoint=False,
        )
        for yaw in headings:
            visible = self.visibility_mask(
                grid, pose, float(yaw), fov, max_range
            )
            eligible = visible & known_free
            gain = int(np.count_nonzero(visible & unseen))
            fraction = gain / max(1, int(np.count_nonzero(eligible)))
            if gain > best[1]:
                best = float(yaw), gain, fraction
        return best

    def _costs_on_map(self, grid, costmap):
        rows, cols = np.indices(grid.data.shape)
        return costmap.sample(*grid.world(cols, rows))
    def _robot_cell(self, grid, pose):
        col, row = grid.cells(pose["x"], pose["y"])
        return int(col), int(row)
    @staticmethod
    def _connected(mask, robot_col, robot_row):
        if not (
            0 <= robot_row < mask.shape[0]
            and 0 <= robot_col < mask.shape[1]
        ):
            return np.zeros_like(mask, bool)
        if not mask[robot_row, robot_col]:
            return np.zeros_like(mask, bool)
        _, labels = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=4
        )
        return labels == labels[robot_row, robot_col]

    def _reachable(self, grid, costs, pose, clearance=0.0):
        free = (
            (grid.data >= 0)
            & (grid.data < self.cfg.occupied)
            & (costs >= 0)
            & (costs < self.cfg.cost_limit)
        )
        if clearance > 0:
            dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
            free &= dist * grid.resolution >= clearance
        return self._connected(free, *self._robot_cell(grid, pose))

    # ------------------------------------------------------------------
    # Local samples
    # ------------------------------------------------------------------
    def _local_candidates(self, grid, reachable, pose):
        out = []
        angle_step = math.radians(self.cfg.sample_angle_deg)
        angles = int(round(2 * math.pi / angle_step))
        distances = [
            float(distance) for distance in self.cfg.sample_distances
            if 0.0 < float(distance) <= self.cfg.sample_radius
        ]

        # Preserve meaningful near, medium and far choices on every direction.
        # Nav2 owns path planning; this stage only requires a reachable endpoint.
        for ring_index, radius in enumerate(distances):
            for angle_index in range(angles):
                yaw = pose["yaw"] + angle_index * angle_step
                x = pose["x"] + radius * math.cos(yaw)
                y = pose["y"] + radius * math.sin(yaw)
                col, row = grid.cells(x, y)
                if not (grid.inside(col, row) and reachable[row, col]):
                    continue
                out.append({
                    "id": str(ring_index * angles + angle_index + 1),
                    "kind": "local",
                    "x": float(x), "y": float(y), "yaw": float(yaw),
                })


        if self.cfg.include_rotations and reachable.any():
            for relative in (math.pi / 2.0, -math.pi / 2.0, math.pi):
                degrees = math.degrees(math.atan2(math.sin(relative), math.cos(relative)))
                if math.isclose(abs(degrees), 180.0):
                    degrees = 180.0
                yaw = pose["yaw"] + relative
                out.append({
                    "id": f"R{degrees:+g}",
                    "kind": "rotation",
                    "x": pose["x"], "y": pose["y"], "yaw": float(yaw),
                })
        return out

    # ------------------------------------------------------------------
    # Frontiers
    # ------------------------------------------------------------------
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
    def _frontier_samples(self, rows, cols, max_samples=5):
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

            samples = self._frontier_samples(rows, cols, 5)

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

    # ------------------------------------------------------------------
    # Camera coverage on the occupancy grid
    # ------------------------------------------------------------------
    def visibility_mask(self, grid, pose, center_yaw, fov_deg, max_range):
        """Return grid cells visible inside an obstacle-clipped camera cone."""
        visible = np.zeros_like(grid.data, dtype=bool)
        ray_count = max(
            31,
            math.ceil(math.radians(fov_deg) * max_range / grid.resolution),
        )
        distances = np.arange(
            grid.resolution / 2.0,
            max_range + grid.resolution / 2.0,
            grid.resolution / 2.0,
        )
        half_fov = math.radians(fov_deg / 2.0)
        for angle in np.linspace(
            center_yaw - half_fov, center_yaw + half_fov, ray_count
        ):
            cols, rows = grid.cells(
                pose["x"] + distances * math.cos(angle),
                pose["y"] + distances * math.sin(angle),
            )
            inside = grid.inside(cols, rows)
            if not np.all(inside):
                cols, rows = cols[:int(np.argmax(~inside))], rows[:int(np.argmax(~inside))]
            if not len(rows):
                continue
            hit = np.flatnonzero(grid.data[rows, cols] >= self.cfg.occupied)
            if len(hit):
                cols, rows = cols[:int(hit[0])], rows[:int(hit[0])]
            visible[rows, cols] = True
        return visible

    def coverage_counts(self, grid, overlay):
        """Count how many stored camera observations covered each map cell."""
        counts = np.zeros_like(grid.data, dtype=np.uint16)
        if not isinstance(overlay, dict):
            return counts
        fov = float(overlay.get(
            "camera_horizontal_fov_deg", self.cfg.camera_horizontal_fov_deg
        ))
        max_range = float(overlay.get(
            "camera_reliable_range_m", self.cfg.camera_fov_max_range_m
        ))
        frame_id = str(overlay.get("frame_id") or "")
        for pose in overlay.get("search_poses") or []:
            if frame_id and pose.get("frame_id") != frame_id:
                continue
            for view in pose.get("views") or []:
                center = pose["yaw"] + math.radians(
                    view["pan"] - self.cfg.camera_center_pan_deg
                )
                counts += self.visibility_mask(
                    grid, pose, center, fov, max_range
                ).astype(np.uint16)
        return counts

        # ------------------------------------------------------------------
    # Door detection
    # ------------------------------------------------------------------
    @staticmethod
    def _at(array, row, col, default=0.0):
        r, c = int(round(row)), int(round(col))
        if 0 <= r < array.shape[0] and 0 <= c < array.shape[1]:
            return float(array[r, c])
        return default

    def _line_free(self, free, row, col, angle, distance):
        for d in np.arange(0, distance + 0.5, 0.5):
            r = row + math.sin(angle) * d
            c = col + math.cos(angle) * d
            if not self._at(free, r, c, 0):
                return False
        return True
    def _obstacle_components(self, grid):
        occupied = grid.data >= self.cfg.occupied

        count, labels = cv2.connectedComponents(
            occupied.astype(np.uint8),
            connectivity=8,
        )

        lengths = np.zeros(count, dtype=np.float32)

        for label in range(1, count):
            rows, cols = np.nonzero(labels == label)
            if len(rows) == 0:
                continue

            if len(rows) == 1:
                lengths[label] = grid.resolution
                continue

            points = np.column_stack((cols, rows)).astype(np.float32)
            centered = points - points.mean(axis=0)

            _, _, axes = np.linalg.svd(centered, full_matrices=False)
            axis = axes[0]
            projection = centered @ axis

            length_cells = float(projection.max() - projection.min()) + 1.0
            lengths[label] = length_cells * grid.resolution

        return labels, lengths
    def _wall(self, grid, obstacle_labels, obstacle_lengths,
            row, col, angle, max_distance_m):
        max_cells = max_distance_m / grid.resolution
        previous = None

        for distance in np.arange(0.5, max_cells + 0.5, 0.5):
            c = int(round(col + math.cos(angle) * distance))
            r = int(round(row + math.sin(angle) * distance))

            if not grid.inside(c, r) or grid.data[r, c] < 0:
                return None

            if (r, c) == previous:
                continue
            previous = (r, c)

            if grid.data[r, c] < self.cfg.occupied:
                continue

            label = int(obstacle_labels[r, c])
            wall_length = float(obstacle_lengths[label])

            if wall_length < self.cfg.door_wall_min_length:
                continue

            return distance * grid.resolution, c, r, wall_length, label

        return None
    def _door_at(self, grid, free, clearance,
                obstacle_labels, obstacle_lengths, row, col):
        probe = self.cfg.door_probe / grid.resolution
        wall_range = self.cfg.door_max_width / 2 + 0.35
        center_clearance = float(clearance[row, col])

        best = None

        for passage in np.linspace(0, math.pi, self.cfg.door_angles, endpoint=False):
            # Passage must be open in both directions.
            if not self._line_free(free, row, col, passage, probe):
                continue
            if not self._line_free(free, row, col, passage + math.pi, probe):
                continue

            # Find walls perpendicular to passage.
            cross = passage + math.pi / 2

            a = self._wall(grid, obstacle_labels, obstacle_lengths,row, col, cross, wall_range,)
            b = self._wall(grid, obstacle_labels, obstacle_lengths,row, col, cross + math.pi, wall_range)

            if not a or not b:
                continue

            # Opening must be door-sized.
            width = a[0] + b[0] - grid.resolution

            if not self.cfg.door_min_width <= width <= self.cfg.door_max_width:
                continue

            # Space should widen beyond the bottleneck.
            ar = row + math.sin(passage) * probe
            ac = col + math.cos(passage) * probe
            br = row - math.sin(passage) * probe
            bc = col - math.cos(passage) * probe

            side_a = self._at(clearance, ar, ac)
            side_b = self._at(clearance, br, bc)

            widening_a = float(side_a - center_clearance)
            widening_b = float(side_b - center_clearance)
            widening = max(widening_a, widening_b)

            if widening < self.cfg.door_min_widening:
                continue

            # Only used to choose the best valid orientation.
            balance = 1.0 - abs(a[0] - b[0]) / max(
                a[0] + b[0],
                grid.resolution,
            )

            candidate = {
                "width": float(width),
                "passage": float(passage),
                "wall_a": [a[1], a[2]],
                "wall_b": [b[1], b[2]],
                "wall_a_length_m": float(a[3]),
                "wall_b_length_m": float(b[3]),
                "widening_a_m": widening_a,
                "widening_b_m": widening_b,
                "widening_m": widening,
                "balance": float(balance),
            }

            if best is None or (
                candidate["balance"],
                candidate["widening_m"],
            ) > (
                best["balance"],
                best["widening_m"],
            ):
                best = candidate

        return best
    def _doors(self, grid, costs, pose):
        map_free = (grid.data >= 0) & (grid.data < self.cfg.occupied)
        connected = self._reachable(grid, costs, pose)

        clearance = cv2.distanceTransform(
            map_free.astype(np.uint8), cv2.DIST_L2, 5
        )
        clearance *= grid.resolution

        ridge = clearance >= cv2.dilate(
            clearance, np.ones((3, 3), np.float32)
        ) - 1e-6

        candidates = (
            connected
            & ridge
            & (clearance >= 0.5 * self.cfg.door_min_width)
            & (clearance <= 0.5 * self.cfg.door_max_width)
        )

        obstacle_labels, obstacle_lengths = self._obstacle_components(grid)

        _, labels = cv2.connectedComponents(
            candidates.astype(np.uint8), connectivity=8
        )

        hypotheses = []

        for label in range(1, labels.max() + 1):
            rows, cols = np.nonzero(labels == label)
            if not len(rows):
                continue

            samples = np.linspace(
                0, len(rows) - 1, min(6, len(rows)), dtype=int
            )

            for i in np.unique(samples):
                row, col = int(rows[i]), int(cols[i])

                door = self._door_at(
                    grid, map_free, clearance,
                    obstacle_labels, obstacle_lengths,
                    row, col,
                )

                if not door:
                    continue

                x, y = grid.world(col, row)
                wa_x, wa_y = grid.world(*door["wall_a"])
                wb_x, wb_y = grid.world(*door["wall_b"])
                passage_yaw = grid.origin_yaw + door["passage"]

                hypotheses.append({
                    "x": float(x),
                    "y": float(y),
                    "cell": [col, row],
                    "width": round(door["width"], 3),
                    "confirmed": True,
                    "passage_yaw": float(passage_yaw),
                    "wall_a": door["wall_a"],
                    "wall_b": door["wall_b"],
                    "wall_a_xy": [float(wa_x), float(wa_y)],
                    "wall_b_xy": [float(wb_x), float(wb_y)],
                    "wall_a_length_m": round(door["wall_a_length_m"], 3),
                    "wall_b_length_m": round(door["wall_b_length_m"], 3),
                    "widening_m": round(door["widening_m"], 3),
                    "balance": round(door["balance"], 3),
                })

        hypotheses.sort(
            key=lambda d: (
                min(d["wall_a_length_m"], d["wall_b_length_m"]),
                d["widening_m"],
                d["balance"],
            ),
            reverse=True,
        )

        doors = []

        for door in hypotheses:
            if any(
                math.hypot(
                    door["x"] - other["x"],
                    door["y"] - other["y"],
                ) < self.cfg.door_min_separation
                for other in doors
            ):
                continue

            doors.append(door)

        return sorted(doors, key=lambda d: (d["y"], d["x"])), candidates

    # ------------------------------------------------------------------
    # Room segmentation
    # ------------------------------------------------------------------
    def _rooms(self, grid, pose, doors):
        map_free = (grid.data >= 0) & (grid.data < self.cfg.occupied)
        free = self._connected(map_free, *self._robot_cell(grid, pose)).astype(np.uint8)

        for door in doors:
            if door["confirmed"]:
                cv2.line(
                    free,
                    tuple(door["wall_a"]),
                    tuple(door["wall_b"]),
                    0,
                    max(1, round(0.12 / grid.resolution)),
                )

        count, labels, stats, centers = cv2.connectedComponentsWithStats(free, connectivity=4)
        robot_col, robot_row = self._robot_cell(grid, pose)
        robot_label = labels[robot_row, robot_col] if grid.inside(robot_col, robot_row) else 0
        min_cells = self.cfg.room_min_area / grid.resolution ** 2

        rooms = []
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_cells:
                continue
            x, y = grid.world(*centers[label])
            rooms.append({
                "id": f"R{len(rooms) + 1}",
                "x": float(x), "y": float(y),
                "area": round(float(area * grid.resolution ** 2), 2),
                "current": bool(label == robot_label),
            })

        current_mask = labels == robot_label if robot_label else np.zeros_like(labels, bool)
        return rooms, current_mask
