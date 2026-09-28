"""Map analysis, candidate generation, coverage sampling, and search ranking."""

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class LogicConfig:
    # Occupancy / costmap
    occupied: int = 50
    cost_limit: int = 99
    clearance: float = 0.10

    # ToF approach ray
    approach_ray_radius_m: float = 0.15
    approach_obstacle_min_cells: int = 3
    approach_obstacle_standoff_m: float = 0.25
    approach_cost_threshold: int = 253

    # Local navigation
    sample_radius: float = 3.0
    nudge_distance_m: float = 0.25
    candidate_min_separation_m: float = 0.30

    # Frontiers
    frontier_hole_min_area: float = 0.5
    frontier_min_length: float = 0.30
    frontier_gain_radius: float = 1.0
    frontier_min_gain: float = 0.25
    frontier_min_separation: float = 0.75
    frontier_rays: int = 120
    frontier_samples_per_component: int = 5

    # Geometric exit-room navigation
    exit_open_clearance_m: float = 0.65
    exit_component_min_area_m2: float = 1.0
    exit_frontier_revisit_m: float = 0.60
    exit_component_lookup_radius_m: float = 1.50
    exit_wall_min_length_m: float = 1.00
    exit_recovery_max_moves: int = 3
    exit_recovery_min_move_m: float = 0.35
    exit_recovery_max_move_m: float = 1.00
    exit_recovery_revisit_m: float = 0.40
    exit_frontier_travel_penalty_m2_per_m: float = 0.15

    # Search views
    camera_center_pan_deg: float = 95.0
    camera_horizontal_fov_deg: float = 85.0
    camera_fov_max_range_m: float = 2.0
    context_radius_samples: int = 3
    context_bearing_fractions: tuple = (-0.75, 0.0, 0.75)
    exploration_step: float = 0.5
    exploration_headings: int = 8
    exploration_shortlist: int = 16
    exploration_ray_m: float = 4.0
    visibility_min_rays: int = 31


def yaw_of(q):
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z),
    )


# =============================================================================
# Occupancy-grid coordinate conversion
# =============================================================================
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


# =============================================================================
# Map analysis and navigation decisions
# =============================================================================
class MapLogic:
    def __init__(self, config=None):
        self.cfg = config if config is not None else LogicConfig()

    # -------------------------------------------------------------------------
    # Prepared map state
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Target approach navigation
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Camera view geometry
    # -------------------------------------------------------------------------
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

    def _camera_view_parameters(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        return (
            float(payload.get(
                "camera_horizontal_fov_deg",
                self.cfg.camera_horizontal_fov_deg,
            )),
            float(payload.get(
                "camera_reliable_range_m",
                self.cfg.camera_fov_max_range_m,
            )),
        )

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
            horizontal_fov, max_range = self._camera_view_parameters(
                search_overlay
            )
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

    # -------------------------------------------------------------------------
    # Candidate planning entry point
    # -------------------------------------------------------------------------
    def plan_candidates(
        self, prepared, pose, search_overlay=None, *, coverage=None
    ):
        # Map-pose candidates belong only to the find loop.
        overlay = search_overlay if isinstance(search_overlay, dict) else {}
        mode = str(overlay.get("mode") or "exploration")

        if mode in {"context", "visual"}:
            candidates = self._context_candidates(prepared, pose, overlay)
        elif mode == "exploration":
            if coverage is None:
                coverage = self.coverage_counts(prepared["grid"], overlay) > 0
            candidates = self._exploration_candidates(
                prepared, pose, overlay, coverage
            )
        else:
            return []
        return self._deduplicate_candidates(candidates)

    # -------------------------------------------------------------------------
    # Deterministic local navigation
    # -------------------------------------------------------------------------
    def resolve_local_destination(
        self, prepared, pose, command, *, previous_pose=None, person_pose=None
    ):
        """Resolve the fixed local-navigation vocabulary from map data."""
        grid, reachable = prepared["grid"], prepared["reachable"]
        yaw = float(pose["yaw"])
        rotations = {
            "rotate_left": math.pi / 2.0,
            "rotate_right": -math.pi / 2.0,
            "rotate_around": math.pi,
        }
        if command in rotations:
            return self._local_goal(pose["x"], pose["y"], yaw + rotations[command], pose)
        if command == "face_person":
            if not isinstance(person_pose, dict):
                raise ValueError("fresh person pose is unavailable")
            person_yaw = math.atan2(
                float(person_pose["y"]) - pose["y"],
                float(person_pose["x"]) - pose["x"],
            )
            return self._local_goal(pose["x"], pose["y"], person_yaw, pose)
        if command == "previous_position":
            if not isinstance(previous_pose, dict):
                raise ValueError("previous position is unavailable")
            return self._local_goal(
                previous_pose["x"], previous_pose["y"],
                previous_pose.get("yaw", yaw), pose,
            )

        directions = {
            "nudge_forward": (0.0, self.cfg.nudge_distance_m),
            "nudge_backward": (math.pi, self.cfg.nudge_distance_m),
            "nudge_left": (math.pi / 2.0, self.cfg.nudge_distance_m),
            "nudge_right": (-math.pi / 2.0, self.cfg.nudge_distance_m),
            "furthest_forward": (0.0, None),
            "furthest_backward": (math.pi, None),
        }
        if command in directions:
            offset, distance = directions[command]
            endpoint = self._furthest_reachable_on_ray(
                grid, reachable, pose, yaw + offset, max_distance=distance
            )
            if endpoint is None:
                raise ValueError("no reachable point in the requested direction")
            x, y, _ = endpoint
            if math.hypot(x - pose["x"], y - pose["y"]) < grid.resolution:
                raise ValueError("requested direction is blocked")
            return self._local_goal(x, y, yaw, pose)

        rows, cols = np.nonzero(reachable)
        if not len(rows):
            raise ValueError("no reachable map space")
        xs, ys = grid.world(cols, rows)
        local = np.hypot(xs - pose["x"], ys - pose["y"]) <= self.cfg.sample_radius
        if np.any(local):
            rows, cols, xs, ys = rows[local], cols[local], xs[local], ys[local]
        if command == "open_space_middle":
            obstacle = (grid.data < 0) | (grid.data >= self.cfg.occupied)
            clearance = cv2.distanceTransform(
                (~obstacle).astype(np.uint8), cv2.DIST_L2, 5
            )
            index = int(np.argmax(clearance[rows, cols]))
        else:
            raise ValueError(f"unknown local command: {command}")
        x, y = float(xs[index]), float(ys[index])
        goal_yaw = math.atan2(y - pose["y"], x - pose["x"])
        return self._local_goal(x, y, goal_yaw, pose)

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

    # -------------------------------------------------------------------------
    # Shared goal, candidate, and ray helpers
    # -------------------------------------------------------------------------
    def _local_goal(self, x, y, yaw, pose):
        return {
            "x": float(x), "y": float(y), "angle": self._angle(float(yaw)),
            "frame_id": str(pose.get("frame_id") or "map"),
        }

    def _deduplicate_candidates(self, candidates):
        """Keep one useful marker per 30 cm cluster; H remains independent."""
        separation = max(0.0, float(self.cfg.candidate_min_separation_m))
        if separation == 0.0:
            return candidates

        def priority(item):
            if item.get("kind") == "frontier":
                return 4
            if (
                item.get("id") in {"CF", "EF"}
                or "furthest" in str(item.get("selection_reason") or "")
            ):
                return 3
            if item.get("uncovered_cell_count") is not None:
                return 2
            return 1

        indexed = list(enumerate(candidates))
        translations = [
            pair for pair in indexed if pair[1].get("kind") != "rotation"
        ]
        translations.sort(key=lambda pair: (-priority(pair[1]), pair[0]))
        kept = []
        for index, candidate in translations:
            if any(
                math.hypot(
                    candidate["x"] - other["x"],
                    candidate["y"] - other["y"],
                ) < separation
                for _, other in kept
            ):
                continue
            kept.append((index, candidate))

        kept.extend(
            pair for pair in indexed if pair[1].get("kind") == "rotation"
        )
        return [item for _, item in sorted(kept)]

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

    # -------------------------------------------------------------------------
    # Context-clue candidates
    # -------------------------------------------------------------------------
    def _context_candidates(
        self, prepared, pose, overlay, *, max_range_m=None
    ):
        """Offer close clue views and the furthest valid center-ray point."""
        observation = self.observation_fov(overlay)
        if observation is None:
            return []

        grid, reachable = prepared["grid"], prepared["reachable"]
        observation_payload = overlay.get("observation") or {}
        candidate_attributes = {
            key: observation_payload.get(key)
            for key in (
                "candidate_type", "movement_limit", "movements_used",
                "remaining_waypoints", "reassessment_limit",
                "reassessments_used",
            )
            if observation_payload.get(key) is not None
        }
        origin = observation["pose"]
        center = observation["center_yaw_rad"]
        half_fov = math.radians(observation["horizontal_fov_deg"] / 2.0)
        max_range = (
            float(max_range_m)
            if max_range_m is not None
            else observation["max_range_m"]
        )
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
                candidate.update(candidate_attributes)
                candidates.append(candidate)
                used_cells.add(cell)

        candidates.sort(key=lambda item: (
            abs(self._angle(item["yaw"] - center)), item["distance_m"]
        ))

        # The reliable camera range limits close clue-view samples, but it
        # should not prevent a deliberate long move in the clue direction.
        # Replace any ordinary sample in the same cell with a clearly labeled
        # endpoint at the last reachable cell before the center ray is blocked.
        ray_limit = (
            float(max_range_m) if max_range_m is not None else None
        )
        farthest = self._furthest_reachable_on_ray(
            grid, reachable, origin, center, max_distance=ray_limit
        )
        if farthest is not None:
            x, y, cell = farthest

            if math.hypot(x - origin["x"], y - origin["y"]) >= grid.resolution:
                far_candidate = self._describe({
                    "id": "CF",
                    "kind": "context",
                    "x": x,
                    "y": y,
                    "yaw": float(center),
                    "selection_reason": "furthest_reachable_on_center_ray",
                }, pose)
                far_candidate.update(candidate_attributes)
                candidates.append(far_candidate)

        return candidates

    # -------------------------------------------------------------------------
    # Exploration candidates
    # -------------------------------------------------------------------------
    def _exploration_candidates(self, prepared, pose, overlay, coverage):
        """Pick the best unseen view and optionally offer a farther move."""
        grid, reachable = prepared["grid"], prepared["reachable"]
        known_free = self._known_free(grid)
        unseen = known_free & ~coverage
        rows, cols = self._exploration_samples(
            grid, reachable, unseen, coverage, pose
        )
        scored_candidates = []

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
            scored_candidates.append(candidate)

        if not scored_candidates:
            return []
        winner = max(scored_candidates, key=lambda item: (
            item["uncovered_cell_count"], -item["distance_m"]
        ))
        candidates = [winner]
        travel_dx = winner["x"] - pose["x"]
        travel_dy = winner["y"] - pose["y"]
        travel_distance = math.hypot(travel_dx, travel_dy)
        if travel_distance < prepared["grid"].resolution:
            return candidates

        travel_yaw = math.atan2(travel_dy, travel_dx)
        center_fov = float(overlay.get(
            "camera_horizontal_fov_deg", self.cfg.camera_horizontal_fov_deg
        ))
        if abs(self._angle(travel_yaw - pose["yaw"])) > math.radians(
            center_fov / 2.0
        ):
            return candidates

        farthest = self._furthest_reachable_on_ray(
            prepared["grid"], prepared["reachable"], winner, travel_yaw,
            max_distance=self.cfg.exploration_ray_m,
        )
        if farthest is None:
            return candidates
        x, y, _ = farthest
        extension = math.hypot(x - winner["x"], y - winner["y"])
        if extension < prepared["grid"].resolution:
            return candidates

        far = {
            key: value for key, value in winner.items()
            if key not in {
                "uncovered_cell_count", "uncovered_ahead_fraction",
                "ranking_score",
            }
        }
        far.update({
            "id": "EF",
            "x": float(x),
            "y": float(y),
            "yaw": float(travel_yaw),
            "selection_reason": "furthest_reachable_forward_ray",
            "ray_extension_m": round(float(extension), 3),
        })
        candidates.append(self._describe(far, pose))
        return candidates

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
        fov, max_range = self._camera_view_parameters(overlay)
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

    # -------------------------------------------------------------------------
    # Costmap projection and reachability
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Frontiers
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Camera coverage on the occupancy grid
    # -------------------------------------------------------------------------
    def visibility_mask(self, grid, pose, center_yaw, fov_deg, max_range):
        """Return grid cells visible inside an obstacle-clipped camera cone."""
        visible = np.zeros_like(grid.data, dtype=bool)
        ray_count = max(
            self.cfg.visibility_min_rays,
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
        fov, max_range = self._camera_view_parameters(overlay)
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
