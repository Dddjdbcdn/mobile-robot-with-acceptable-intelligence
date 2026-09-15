"""Compact map logic for local samples, frontiers, doors and room segmentation."""

from collections import deque
from dataclasses import dataclass
import json
import math
import struct
import threading
import time
import uuid

import cv2
import numpy as np
import zmq


@dataclass
class Config:
    # Occupancy / costmap
    occupied: int = 50
    cost_limit: int = 99
    clearance: float = 0.10

    # Local navigation samples
    sample_radius: float = 2.0
    sample_step: float = 0.5
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
    door_min_width: float = 0.5
    door_max_width: float = 1.5
    door_probe: float = 0.5
    door_min_widening: float = 0.1
    door_wall_min_length: float = 0.3
    door_angles: int = 12
    door_min_separation: float = 0.5
    room_min_area: float = 1.0

    # Rendering
    image_max_size: int = 1024
    view_margin: float = 0.45
    grid_spacing: float = 1.0


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
        self.cfg = config or Config()

    def process(self, map_msg, cost_msg, pose):
        """Compatibility path that prepares and composes one complete frame."""
        return self.compose(self.prepare(map_msg, cost_msg, pose), pose)

    def prepare(self, map_msg, cost_msg, pose):
        """Build the expensive semantic analysis and pose-free background."""
        grid = Grid(map_msg)
        costmap = Grid(cost_msg)
        costs = self._costs_on_map(grid, costmap)

        reachable = self._reachable(grid, costs, pose, self.cfg.clearance)
        frontiers, frontier_mask = self._frontiers(grid, costs, pose)
        doors, door_candidate_mask = self._doors(grid, costs, pose)
        rooms, current_room_mask = self._rooms(grid, pose, doors)

        for i, frontier in enumerate(frontiers, 1):
            frontier["id"] = f"F{i}"
        for i, door in enumerate(doors, 1):
            door["id"] = f"D{i}"

        background, view = self._render(
            grid, costmap, pose, [], frontiers, frontier_mask,
            doors, door_candidate_mask, rooms, current_room_mask,
            draw_robot=False,
        )
        return {
            "background": background,
            "frame_id": pose["frame_id"],
            "grid": grid,
            "reachable": reachable,
            "frontiers": frontiers,
            "doors": doors,
            "rooms": rooms,
            "view": view,
        }

    def compose(self, prepared, pose):
        """Draw only pose-dependent overlays onto a prepared background."""
        locals_ = self._local_candidates(
            prepared["grid"], prepared["reachable"], pose
        )
        frontiers = [dict(frontier) for frontier in prepared["frontiers"]]
        candidates = locals_ + frontiers
        for candidate in candidates:
            candidate["frame_id"] = pose["frame_id"]
            candidate["distance_m"] = float(math.hypot(
                candidate["x"] - pose["x"], candidate["y"] - pose["y"]
            ))

        image = prepared["background"].copy()
        self._draw_dynamic(image, prepared["view"], pose, locals_)
        metadata = {
            "schema_version": 1,
            "snapshot_id": uuid.uuid4().hex,
            "frame_id": pose["frame_id"],
            "robot_pose": pose,
            "candidates": candidates,
            "frontiers": frontiers,
            "doors": prepared["doors"],
            "rooms": prepared["rooms"],
            "sample_radius_m": self.cfg.sample_radius,
            "image_world_bounds": {
                key: prepared["view"][key]
                for key in ("xmin", "xmax", "ymin", "ymax")
            },
        }
        return image, metadata

    # ------------------------------------------------------------------
    # Shared geometry
    # ------------------------------------------------------------------

    def _costs_on_map(self, grid, costmap):
        rows, cols = np.indices(grid.data.shape)
        return costmap.sample(*grid.world(cols, rows))

    def _robot_cell(self, grid, pose):
        col, row = grid.cells(pose["x"], pose["y"])
        return int(col), int(row)

    @staticmethod
    def _connected(mask, robot_col, robot_row):
        if not (0 <= robot_row < mask.shape[0] and 0 <= robot_col < mask.shape[1]):
            return np.zeros_like(mask, bool)
        if not mask[robot_row, robot_col]:
            return np.zeros_like(mask, bool)
        _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
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
        rings = int(self.cfg.sample_radius / self.cfg.sample_step)
        angles = int(round(2 * math.pi / angle_step))

        for ring in range(1, rings + 1):
            radius = ring * self.cfg.sample_step
            for i in range(angles):
                yaw = pose["yaw"] + i * angle_step
                x = pose["x"] + radius * math.cos(yaw)
                y = pose["y"] + radius * math.sin(yaw)
                col, row = grid.cells(x, y)
                if grid.inside(col, row) and reachable[row, col]:
                    out.append({
                        "id": str((ring - 1) * angles + i + 1),
                        "kind": "local",
                        "x": float(x), "y": float(y), "yaw": float(yaw),
                    })

        if self.cfg.include_rotations and reachable.any():
            for i in range(1, angles):
                relative = i * angle_step
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

        # Build each ray in NumPy. The former inner Python loop called
        # round()/inside() tens of thousands of times per frame.
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

    # ------------------------------------------------------------------
    # LLM-friendly rendering
    # ------------------------------------------------------------------

    def _make_view(self, grid, pose):
        """Crop tightly around explored space while keeping local samples visible."""
        rows, cols = np.nonzero(grid.data >= 0)
        if len(rows):
            xs, ys = grid.world(cols, rows)
        else:
            xs = np.array([pose["x"]])
            ys = np.array([pose["y"]])

        radius = self.cfg.sample_radius
        xs = np.append(xs, [pose["x"] - radius, pose["x"] + radius])
        ys = np.append(ys, [pose["y"] - radius, pose["y"] + radius])

        margin = self.cfg.view_margin + grid.resolution
        xmin, xmax = float(xs.min()) - margin, float(xs.max()) + margin
        ymin, ymax = float(ys.min()) - margin, float(ys.max()) + margin

        span_x = max(xmax - xmin, grid.resolution)
        span_y = max(ymax - ymin, grid.resolution)
        ppm = self.cfg.image_max_size / max(span_x, span_y)

        return {
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "ppm": ppm,
            "width": max(1, round(span_x * ppm)),
            "height": max(1, round(span_y * ppm)),
        }

    @staticmethod
    def _pixel(view, x, y):
        return (
            round((x - view["xmin"]) * view["ppm"]),
            round((view["ymax"] - y) * view["ppm"]),
        )

    @staticmethod
    def _sample_mask(grid, mask, x, y):
        cols, rows = grid.cells(x, y)
        h, w = mask.shape
        inside = grid.inside(cols, rows)
        values = mask[np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)]
        return inside & values

    @staticmethod
    def _heading_tick(image, point, yaw, radius, color):
        """Give a waypoint an explicit orientation without adding another label."""
        direction = (math.cos(yaw), -math.sin(yaw))
        start = (
            round(point[0] + direction[0] * (radius - 1)),
            round(point[1] + direction[1] * (radius - 1)),
        )
        end = (
            round(point[0] + direction[0] * (radius + 10)),
            round(point[1] + direction[1] * (radius + 10)),
        )
        cv2.line(image, start, end, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.line(image, start, end, color, 3, cv2.LINE_AA)

    @staticmethod
    def _centered_text(image, point, text, scale, color, thickness):
        """Center the visible glyph pixels, excluding OpenCV baseline padding."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        size, baseline = cv2.getTextSize(text, font, scale, thickness)
        padding = thickness + 3
        anchor = (padding, padding + size[1])
        mask = np.zeros((
            size[1] + baseline + 2 * padding,
            size[0] + 2 * padding,
        ), np.uint8)
        cv2.putText(
            mask, text, anchor, font, scale, 255, thickness, cv2.LINE_AA
        )
        rows, cols = np.nonzero(mask)
        if len(rows):
            ink_center_x = (float(cols.min()) + float(cols.max())) / 2
            ink_center_y = (float(rows.min()) + float(rows.max())) / 2
            origin = (
                round(point[0] + anchor[0] - ink_center_x),
                round(point[1] + anchor[1] - ink_center_y),
            )
        else:
            origin = (
                point[0] - size[0] // 2,
                point[1] + size[1] // 2,
            )
        cv2.putText(
            image, text, origin, font, scale, color, thickness, cv2.LINE_AA
        )

    @staticmethod
    def _circle_badge(image, point, text, color, radius=12):
        """Draw an OCR-friendly local-pose badge with the ID inside it."""
        cv2.circle(image, point, radius + 3, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.circle(image, point, radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)
        scale = 0.43 if len(text) <= 2 else 0.34
        thickness = 1
        MapLogic._centered_text(
            image, point, text, scale, (255, 255, 255), thickness
        )

    @staticmethod
    def _diamond_badge(image, point, text, color, radius=19):
        """Draw a large, shape-coded frontier badge with its ID centered."""
        def diamond(size):
            return np.array([
                (point[0], point[1] - size), (point[0] + size, point[1]),
                (point[0], point[1] + size), (point[0] - size, point[1]),
            ], np.int32)

        cv2.fillConvexPoly(image, diamond(radius + 3), (35, 35, 35), cv2.LINE_AA)
        cv2.fillConvexPoly(image, diamond(radius + 1), (255, 255, 255), cv2.LINE_AA)
        cv2.fillConvexPoly(image, diamond(radius), color, cv2.LINE_AA)
        scale = 0.52 if len(text) <= 2 else 0.44
        thickness = 1
        MapLogic._centered_text(
            image, point, text, scale, (255, 255, 255), thickness
        )

    @staticmethod
    def _outlined_label(image, point, text, color, scale=0.46):
        """Draw a compact annotation label that remains legible over the map."""
        origin = (point[0] + 7, point[1] - 7)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)

    def _draw_dynamic(self, image, view, pose, locals_):
        """Draw the inexpensive overlays that change with every TF pose."""
        center = self._pixel(view, pose["x"], pose["y"])
        robot_color = (25, 25, 225)
        cv2.circle(image, center, 15, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.circle(image, center, 13, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, center, 10, robot_color, -1, cv2.LINE_AA)
        tip = self._pixel(
            view,
            pose["x"] + 0.30 * math.cos(pose["yaw"]),
            pose["y"] + 0.30 * math.sin(pose["yaw"]),
        )
        cv2.arrowedLine(
            image, center, tip, (255, 255, 255), 8,
            cv2.LINE_AA, tipLength=0.28,
        )
        cv2.arrowedLine(
            image, center, tip, robot_color, 5,
            cv2.LINE_AA, tipLength=0.28,
        )

        # Draw selectable poses last. In particular, the robot heading arrow
        # must not cover the closest forward candidate.
        for item in locals_:
            if item["kind"] == "rotation":
                continue
            point = self._pixel(view, item["x"], item["y"])
            color = (220, 105, 15)  # saturated blue in BGR
            self._heading_tick(image, point, item["yaw"], 12, color)
            self._circle_badge(image, point, item["id"], color)

    def _render(self, grid, costmap, pose, locals_, frontiers,
                frontier_mask, doors, door_candidate_mask, rooms, room_mask,
                draw_robot=True):
        """Render a clean metric map intended for VLM waypoint selection."""
        view = self._make_view(grid, pose)
        xs = view["xmin"] + (np.arange(view["width"]) + 0.5) / view["ppm"]
        ys = view["ymax"] - (np.arange(view["height"]) + 0.5) / view["ppm"]
        x, y = np.meshgrid(xs, ys)

        # Reuse the map indices for occupancy and semantic overlay masks. At 1280
        # pixels this avoids two additional 1.6-million-element transforms.
        map_cols, map_rows = grid.cells(x, y)
        map_inside = grid.inside(map_cols, map_rows)
        map_rows_clipped = np.clip(map_rows, 0, grid.data.shape[0] - 1)
        map_cols_clipped = np.clip(map_cols, 0, grid.data.shape[1] - 1)
        occupancy = np.where(
            map_inside,
            grid.data[map_rows_clipped, map_cols_clipped],
            -1,
        )
        costs = costmap.sample(x, y)
        free = (occupancy >= 0) & (occupancy < self.cfg.occupied)

        # Keep the base map nearly monochrome. Candidate colors then have one
        # unambiguous meaning instead of competing with the costmap and rooms.
        image = np.full((view["height"], view["width"], 3), (62, 62, 62), np.uint8)
        image[map_inside & (occupancy < 0)] = (112, 112, 112)
        image[free] = (246, 246, 246)
        image[occupancy >= self.cfg.occupied] = (24, 24, 24)
        image[free & (costs > 0)] = (224, 232, 244)
        image[free & (costs >= self.cfg.cost_limit)] = (185, 198, 232)
        image[free & (costs < 0)] = (190, 190, 190)

        # Current-room tint remains subtle so walls and costmap stay legible.
        room_pixels = map_inside & room_mask[map_rows_clipped, map_cols_clipped]
        if np.any(room_pixels):
            tint = np.full_like(image, (225, 238, 225))
            image[room_pixels] = cv2.addWeighted(
                image[room_pixels], 0.88, tint[room_pixels], 0.12, 0
            )

        # Show the exact ridge/clearance candidate mask used by _doors(). A
        # fixed-width halo keeps the usually one-cell-wide ridges visible after
        # the full map is down-scaled and JPEG-compressed.
        door_candidate_pixels = (
            map_inside
            & door_candidate_mask[map_rows_clipped, map_cols_clipped]
        )
        door_candidate_core = door_candidate_pixels.astype(np.uint8)
        door_candidate_halo = cv2.dilate(
            door_candidate_core, np.ones((7, 7), np.uint8)
        ) > 0
        door_candidate_halo_only = door_candidate_halo & ~door_candidate_pixels
        if np.any(door_candidate_halo_only):
            halo_tint = np.full_like(image, (225, 150, 225))
            image[door_candidate_halo_only] = cv2.addWeighted(
                image[door_candidate_halo_only], 0.70,
                halo_tint[door_candidate_halo_only], 0.30, 0,
            )
        if np.any(door_candidate_pixels):
            candidate_tint = np.full_like(image, (205, 55, 205))
            image[door_candidate_pixels] = cv2.addWeighted(
                image[door_candidate_pixels], 0.45,
                candidate_tint[door_candidate_pixels], 0.55, 0,
            )

        frontier_pixels = map_inside & frontier_mask[map_rows_clipped, map_cols_clipped]
        # A fixed-width halo survives down-scaling and JPEG compression even
        # when the actual frontier is only a one-cell-wide contour.
        frontier_core = frontier_pixels.astype(np.uint8)
        frontier_halo = cv2.dilate(frontier_core, np.ones((7, 7), np.uint8)) > 0
        frontier_halo_only = frontier_halo & ~frontier_pixels
        if np.any(frontier_halo_only):
            halo_tint = np.full_like(image, (185, 245, 185))
            image[frontier_halo_only] = cv2.addWeighted(
                image[frontier_halo_only], 0.85,
                halo_tint[frontier_halo_only], 0.15, 0,
            )
        if np.any(frontier_pixels):
            core_tint = np.full_like(image, (35, 220, 35))
            image[frontier_pixels] = cv2.addWeighted(
                image[frontier_pixels], 0.60,
                core_tint[frontier_pixels], 0.40, 0,
            )

        # Metric grid helps the VLM reason about distance and direction.
        spacing = self.cfg.grid_spacing
        x0 = math.ceil(view["xmin"] / spacing)
        x1 = math.floor(view["xmax"] / spacing)
        y0 = math.ceil(view["ymin"] / spacing)
        y1 = math.floor(view["ymax"] / spacing)
        for i in range(x0, x1 + 1):
            px = self._pixel(view, i * spacing, view["ymin"])[0]
            if 0 <= px < view["width"]:
                visible = free[:, px] & ~frontier_halo[:, px]
                image[visible, px] = (218, 218, 218)
        for i in range(y0, y1 + 1):
            py = self._pixel(view, view["xmin"], i * spacing)[1]
            if 0 <= py < view["height"]:
                visible = free[py] & ~frontier_halo[py]
                image[py, visible] = (218, 218, 218)

        # Local translation poses are blue numbered circles. Frontier goals are
        # deliberately larger green diamonds, so color, shape and ID all agree.
        for item in locals_ + frontiers:
            if item["kind"] == "rotation":
                continue
            p = self._pixel(view, item["x"], item["y"])
            if item["kind"] == "frontier":
                color = (35, 185, 35)
                # Clear noisy occupancy/frontier pixels immediately around the
                # badge. Draw this first so the heading tick remains visible.
                clearance = np.array([
                    (p[0], p[1] - 26), (p[0] + 26, p[1]),
                    (p[0], p[1] + 26), (p[0] - 26, p[1]),
                ], np.int32)
                cv2.fillConvexPoly(
                    image, clearance, (255, 255, 255), cv2.LINE_AA
                )
                self._heading_tick(image, p, item["yaw"], 19, color)
                self._diamond_badge(image, p, item["id"], color)
            else:
                color = (220, 105, 15)
                self._heading_tick(image, p, item["yaw"], 12, color)
                self._circle_badge(image, p, item["id"], color)

        # Accepted doors use saturated color and a strong white keyline so they
        # remain distinct from the faint raw candidate mask.
        for door in doors:
            a = self._pixel(view, *door["wall_a_xy"])
            b = self._pixel(view, *door["wall_b_xy"])
            center = self._pixel(view, door["x"], door["y"])
            color = (225, 30, 225) if door["confirmed"] else (0, 145, 255)
            cv2.line(image, a, b, (255, 255, 255), 7, cv2.LINE_AA)
            cv2.line(image, a, b, color, 4, cv2.LINE_AA)
            self._outlined_label(image, center, door["id"], color)

        if draw_robot:
            self._draw_dynamic(image, view, pose, locals_)

        return image, view


class MapImageStream:
    """Two-rate ROS wrapper: cached semantic map plus a fast pose overlay."""

    def __init__(self, node, tf_buffer, context):
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

        self.node = node
        self.tf = tf_buffer
        self.base_frame = node.declare_parameter(
            "map_image_base_frame", "base_footprint"
        ).value
        bind = node.declare_parameter(
            "map_image_bind", "tcp://127.0.0.1:5559"
        ).value
        image_max_size = int(
            node.declare_parameter("map_image_max_size", 1024).value
        )
        self.target_hz = float(
            node.declare_parameter("map_image_publish_hz", 5.0).value
        )
        self.analysis_hz = float(
            node.declare_parameter("map_image_analysis_hz", 1.0).value
        )
        if image_max_size <= 0:
            raise ValueError("map_image_max_size must be positive")
        if self.target_hz <= 0:
            raise ValueError("map_image_publish_hz must be positive")
        if self.analysis_hz <= 0:
            raise ValueError("map_image_analysis_hz must be positive")

        self.logic = MapLogic(Config(image_max_size=image_max_size))
        self.socket = context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(bind)

        self.state_lock = threading.Lock()
        self.map_msg = None
        self.cost_msg = None
        self.prepared = None
        self.stop_event = threading.Event()
        self.sequence = 0
        self.last_sent_at = None
        self.publish_times = deque(maxlen=30)
        self.last_rate_log = time.monotonic()

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        node.create_subscription(OccupancyGrid, "/map", self._map, qos)
        node.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self._cost, qos
        )

        self.analysis_thread = threading.Thread(
            target=self._analysis_loop,
            name="map-analysis",
            daemon=True,
        )
        self.analysis_thread.start()
        self.timer = node.create_timer(1.0 / self.target_hz, self.publish)

    def _map(self, msg):
        with self.state_lock:
            self.map_msg = msg

    def _cost(self, msg):
        with self.state_lock:
            self.cost_msg = msg

    def _pose(self, frame_id):
        import rclpy.time
        try:
            transform = self.tf.lookup_transform(
                frame_id,
                self.base_frame,
                rclpy.time.Time(),
            )
            position = transform.transform.translation
            orientation = transform.transform.rotation
        except Exception:
            return None

        return {
            "x": float(position.x),
            "y": float(position.y),
            "yaw": yaw_of(orientation),
            "frame_id": frame_id,
        }

    def _analysis_loop(self):
        """Refresh expensive map semantics without blocking the ROS executor."""
        period = 1.0 / self.analysis_hz
        next_refresh = time.monotonic()
        while not self.stop_event.is_set():
            delay = max(0.0, next_refresh - time.monotonic())
            if self.stop_event.wait(delay):
                break

            with self.state_lock:
                map_msg = self.map_msg
                cost_msg = self.cost_msg
            if map_msg is None or cost_msg is None:
                next_refresh = time.monotonic() + 0.1
                continue

            pose = self._pose(map_msg.header.frame_id)
            if pose is None:
                next_refresh = time.monotonic() + 0.1
                continue

            started_at = time.monotonic()
            try:
                prepared = self.logic.prepare(map_msg, cost_msg, pose)
            except Exception as error:
                self.node.get_logger().warning(f"Map analysis: {error}")
                next_refresh = time.monotonic() + period
                continue

            finished_at = time.monotonic()
            prepared["analysis_ms"] = round(
                (finished_at - started_at) * 1000, 1
            )
            prepared["prepared_at_monotonic"] = finished_at
            prepared["prepared_at_unix_ns"] = time.time_ns()
            with self.state_lock:
                self.prepared = prepared

            # Maintain a start-to-start rate without spinning if analysis itself
            # takes longer than its requested period.
            next_refresh = max(next_refresh + period, finished_at)

    def publish(self):
        with self.state_lock:
            prepared = self.prepared
        if prepared is None:
            return

        pose = self._pose(prepared["frame_id"])
        if pose is None:
            return

        started_at = time.monotonic()
        captured_at_unix_ns = time.time_ns()
        try:
            image, metadata = self.logic.compose(prepared, pose)
            ok, jpeg = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ok:
                return
            encoded_at = time.monotonic()
            self.sequence += 1
            metadata.update({
                "sequence": self.sequence,
                "target_hz": self.target_hz,
                "analysis_hz": self.analysis_hz,
                "analysis_ms": prepared["analysis_ms"],
                "analysis_age_ms": round(
                    (started_at - prepared["prepared_at_monotonic"]) * 1000, 1
                ),
                "captured_at_unix_ns": captured_at_unix_ns,
                "generated_at_unix_ns": time.time_ns(),
                "render_ms": round((encoded_at - started_at) * 1000, 1),
            })
            header = json.dumps(metadata).encode()
            payload = struct.pack("!I", len(header)) + header + jpeg.tobytes()
            self.socket.send_multipart(
                [b"map/image", payload], flags=zmq.NOBLOCK
            )

        except zmq.Again:
            self.node.get_logger().warning(
                "Map image dropped: subscriber is too slow"
            )
        except Exception as error:
            self.node.get_logger().warning(f"Map image: {error}")

    def close(self):
        self.timer.cancel()
        self.stop_event.set()
        self.analysis_thread.join(timeout=2.0)
        self.socket.close(0)
