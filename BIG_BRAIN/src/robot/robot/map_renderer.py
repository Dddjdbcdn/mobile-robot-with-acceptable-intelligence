"""Standalone OpenCV rendering for analyzed map snapshots."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class RenderConfig:
    image_max_size: int = 1024
    view_margin: float = 0.45
    grid_spacing: float = 1.0
    live_fov_enabled: bool = False
    fov_alpha: float = 0.32
    coverage_color: tuple = (255, 220, 175)
    repeated_coverage_color: tuple = (220, 205, 255)
    sample_radius: float = 3.0
    occupied: int = 50
    cost_limit: int = 99


class MapRenderer:
    """Convert prepared map data and overlays into an image."""

    def __init__(self):
        self.cfg = RenderConfig()

    def render_background(self, prepared, pose):
        return self._render(
            prepared["grid"], prepared["costmap"], pose, [], [],
            prepared["frontier_mask"], prepared["doors"],
            prepared["door_candidate_mask"], prepared["rooms"],
            prepared["current_room_mask"], draw_robot=False,
        )

    def render_snapshot(
        self, prepared, pose, candidates, coverage_counts,
        search_overlay=None, live_fov_mask=None,
        observation=None, observation_mask=None,
    ):
        image = prepared["background"].copy()
        view = prepared["view"]
        if isinstance(search_overlay, dict):
            self._draw_search_overlay(
                image, view, search_overlay, coverage_counts
            )
        if self.cfg.live_fov_enabled:
            self._draw_camera_fov(image, view, live_fov_mask)
        self._draw_observation_fov(
            image, view, observation, observation_mask
        )

        mode = str((search_overlay or {}).get("mode") or "")
        frontiers = (
            prepared["frontiers"] if mode == "exploration"
            else [item for item in candidates if item.get("kind") == "frontier"]
        )
        self._draw_frontier_candidates(
            image, view, frontiers, debug=mode == "exploration"
        )
        self._draw_dynamic(
            image, view, pose,
            [item for item in candidates if item.get("kind") != "frontier"],
        )
        if mode == "exploration" and candidates:
            self._draw_selected_pose(image, view, candidates[0])
        return image

    def _draw_observation_fov(
        self, image, view, observation, grid_mask
    ):
        if observation is None or grid_mask is None:
            return
        mask = self._project_grid(view, grid_mask).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (0, 140, 255), 3, cv2.LINE_AA)

    def _draw_frontier_candidates(self, image, view, frontiers, debug=False):
        for item in frontiers:
            point = self._pixel(view, item["x"], item["y"])
            color = (110, 165, 110) if debug else (35, 185, 35)
            self._heading_tick(image, point, item["yaw"], 19, color)
            self._diamond_badge(image, point, item["id"], color)

    def _draw_selected_pose(self, image, view, candidate):
        point = self._pixel(view, candidate["x"], candidate["y"])
        cv2.circle(image, point, 25, (0, 165, 255), 4, cv2.LINE_AA)
        self._outlined_label(
            image, (point[0] + 28, point[1] - 18),
            f"NEXT {candidate['id']}", (0, 165, 255), scale=0.54,
        )

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
    def _project_grid(view, values):
        projected = np.zeros(
            (view["height"], view["width"]), dtype=values.dtype
        )
        inside = view["map_inside"]
        projected[inside] = values[
            view["map_rows"][inside], view["map_cols"][inside]
        ]
        return projected

    @staticmethod
    def _heading_tick(image, point, yaw, radius, color):
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
            origin = (
                round(point[0] + anchor[0] - (
                    float(cols.min()) + float(cols.max())
                ) / 2),
                round(point[1] + anchor[1] - (
                    float(rows.min()) + float(rows.max())
                ) / 2),
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
        cv2.circle(image, point, radius + 3, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.circle(image, point, radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)
        scale = 0.43 if len(text) <= 2 else 0.34
        MapRenderer._centered_text(
            image, point, text, scale, (255, 255, 255), 1
        )

    @staticmethod
    def _diamond_badge(image, point, text, color, radius=19):
        def diamond(size):
            return np.array([
                (point[0], point[1] - size),
                (point[0] + size, point[1]),
                (point[0], point[1] + size),
                (point[0] - size, point[1]),
            ], np.int32)
        cv2.fillConvexPoly(
            image, diamond(radius + 3), (35, 35, 35), cv2.LINE_AA
        )
        cv2.fillConvexPoly(
            image, diamond(radius + 1), (255, 255, 255), cv2.LINE_AA
        )
        cv2.fillConvexPoly(image, diamond(radius), color, cv2.LINE_AA)
        scale = 0.52 if len(text) <= 2 else 0.44
        MapRenderer._centered_text(
            image, point, text, scale, (255, 255, 255), 1
        )

    @staticmethod
    def _outlined_label(image, point, text, color, scale=0.46):
        origin = (point[0] + 7, point[1] - 7)
        cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
            scale, (255, 255, 255), 4, cv2.LINE_AA,
        )
        cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, 2, cv2.LINE_AA,
        )

    def _draw_search_overlay(self, image, view, search_overlay, coverage_counts):
        """Draw accumulated find coverage and path before dynamic pose labels."""
        rendered_counts = self._project_grid(view, coverage_counts)
        coverage = rendered_counts > 0
        repeated = rendered_counts >= 2
        if np.any(coverage):
            channel_span = np.max(image, axis=2) - np.min(image, axis=2)
            visible = coverage & (np.min(image, axis=2) > 150) & (channel_span < 90)
            tint = np.full_like(image, self.cfg.coverage_color)
            image[visible] = cv2.addWeighted(
                image[visible], 1.0 - self.cfg.fov_alpha,
                tint[visible], self.cfg.fov_alpha, 0,
            )
            repeated_visible = repeated & visible
            if np.any(repeated_visible):
                repeated_tint = np.full_like(image, self.cfg.repeated_coverage_color)
                image[repeated_visible] = cv2.addWeighted(
                    image[repeated_visible], 1.0 - self.cfg.fov_alpha,
                    repeated_tint[repeated_visible], self.cfg.fov_alpha, 0,
                )

        points = []
        for history_pose in search_overlay.get("search_poses") or []:
            try:
                point = self._pixel(
                    view, float(history_pose["x"]), float(history_pose["y"])
                )
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= point[0] < view["width"] and 0 <= point[1] < view["height"]:
                points.append(point)
        for start, end in zip(points, points[1:]):
            cv2.line(image, start, end, (180, 0, 180), 3, cv2.LINE_AA)
        if points:
            start = points[0]
            cv2.circle(image, start, 18, (20, 20, 20), 7, cv2.LINE_AA)
            cv2.circle(image, start, 18, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.circle(image, start, 18, (0, 235, 255), 2, cv2.LINE_AA)
            origin = (start[0] + 21, start[1] + 4)
            cv2.putText(image, "S", origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.putText(image, "S", origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 235, 255), 2, cv2.LINE_AA)
    def _draw_camera_fov(self, image, view, coverage):
        """Tint the current camera footprint without changing search coverage."""
        if coverage is None or not np.any(coverage):
            return
        coverage = self._project_grid(view, coverage)
        channel_span = np.max(image, axis=2) - np.min(image, axis=2)
        visible = (
            coverage
            & (np.min(image, axis=2) > 150)
            & (channel_span < 90)
        )
        tint = np.full_like(image, self.cfg.coverage_color)
        image[visible] = cv2.addWeighted(
            image[visible], 1.0 - self.cfg.fov_alpha,
            tint[visible], self.cfg.fov_alpha, 0,
        )

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
        view["map_inside"] = map_inside
        view["map_rows"] = map_rows_clipped
        view["map_cols"] = map_cols_clipped
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

