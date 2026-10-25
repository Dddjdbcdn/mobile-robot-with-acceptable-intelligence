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
    live_fov_enabled: bool = True
    fov_alpha: float = 0.32
    coverage_color: tuple = (255, 220, 175)
    repeated_coverage_color: tuple = (220, 205, 255)
    sample_radius: float = 3.0
    occupied: int = 50
    cost_limit: int = 99
    # Candidate markers keep the same proportions in the robot-centred crop.
    pose_reference_crop_px: int = 768
    pose_circle_radius_px: int = 12
    frontier_diamond_radius_px: int = 19


class MapRenderer:
    """Convert prepared map data and overlays into an image."""

    def __init__(self):
        self.cfg = RenderConfig()

    def render_grid_diagnostic(self, msg, pose, layer, overlays=None):
        """Render one raw OccupancyGrid for the on-robot diagnostic display."""
        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0:
            raise ValueError("OccupancyGrid has empty dimensions")
        values = np.asarray(msg.data, dtype=np.int16)
        if values.size != width * height:
            raise ValueError("OccupancyGrid data does not match its dimensions")
        values = values.reshape((height, width))

        image = np.full((height, width, 3), (112, 112, 112), np.uint8)
        known = values >= 0
        if layer == "slam":
            image[known] = (246, 246, 246)
            occupied = values >= self.cfg.occupied
            image[occupied] = (24, 24, 24)
            partial = known & ~occupied & (values > 0)
            if np.any(partial):
                shade = (246 - values[partial] * 1.4).clip(80, 246).astype(np.uint8)
                image[partial] = np.column_stack((shade, shade, shade))
        else:
            image[known] = (246, 246, 246)
            inflated = known & (values > 0) & (values < self.cfg.cost_limit)
            if np.any(inflated):
                strength = values[inflated].astype(np.float32) / self.cfg.cost_limit
                image[inflated, 0] = (235 - 105 * strength).astype(np.uint8)
                image[inflated, 1] = (238 - 145 * strength).astype(np.uint8)
                image[inflated, 2] = 245
            image[values >= self.cfg.cost_limit] = (45, 45, 210)

        # OccupancyGrid row zero is at the map origin; screen row zero is at
        # the top. Flip before converting world coordinates to pixels.
        image = np.flipud(image).copy()
        max_side = max(width, height)
        scale = min(1.0, float(self.cfg.image_max_size) / max_side)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_NEAREST,
            )

        origin = msg.info.origin
        quaternion = origin.orientation
        origin_yaw = math.atan2(
            2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )
        dx = float(pose["x"]) - float(origin.position.x)
        dy = float(pose["y"]) - float(origin.position.y)
        col = (math.cos(origin_yaw) * dx + math.sin(origin_yaw) * dy) / msg.info.resolution
        row = (-math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy) / msg.info.resolution
        point = (round(col * scale), round((height - 1 - row) * scale))
        def grid_pixel(item):
            item_dx = float(item[0]) - float(origin.position.x)
            item_dy = float(item[1]) - float(origin.position.y)
            item_col = (
                math.cos(origin_yaw) * item_dx
                + math.sin(origin_yaw) * item_dy
            ) / msg.info.resolution
            item_row = (
                -math.sin(origin_yaw) * item_dx
                + math.cos(origin_yaw) * item_dy
            ) / msg.info.resolution
            return (
                round(item_col * scale),
                round((height - 1 - item_row) * scale),
            )

        self._draw_diagnostic_overlays(
            image, overlays or {}, grid_pixel,
            pixels_per_meter=scale / float(msg.info.resolution),
        )
        radius = max(7, round(0.20 / float(msg.info.resolution) * scale))
        yaw = float(pose["yaw"]) - origin_yaw
        arrow_length = max(radius * 2, 22)
        tip = (
            round(point[0] + arrow_length * math.cos(yaw)),
            round(point[1] - arrow_length * math.sin(yaw)),
        )
        cv2.circle(image, point, radius, (30, 210, 80), 3, cv2.LINE_AA)
        cv2.arrowedLine(
            image, point, tip, (30, 210, 80), 4, cv2.LINE_AA, tipLength=0.35
        )
        return image

    def render_background(self, prepared, pose, view_size_m=None):
        return self._render(
            prepared["grid"], prepared["costmap"], pose, [], [],
            prepared["frontier_mask"], draw_robot=False,
            view_size_m=view_size_m,
        )

    @staticmethod
    def rotate_180(image):
        """Rotate the complete rendered map for a viewer facing the robot."""
        return cv2.rotate(image, cv2.ROTATE_180)

    def render_snapshot(
        self, prepared, pose, candidates, coverage_counts,
        search_overlay=None, live_fov_mask=None,
        observation=None, observation_mask=None,
        pose_marker_scale=1.0, frontiers=None,
        pending_navigation_pose=None,
        diagnostic_overlays=None,
    ):
        image = prepared["background"].copy()
        view = prepared["view"]
        self._draw_diagnostic_overlays(
            image,
            diagnostic_overlays or {},
            lambda item: self._pixel(view, item[0], item[1]),
            pixels_per_meter=float(view["ppm"]),
        )
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
        if frontiers is None:
            frontiers = [
                item for item in candidates if item.get("kind") == "frontier"
            ]
        self._draw_frontier_candidates(
            image, view, frontiers, marker_scale=pose_marker_scale
        )
        self._draw_dynamic(
            image, view, pose,
            [item for item in candidates if item.get("kind") != "frontier"],
            marker_scale=pose_marker_scale,
        )
        if pending_navigation_pose is not None:
            self._draw_pending_navigation_pose(
                image, view, pending_navigation_pose,
                marker_scale=pose_marker_scale,
            )
        if mode == "exploration" and len(candidates) == 1:
            self._draw_selected_pose(
                image, view, candidates[0], marker_scale=pose_marker_scale
            )
        return image

    @staticmethod
    def _draw_diagnostic_overlays(
        image, overlays, pixel_of, pixels_per_meter,
    ):
        """Draw low-rate operator diagnostics below navigation markers."""
        plan = overlays.get("plan") or []
        if len(plan) >= 2:
            points = np.asarray([pixel_of(item) for item in plan], np.int32)
            cv2.polylines(image, [points], False, (220, 145, 30), 3, cv2.LINE_AA)

        for item in overlays.get("lidar") or []:
            cv2.circle(image, pixel_of(item), 2, (0, 210, 255), -1, cv2.LINE_AA)

        footprint = overlays.get("footprint") or []
        if len(footprint) >= 3:
            points = np.asarray([pixel_of(item) for item in footprint], np.int32)
            cv2.polylines(image, [points], True, (30, 210, 80), 3, cv2.LINE_AA)

        goal = overlays.get("goal")
        if goal is not None:
            center = pixel_of((goal["x"], goal["y"]))
            radius = max(7, round(0.14 * pixels_per_meter))
            yaw = float(goal.get("yaw", 0.0))
            length = max(radius * 2, round(0.30 * pixels_per_meter))
            tip = pixel_of((
                float(goal["x"]) + length / pixels_per_meter * math.cos(yaw),
                float(goal["y"]) + length / pixels_per_meter * math.sin(yaw),
            ))
            cv2.circle(image, center, radius, (220, 40, 220), 3, cv2.LINE_AA)
            cv2.arrowedLine(
                image, center, tip, (220, 40, 220), 3,
                cv2.LINE_AA, tipLength=0.35,
            )

    def _draw_pending_navigation_pose(
        self, image, view, pose, marker_scale=1.0
    ):
        """Draw the destination currently being resolved or driven toward."""
        point = self._pixel(view, pose["x"], pose["y"])
        yaw = float(pose.get("yaw", pose.get("angle", 0.0)))
        arrow_length_m = max(0.25, 28.0 / float(view["ppm"]))
        tip = self._pixel(
            view,
            float(pose["x"]) + arrow_length_m * math.cos(yaw),
            float(pose["y"]) + arrow_length_m * math.sin(yaw),
        )
        color = (0, 165, 255)
        radius = max(4, round(22 * marker_scale))
        thickness = max(1, round(4 * marker_scale))
        cv2.circle(image, point, radius, color, thickness, cv2.LINE_AA)
        cv2.arrowedLine(
            image, point, tip, color, thickness, cv2.LINE_AA, tipLength=0.35
        )
        self._outlined_label(
            image, (point[0] + radius, point[1] - radius),
            "PENDING", color, scale=0.54,
        )

    def pose_marker_scale_for_crop(self, view, crop_size_m):
        """Scale pose badges to a fixed fraction of the delivered crop."""
        if crop_size_m is None:
            return 1.0
        crop_pixels = float(crop_size_m) * float(view["ppm"])
        return crop_pixels / float(self.cfg.pose_reference_crop_px)

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

    def _draw_frontier_candidates(
        self, image, view, frontiers, debug=False, marker_scale=1.0
    ):
        radius = max(2, round(
            self.cfg.frontier_diamond_radius_px * marker_scale
        ))
        for item in frontiers:
            point = self._pixel(view, item["x"], item["y"])
            color = (110, 165, 110) if debug else (35, 185, 35)
            self._diamond_badge(
                image, point, item["id"], color, radius=radius
            )

    def _draw_selected_pose(
        self, image, view, candidate, marker_scale=1.0
    ):
        point = self._pixel(view, candidate["x"], candidate["y"])
        radius = max(4, round(25 * marker_scale))
        thickness = max(1, round(4 * marker_scale))
        cv2.circle(
            image, point, radius, (0, 165, 255), thickness, cv2.LINE_AA
        )
        self._outlined_label(
            image, (point[0] + 28, point[1] - 18),
            f"NEXT {candidate['id']}", (0, 165, 255), scale=0.54,
        )

    def _make_view(self, grid, pose, view_size_m=None):
        """Build a robot-relative view where its current heading is always up."""
        if view_size_m is not None:
            size = float(view_size_m)
            if not math.isfinite(size) or size <= 0.0:
                raise ValueError("view_size_m must be a positive finite number")
            half = size / 2.0
            return {
                "xmin": -half,
                "xmax": half,
                "ymin": -half,
                "ymax": half,
                "ppm": self.cfg.image_max_size / size,
                "width": self.cfg.image_max_size,
                "height": self.cfg.image_max_size,
                "origin_x": float(pose["x"]),
                "origin_y": float(pose["y"]),
                "heading_yaw": float(pose["yaw"]),
            }

        rows, cols = np.nonzero(grid.data >= 0)
        if len(rows):
            xs, ys = grid.world(cols, rows)
        else:
            xs = np.array([pose["x"]])
            ys = np.array([pose["y"]])

        dx = np.asarray(xs) - float(pose["x"])
        dy = np.asarray(ys) - float(pose["y"])
        yaw = float(pose["yaw"])
        # Horizontal image motion is robot-right; vertical image motion is
        # robot-forward. Keeping these as metric coordinates makes cropping
        # and candidate interpretation independent of the world-map yaw.
        right = dx * math.sin(yaw) - dy * math.cos(yaw)
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        radius = self.cfg.sample_radius
        right = np.append(right, [-radius, radius])
        forward = np.append(forward, [-radius, radius])

        margin = self.cfg.view_margin + grid.resolution
        xmin, xmax = float(right.min()) - margin, float(right.max()) + margin
        ymin, ymax = float(forward.min()) - margin, float(forward.max()) + margin

        span_x = max(xmax - xmin, grid.resolution)
        span_y = max(ymax - ymin, grid.resolution)
        ppm = self.cfg.image_max_size / max(span_x, span_y)

        return {
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "ppm": ppm,
            "width": max(1, round(span_x * ppm)),
            "height": max(1, round(span_y * ppm)),
            "origin_x": float(pose["x"]),
            "origin_y": float(pose["y"]),
            "heading_yaw": yaw,
        }
    @staticmethod
    def _pixel(view, x, y):
        dx = float(x) - view.get("origin_x", 0.0)
        dy = float(y) - view.get("origin_y", 0.0)
        yaw = view.get("heading_yaw", 0.0)
        right = dx * math.sin(yaw) - dy * math.cos(yaw)
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        return (
            round((right - view["xmin"]) * view["ppm"]),
            round((view["ymax"] - forward) * view["ppm"]),
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
        factor = radius / 12.0
        outer = max(1, round(3 * factor))
        keyline = max(1, round(factor))
        cv2.circle(
            image, point, radius + outer, (35, 35, 35), -1, cv2.LINE_AA
        )
        cv2.circle(
            image, point, radius + keyline,
            (255, 255, 255), -1, cv2.LINE_AA,
        )
        cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)
        scale = (0.43 if len(text) <= 2 else 0.34) * factor
        MapRenderer._centered_text(
            image, point, text, scale, (255, 255, 255),
            max(1, round(factor)),
        )

    @staticmethod
    def _diamond_badge(image, point, text, color, radius=19):
        factor = radius / 19.0
        outer = max(1, round(3 * factor))
        keyline = max(1, round(factor))

        def diamond(size):
            return np.array([
                (point[0], point[1] - size),
                (point[0] + size, point[1]),
                (point[0], point[1] + size),
                (point[0] - size, point[1]),
            ], np.int32)
        cv2.fillConvexPoly(
            image, diamond(radius + outer), (35, 35, 35), cv2.LINE_AA
        )
        cv2.fillConvexPoly(
            image, diamond(radius + keyline), (255, 255, 255), cv2.LINE_AA
        )
        cv2.fillConvexPoly(image, diamond(radius), color, cv2.LINE_AA)
        scale = (0.52 if len(text) <= 2 else 0.44) * factor
        MapRenderer._centered_text(
            image, point, text, scale, (255, 255, 255),
            max(1, round(factor)),
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
        """Draw the obstacle-clipped cone corresponding to current Image 2."""
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
        mask = coverage.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (210, 185, 20), 3, cv2.LINE_AA)

    def _draw_dynamic(
        self, image, view, pose, locals_, marker_scale=1.0
    ):
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
        badge_radius = max(2, round(
            self.cfg.pose_circle_radius_px * marker_scale
        ))
        for item in locals_:
            if item["kind"] == "rotation":
                continue
            point = self._pixel(view, item["x"], item["y"])
            color = (220, 105, 15)  # saturated blue in BGR
            self._circle_badge(
                image, point, item["id"], color, radius=badge_radius
            )
        if any(item.get("kind") == "rotation" for item in locals_):
            self._outlined_label(
                image, (center[0] + 17, center[1] + 24),
                "H", robot_color, scale=0.5,
            )
    def _render(
        self, grid, costmap, pose, locals_, frontiers, frontier_mask,
        draw_robot=True, view_size_m=None,
    ):
        """Render a clean metric map intended for VLM waypoint selection."""
        view = self._make_view(grid, pose, view_size_m=view_size_m)
        rights = view["xmin"] + (np.arange(view["width"]) + 0.5) / view["ppm"]
        forwards = view["ymax"] - (np.arange(view["height"]) + 0.5) / view["ppm"]
        right, forward = np.meshgrid(rights, forwards)
        yaw = view["heading_yaw"]
        x = view["origin_x"] + right * math.sin(yaw) + forward * math.cos(yaw)
        y = view["origin_y"] - right * math.cos(yaw) + forward * math.sin(yaw)

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
        # unambiguous meaning instead of competing with the costmap.
        image = np.full((view["height"], view["width"], 3), (62, 62, 62), np.uint8)
        image[map_inside & (occupancy < 0)] = (112, 112, 112)
        image[free] = (246, 246, 246)
        image[occupancy >= self.cfg.occupied] = (24, 24, 24)
        image[free & (costs > 0)] = (224, 232, 244)
        image[free & (costs >= self.cfg.cost_limit)] = (185, 198, 232)
        image[free & (costs < 0)] = (190, 190, 190)

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

        # Draw the robot-relative metric grid over the rendered map.
        spacing = self.cfg.grid_spacing
        x0 = math.ceil(view["xmin"] / spacing)
        x1 = math.floor(view["xmax"] / spacing)
        y0 = math.ceil(view["ymin"] / spacing)
        y1 = math.floor(view["ymax"] / spacing)
        for i in range(x0, x1 + 1):
            px = round((i * spacing - view["xmin"]) * view["ppm"])
            if 0 <= px < view["width"]:
                visible = free[:, px] & ~frontier_halo[:, px]
                image[visible, px] = (218, 218, 218)
        for i in range(y0, y1 + 1):
            py = round((view["ymax"] - i * spacing) * view["ppm"])
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
                # Clear noisy occupancy/frontier pixels around the badge so
                # the candidate remains legible after JPEG compression.
                clearance = np.array([
                    (p[0], p[1] - 26), (p[0] + 26, p[1]),
                    (p[0], p[1] + 26), (p[0] - 26, p[1]),
                ], np.int32)
                cv2.fillConvexPoly(
                    image, clearance, (255, 255, 255), cv2.LINE_AA
                )
                self._diamond_badge(image, p, item["id"], color)
            else:
                color = (220, 105, 15)
                self._circle_badge(image, p, item["id"], color)

        if draw_robot:
            self._draw_dynamic(image, view, pose, locals_)

        return image, view
