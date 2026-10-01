"""Context and exploration candidate planning."""

import math

import cv2
import numpy as np


class SearchPlanningMixin:
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

