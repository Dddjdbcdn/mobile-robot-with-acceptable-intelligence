"""Camera geometry, visibility, and coverage tracking."""

import math

import numpy as np


class VisibilityMixin:
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

