"""Deterministic local navigation commands."""

import math


class LocalNavigationMixin:
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
            person_yaw = self._person_facing_yaw(
                pose["x"], pose["y"], person_pose
            )
            if person_yaw is None:
                raise ValueError("fresh person pose is unavailable")
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
            arrival_yaw = self._person_facing_yaw(x, y, person_pose)
            return self._local_goal(
                x, y, yaw if arrival_yaw is None else arrival_yaw, pose
            )

        if command == "open_space_middle":
            step = self.resolve_open_space_middle_step(
                prepared, pose, person_pose=person_pose
            )
            if step["phase"] != "move_to_room_core":
                raise ValueError(
                    "no sufficiently open room core is mapped yet"
                )
            return step["destination"]
        raise ValueError(f"unknown local command: {command}")
