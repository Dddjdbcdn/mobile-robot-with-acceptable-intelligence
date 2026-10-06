"""Deterministic local navigation commands."""

import math


class LocalNavigationMixin:
    PERSON_RELATIVE_COMMANDS = {
        "move_to_person_front": 0.0,
        "move_to_person_right": -math.pi / 2.0,
        "move_to_person_left": math.pi / 2.0,
    }

    def resolve_local_destination(
        self,
        prepared,
        pose,
        command,
        *,
        previous_pose=None,
        person_pose=None,
        face_person=False,
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
        if command in self.PERSON_RELATIVE_COMMANDS:
            return self._person_relative_goal(
                prepared, pose, person_pose, command
            )
        if command == "previous_position":
            if not isinstance(previous_pose, dict):
                raise ValueError("previous position is unavailable")
            arrival_yaw = (
                self._person_facing_yaw(
                    previous_pose["x"], previous_pose["y"], person_pose
                )
                if face_person
                else None
            )
            return self._local_goal(
                previous_pose["x"], previous_pose["y"],
                (
                    previous_pose.get("yaw", yaw)
                    if arrival_yaw is None
                    else arrival_yaw
                ),
                pose,
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
            arrival_yaw = (
                self._person_facing_yaw(x, y, person_pose)
                if face_person
                else None
            )
            return self._local_goal(
                x, y, yaw if arrival_yaw is None else arrival_yaw, pose
            )

        raise ValueError(f"unknown local command: {command}")

    def _person_relative_goal(self, prepared, pose, person_pose, command):
        """Choose the nearest safe pose on a person-relative ray."""
        person_yaw = self._person_facing_yaw(
            pose["x"], pose["y"], person_pose
        )
        if person_yaw is None:
            raise ValueError("fresh person pose is unavailable")

        person_x = float(person_pose["x"])
        person_y = float(person_pose["y"])
        # The front point lies on the person-to-robot ray. Positive angular
        # offsets move counterclockwise around the person (the person's left),
        # while negative offsets move clockwise (the person's right).
        front_angle = math.atan2(
            float(pose["y"]) - person_y,
            float(pose["x"]) - person_x,
        )
        requested_offset = self.PERSON_RELATIVE_COMMANDS[command]
        offsets = [requested_offset]
        if requested_offset:
            step = math.radians(max(
                0.1, float(self.cfg.person_navigation_angle_step_deg)
            ))
            sign = 1.0 if requested_offset > 0.0 else -1.0
            angle = abs(requested_offset) - step
            while angle > 0.0:
                offsets.append(sign * angle)
                angle -= step
            offsets.append(0.0)

        grid = prepared["grid"]
        reachable = prepared["reachable"]
        for offset in offsets:
            nearest = self._nearest_person_ray_pose(
                grid,
                reachable,
                person_x,
                person_y,
                front_angle + offset,
            )
            if nearest is None:
                continue
            x, y, distance = nearest
            goal = self._local_goal(
                x, y, math.atan2(person_y - y, person_x - x), pose
            )
            goal.update({
                "person_distance_m": distance,
                "requested_person_angle_deg": math.degrees(requested_offset),
                "resolved_person_angle_deg": math.degrees(offset),
                "used_angle_fallback": not math.isclose(
                    offset, requested_offset, abs_tol=1e-9
                ),
            })
            return goal

        raise ValueError("no safe person-relative pose is reachable")

    def _nearest_person_ray_pose(
        self, grid, reachable, person_x, person_y, yaw
    ):
        """Return the nearest reachable cell center on a ray from the person."""
        step = max(grid.resolution / 2.0, 1e-6)
        max_distance = max(
            step, float(self.cfg.person_navigation_max_distance_m)
        )
        origin_cell = tuple(
            int(value) for value in grid.cells(person_x, person_y)
        )
        last_cell = None
        distance = 0.0
        while distance <= max_distance + 1e-9:
            x = person_x + distance * math.cos(yaw)
            y = person_y + distance * math.sin(yaw)
            col, row = (int(value) for value in grid.cells(x, y))
            cell = (col, row)
            if cell != last_cell:
                last_cell = cell
                if (
                    cell != origin_cell
                    and grid.inside(col, row)
                    and reachable[row, col]
                ):
                    cell_x, cell_y = grid.world(col, row)
                    cell_x, cell_y = float(cell_x), float(cell_y)
                    return (
                        cell_x,
                        cell_y,
                        math.hypot(cell_x - person_x, cell_y - person_y),
                    )
            distance += step
        return None
