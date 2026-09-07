"""Local semantic room geometry derived from a ROS occupancy grid."""

from __future__ import annotations

import math


def _normalize(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class RoomGeometryEstimator:
    DIRECTIONS = {
        "front": 0.0,
        "left": math.pi / 2.0,
        "back": math.pi,
        "right": -math.pi / 2.0,
    }
    CORNERS = {
        "front_left": math.pi / 4.0,
        "back_left": 3.0 * math.pi / 4.0,
        "back_right": -3.0 * math.pi / 4.0,
        "front_right": -math.pi / 4.0,
    }

    def __init__(self, occupied_threshold=50, max_range=8.0):
        self.occupied_threshold = occupied_threshold
        self.max_range = max_range
        self.wall_inset = 0.25
        self.corner_inset = 0.25

    def estimate(self, grid, robot_x, robot_y, robot_yaw):
        info = grid.info
        data = grid.data
        resolution = float(info.resolution)
        origin = info.origin.position
        orientation = info.origin.orientation
        origin_yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2),
        )

        def blocked(x, y):
            dx, dy = x - origin.x, y - origin.y
            c, s = math.cos(origin_yaw), math.sin(origin_yaw)
            col = int(math.floor((c * dx + s * dy) / resolution))
            row = int(math.floor((-s * dx + c * dy) / resolution))
            if not (0 <= col < info.width and 0 <= row < info.height):
                return True
            value = data[row * info.width + col]
            return value < 0 or value >= self.occupied_threshold

        step = max(resolution * 0.5, 0.02)

        def ray(relative_angle):
            angle = robot_yaw + relative_angle
            last_x, last_y, distance = robot_x, robot_y, 0.0
            while distance < self.max_range:
                distance += step
                x = robot_x + distance * math.cos(angle)
                y = robot_y + distance * math.sin(angle)
                if blocked(x, y):
                    break
                last_x, last_y = x, y
            return last_x, last_y, math.hypot(last_x - robot_x, last_y - robot_y)

        boundary = [ray(math.tau * index / 72.0) for index in range(72)]
        polygon = [(x, y) for x, y, _ in boundary]
        center_x, center_y = self._polygon_centroid(polygon, robot_x, robot_y)

        walls = {}
        for name, relative_angle in self.DIRECTIONS.items():
            hit_x, hit_y, distance = ray(relative_angle)
            walls[name] = self._inset_goal(
                hit_x, hit_y, robot_x, robot_y, self.wall_inset,
                face_x=hit_x, face_y=hit_y,
                distance=distance,
            )

        corners = {}
        for name, relative_angle in self.CORNERS.items():
            hit_x, hit_y, distance = ray(relative_angle)
            corners[name] = self._inset_goal(
                hit_x, hit_y, robot_x, robot_y, self.corner_inset,
                face_x=hit_x, face_y=hit_y,
                distance=distance,
            )

        nearest_wall = min(walls, key=lambda name: walls[name]["distance"])
        nearest_corner = min(corners, key=lambda name: corners[name]["distance"])
        return {
            "frame_id": "map",
            "center": {
                "x": center_x,
                "y": center_y,
                "yaw": math.atan2(robot_y - center_y, robot_x - center_x),
            },
            "walls": walls,
            "corners": corners,
            "most_meaningful_wall": nearest_wall,
            "most_meaningful_corner": nearest_corner,
        }

    @staticmethod
    def _polygon_centroid(points, fallback_x, fallback_y):
        twice_area = 0.0
        x_sum = 0.0
        y_sum = 0.0
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            cross = x1 * y2 - x2 * y1
            twice_area += cross
            x_sum += (x1 + x2) * cross
            y_sum += (y1 + y2) * cross
        if abs(twice_area) < 1e-9:
            return fallback_x, fallback_y
        return x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area)

    @staticmethod
    def _inset_goal(x, y, robot_x, robot_y, inset, face_x, face_y, distance):
        if distance > 1e-6:
            scale = max(0.0, distance - inset) / distance
            goal_x = robot_x + (x - robot_x) * scale
            goal_y = robot_y + (y - robot_y) * scale
        else:
            goal_x, goal_y = robot_x, robot_y
        return {
            "x": goal_x,
            "y": goal_y,
            "yaw": _normalize(math.atan2(face_y - goal_y, face_x - goal_x)),
            "distance": distance,
            "boundary_x": x,
            "boundary_y": y,
        }
