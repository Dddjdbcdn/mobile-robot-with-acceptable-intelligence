"""Occupancy-grid coordinate conversion."""

import math

import numpy as np


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

