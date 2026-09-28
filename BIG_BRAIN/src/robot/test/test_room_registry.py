import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from robot.map.room_registry import RoomRegistry


class FakeGrid:
    resolution = 1.0

    def __init__(self):
        self.shape = (3, 7)

    def cells(self, x, y):
        return int(round(x)), int(round(y))

    def inside(self, col, row):
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]

    def world(self, cols, rows):
        return np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)


class FakeConfig:
    exit_component_lookup_radius_m = 1.0


class FakeLogic:
    cfg = FakeConfig()

    @staticmethod
    def _exit_open_components(grid, reachable):
        labels = np.zeros(grid.shape, dtype=np.int32)
        labels[:, :3] = 1
        labels[:, 4:] = 2
        return labels, {1: 9.0, 2: 9.0}

    @staticmethod
    def _nearest_component_label(grid, labels, x, y, max_distance_m):
        col, row = grid.cells(x, y)
        return int(labels[row, col]) if grid.inside(col, row) else 0


def test_registry_numbers_rooms_and_remembers_connections():
    registry = RoomRegistry(FakeLogic())
    prepared = {
        "grid": FakeGrid(),
        "reachable": np.ones((3, 7), dtype=bool),
    }
    room_1_pose = {"x": 1.0, "y": 1.0, "yaw": 0.0, "frame_id": "map"}
    room_2_pose = {"x": 5.0, "y": 1.0, "yaw": 1.0, "frame_id": "map"}

    room_1 = registry.ensure_startup_room(prepared, room_1_pose)
    same_room = registry.observe_room(prepared, room_1_pose)
    room_2 = registry.observe_room(prepared, room_2_pose, connected_from=1)

    assert room_1["room_id"] == 1
    assert same_room["room_id"] == 1
    assert room_2["room_id"] == 2
    assert registry.room_at(prepared, room_2_pose)["room_id"] == 2
    assert registry.destination(1) == room_1["anchor"]
    assert registry.snapshot()["connections"] == [[1, 2]]


def test_registry_rejects_an_unknown_room():
    registry = RoomRegistry(FakeLogic())
    try:
        registry.destination(7)
    except ValueError as error:
        assert str(error) == "unknown room: 7"
    else:
        raise AssertionError("unknown room should not resolve")
