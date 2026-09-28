"""Persistent runtime identities for geometric room components."""

from __future__ import annotations

import threading
from copy import deepcopy

import numpy as np


class RoomRegistry:
    """Match transient component labels to stable Room 1, Room 2, ... IDs."""

    def __init__(self, logic):
        self.logic = logic
        self._lock = threading.Lock()
        self._rooms: dict[int, dict] = {}
        self._edges: set[tuple[int, int]] = set()

    @staticmethod
    def _exact_label(grid, labels, pose) -> int:
        col, row = (int(value) for value in grid.cells(pose["x"], pose["y"]))
        if not grid.inside(col, row):
            return 0
        return int(labels[row, col])

    @staticmethod
    def _anchor_for_label(grid, labels, label, pose) -> dict:
        rows, cols = np.nonzero(labels == label)
        xs, ys = grid.world(cols, rows)
        index = int(np.argmin(
            (xs - float(pose["x"])) ** 2
            + (ys - float(pose["y"])) ** 2
        ))
        return {
            "x": float(xs[index]),
            "y": float(ys[index]),
            "yaw": float(pose.get("yaw", 0.0)),
            "frame_id": str(pose.get("frame_id") or "map"),
        }

    def _observation(self, prepared, pose):
        grid = prepared["grid"]
        labels, areas = self.logic._exit_open_components(
            grid, prepared["reachable"]
        )
        label = self._exact_label(grid, labels, pose)
        if label == 0:
            label = self.logic._nearest_component_label(
                grid, labels, pose["x"], pose["y"],
                self.logic.cfg.exit_component_lookup_radius_m,
            )
        if label == 0:
            return None
        return grid, labels, areas, label

    def _matching_room_id(self, grid, labels, label):
        for room_id, room in self._rooms.items():
            if self._exact_label(grid, labels, room["anchor"]) == label:
                return room_id
        return None

    def ensure_startup_room(self, prepared, pose):
        """Register the first observed open component as Room 1 exactly once."""
        with self._lock:
            if self._rooms:
                return deepcopy(self._rooms[1])
            observation = self._observation(prepared, pose)
            if observation is None:
                return None
            grid, labels, areas, label = observation
            room = {
                "room_id": 1,
                "name": "Room 1",
                "anchor": self._anchor_for_label(grid, labels, label, pose),
                "area_m2": float(areas.get(label, 0.0)),
            }
            self._rooms[1] = room
            return deepcopy(room)

    def observe_room(self, prepared, pose, connected_from=None):
        """Return the known room at pose or register the component as a new one."""
        with self._lock:
            observation = self._observation(prepared, pose)
            if observation is None:
                return None
            grid, labels, areas, label = observation
            room_id = self._matching_room_id(grid, labels, label)
            if room_id is None:
                room_id = len(self._rooms) + 1
                self._rooms[room_id] = {
                    "room_id": room_id,
                    "name": f"Room {room_id}",
                    "anchor": self._anchor_for_label(
                        grid, labels, label, pose
                    ),
                    "area_m2": float(areas.get(label, 0.0)),
                }
            if connected_from is not None and int(connected_from) != room_id:
                self._edges.add(tuple(sorted((int(connected_from), room_id))))
            return deepcopy(self._rooms[room_id])

    def room_at(self, prepared, pose):
        """Match the current pose without creating a new room identity."""
        with self._lock:
            observation = self._observation(prepared, pose)
            if observation is None:
                return None
            grid, labels, _, label = observation
            room_id = self._matching_room_id(grid, labels, label)
            return deepcopy(self._rooms[room_id]) if room_id is not None else None

    def destination(self, room_id) -> dict:
        with self._lock:
            try:
                room = self._rooms[int(room_id)]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"unknown room: {room_id}") from error
            return dict(room["anchor"])

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "rooms": [
                    deepcopy(self._rooms[key]) for key in sorted(self._rooms)
                ],
                "connections": [list(edge) for edge in sorted(self._edges)],
            }
