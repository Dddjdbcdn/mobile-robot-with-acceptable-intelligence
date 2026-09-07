from __future__ import annotations

import asyncio
import math

from actions.action_result import ActionResult
from cognition.state import robot_state


class NavigateSemanticAction:
    """Navigate to a map-relative room landmark computed by ROS2."""

    DESTINATIONS = {
        "center": ("center", None),
        "front_wall": ("walls", "front"),
        "left_wall": ("walls", "left"),
        "back_wall": ("walls", "back"),
        "right_wall": ("walls", "right"),
        "front_left_corner": ("corners", "front_left"),
        "back_left_corner": ("corners", "back_left"),
        "back_right_corner": ("corners", "back_right"),
        "front_right_corner": ("corners", "front_right"),
    }

    def __init__(self, send_robot_command, semantic_memory=None):
        self.send_robot_command = send_robot_command
        self.semantic_memory = semantic_memory
        self.active = False
        self.action_id = None
        self.target = None
        self.completion_future = None

    async def start(self, destination, action_id, object_a=None, object_b=None):
        destination = str(destination or "").strip().lower().replace("-", "_")
        if self.active:
            return ActionResult(
                action_id=action_id,
                action_type="navigate_semantic",
                status="already_running",
                target=destination,
                outcome="already_running",
                reason_code="NAVIGATION_BUSY",
                retryable=True,
            )

        geometry = robot_state.get("room_geometry") or {}
        resolved = self._resolve(destination, geometry, object_a, object_b)
        if resolved is None:
            return ActionResult(
                action_id=action_id,
                action_type="navigate_semantic",
                status="failed",
                target=destination or None,
                outcome="room_geometry_unavailable",
                reason_code="ROOM_GEOMETRY_UNAVAILABLE",
                retryable=True,
                data={"valid_destinations": [*self.DESTINATIONS, "nearest_wall", "nearest_corner", "between_objects"]},
            )

        self.active = True
        self.action_id = action_id
        self.target = destination
        self.completion_future = asyncio.get_running_loop().create_future()
        try:
            feedback = await self.send_robot_command({
                "command": "navigate_to_pose",
                "action_id": action_id,
                "frame_id": "map",
                "x": resolved["x"],
                "y": resolved["y"],
                "angle": resolved["yaw"],
            })
        except Exception as error:
            return self._complete(
                "failed",
                "command_failed",
                "NAVIGATION_COMMAND_FAILED",
                {"error": f"{type(error).__name__}: {error}"},
            )
        if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
            return self._complete(
                "failed", "rejected", "NAVIGATION_REJECTED",
                {"message": str(feedback)},
            )
        return ActionResult(
            action_id=action_id,
            action_type="navigate_semantic",
            status="running",
            target=destination,
            outcome="navigating",
            data={"destination": resolved, "frame_id": "map"},
        )

    def _resolve(self, destination, geometry, object_a=None, object_b=None):
        if destination == "between_objects":
            if self.semantic_memory is None:
                return None
            first = self.semantic_memory.recall(object_a)
            second = self.semantic_memory.recall(object_b)
            if not isinstance(first, dict) or not isinstance(second, dict):
                return None
            first_x, first_y = float(first["map_x"]), float(first["map_y"])
            second_x, second_y = float(second["map_x"]), float(second["map_y"])
            return {
                "x": (first_x + second_x) / 2.0,
                "y": (first_y + second_y) / 2.0,
                "yaw": math.atan2(second_y - first_y, second_x - first_x),
            }
        if destination == "nearest_wall":
            destination = f"{geometry.get('most_meaningful_wall')}_wall"
        elif destination == "nearest_corner":
            destination = f"{geometry.get('most_meaningful_corner')}_corner"
        route = self.DESTINATIONS.get(destination)
        if route is None:
            return None
        group, name = route
        value = geometry.get(group) if name is None else (geometry.get(group) or {}).get(name)
        if not isinstance(value, dict) or not all(
            key in value for key in ("x", "y", "yaw")
        ):
            return None
        return {key: float(value[key]) for key in ("x", "y", "yaw")}

    async def wait_until_finished(self):
        return await self.completion_future

    async def stop(self, reason_code="USER_REQUESTED"):
        if not self.active:
            return None
        await self.send_robot_command({
            "command": "stop_moving", "action_id": self.action_id
        })
        return self._complete("cancelled", "stopped", reason_code)

    def handle_navigation_event(self, payload):
        if not self.active or payload.get("event") != "navigation":
            return False
        if payload.get("action_id") not in {None, self.action_id}:
            return False
        succeeded = str(payload.get("status")) == "Goal Reached"
        self._complete(
            "succeeded" if succeeded else "failed",
            "reached" if succeeded else "navigation_failed",
            None if succeeded else "NAVIGATION_FAILED",
        )
        return True

    def _complete(self, status, outcome, reason_code=None, data=None):
        result = ActionResult(
            action_id=self.action_id or "unassigned",
            action_type="navigate_semantic",
            status=status,
            target=self.target,
            outcome=outcome,
            reason_code=reason_code,
            retryable=status == "failed",
            data=data or {},
        )
        future = self.completion_future
        self.active = False
        self.action_id = None
        self.target = None
        if future is not None and not future.done():
            future.set_result(result)
        return result
