"""One cognition-side lifecycle for all foreground navigation."""
from __future__ import annotations

import asyncio
import math

from actions.action_result import ActionResult


SEQUENCE_COMMANDS = {
    "open_space_middle", "exit_room", "go_to_another_room",
}

LOCAL_COMMANDS = {
    "nudge_forward", "nudge_backward", "nudge_left", "nudge_right",
    "rotate_left", "rotate_right", "rotate_around", "furthest_forward",
    "furthest_backward", "previous_position", "face_person", "go_to_room",
    "return_to_initial_place",
    "move_to_person_front", "move_to_person_right", "move_to_person_left",
}
NAVIGATION_COMMANDS = LOCAL_COMMANDS | SEQUENCE_COMMANDS

PERSON_RELATIVE_COMMANDS = {
    "move_to_person_front", "move_to_person_right", "move_to_person_left",
}


class NavigationAction:
    """Start, observe, and cancel the single foreground navigation action."""

    def __init__(self, send_robot_command):
        self.send_robot_command = send_robot_command
        self.active = False
        self.mode = None
        self.action_id = None
        self.target = None
        self.room_id = None
        self.completion_future = None
        self._result_data = {}

    @property
    def action_type(self):
        return (
            "approach_action" if self.mode == "approach"
            else "explicit_navigation"
        )

    async def start_local(
        self, command, action_id, room_id=None, face_person=False
    ):
        command = str(command or "").strip().lower()
        if self.active:
            return self._busy_result(
                action_id, command, "explicit_navigation"
            )
        if command not in NAVIGATION_COMMANDS:
            return ActionResult(
                action_id, "explicit_navigation", "failed",
                target=command or None,
                reason_code="UNKNOWN_LOCAL_NAVIGATION_COMMAND",
                data={"valid_commands": sorted(NAVIGATION_COMMANDS)},
            )
        if command == "go_to_room":
            try:
                room_id = int(room_id)
            except (TypeError, ValueError):
                return ActionResult(
                    action_id, "explicit_navigation", "failed",
                    target=command, reason_code="ROOM_ID_REQUIRED",
                )
            if room_id < 1:
                return ActionResult(
                    action_id, "explicit_navigation", "failed",
                    target=command, reason_code="ROOM_ID_INVALID",
                )

        mode = "sequence" if command in SEQUENCE_COMMANDS else "local"
        payload = {
            "command": (
                "navigate_sequence" if mode == "sequence" else "navigate_local"
            ),
            "action_id": action_id,
            "sequence_command" if mode == "sequence" else "local_command": command,
            "face_person": bool(face_person),
        }
        if command == "go_to_room":
            payload["room_id"] = room_id
        return await self._start(
            mode=mode,
            action_id=action_id,
            target=command,
            payload=payload,
            running_outcome="executing_explicit_navigation",
            room_id=room_id,
        )

    async def start_approach(
        self, location, action_id, target=None, standoff_m=None
    ):
        if self.active:
            return self._busy_result(action_id, target, "approach_action")
        required = ("x", "y", "angle", "tof_range")
        valid = isinstance(location, dict) and all(
            name in location
            and not isinstance(location[name], bool)
            and isinstance(location[name], (int, float))
            and math.isfinite(float(location[name]))
            for name in required
        )
        if valid:
            valid = float(location["tof_range"]) > 0.0
        if not valid:
            return ActionResult(
                action_id, "approach_action", "failed", target=target,
                outcome="invalid_location",
                reason_code="INVALID_APPROACH_LOCATION",
            )

        raw_destination = {name: location[name] for name in required}
        payload = {
            "command": "navigate_to_approach",
            "action_id": action_id,
            "frame_id": "base_footprint",
            **raw_destination,
        }
        if standoff_m is not None:
            payload["standoff_m"] = float(standoff_m)
        return await self._start(
            mode="approach",
            action_id=action_id,
            target=target,
            payload=payload,
            running_outcome="approaching",
            result_data={
                "approach_location": dict(location),
                "raw_destination": dict(raw_destination),
            },
        )

    async def _start(
        self,
        *,
        mode,
        action_id,
        target,
        payload,
        running_outcome,
        room_id=None,
        result_data=None,
    ):
        self.active = True
        self.mode = mode
        self.action_id = action_id
        self.target = target
        self.room_id = room_id
        self._result_data = dict(result_data or {})
        self.completion_future = asyncio.get_running_loop().create_future()
        try:
            response = await self.send_robot_command(payload)
        except Exception as error:
            return self._finish(
                "failed", "command_failed", "NAVIGATION_COMMAND_FAILED",
                {"error": f"{type(error).__name__}: {error}"},
            )
        if not isinstance(response, dict) or response.get("status") != "accepted":
            message = (
                response.get("message", "navigation_rejected")
                if isinstance(response, dict) else str(response)
            )
            return self._finish(
                "failed", "rejected", "NAVIGATION_REJECTED",
                {"message": message},
            )
        if not self.active and self.completion_future.done():
            return self.completion_future.result()
        return ActionResult(
            self.action_id,
            self.action_type,
            "running",
            target=self.target,
            outcome=running_outcome,
            data=dict(self._result_data),
        )

    def handle_navigation_event(self, payload):
        if (
            not self.active
            or payload.get("event") != "navigation"
            or payload.get("action_id") != self.action_id
            or payload.get("mode") != self.mode
        ):
            return False
        status = str(payload.get("status") or "failed")
        if status not in {"succeeded", "failed", "cancelled"}:
            return False
        self._finish(
            status,
            payload.get("outcome") or (
                "navigation_reached" if status == "succeeded"
                else "navigation_failed"
            ),
            payload.get("reason_code"),
            payload.get("data"),
        )
        return True

    async def wait_until_finished(self):
        return await asyncio.shield(self.completion_future)

    async def stop(self, reason_code="USER_REQUESTED"):
        if not self.active:
            return None
        try:
            await self.send_robot_command({
                "command": "stop_moving", "action_id": self.action_id,
            })
        finally:
            return self._finish("cancelled", "stopped", reason_code)

    def _busy_result(self, action_id, target, action_type):
        return ActionResult(
            action_id,
            action_type,
            "already_running",
            target=target,
            outcome="already_running",
            reason_code="NAVIGATION_BUSY",
            retryable=True,
            data={"active_action_id": self.action_id},
        )

    def _finish(self, status, outcome, reason_code=None, data=None):
        result_data = dict(self._result_data)
        result_data.update(data or {})
        result = ActionResult(
            self.action_id or "unassigned",
            self.action_type,
            status,
            target=self.target,
            outcome=outcome,
            reason_code=reason_code,
            retryable=status == "failed",
            data=result_data,
        )
        completion_future = self.completion_future
        self.active = False
        self.mode = None
        self.room_id = None
        if completion_future is not None and not completion_future.done():
            completion_future.set_result(result)
        return result
