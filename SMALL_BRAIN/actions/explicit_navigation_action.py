"""Deterministic local navigation; no rendered map is sent to the LLM."""
from __future__ import annotations

import asyncio
import time
import uuid

from actions.action_result import ActionResult


LOCAL_COMMANDS = {
    "nudge_forward", "nudge_backward", "nudge_left", "nudge_right",
    "rotate_left", "rotate_right", "rotate_around", "furthest_forward",
    "furthest_backward", "open_space_middle", "previous_position",
    "face_person", "exit_room", "go_to_another_room", "go_to_room",
    "return_to_initial_place",
}

ROOM_STEP_COMMANDS = {"exit_room", "go_to_another_room"}


class ExplicitNavigationAction:
    """Ask ROS map logic to resolve and execute one named local movement."""

    NAVIGATION_TIMEOUT = 90.0
    EXIT_MAX_PASSES = 12
    OPEN_SPACE_MAX_PASSES = 4
    # SLAM publishes /map every 10 s and analysis runs at 0.5 Hz. Allow one
    # full publication interval plus the following analysis cycle.
    MAP_REFRESH_TIMEOUT = 15.0

    def __init__(
        self, send_robot_command, request_map_snapshot=None,
        save_map_snapshot=None,
    ):
        self.send_robot_command = send_robot_command
        self.request_map_snapshot = request_map_snapshot
        self.save_map_snapshot = save_map_snapshot
        self.active = False
        self.action_id = None
        self.target = None
        self.completion_future = None
        self._navigation_future = None
        self._worker = None
        self.room_id = None
        self._map_render_data = {}

    async def start(self, command, action_id, room_id=None):
        command = str(command or "").strip().lower()
        if self.active:
            return ActionResult(
                action_id, "explicit_navigation", "already_running",
                reason_code="NAVIGATION_BUSY", retryable=True,
            )
        if command not in LOCAL_COMMANDS:
            return ActionResult(
                action_id, "explicit_navigation", "failed", target=command or None,
                reason_code="UNKNOWN_LOCAL_NAVIGATION_COMMAND",
                data={"valid_commands": sorted(LOCAL_COMMANDS)},
            )
        if command == "go_to_room":
            try:
                room_id = int(room_id)
            except (TypeError, ValueError):
                return ActionResult(
                    action_id, "explicit_navigation", "failed", target=command,
                    reason_code="ROOM_ID_REQUIRED",
                )
            if room_id < 1:
                return ActionResult(
                    action_id, "explicit_navigation", "failed", target=command,
                    reason_code="ROOM_ID_INVALID",
                )

        self.active = True
        self.action_id = action_id
        self.target = command
        self.room_id = room_id
        self._map_render_data = {}
        self.completion_future = asyncio.get_running_loop().create_future()
        self._navigation_future = asyncio.get_running_loop().create_future()
        self._worker = asyncio.create_task(self._run(), name=f"local-nav-{action_id}")
        return ActionResult(
            action_id, "explicit_navigation", "running", target=command,
            outcome="executing_explicit_navigation",
        )

    async def _run(self):
        try:
            if self.target in ROOM_STEP_COMMANDS:
                return await self._run_exit_room()
            if self.target == "open_space_middle":
                return await self._run_open_space_middle()
            payload = {
                "command": "navigate_local",
                "action_id": self.action_id,
                "local_command": self.target,
            }
            if self.target == "go_to_room":
                payload["room_id"] = self.room_id
            feedback = await self.send_robot_command(payload)
            pending_pose = (
                feedback.get("destination")
                if isinstance(feedback, dict) else None
            )
            await self._capture_map_render(pending_pose)
            if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
                return self._finish(
                    "failed", "rejected", "LOCAL_NAVIGATION_REJECTED",
                    {"message": str(feedback)},
                )
            event = await asyncio.wait_for(
                self._navigation_future, self.NAVIGATION_TIMEOUT
            )
            if event.get("status") != "Goal Reached":
                return self._finish(
                    "failed", "navigation_failed", "NAVIGATION_FAILED",
                    {"robot_status": event.get("status")},
                )
            self._finish(
                "succeeded", "explicit_navigation_completed",
                data={"destination": feedback.get("destination")},
            )
        except asyncio.CancelledError:
            return
        except TimeoutError:
            await self.send_robot_command({"command": "stop_moving", "action_id": self.action_id})
            self._finish("failed", "navigation_timeout", "NAVIGATION_TIMEOUT")
        except Exception as error:
            self._finish(
                "failed", "command_failed", "NAVIGATION_COMMAND_FAILED",
                {"error": f"{type(error).__name__}: {error}"},
            )

    async def _run_open_space_middle(self):
        state = None
        after_revision = None
        after_map_received_at = None
        destinations = []
        for pass_index in range(self.OPEN_SPACE_MAX_PASSES):
            feedback = await self._request_open_space_step(
                state, after_revision, after_map_received_at
            )
            if feedback.get("status") != "accepted":
                return self._finish(
                    "failed", "open_space_blocked", "OPEN_SPACE_BLOCKED",
                    {"message": str(feedback), "passes": pass_index},
                )

            state = feedback.get("open_space_state")
            after_revision = feedback.get("map_revision")
            phase = feedback.get("open_space_phase")
            destinations.append({
                "phase": phase,
                "recovery_reason": feedback.get("recovery_reason"),
                "destination": feedback.get("destination"),
            })
            event = await asyncio.wait_for(
                self._navigation_future, self.NAVIGATION_TIMEOUT
            )
            if event.get("status") != "Goal Reached":
                return self._finish(
                    "failed", "navigation_failed", "NAVIGATION_FAILED",
                    {"robot_status": event.get("status"),
                     "passes": pass_index + 1},
                )
            if phase == "move_to_room_core":
                return self._finish(
                    "succeeded", "open_space_middle_reached",
                    data={
                        "passes": pass_index + 1,
                        "destinations": destinations,
                        "destination": feedback.get("destination"),
                    },
                )

            after_map_received_at = time.time_ns()
            self._navigation_future = asyncio.get_running_loop().create_future()

        return self._finish(
            "failed", "open_space_limit", "OPEN_SPACE_PASS_LIMIT",
            {"passes": self.OPEN_SPACE_MAX_PASSES,
             "destinations": destinations},
        )

    async def _request_open_space_step(
        self, state, after_revision, after_map_received_at
    ):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.MAP_REFRESH_TIMEOUT
        while True:
            payload = {
                "command": "navigate_local",
                "action_id": self.action_id,
                "local_command": self.target,
                "open_space_state": state,
            }
            if after_revision is not None:
                payload["after_map_revision"] = after_revision
            if after_map_received_at is not None:
                payload["after_map_received_at_unix_ns"] = (
                    after_map_received_at
                )
            feedback = await self.send_robot_command(payload)
            if not isinstance(feedback, dict):
                await self._capture_map_render()
                return {"status": "error", "message": str(feedback)}
            if feedback.get("status") != "map_updating":
                await self._capture_map_render(feedback.get("destination"))
                return feedback
            if loop.time() >= deadline:
                return {
                    "status": "error",
                    "message": "Map did not refresh after the previous pass",
                }
            await asyncio.sleep(0.25)

    async def _run_exit_room(self):
        state = None
        after_revision = None
        after_map_received_at = None
        destinations = []
        for pass_index in range(self.EXIT_MAX_PASSES):
            feedback = await self._request_exit_step(
                state, after_revision, after_map_received_at
            )
            if feedback.get("status") == "complete":
                entered_room = feedback.get("room")
                outcome = (
                    "room_entered"
                    if self.target == "go_to_another_room"
                    else "room_exit_ready"
                )
                return self._finish(
                    "succeeded", outcome,
                    data={
                        "passes": pass_index,
                        "destinations": destinations,
                        "exit_phase": feedback.get("exit_phase"),
                        "crossing_destination": feedback.get(
                            "crossing_destination"
                        ),
                        "room": entered_room,
                        "room_graph": feedback.get("room_graph"),
                    },
                )
            if feedback.get("status") != "accepted":
                return self._finish(
                    "failed", "exit_room_blocked", "EXIT_ROOM_BLOCKED",
                    {"message": str(feedback), "passes": pass_index},
                )

            state = feedback.get("exit_state")
            after_revision = feedback.get("map_revision")
            destinations.append({
                "phase": feedback.get("exit_phase"),
                "recovery_reason": feedback.get("recovery_reason"),
                "destination": feedback.get("destination"),
            })
            event = await asyncio.wait_for(
                self._navigation_future, self.NAVIGATION_TIMEOUT
            )
            if event.get("status") != "Goal Reached":
                return self._finish(
                    "failed", "navigation_failed", "NAVIGATION_FAILED",
                    {"robot_status": event.get("status"), "passes": pass_index + 1},
                )
            after_map_received_at = time.time_ns()
            self._navigation_future = asyncio.get_running_loop().create_future()

        self._finish(
            "failed", "exit_room_limit", "EXIT_ROOM_PASS_LIMIT",
            {"passes": self.EXIT_MAX_PASSES, "destinations": destinations},
        )

    async def _request_exit_step(
        self, state, after_revision, after_map_received_at
    ):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.MAP_REFRESH_TIMEOUT
        while True:
            payload = {
                "command": "navigate_local",
                "action_id": self.action_id,
                "local_command": self.target,
                "exit_state": state,
            }
            if after_revision is not None:
                payload["after_map_revision"] = after_revision
            if after_map_received_at is not None:
                payload["after_map_received_at_unix_ns"] = (
                    after_map_received_at
                )
            feedback = await self.send_robot_command(payload)
            if not isinstance(feedback, dict):
                await self._capture_map_render()
                return {"status": "error", "message": str(feedback)}
            if feedback.get("status") != "map_updating":
                pending_pose = (
                    feedback.get("destination")
                    or feedback.get("crossing_destination")
                )
                await self._capture_map_render(pending_pose)
                return feedback
            if loop.time() >= deadline:
                return {
                    "status": "error",
                    "message": "Map did not refresh after the previous pass",
                }
            await asyncio.sleep(0.25)

    async def _capture_map_render(self, pending_navigation_pose=None):
        """Replace the saved map with the render used by this map action."""
        if self.request_map_snapshot is None or self.save_map_snapshot is None:
            return
        request_id = uuid.uuid4().hex
        try:
            request = {
                "schema_version": 1,
                "operation": "snapshot",
                "action_id": self.action_id,
                "request_id": request_id,
                "render_frontiers": True,
            }
            if isinstance(pending_navigation_pose, dict):
                request["pending_navigation_pose"] = pending_navigation_pose
            snapshot = await self.request_map_snapshot(request)
            metadata = snapshot.get("metadata") or {}
            if str(metadata.get("snapshot_request_id")) != request_id:
                raise RuntimeError("Map service returned the wrong request")
            saved = self.save_map_snapshot(snapshot)
            if saved is not None:
                self._map_render_data["map_renders"] = saved
            self._map_render_data.pop("map_render_save_error", None)
        except Exception as error:
            # Debug persistence must not prevent deterministic navigation.
            self._map_render_data["map_render_save_error"] = (
                f"{type(error).__name__}: {error}"
            )

    def handle_navigation_event(self, payload):
        if (
            self.active and payload.get("event") == "navigation"
            and payload.get("action_id") == self.action_id
            and self._navigation_future is not None
            and not self._navigation_future.done()
        ):
            self._navigation_future.set_result(dict(payload))
            return True
        return False

    async def wait_until_finished(self):
        return await asyncio.shield(self.completion_future)

    async def stop(self, reason_code="USER_REQUESTED"):
        if not self.active:
            return None
        await self.send_robot_command({"command": "stop_moving", "action_id": self.action_id})
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        return self._finish("cancelled", "stopped", reason_code)

    def _finish(self, status, outcome, reason_code=None, data=None):
        result_data = dict(self._map_render_data)
        result_data.update(data or {})
        result = ActionResult(
            self.action_id or "unassigned", "explicit_navigation", status,
            target=self.target, outcome=outcome, reason_code=reason_code,
            retryable=status == "failed", data=result_data,
        )
        self.active = False
        self.room_id = None
        if self.completion_future is not None and not self.completion_future.done():
            self.completion_future.set_result(result)
        return result
