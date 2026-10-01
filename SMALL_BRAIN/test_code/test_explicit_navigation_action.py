import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actions.explicit_navigation_action import (
    LOCAL_COMMANDS,
    ExplicitNavigationAction,
)


class ExplicitNavigationActionTests(unittest.IsolatedAsyncioTestCase):
    async def test_map_using_action_overwrites_latest_render(self):
        action = None
        requested = []

        async def request_map_snapshot(payload):
            requested.append(dict(payload))
            return {
                "jpeg_bytes": b"map",
                "metadata": {
                    "snapshot_request_id": payload["request_id"],
                },
            }

        async def send_robot_command(_payload):
            asyncio.get_running_loop().call_soon(
                action.handle_navigation_event,
                {
                    "event": "navigation", "action_id": "nudge-1",
                    "status": "Goal Reached",
                },
            )
            return {"status": "accepted", "destination": {"x": 1.0}}

        save_map_snapshot = Mock(
            return_value={"map_crop": "latest_map_crop.jpg"}
        )
        action = ExplicitNavigationAction(
            send_robot_command,
            request_map_snapshot=request_map_snapshot,
            save_map_snapshot=save_map_snapshot,
        )

        await action.start("nudge_forward", "nudge-1")
        result = await action.wait_until_finished()

        self.assertEqual(requested[0]["action_id"], "nudge-1")
        self.assertEqual(requested[0]["operation"], "snapshot")
        self.assertTrue(requested[0]["render_frontiers"])
        self.assertEqual(
            requested[0]["pending_navigation_pose"], {"x": 1.0}
        )
        save_map_snapshot.assert_called_once()
        snapshot = save_map_snapshot.call_args.args[0]
        self.assertEqual(snapshot["jpeg_bytes"], b"map")
        self.assertEqual(
            result.data["map_renders"]["map_crop"],
            "latest_map_crop.jpg",
        )

    async def test_exit_room_refreshes_saved_render_for_each_map_step(self):
        snapshots = []
        saves = []

        async def request_map_snapshot(payload):
            snapshots.append(dict(payload))
            return {
                "jpeg_bytes": f"map-{len(snapshots)}".encode(),
                "metadata": {
                    "snapshot_request_id": payload["request_id"],
                },
            }

        responses = [
            {
                "status": "accepted", "exit_phase": "explore_room_frontier",
                "exit_state": {"passes": 1}, "map_revision": 10,
                "destination": {"x": 1.0, "y": 0.0},
            },
            {
                "status": "complete", "exit_phase": "crossing_ready",
                "exit_state": {"passes": 2}, "map_revision": 20,
                "crossing_destination": {"x": 2.0, "y": 0.0},
            },
        ]
        action = None

        async def send_robot_command(_payload):
            response = responses.pop(0)
            if response["status"] == "accepted":
                asyncio.get_running_loop().call_soon(
                    action.handle_navigation_event,
                    {
                        "event": "navigation", "action_id": "exit-save",
                        "status": "Goal Reached",
                    },
                )
            return response

        action = ExplicitNavigationAction(
            send_robot_command,
            request_map_snapshot=request_map_snapshot,
            save_map_snapshot=lambda item: saves.append(item["jpeg_bytes"]),
        )

        await action.start("exit_room", "exit-save")
        result = await action.wait_until_finished()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(saves, [b"map-1", b"map-2"])
        self.assertEqual(
            snapshots[0]["pending_navigation_pose"],
            {"x": 1.0, "y": 0.0},
        )
        self.assertEqual(
            snapshots[1]["pending_navigation_pose"],
            {"x": 2.0, "y": 0.0},
        )

    async def test_open_space_explores_then_moves_to_room_core(self):
        responses = [
            {
                "status": "accepted",
                "open_space_phase": "stabilize_room_map",
                "open_space_state": {"recovery_moves": 1, "passes": 1},
                "recovery_reason": "no_room_core",
                "map_revision": 10,
                "destination": {"x": 0.5, "y": 0.0},
            },
            {"status": "map_updating"},
            {
                "status": "accepted",
                "open_space_phase": "move_to_room_core",
                "open_space_state": {"recovery_moves": 1, "passes": 2},
                "map_revision": 20,
                "destination": {"x": 1.5, "y": 0.5},
            },
        ]
        requests = []
        action = None

        async def send_robot_command(payload):
            requests.append(dict(payload))
            response = responses.pop(0)
            if response["status"] == "accepted":
                asyncio.get_running_loop().call_soon(
                    action.handle_navigation_event,
                    {
                        "event": "navigation", "action_id": "open-1",
                        "status": "Goal Reached",
                    },
                )
            return response

        action = ExplicitNavigationAction(send_robot_command)
        with patch(
            "actions.explicit_navigation_action.asyncio.sleep",
            new=AsyncMock(),
        ):
            await action.start("open_space_middle", "open-1")
            result = await action.wait_until_finished()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.outcome, "open_space_middle_reached")
        self.assertEqual(result.data["passes"], 2)
        self.assertEqual(result.data["destination"]["x"], 1.5)
        self.assertEqual(requests[1]["after_map_revision"], 10)
        self.assertEqual(requests[2]["after_map_revision"], 10)
        self.assertEqual(
            requests[2]["open_space_state"]["recovery_moves"], 1
        )

    async def test_exit_room_stops_when_crossing_is_ready(self):
        responses = [
            {
                "status": "accepted", "exit_phase": "explore_room_frontier",
                "exit_state": {"passes": 1}, "map_revision": 10,
                "destination": {"x": 1.0, "y": 0.0},
            },
            {"status": "map_updating"},
            {
                "status": "complete", "exit_phase": "crossing_ready",
                "exit_state": {"passes": 2}, "map_revision": 20,
                "crossing_destination": {"x": 2.0, "y": 0.0},
                "room_graph": {"rooms": [{"room_id": 1}]},
            },
        ]
        requests = []
        action = None

        async def send_robot_command(payload):
            requests.append(dict(payload))
            response = responses.pop(0)
            if response["status"] == "accepted":
                asyncio.get_running_loop().call_soon(
                    action.handle_navigation_event,
                    {
                        "event": "navigation", "action_id": "exit-1",
                        "status": "Goal Reached",
                    },
                )
            return response

        action = ExplicitNavigationAction(send_robot_command)
        with patch("actions.explicit_navigation_action.asyncio.sleep", new=AsyncMock()):
            started = await action.start("exit_room", "exit-1")
            result = await action.wait_until_finished()

        self.assertEqual(started.status, "running")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.outcome, "room_exit_ready")
        self.assertEqual(result.data["passes"], 1)
        self.assertEqual(result.data["exit_phase"], "crossing_ready")
        self.assertEqual(result.data["crossing_destination"]["x"], 2.0)
        self.assertEqual(requests[1]["after_map_revision"], 10)
        self.assertEqual(requests[2]["after_map_revision"], 10)
        self.assertIn("after_map_received_at_unix_ns", requests[1])
        self.assertEqual(
            requests[2]["after_map_received_at_unix_ns"],
            requests[1]["after_map_received_at_unix_ns"],
        )

    async def test_go_to_another_room_crosses_and_returns_registered_room(self):
        responses = [
            {
                "status": "accepted", "exit_phase": "cross_room_boundary",
                "exit_state": {"passes": 1, "origin_room_id": 1},
                "map_revision": 10,
                "destination": {"x": 2.0, "y": 0.0},
            },
            {
                "status": "complete", "exit_phase": "outside",
                "exit_state": {"passes": 1, "origin_room_id": 1},
                "map_revision": 20,
                "room": {"room_id": 2, "name": "Room 2"},
                "room_graph": {
                    "rooms": [{"room_id": 1}, {"room_id": 2}],
                    "connections": [[1, 2]],
                },
            },
        ]
        requests = []
        action = None

        async def send_robot_command(payload):
            requests.append(dict(payload))
            response = responses.pop(0)
            if response["status"] == "accepted":
                asyncio.get_running_loop().call_soon(
                    action.handle_navigation_event,
                    {
                        "event": "navigation", "action_id": "room-2",
                        "status": "Goal Reached",
                    },
                )
            return response

        action = ExplicitNavigationAction(send_robot_command)
        started = await action.start("go_to_another_room", "room-2")
        result = await action.wait_until_finished()

        self.assertEqual(started.status, "running")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.outcome, "room_entered")
        self.assertEqual(result.data["room"]["room_id"], 2)
        self.assertEqual(result.data["room_graph"]["connections"], [[1, 2]])
        self.assertEqual(requests[1]["after_map_revision"], 10)
        self.assertIn("after_map_received_at_unix_ns", requests[1])

    async def test_go_to_room_requires_and_sends_room_id(self):
        action = ExplicitNavigationAction(AsyncMock())
        missing = await action.start("go_to_room", "known-room")
        self.assertEqual(missing.status, "failed")
        self.assertEqual(missing.reason_code, "ROOM_ID_REQUIRED")

        requests = []
        action = None

        async def send_robot_command(payload):
            requests.append(dict(payload))
            asyncio.get_running_loop().call_soon(
                action.handle_navigation_event,
                {
                    "event": "navigation", "action_id": "known-room",
                    "status": "Goal Reached",
                },
            )
            return {"status": "accepted", "destination": {"x": 1.0}}

        action = ExplicitNavigationAction(send_robot_command)
        await action.start("go_to_room", "known-room", room_id=2)
        result = await action.wait_until_finished()
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(requests[0]["room_id"], 2)

    def test_removed_wall_and_corner_commands_are_not_public(self):
        self.assertNotIn("nearest_wall", LOCAL_COMMANDS)
        self.assertNotIn("nearest_corner", LOCAL_COMMANDS)
        self.assertIn("exit_room", LOCAL_COMMANDS)
        self.assertIn("go_to_another_room", LOCAL_COMMANDS)
        self.assertIn("return_to_initial_place", LOCAL_COMMANDS)


if __name__ == "__main__":
    unittest.main()
