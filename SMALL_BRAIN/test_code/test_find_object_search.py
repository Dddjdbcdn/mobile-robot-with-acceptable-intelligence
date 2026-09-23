import asyncio
import math
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import AsyncMock, Mock, call

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BIG_BRAIN = ROOT.parent / "BIG_BRAIN" / "src" / "robot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BIG_BRAIN))

from actions.navigate_action import (
    MAP_GUIDANCE,
    MODE_GUIDANCE,
    NavigateAction,
    NavigationError,
)
from actions.action_result import ActionResult
from actions.approach_action import ApproachAction
from actions.search_action import SearchAction
from actions.track_action import TrackAction
from cognition.goal_executor import (
    FindLoopConfig,
    GoalExecutor,
)
from cognition.follow_executor import FollowConfig, FollowExecutor
from cognition.state import robot_state
from robot.map_logic import LogicConfig, MapLogic
from robot.map_renderer import MapRenderer


def candidate(candidate_id, kind, x, y):
    return {
        "id": candidate_id,
        "kind": kind,
        "x": x,
        "y": y,
        "yaw": math.atan2(y, x),
        "frame_id": "map",
        "distance_m": math.hypot(x, y),
    }


class FindObjectCandidateTests(unittest.TestCase):
    class Grid:
        resolution = 0.1

        def __init__(self):
            self.data = np.zeros((61, 61), dtype=np.int16)

        def cells(self, x, y):
            return (
                np.floor((np.asarray(x) + 3.0) / self.resolution).astype(int),
                np.floor((np.asarray(y) + 3.0) / self.resolution).astype(int),
            )

        def world(self, col, row):
            return (
                (np.asarray(col) + 0.5) * self.resolution - 3.0,
                (np.asarray(row) + 0.5) * self.resolution - 3.0,
            )

        def inside(self, col, row):
            return (
                (col >= 0) & (row >= 0)
                & (col < self.data.shape[1])
                & (row < self.data.shape[0])
            )

    def setUp(self):
        self.logic = MapLogic()
        self.grid = self.Grid()
        self.pose = {
            "x": 0.0, "y": 0.0, "yaw": 0.0, "frame_id": "map"
        }
        self.prepared = {
            "grid": self.grid,
            "reachable": np.ones_like(self.grid.data, dtype=bool),
            "frontiers": [],
        }

    def context_overlay(self):
        return {
            "mode": "context",
            "camera_horizontal_fov_deg": 60.0,
            "camera_reliable_range_m": 2.0,
            "observation": {
                "robot_pose": self.pose,
                "pan_angle": 95.0,
            },
            "search_poses": [],
        }

    def destination_overlay(self):
        overlay = self.context_overlay()
        overlay.update({"mode": "destination", "search_poses": []})
        overlay["observation"]["candidate_type"] = "destination"
        return overlay

    def test_goal_candidates_do_not_require_an_overlay(self):
        selected = self.logic.plan_candidates(self.prepared, self.pose)

        self.assertTrue(any(item["kind"] == "local" for item in selected))
        self.assertEqual(
            len([item for item in selected if item["kind"] == "rotation"]),
            3,
        )

    def test_context_samples_the_clue_cone_and_one_backup(self):
        selected = self.logic.plan_candidates(
            self.prepared, self.pose, self.context_overlay()
        )
        cone = [item for item in selected if item["id"] != "CB"]
        self.assertTrue(cone)
        self.assertEqual(selected[-1]["id"], "CB")
        self.assertEqual(
            selected[-1]["selection_reason"], "reverse_for_wider_view"
        )
        for item in cone:
            bearing = math.atan2(item["y"], item["x"])
            self.assertLessEqual(abs(bearing), math.radians(30.0))
            self.assertEqual(item["kind"], "context")

    def test_context_sampling_parameters_come_from_config(self):
        logic = MapLogic(LogicConfig(
            context_radius_samples=1,
            context_bearing_fractions=(0.0,),
            context_backup_distance_m=0.25,
        ))

        selected = logic.plan_candidates(
            self.prepared, self.pose, self.context_overlay()
        )

        cone = [
            item for item in selected
            if item["id"].startswith("C") and item["id"] not in ("CB", "CF")
        ]
        self.assertEqual(len(cone), 1)
        self.assertAlmostEqual(cone[0]["x"], 0.0)
        self.assertAlmostEqual(cone[0]["y"], 0.0)
        self.assertAlmostEqual(selected[-1]["x"], -0.25)
        self.assertAlmostEqual(selected[-1]["y"], 0.0)

    def approach_destination(self, target_x, target_y, **overrides):
        config = LogicConfig()
        parameters = {
            "approach_ray_radius_m": 0.18,
            "approach_obstacle_standoff_m": 0.20,
            "approach_obstacle_min_cells": 3,
        }
        parameters.update(overrides)
        for name, value in parameters.items():
            setattr(config, name, value)
        self.prepared["frame_id"] = "map"
        return MapLogic(config).resolve_approach_destination(
            self.prepared,
            self.pose,
            target_x,
            target_y,
        )

    def add_approach_obstacle(self, x, y, half_width=0.05):
        x1 = int(self.grid.cells(x - half_width, y)[0])
        x2 = int(self.grid.cells(x + half_width, y)[0])
        y1 = int(self.grid.cells(x, y - half_width)[1])
        y2 = int(self.grid.cells(x, y + half_width)[1])
        self.grid.data[y1:y2 + 1, x1:x2 + 1] = 100

    def test_approach_ray_applies_standoff_before_clear_target(self):
        self.add_approach_obstacle(1.0, 0.0, half_width=0.10)

        destination = self.approach_destination(0.50, 0.0)

        self.assertFalse(destination["was_clamped"])
        self.assertEqual(destination["resolution"], "standoff_from_target")
        self.assertAlmostEqual(destination["x"], 0.30, places=2)

    def test_approach_ray_uses_standoff_for_target_on_obstacle(self):
        self.add_approach_obstacle(1.0, 0.0, half_width=0.10)

        destination = self.approach_destination(1.05, 0.0)

        self.assertTrue(destination["was_clamped"])
        self.assertEqual(destination["resolution"], "standoff_from_obstacle")
        self.assertLess(destination["x"], 1.0)
        self.assertIsNotNone(destination["obstacle_distance_m"])

    def test_approach_ray_clamps_far_return_before_obstacle(self):
        self.add_approach_obstacle(1.0, 0.0, half_width=0.10)

        destination = self.approach_destination(2.0, 0.0)

        self.assertTrue(destination["was_clamped"])
        self.assertEqual(destination["resolution"], "standoff_from_obstacle")
        self.assertLess(destination["resolved_distance_m"], 1.0)
        self.assertAlmostEqual(
            destination["resolved_distance_m"],
            destination["obstacle_distance_m"] - 0.20,
            places=6,
        )

    def test_approach_ray_inflation_catches_off_axis_obstacle(self):
        self.add_approach_obstacle(1.0, 0.15, half_width=0.08)

        destination = self.approach_destination(
            2.0, 0.0, approach_ray_radius_m=0.20
        )

        self.assertTrue(destination["was_clamped"])
        self.assertLess(destination["obstacle_distance_m"], 1.1)

    def test_approach_ray_ignores_tiny_obstacle_component(self):
        col, row = self.grid.cells(1.0, 0.0)
        self.grid.data[int(row), int(col)] = 100

        destination = self.approach_destination(
            2.0, 0.0, approach_obstacle_min_cells=2
        )

        self.assertFalse(destination["was_clamped"])
        self.assertIsNone(destination["obstacle_distance_m"])

    def test_approach_ray_moves_blocked_target_to_reachable_cell(self):
        target_x, target_y = 0.50, 0.0
        standoff_x = target_x - 0.20
        col, row = self.grid.cells(standoff_x, target_y)
        self.prepared["reachable"][int(row), int(col)] = False

        destination = self.approach_destination(target_x, target_y)

        dest_col, dest_row = self.grid.cells(destination["x"], destination["y"])
        self.assertTrue(destination["moved_to_reachable"])
        self.assertTrue(
            self.prepared["reachable"][int(dest_row), int(dest_col)]
        )
        self.assertAlmostEqual(destination["y"], 0.0)
        self.assertGreaterEqual(destination["x"], 0.0)
        self.assertLess(destination["x"], standoff_x)
        self.assertAlmostEqual(destination["angle"], 0.0)

    def test_context_excludes_frontiers_but_keeps_furthest_point(self):
        self.prepared["frontiers"] = [
            candidate("F1", "frontier", 2.8, 0.4),
            candidate("F2", "frontier", 0.0, 2.8),
        ]

        selected = self.logic.plan_candidates(
            self.prepared, self.pose, self.context_overlay()
        )

        selected_ids = {item["id"] for item in selected}
        self.assertIn("CF", selected_ids)
        self.assertNotIn("F1", selected_ids)
        self.assertNotIn("F2", selected_ids)

    def test_context_adds_furthest_reachable_center_ray_candidate(self):
        # The center ray is free through x=2.45 and blocked from x=2.5.
        self.grid.data[:, 55:] = 100
        self.prepared["reachable"][:, 55:] = False

        selected = self.logic.plan_candidates(
            self.prepared, self.pose, self.context_overlay()
        )

        farthest = next(item for item in selected if item["id"] == "CF")
        self.assertEqual(
            farthest["selection_reason"],
            "furthest_reachable_on_center_ray",
        )
        self.assertAlmostEqual(farthest["x"], 2.45)
        self.assertAlmostEqual(farthest["y"], 0.05)
        self.assertAlmostEqual(farthest["yaw"], 0.0)

    def test_local_candidate_metadata_is_inherited_and_frontiers_are_filtered(self):
        self.prepared["frontiers"] = [
            candidate("F1", "frontier", 2.8, 0.0),
        ]
        overlay = self.context_overlay()
        overlay["observation"].update({
            "candidate_type": "local",
            "movement_limit": 2,
            "movements_used": 0,
            "remaining_waypoints": 2,
        })

        selected = self.logic.plan_candidates(self.prepared, self.pose, overlay)

        self.assertTrue(selected)
        self.assertIn("CF", {item["id"] for item in selected})
        self.assertNotIn("F1", {item["id"] for item in selected})
        self.assertTrue(all(item["candidate_type"] == "local" for item in selected))
        self.assertTrue(all(item["remaining_waypoints"] == 2 for item in selected))

    def test_destination_uses_wide_context_candidates_without_frontiers(self):
        self.prepared["frontiers"] = [
            candidate("F1", "frontier", 2.8, 0.0),
        ]
        self.prepared["doors"] = [{
            "id": "D1", "confirmed": True, "x": 1.0, "y": 0.0,
        }]

        selected = self.logic.plan_candidates(
            self.prepared, self.pose, self.destination_overlay()
        )

        self.assertGreater(len(selected), 1)
        self.assertTrue(all(item["kind"] == "context" for item in selected))
        self.assertNotIn("F1", {item["id"] for item in selected})
        self.assertGreater(max(item["distance_m"] for item in selected), 2.0)
        self.assertLessEqual(
            max(item["distance_m"] for item in selected),
            self.logic.cfg.destination_context_radius_m + self.grid.resolution,
        )

    def _exploration_options_at(self, x, y):
        col, row = self.grid.cells(x, y)
        coverage = np.ones_like(self.grid.data, dtype=bool)
        with (
            unittest.mock.patch.object(
                self.logic,
                "_exploration_samples",
                return_value=(np.asarray([row]), np.asarray([col])),
            ),
            unittest.mock.patch.object(
                self.logic,
                "_best_unseen_view",
                return_value=(0.0, 10, 1.0),
            ),
        ):
            return self.logic._exploration_candidates(
                self.prepared, self.pose,
                {"camera_horizontal_fov_deg": 60.0}, coverage,
            )

    def test_exploration_offers_near_and_far_for_forward_candidate(self):
        options = self._exploration_options_at(0.5, 0.0)

        self.assertEqual([item["id"] for item in options], ["E1", "EF"])
        self.assertGreater(options[1]["x"], options[0]["x"])
        self.assertEqual(
            options[1]["selection_reason"], "furthest_reachable_forward_ray"
        )
        self.assertNotIn("uncovered_cell_count", options[1])
        self.assertNotIn("ranking_score", options[1])

    def test_exploration_does_not_offer_far_pose_outside_center_cone(self):
        options = self._exploration_options_at(0.0, 0.5)

        self.assertEqual([item["id"] for item in options], ["E1"])

    def test_exploration_far_pose_stops_before_unreachable_cell(self):
        obstacle_col, _ = self.grid.cells(1.0, 0.0)
        self.prepared["reachable"][:, int(obstacle_col)] = False

        options = self._exploration_options_at(0.5, 0.0)

        self.assertEqual([item["id"] for item in options], ["E1", "EF"])
        self.assertLess(options[1]["x"], 1.0)

    def test_exploration_heading_reveals_the_selected_unseen_area(self):
        coverage = np.ones_like(self.grid.data, dtype=bool)
        coverage[30:51, 24:37] = False
        winner = self.logic._exploration_candidates(
            self.prepared, self.pose, {"search_poses": []}, coverage
        )[0]
        unseen = ~coverage
        visible = self.logic.visibility_mask(
            self.grid, winner, winner["yaw"], 60.0, 2.0
        )
        opposite = self.logic.visibility_mask(
            self.grid, winner, winner["yaw"] + math.pi, 60.0, 2.0
        )
        self.assertEqual(
            np.count_nonzero(visible & unseen),
            winner["uncovered_cell_count"],
        )
        self.assertGreaterEqual(
            np.count_nonzero(visible & unseen),
            np.count_nonzero(opposite & unseen),
        )

    def test_exploration_samples_only_seen_cells_plus_robot(self):
        coverage = np.zeros_like(self.grid.data, dtype=bool)
        coverage[10, 10] = True
        coverage[20, 20] = True
        unseen = ~coverage

        rows, cols = self.logic._exploration_samples(
            self.grid, self.prepared["reachable"], unseen, coverage, self.pose
        )

        robot_col, robot_row = self.grid.cells(self.pose["x"], self.pose["y"])
        sampled = set(zip(rows.tolist(), cols.tolist()))
        self.assertEqual(sampled, {
            (10, 10),
            (20, 20),
            (int(robot_row), int(robot_col)),
        })

    def test_exploration_does_not_fall_back_to_frontiers(self):
        self.prepared["frontiers"] = [{
            "id": "F1", "kind": "frontier",
            "x": 0.1, "y": 0.0, "yaw": 0.0,
            "information_gain_m2": 2.0,
        }]
        coverage = np.ones_like(self.grid.data, dtype=bool)

        options = self.logic._exploration_candidates(
            self.prepared, self.pose, {"search_poses": []}, coverage
        )

        self.assertEqual(options, [])


class UnifiedSearchPoseTests(unittest.TestCase):
    def setUp(self):
        self.executor = GoalExecutor.__new__(GoalExecutor)
        self.executor.find_loop = FindLoopConfig()
        self.executor._search_waypoint_count = 2
        self.executor._search_poses = []

    def test_views_share_pose_but_robot_rotation_creates_new_pose(self):
        pose = {
            "x": 1.0, "y": 2.0, "yaw": 0.0,
            "frame_id": "map", "kind": "rotation",
        }
        self.executor._remember_search_pose("arrived", pose=pose)
        frames = [
            {"robot_pose": pose, "pan_angle": 35.0, "tilt_angle": 90.0},
            {"robot_pose": pose, "pan_angle": 155.0, "tilt_angle": 90.0},
        ]
        self.executor._remember_search_pose(
            "scan", frames=frames
        )
        self.executor._remember_search_pose(
            "scan", frames=frames[:1]
        )
        self.executor._remember_search_pose(
            "turned",
            pose={**pose, "yaw": math.pi / 2.0},
        )

        self.assertEqual(len(self.executor._search_poses), 2)
        self.assertEqual(len(self.executor._search_poses[0]["views"]), 2)
        self.assertEqual(self.executor._search_poses[0]["kind"], "rotation")
        self.assertAlmostEqual(
            self.executor._search_poses[1]["yaw"], math.pi / 2.0
        )


class NavigationSearchPoseTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _navigation_for_capture(camera, request_snapshot=None):
        navigation = NavigateAction.__new__(NavigateAction)
        navigation.camera = camera
        navigation.request_map_snapshot = request_snapshot
        navigation.action_id = "find-1:explore-1"
        navigation._overlay_action_id = "find-1"
        navigation._overlay_revision = 2
        navigation._mode = "exploration"
        navigation._save_map_snapshot = Mock()
        return navigation

    async def test_capture_uses_the_exact_map_reply(self):
        async def request_snapshot(payload):
            return {
                "jpeg_bytes": b"jpeg",
                "metadata": {
                    "snapshot_id": "snapshot-1",
                    "snapshot_request_id": payload["request_id"],
                    "candidates": [candidate("F1", "frontier", 1.0, 0.0)],
                    "search_overlay": {"action_id": "find-1", "revision": 2},
                },
            }
        camera = Mock()
        navigation = self._navigation_for_capture(camera, request_snapshot)

        captured, camera_jpeg = await navigation._capture()

        self.assertEqual(captured["metadata"]["snapshot_id"], "snapshot-1")
        self.assertIsNone(camera_jpeg)
        camera.snapshot.assert_not_called()
        navigation._save_map_snapshot.assert_called_once_with(captured)

    async def test_two_pose_exploration_centers_and_captures_camera(self):
        async def request_snapshot(payload):
            return {
                "jpeg_bytes": b"map-jpeg",
                "metadata": {
                    "snapshot_id": "snapshot-2",
                    "snapshot_request_id": payload["request_id"],
                    "candidates": [
                        candidate("E1", "exploration", 0.5, 0.0),
                        candidate("EF", "exploration", 2.0, 0.0),
                    ],
                    "search_overlay": {"action_id": "find-1", "revision": 2},
                },
            }

        frame = type("Frame", (), {
            "captured_at": float("inf"),
            "tracking_bgr": np.zeros((4, 4, 3), dtype=np.uint8),
        })()
        camera = Mock()
        camera.snapshot.return_value = frame
        navigation = self._navigation_for_capture(camera, request_snapshot)
        navigation.see_action = Mock()
        navigation.see_action.move_to_region = AsyncMock(
            return_value=ActionResult(
                "camera", "move_camera", "succeeded"
            )
        )

        captured, camera_jpeg = await navigation._capture()

        self.assertEqual(captured["metadata"]["snapshot_id"], "snapshot-2")
        self.assertTrue(camera_jpeg)
        navigation.see_action.move_to_region.assert_awaited_once_with(
            region="center",
            action_id="find-1:explore-1:center-camera",
        )
        camera.snapshot.assert_called_once_with()

    async def test_capture_reports_persistent_map_stream_failure(self):
        async def request_snapshot(_payload):
            raise RuntimeError("Map is not ready")

        navigation = self._navigation_for_capture(Mock(), request_snapshot)

        with self.assertRaises(NavigationError) as raised:
            await navigation._capture()

        self.assertEqual(raised.exception.code, "MAP_STREAM_UNAVAILABLE")
        self.assertIn("Map is not ready", str(raised.exception))

    async def test_capture_requests_a_matching_standalone_snapshot(self):
        request = {}

        async def request_snapshot(payload):
            request.update(payload)
            return {
                "jpeg_bytes": b"jpeg",
                "metadata": {
                    "snapshot_id": "snapshot-1",
                    "snapshot_request_id": payload["request_id"],
                    "candidates": [candidate("F1", "frontier", 1.0, 0.0)],
                    "search_overlay": None,
                },
            }

        navigation = self._navigation_for_capture(Mock(), request_snapshot)
        navigation._overlay_action_id = None
        navigation.action_id = "navigate-1"
        navigation.request_map_snapshot = request_snapshot

        captured, camera_jpeg = await navigation._capture()

        self.assertEqual(request["operation"], "snapshot")
        self.assertEqual(request["action_id"], "navigate-1")
        self.assertNotIn("crop_size_m", request)
        self.assertEqual(
            captured["metadata"]["snapshot_request_id"],
            request["request_id"],
        )
        self.assertIsNone(camera_jpeg)

    async def test_goal_capture_publishes_observed_pose_and_refreshes_map(self):
        requests = []

        async def request_snapshot(payload):
            requests.append(dict(payload))
            return {
                "jpeg_bytes": b"map-jpeg",
                "metadata": {
                    "snapshot_id": f"snapshot-{len(requests)}",
                    "snapshot_request_id": payload["request_id"],
                    "robot_pose": {
                        "x": 1.0, "y": 2.0, "yaw": 0.25,
                        "frame_id": "map",
                    },
                    "candidates": [candidate("1", "local", 1.5, 2.0)],
                    "search_overlay": {
                        "action_id": "navigate-1",
                        "revision": payload["overlay_revision"],
                    },
                },
            }

        frame = type("Frame", (), {
            "captured_at": float("inf"),
            "tracking_bgr": np.zeros((4, 4, 3), dtype=np.uint8),
        })()
        camera = Mock()
        camera.snapshot.return_value = frame
        navigation = self._navigation_for_capture(camera, request_snapshot)
        navigation.action_id = "navigate-1"
        navigation._mode = "goal"
        navigation._owns_goal_overlay = True
        navigation._goal_search_poses = []
        navigation._overlay_action_id = "navigate-1"
        navigation._overlay_revision = 0
        navigation.map_crop_size_m = 4.0
        navigation.send_map_overlay = AsyncMock()

        captured, camera_jpeg = await navigation._capture()

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["overlay_revision"], 0)
        self.assertEqual(requests[1]["overlay_revision"], 1)
        self.assertEqual(captured["metadata"]["snapshot_id"], "snapshot-2")
        self.assertTrue(camera_jpeg)
        overlay = navigation.send_map_overlay.await_args.args[0]
        self.assertEqual(overlay["operation"], "set")
        self.assertEqual(overlay["mode"], "goal")
        self.assertEqual(overlay["revision"], 1)
        self.assertEqual(overlay["search_poses"][0]["reason"], "navigation_start")
        self.assertEqual(overlay["search_poses"][0]["x"], 1.0)
        self.assertEqual(overlay["search_poses"][0]["y"], 2.0)

    async def test_capture_rejects_a_mismatched_context_overlay(self):
        async def request_snapshot(payload):
            return {
                "jpeg_bytes": b"jpeg",
                "metadata": {
                    "snapshot_id": "snapshot-context",
                    "snapshot_request_id": payload["request_id"],
                    "candidates": [candidate("F1", "frontier", 1.0, 0.0)],
                    "search_overlay": {
                        "action_id": "another-find",
                        "revision": 1,
                    },
                },
            }
        navigation = self._navigation_for_capture(Mock(), request_snapshot)
        navigation.action_id = "find-1:context-1"

        with self.assertRaises(NavigationError) as raised:
            await navigation._capture()

        self.assertEqual(raised.exception.code, "MAP_OVERLAY_TIMEOUT")

    async def test_navigation_does_not_record_commanded_destination(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor.action_id = "find-1"
        executor.target = "bottle"
        executor._search_waypoint_count = 0
        executor._completed_steps = []
        executor._publish_map_overlay = AsyncMock(return_value=1)
        executor._remember_search_pose = Mock()
        executor.navigate_action = type("Navigation", (), {})()
        executor.navigate_action.start_find_object_step = AsyncMock(
            return_value=ActionResult(
                "nav", "navigate_action", "succeeded",
                data={"destination": {"x": 1.0, "y": 2.0, "yaw": 0.0}},
            )
        )

        result = await executor._navigate_search_step(
            None, mode="exploration", step="explore_1"
        )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(executor._search_waypoint_count, 1)
        self.assertEqual(executor._completed_steps, ["explore_1"])
        executor._remember_search_pose.assert_not_called()


class DeterministicNavigationSelectionTests(unittest.TestCase):
    def test_navigation_consumes_map_planner_winner_without_reranking(self):
        navigation = NavigateAction.__new__(NavigateAction)
        navigation._candidates = {
            "7": {
                **candidate("7", "local", 1.0, 0.0),
                "selection_reason": "uncovered_local_view",
                "ranking_score": 2.4,
            }
        }
        selection = navigation._deterministic_selection({
            "metadata": {"selected_pose_id": "7"}
        })
        self.assertEqual(selection["pose_id"], "7")
        self.assertEqual(selection["source"], "map_planner")
        self.assertEqual(selection["ranking_score"], 2.4)


class ContextualClueContractTests(unittest.TestCase):
    def test_candidate_requires_clue_and_type(self):
        self.assertTrue(SearchAction._contextual_clue_is_valid({
            "result": "candidate",
            "candidate_type": "local",
            "contextual_clue": "A desk surface is visible on the right.",
        }))
        self.assertFalse(SearchAction._contextual_clue_is_valid({
            "result": "candidate",
            "contextual_clue": "A desk surface is visible on the right.",
        }))
        self.assertFalse(SearchAction._contextual_clue_is_valid({
            "result": "not_found",
            "candidate_type": None,
            "contextual_clue": None,
        }))

    def test_local_and_destination_have_explicit_independent_budgets(self):
        config = FindLoopConfig()
        self.assertEqual(config.max_local_waypoints, 2)
        self.assertEqual(config.max_destination_waypoints, 10)

    def test_exploration_prompt_explains_near_and_far_tradeoff(self):
        guidance = MODE_GUIDANCE["exploration"]
        self.assertIn("EF", guidance)
        self.assertIn("may not have been visually", guidance)
        self.assertIn("center-facing camera view", guidance)

    def test_vision_selected_modes_share_one_map_guide(self):
        for mode in ("goal", "context", "destination", "exploration"):
            prompt = MAP_GUIDANCE + "\n\n" + MODE_GUIDANCE[mode]
            self.assertTrue(prompt.startswith(MAP_GUIDANCE))
            self.assertIn(MODE_GUIDANCE[mode], prompt)

    def test_batch_comparison_always_reuses_center_with_side_views(self):
        search = SearchAction.__new__(SearchAction)
        search.current_initial_frame = {
            "image_id": "old", "pan_position": "current", "jpeg_bytes": b"center"
        }
        sides = [
            {"image_id": "old-left", "pan_position": "leftmost", "jpeg_bytes": b"left"},
            {"image_id": "old-right", "pan_position": "rightmost", "jpeg_bytes": b"right"},
        ]
        frames = search._comparison_frames(sides, "center")
        self.assertEqual(
            [(frame["image_id"], frame["pan_position"]) for frame in frames],
            [("image_1", "leftmost"), ("image_2", "center"), ("image_3", "rightmost")],
        )


class ContextualClueLoopTests(unittest.IsolatedAsyncioTestCase):
    def executor(self, *, config=None):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor.find_loop = config or FindLoopConfig()
        executor.action_id = "find-1"
        executor.target = "water bottle"
        executor._search_poses = []
        executor._search_waypoint_count = 0
        executor._completed_steps = []
        executor._approach_result_data = {}
        executor._approach_follow = False
        executor.search_action = type(
            "Search", (), {"last_observation_frame": None}
        )()
        executor.track_action = type("Tracker", (), {"active": False})()
        executor._search_limit_result = lambda: None
        executor._move_camera_for_goal = AsyncMock(return_value=ActionResult(
            "camera", "see_action", "succeeded",
        ))
        executor._navigate_search_step = AsyncMock(return_value=ActionResult(
            "nav", "navigate_action", "succeeded",
        ))
        executor._track_approach_and_reacquire = AsyncMock(return_value=ActionResult(
            "verified", "track_action", "succeeded",
        ))
        return executor

    @staticmethod
    def clue(candidate_type, text="useful clue"):
        return {
            "jpeg_bytes": b"frame",
            "contextual_clue": text,
            "candidate_type": candidate_type,
        }

    async def test_context_navigation_repeats_until_found(self):
        executor = self.executor()
        results = [
            (ActionResult("scan-1", "search_action", "failed",
                          reason_code="SEARCH_CONTEXTUAL_CLUE"),
             self.clue("local")),
            (ActionResult("scan-2", "search_action", "succeeded"), None),
        ]

        async def search(*_args, **_kwargs):
            result, observation = results.pop(0)
            executor.search_action.last_observation_frame = observation
            return result

        executor._search = AsyncMock(side_effect=search)
        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(executor._search.await_count, 2)
        navigation_call = executor._navigate_search_step.await_args
        self.assertEqual(navigation_call.args[0]["candidate_type"], "local")
        self.assertEqual(navigation_call.args[0]["movement_limit"], 2)
        executor._track_approach_and_reacquire.assert_awaited_once_with(0)

    async def test_destination_candidate_routes_to_vision_selected_mode(self):
        executor = self.executor()
        results = [
            (ActionResult("scan-1", "search_action", "failed",
                          reason_code="SEARCH_CONTEXTUAL_CLUE"),
             self.clue("destination")),
            (ActionResult("scan-2", "search_action", "succeeded"), None),
        ]

        async def search(*_args, **_kwargs):
            result, observation = results.pop(0)
            executor.search_action.last_observation_frame = observation
            return result

        executor._search = AsyncMock(side_effect=search)
        result = await executor._search_environment()
        self.assertEqual(result.status, "succeeded")
        navigation_call = executor._navigate_search_step.await_args
        self.assertEqual(navigation_call.kwargs["mode"], "destination")

    async def test_visual_candidate_is_verified_exactly_like_found(self):
        executor = self.executor()
        executor.search_action.last_observation_frame = self.clue("visual")
        executor._search = AsyncMock(return_value=ActionResult(
            "scan", "search_action", "failed",
            reason_code="SEARCH_CONTEXTUAL_CLUE",
        ))

        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        executor._track_approach_and_reacquire.assert_awaited_once_with(0)
        executor._search.assert_awaited_once()
        executor._navigate_search_step.assert_not_awaited()

    async def test_speculative_candidate_is_treated_as_not_found(self):
        executor = self.executor()
        executor.search_action.last_observation_frame = self.clue("speculative")
        executor._search = AsyncMock(return_value=ActionResult(
            "scan", "search_action", "failed",
            reason_code="SEARCH_CONTEXTUAL_CLUE",
        ))
        executor._navigate_search_step.return_value = ActionResult(
            "nav", "navigate_action", "failed",
            reason_code="NO_NAVIGATION_CANDIDATES",
        )

        result = await executor._search_environment()

        self.assertEqual(result.reason_code, "OBJECT_SEARCH_EXHAUSTED")
        executor._track_approach_and_reacquire.assert_not_awaited()
        executor._navigate_search_step.assert_awaited_once_with(
            None, mode="exploration", step="explore_1"
        )

    async def test_local_budget_is_consumed_by_repeated_frames(self):
        executor = self.executor()

        async def search(*_args, **_kwargs):
            executor.search_action.last_observation_frame = self.clue("local")
            return ActionResult(
                "scan", "search_action", "failed",
                reason_code="SEARCH_CONTEXTUAL_CLUE",
            )

        executor._search = AsyncMock(side_effect=search)
        executor._navigate_search_step.side_effect = [
            ActionResult("nav-1", "navigate_action", "succeeded"),
            ActionResult("nav-2", "navigate_action", "succeeded"),
            ActionResult("nav-3", "navigate_action", "failed",
                         reason_code="NO_NAVIGATION_CANDIDATES"),
        ]

        result = await executor._search_environment()

        self.assertEqual(result.reason_code, "OBJECT_SEARCH_EXHAUSTED")
        context_calls = [
            item for item in executor._navigate_search_step.await_args_list
            if item.kwargs["mode"] == "context"
        ]
        self.assertEqual(
            [item.args[0]["remaining_waypoints"] for item in context_calls],
            [2, 1],
        )

    async def test_changed_contextual_type_resets_the_loop_budget(self):
        executor = self.executor(config=FindLoopConfig(
            max_local_waypoints=1,
            max_destination_waypoints=1,
        ))
        observations = [self.clue("local"), self.clue("destination"), None]

        async def search(*_args, **_kwargs):
            observation = observations.pop(0)
            executor.search_action.last_observation_frame = observation
            return ActionResult(
                "scan", "search_action",
                "succeeded" if observation is None else "failed",
                reason_code=None if observation is None else "SEARCH_CONTEXTUAL_CLUE",
            )

        executor._search = AsyncMock(side_effect=search)
        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(
            [call.kwargs["mode"] for call in
             executor._navigate_search_step.await_args_list],
            ["context", "destination"],
        )
        self.assertEqual(
            [call.args[0]["remaining_waypoints"] for call in
             executor._navigate_search_step.await_args_list],
            [1, 1],
        )

    async def test_blocked_exploration_exhausts_search(self):
        executor = self.executor(config=FindLoopConfig(max_exploration_waypoints=1))
        executor._search = AsyncMock(return_value=ActionResult(
            "scan", "search_action", "failed", reason_code="TARGET_NOT_VISIBLE",
        ))
        blocked = ActionResult(
            "nav", "navigate_action", "failed",
            reason_code="NO_NAVIGATION_CANDIDATES",
        )
        executor._navigate_search_step = AsyncMock(return_value=blocked)

        result = await executor._search_environment()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason_code, "OBJECT_SEARCH_EXHAUSTED")
        executor._navigate_search_step.assert_awaited_once_with(
            None,
            mode="exploration",
            step="explore_1",
        )

    async def test_exploration_does_not_forward_a_stale_clue_frame(self):
        executor = self.executor(config=FindLoopConfig(max_exploration_waypoints=1))
        results = [
            ActionResult("miss", "search_action", "failed",
                         reason_code="TARGET_NOT_VISIBLE"),
            ActionResult("found", "search_action", "succeeded"),
        ]

        async def search(*_args, **_kwargs):
            executor.search_action.last_observation_frame = None
            return results.pop(0)

        executor._search = AsyncMock(side_effect=search)

        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        executor._navigate_search_step.assert_awaited_once_with(
            None,
            mode="exploration",
            step="explore_1",
        )
        self.assertEqual(executor._search.await_args_list[1].args[0],
                         "explore_1_center_scan")

    async def test_inspection_propagates_operational_scan_failure(self):
        executor = self.executor()
        failure = ActionResult(
            "scan", "search_action", "failed",
            reason_code="CAMERA_FAILURE",
        )
        executor._search = AsyncMock(return_value=failure)
        executor._navigate_search_step = AsyncMock()

        result = await executor._search_environment()

        self.assertIs(result, failure)
        executor._navigate_search_step.assert_not_awaited()

    async def test_failed_reacquisition_navigates_without_rescanning_arrival(self):
        executor = self.executor()
        scans = [
            ActionResult("found-1", "search_action", "succeeded"),
            ActionResult("found-2", "search_action", "succeeded"),
        ]

        async def search(*_args, **_kwargs):
            executor.search_action.last_observation_frame = None
            return scans.pop(0)

        attempts = 0

        async def verify(_attempt):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                executor.search_action.last_observation_frame = self.clue("local")
                return ActionResult(
                    "reacquire", "search_action", "failed",
                    reason_code="TARGET_NOT_VISIBLE",
                )
            return ActionResult("verified", "track_action", "succeeded")

        executor._search = AsyncMock(side_effect=search)
        executor._track_approach_and_reacquire = AsyncMock(side_effect=verify)

        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(executor._search.await_count, 2)
        self.assertEqual(executor._move_camera_for_goal.await_count, 2)
        executor._navigate_search_step.assert_awaited_once()
        self.assertEqual(
            executor._navigate_search_step.await_args.kwargs["mode"], "context"
        )


class FollowExecutorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saved_camera = dict(robot_state.get("camera") or {})

    def tearDown(self):
        robot_state["camera"].clear()
        robot_state["camera"].update(self.saved_camera)

    def test_distance_uses_object_xy_instead_of_raw_tof(self):
        executor = FollowExecutor.__new__(FollowExecutor)
        executor.config = FollowConfig(target_max_age_seconds=1.0)
        robot_state["camera"].update({
            "object_x": 3.0,
            "object_y": 4.0,
            "camera_tof_range": 0.2,
            "timestamp": time.monotonic(),
        })

        self.assertEqual(executor._fresh_distance(), 5.0)

    async def test_acquisition_searches_then_tracks_without_approaching(self):
        search = type("Search", (), {"active": False})()
        search.start_searching = AsyncMock(return_value=ActionResult(
            "search", "search_action", "succeeded", target="person"
        ))
        tracker = type("Tracker", (), {"active": False})()
        tracker.start_tracking = AsyncMock(return_value=ActionResult(
            "track", "track_action", "running", target="person"
        ))
        tracker.wait_until_stable = AsyncMock(return_value=True)
        tracker.stop_tracking = AsyncMock()
        approach = type("Approach", (), {"active": False})()
        approach.start_approaching = AsyncMock(return_value=ActionResult(
            "follow", "approach_action", "running", target="person"
        ))
        goal_executor = type("Goal", (), {"search_action": search})()
        goal_executor.start = AsyncMock()
        executor = FollowExecutor(goal_executor, tracker, approach)
        executor.action_id = "follow-1"
        executor.radius_m = 1.25
        executor._approach_count = 0

        result = await executor._acquire_target("initial")

        self.assertEqual(result.status, "succeeded")
        search.start_searching.assert_awaited_once_with(
            target="person",
            action_id="follow-1:initial:search",
            effort="best_effort",
            initial_view_only=False,
        )
        tracker.start_tracking.assert_awaited_once()
        approach.start_approaching.assert_not_awaited()
        goal_executor.start.assert_not_awaited()

    async def test_recovery_reacquires_without_running_find_object(self):
        executor = FollowExecutor.__new__(FollowExecutor)
        executor.config = FollowConfig(
            tracking_recovery_seconds=0.0
        )
        executor._stop_requested = False
        executor._recovery_count = 0
        executor.track_action = type("Tracker", (), {"active": True})()
        executor.track_action.stop_tracking = AsyncMock()
        executor.approach_action = type("Approach", (), {"active": False})()
        executor._fresh_distance = Mock(return_value=None)
        executor._acquire_target = AsyncMock(return_value=ActionResult(
            "acquire", "follow_action", "succeeded", target="person"
        ))

        recovered = await executor._recover_tracking()

        self.assertTrue(recovered)
        executor.track_action.stop_tracking.assert_awaited_once_with(
            "TRACKING_RECOVERY_TIMEOUT"
        )
        executor._acquire_target.assert_awaited_once_with("reacquire:1")

    def test_approach_standoff_is_inside_follow_radius(self):
        executor = FollowExecutor.__new__(FollowExecutor)
        executor.config = FollowConfig(approach_standoff_ratio=0.8)
        executor.radius_m = 1.25

        self.assertEqual(executor._approach_standoff_m(), 1.0)


class PostApproachReacquisitionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saved_pose = robot_state.get("pose")
        self.saved_camera = dict(robot_state.get("camera") or {})

    def tearDown(self):
        robot_state["pose"] = self.saved_pose
        robot_state["camera"].clear()
        robot_state["camera"].update(self.saved_camera)

    def test_person_stability_uses_pose_mask_and_bounded_map_motion(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker.camera = Mock()
        tracker.camera.snapshot.return_value = type("Frame", (), {
            "tracking_bgr": np.zeros((360, 640, 3), dtype=np.uint8),
        })()
        tracker.stable = False
        tracker.person_stable = False
        tracker.person_stable_tick = 0
        tracker._last_valid_person_point = None
        tracker._last_valid_person_timestamp = None

        person = {
            "bbox": {"x1": 240, "y1": 80, "x2": 400, "y2": 340},
            "keypoints": {
                "left_shoulder": {
                    "x": 275, "y": 130, "confidence": 0.9,
                },
                "right_shoulder": {
                    "x": 365, "y": 130, "confidence": 0.9,
                },
                "left_hip": {"x": 290, "y": 230, "confidence": 0.9},
                "right_hip": {"x": 350, "y": 230, "confidence": 0.9},
            },
        }
        now = time.monotonic()
        robot_state["camera"].update({
            "camera_tof_range": 2.0,
            "object_x": 2.0,
            "object_y": 0.0,
            "object_map_x": 3.0,
            "object_map_y": 4.0,
            "timestamp": now - 0.2,
        })

        self.assertFalse(tracker._update_person_stability(person))
        robot_state["camera"].update({
            "object_map_x": 3.1,
            "timestamp": now - 0.1,
        })
        self.assertTrue(tracker._update_person_stability(person))
        self.assertFalse(tracker.stable)

        robot_state["camera"].update({
            "object_map_x": 5.0,
            "timestamp": now,
        })
        self.assertFalse(tracker._update_person_stability(person))
        self.assertFalse(tracker.person_stable)

    async def test_stable_tracking_saves_map_location_without_semantic_memory(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker._memory_recorded_for_session = False
        tracker.semantic_memory = None
        tracker.target = "bottle"
        tracker.detection_confidence = 0.88
        tracker.last_stable_target_location = None
        robot_state["pose"] = {"x": 1.0, "y": 2.0, "yaw": 0.3}
        robot_state["camera"].update({
            "object_map_x": 2.5,
            "object_map_y": 2.75,
            "camera_tof_range": 1.7,
            "pan_angle": 110.0,
            "tilt_angle": 92.0,
            "timestamp": 12.0,
        })

        await tracker._remember_stable_target()

        self.assertEqual(tracker.last_stable_target_location["target"], "bottle")
        self.assertEqual(tracker.last_stable_target_location["map_x"], 2.5)
        self.assertEqual(tracker.last_stable_target_location["map_y"], 2.75)
        self.assertTrue(tracker._memory_recorded_for_session)

    async def test_approach_finishes_immediately_on_navigation_success(self):
        approach = ApproachAction.__new__(ApproachAction)
        approach.active = True
        approach.action_id = "approach-1"
        approach.target = "bottle"
        approach._navigation_attempts = 1
        approach._last_destination = {"x": 1.0, "y": 0.0, "angle": 0.0}
        approach.completion_future = asyncio.get_running_loop().create_future()

        handled = approach.handle_navigation_event({
            "event": "navigation",
            "action_id": "approach-1",
            "status": "Goal Reached",
        })
        result = await approach.wait_until_finished()

        self.assertTrue(handled)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.outcome, "navigation_reached")
        self.assertFalse(approach.active)

    async def test_approach_sends_one_resolved_navigation_command(self):
        tracker = type("Tracker", (), {
            "active": True,
            "target": "bottle",
            "stable": True,
        })()
        resolved = {
            "x": 2.0,
            "y": 3.0,
            "angle": 0.4,
            "frame_id": "map",
            "resolution": "standoff_from_obstacle",
            "was_clamped": True,
        }
        send_robot_command = AsyncMock(return_value={
            "status": "accepted",
            "destination": resolved,
        })
        approach = ApproachAction(send_robot_command, tracker)
        robot_state["camera"].update({
            "object_x": 4.0,
            "object_y": 0.5,
            "object_angle": 0.12,
            "camera_tof_range": 4.1,
        })

        result = await approach.start_approaching("bottle", "approach-2")

        self.assertEqual(result.status, "running")
        self.assertEqual(result.data["destination"], resolved)
        self.assertEqual(
            send_robot_command.await_args_list,
            [
                call({
                    "command": "navigate_to_approach",
                    "action_id": "approach-2",
                    "frame_id": "base_footprint",
                    "x": 4.0,
                    "y": 0.5,
                    "angle": 0.12,
                    "tof_range": 4.1,
                }),
            ],
        )

    async def test_follow_starts_once_then_updates_the_same_navigation(self):
        tracker = type("Tracker", (), {
            "active": True,
            "target": "person",
            "stable": False,
            "person_stable": True,
        })()
        tracker.wait_until_person_stable = AsyncMock(return_value=True)
        tracker.wait_until_stable = AsyncMock(return_value=True)
        send_robot_command = AsyncMock(return_value={
            "status": "accepted",
            "destination": {"x": 1.0, "y": 0.0, "frame_id": "map"},
        })
        approach = ApproachAction(send_robot_command, tracker)
        robot_state["camera"].update({
            "object_x": 2.0,
            "object_y": 0.0,
            "object_angle": 0.0,
            "camera_tof_range": 2.0,
        })

        started = await approach.start_approaching(
            "person", "follow-1", standoff_m=1.0, follow=True
        )
        robot_state["camera"]["object_x"] = 2.5
        updated = await approach.start_approaching(
            "person", "ignored-update-id", standoff_m=1.0, follow=True
        )

        self.assertEqual(started.outcome, "following")
        self.assertEqual(updated.outcome, "follow_goal_updated")
        self.assertTrue(approach.active)
        self.assertTrue(approach.following)
        self.assertEqual(
            [item.args[0]["command"] for item in send_robot_command.await_args_list],
            ["navigate_to_follow", "navigate_to_follow"],
        )
        self.assertEqual(
            [item.args[0]["action_id"] for item in send_robot_command.await_args_list],
            ["follow-1", "follow-1"],
        )
        self.assertEqual(
            send_robot_command.await_args_list[1].args[0]["x"], 2.5
        )
        tracker.wait_until_person_stable.assert_not_awaited()
        tracker.wait_until_stable.assert_not_awaited()

    async def test_follow_waits_for_quick_person_stability_not_strict_stability(self):
        tracker = type("Tracker", (), {
            "active": True,
            "target": "person",
            "stable": False,
            "person_stable": False,
        })()
        tracker.wait_until_person_stable = AsyncMock(return_value=True)
        tracker.wait_until_stable = AsyncMock(return_value=False)
        approach = ApproachAction(
            AsyncMock(return_value={"status": "accepted"}), tracker
        )
        robot_state["camera"].update({
            "object_x": 2.0,
            "object_y": 0.0,
            "object_angle": 0.0,
            "camera_tof_range": 2.0,
        })

        result = await approach.start_approaching(
            "person", "follow-quick", standoff_m=0.8, follow=True
        )

        self.assertEqual(result.status, "running")
        tracker.wait_until_person_stable.assert_awaited_once_with(timeout=2.0)
        tracker.wait_until_stable.assert_not_awaited()

    async def test_goal_executor_starts_follow_without_waiting_for_completion(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor.target = "person"
        executor.action_id = "follow-root"
        executor._approach_standoff_m = 1.0
        executor._approach_follow = True
        executor._approach_result_data = {}
        executor._step_id = lambda step: f"follow-root:{step}"
        executor.approach_action = type("Approach", (), {})()
        executor.approach_action.start_approaching = AsyncMock(
            return_value=ActionResult(
                "follow-root:approach", "approach_action", "running",
                target="person", outcome="following",
                data={"destination": {"x": 1.0, "y": 0.0}},
            )
        )

        result = await executor._approach()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.outcome, "follow_navigation_started")
        executor.approach_action.start_approaching.assert_awaited_once_with(
            target="person",
            action_id="follow-root:approach",
            standoff_m=1.0,
            follow=True,
        )

    async def test_track_approach_and_reacquire_uses_max_effort(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor.action_id = "find-1"
        executor.target = "bottle"
        executor._completed_steps = []
        executor._approach_result_data = {}
        executor._search_waypoint_count = 3
        executor._approach_follow = False
        executor.track_action = type("Tracker", (), {})()
        executor.track_action.active = True
        executor.track_action.stop_tracking = AsyncMock()
        executor._approach = AsyncMock(return_value=ActionResult(
            "approach", "approach_action", "succeeded", target="bottle",
        ))
        executor._search = AsyncMock(return_value=ActionResult(
            "search", "search_action", "succeeded", target="bottle"
        ))
        executor._track = AsyncMock(side_effect=[
            ActionResult("track", "track_action", "succeeded", target="bottle"),
            ActionResult("retrack", "track_action", "succeeded", target="bottle"),
        ])

        result = await executor._track_approach_and_reacquire(0)

        self.assertEqual(result.status, "succeeded")
        executor._approach.assert_awaited_once_with(step="approach")
        executor.track_action.stop_tracking.assert_awaited_once_with(
            reason_code="APPROACH_NAVIGATION_COMPLETE",
            status="succeeded",
            outcome="ready_for_post_approach_reacquisition",
            reset_camera=False,
        )
        executor._search.assert_awaited_once_with(
            "post_approach_search_0",
            effort="best_effort",
            initial_view_only=False,
            sweep_directions=None,
        )
        self.assertEqual(
            executor._track.await_args_list,
            [
                call(allow_grounding_dino=True, step="track"),
                call(allow_grounding_dino=True, step="post_approach_track_0"),
            ],
        )
        self.assertEqual(executor._search_waypoint_count, 4)
        self.assertTrue(executor._approach_result_data["verified"])

    async def test_run_delegates_the_complete_goal_to_search_environment(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor._approach_follow = False
        executor._search_environment = AsyncMock(return_value=ActionResult(
            "verified", "track_action", "succeeded",
        ))
        executor._complete = Mock()

        await executor._run()

        executor._search_environment.assert_awaited_once_with()
        executor._complete.assert_called_once_with(
            status="succeeded",
            outcome="found_reached_and_reacquired",
        )


class CenterFirstSearchTests(unittest.IsolatedAsyncioTestCase):
    def search(self):
        search = SearchAction.__new__(SearchAction)
        search.active = True
        search.search_id = "search-1"
        search.pending_request_id = "request-1"
        search.pan_angle = 95.0
        search.tilt_angle = 90.0
        search.current_initial_frame = {"jpeg_bytes": b"center"}
        search.current_candidate = None
        search.sweep_pan_positions = ("leftmost", "rightmost")
        search.capture_sweep_batch = AsyncMock()
        search.move_to_found_target = AsyncMock()
        search.move_to_candidate_frame = AsyncMock()
        search.complete_searching = AsyncMock()
        return search

    async def test_center_candidate_is_saved_then_ranked_with_sides(self):
        search = self.search()
        await search.assess_frame_search(
            {
                "result": "candidate",
                "candidate_type": "speculative",
                "target_position": None,
                "contextual_clue": "A cabinet may contain the target.",
            },
            {
                "kind": "visual_search_assessment",
                "search_id": "search-1",
                "request_id": "request-1",
            },
        )
        self.assertEqual(search.current_candidate["frame"], search.current_initial_frame)
        search.capture_sweep_batch.assert_awaited_once()
        search.complete_searching.assert_not_awaited()

    async def test_center_not_found_still_captures_eligible_sides(self):
        search = self.search()
        await search.assess_frame_search(
            {
                "result": "not_found",
                "candidate_type": None,
                "target_position": None,
                "contextual_clue": None,
            },
            {
                "kind": "visual_search_assessment",
                "search_id": "search-1",
                "request_id": "request-1",
            },
        )
        search.capture_sweep_batch.assert_awaited_once()
        search.complete_searching.assert_not_awaited()


class SearchFrameExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_best_effort_exports_only_middle_layer_frames(self):
        search = SearchAction.__new__(SearchAction)
        search.active = True
        search.action_id = "search-1"
        search.target = "bottle"
        search.effort = "best_effort"
        search.initial_view_only = False
        search.sweep_tilt_order = ("center", "upmost", "downmost")
        search.batch_results = []
        search.first_frame_result = None
        search.pan_angle = 95.0
        search.tilt_angle = 30.0
        search.last_detection_source = None
        search._yolo_owner = None
        search.completion_future = asyncio.get_running_loop().create_future()
        pose = {"x": 1.0, "y": 2.0, "yaw": 0.0, "frame_id": "map"}
        middle = {
            "jpeg_bytes": b"middle", "pan_angle": 95.0, "tilt_angle": 90.0,
            "tilt_position": "center", "robot_pose": pose,
        }
        upper = {
            "jpeg_bytes": b"upper", "pan_angle": 95.0, "tilt_angle": 30.0,
            "tilt_position": "upmost", "robot_pose": pose,
        }
        lower = {
            "jpeg_bytes": b"lower", "pan_angle": 95.0, "tilt_angle": 120.0,
            "tilt_position": "downmost", "robot_pose": pose,
        }
        search.captured_frames = [middle, upper, lower]
        search.current_candidate = {
            "assessment": {
                "contextual_clue": "possible bottle above",
                "candidate_type": "visual",
            },
            "frame": upper,
        }

        await search.complete_searching(
            "failed", "contextual_clue", "SEARCH_CONTEXTUAL_CLUE"
        )

        self.assertEqual(
            [frame["jpeg_bytes"] for frame in search.last_coverage_frames],
            [b"middle"],
        )
        self.assertIsNone(search.last_observation_frame)


class CameraCoverageTests(unittest.TestCase):
    class Grid:
        resolution = 0.02

        def __init__(self, wall=True):
            self.data = np.zeros((201, 201), dtype=np.int16)
            if wall:
                self.data[:, 137:138] = 100

        def cells(self, x, y):
            return (
                np.floor((np.asarray(x) + 2.0) / self.resolution).astype(int),
                np.floor((np.asarray(y) + 2.0) / self.resolution).astype(int),
            )

        def inside(self, col, row):
            return (col >= 0) & (row >= 0) & (col < 201) & (row < 201)

    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0, "frame_id": "map"}

    def setUp(self):
        self.logic = MapLogic()
        self.renderer = MapRenderer()
        self.overlay = {
            "frame_id": "map",
            "camera_horizontal_fov_deg": 60.0,
            "camera_reliable_range_m": 2.0,
            "search_poses": [{
                **self.pose,
                "kind": "pose",
                "views": [{"pan": 95.0, "tilt": 90.0}],
            }],
        }

    def cell(self, grid, x, y):
        col, row = grid.cells(x, y)
        return int(row), int(col)

    def test_coverage_is_an_obstacle_clipped_grid(self):
        grid = self.Grid(wall=True)
        coverage = self.logic.coverage_counts(grid, self.overlay) > 0
        self.assertTrue(coverage[self.cell(grid, 0.5, 0.0)])
        self.assertFalse(coverage[self.cell(grid, 1.5, 0.0)])

    def test_coverage_stops_at_reliable_range(self):
        grid = self.Grid(wall=False)
        self.overlay["search_poses"][0]["x"] = -0.5
        coverage = self.logic.coverage_counts(grid, self.overlay) > 0
        self.assertTrue(coverage[self.cell(grid, 1.4, 0.0)])
        self.assertFalse(coverage[self.cell(grid, 1.8, 0.0)])

    def test_overlapping_observations_increment_grid_counts(self):
        grid = self.Grid(wall=False)
        once = self.logic.coverage_counts(grid, self.overlay)
        self.overlay["search_poses"][0]["views"].append({
            "pan": 100.0, "tilt": 90.0,
        })
        twice = self.logic.coverage_counts(grid, self.overlay)
        self.assertEqual(int(twice.max()), 2)
        self.assertTrue(np.all((once > 0) <= (twice > 0)))

    def test_renderer_projects_grid_coverage_before_tinting(self):
        grid = self.Grid(wall=False)
        rows, cols = np.indices(grid.data.shape)
        view = {
            "height": 201, "width": 201,
            "map_inside": np.ones((201, 201), dtype=bool),
            "map_rows": rows, "map_cols": cols,
        }
        counts = self.logic.coverage_counts(grid, self.overlay)
        image = np.full((201, 201, 3), 246, dtype=np.uint8)
        self.renderer._draw_search_overlay(
            image, view, self.overlay, counts
        )
        row, col = self.cell(grid, 1.0, 0.0)
        self.assertGreater(int(image[row, col, 0]), int(image[row, col, 2]))


class LiveCameraFovTests(unittest.TestCase):
    def test_live_view_rotates_with_servo_pan(self):
        grid = CameraCoverageTests.Grid(wall=False)
        logic = MapLogic()
        pose = CameraCoverageTests.pose
        center = pose["yaw"] + math.radians(155.0 - 95.0)
        visible = logic.visibility_mask(grid, pose, center, 60.0, 2.0)
        aimed = CameraCoverageTests().cell(
            grid,
            0.5 * math.cos(math.radians(60.0)),
            0.5 * math.sin(math.radians(60.0)),
        )
        straight = CameraCoverageTests().cell(grid, 0.5, 0.0)
        self.assertTrue(visible[aimed])
        self.assertFalse(visible[straight])


class DetectorFirstTests(unittest.TestCase):
    def test_search_uses_best_matching_yolo_box(self):
        search = SearchAction.__new__(SearchAction)
        search.target = "water bottle"
        search.camera = type("Camera", (), {"tracking_size": (640, 360)})()
        search.yolo = type(
            "Yolo",
            (),
            {
                "detections": [
                    {
                        "class": "bottle",
                        "confidence": 0.6,
                        "bbox": {"x1": 0, "y1": 0, "x2": 64, "y2": 36},
                    },
                    {
                        "class": "bottle",
                        "confidence": 0.9,
                        "bbox": {"x1": 288, "y1": 162, "x2": 352, "y2": 198},
                    },
                ]
            },
        )()
        result = search._best_yolo_candidate()
        self.assertEqual(result["confidence"], 0.9)
        self.assertAlmostEqual(result["position"]["x"], 0.5)
        self.assertAlmostEqual(result["position"]["y"], 0.5)


class FoundTransitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_search_returns_found_without_starting_tracking(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor.action_id = "find-1"
        executor.target = "fan"
        executor._completed_steps = []
        executor.search_action = type(
            "Search", (), {
                "active": False,
                "last_found_target": "fan",
                "last_coverage_frames": [],
                "last_observation_frame": None,
            }
        )()
        candidate = ActionResult(
            "find-1:scan",
            "search_action",
            "succeeded",
            target="fan",
            outcome="found",
        )
        executor.search_action.start_searching = AsyncMock(return_value=candidate)
        executor._remember_search_pose = Mock(return_value=False)
        executor._track = AsyncMock()

        result = await executor._search(
            "initial_center_scan", initial_view_only=False
        )

        self.assertIs(result, candidate)
        self.assertEqual(executor.search_action.last_found_target, "fan")
        executor._track.assert_not_awaited()


class LocalDistanceCandidateTests(unittest.TestCase):
    class Grid:
        def cells(self, x, y):
            return int(round(x)) + 5, int(round(y)) + 5

        def inside(self, col, row):
            return 0 <= col < 11 and 0 <= row < 11

    def test_map_offers_four_useful_distances_per_ray(self):
        logic = MapLogic()
        logic.cfg = LogicConfig(
            sample_radius=3.0,
            sample_step=0.5,
            sample_angle_deg=45.0,
            include_rotations=True,
        )
        candidates = logic._local_candidates(
            self.Grid(),
            np.ones((11, 11), dtype=bool),
            {"x": 0.0, "y": 0.0, "yaw": 0.0},
        )
        translations = [
            item for item in candidates if item["kind"] == "local"
        ]
        rotations = [
            item for item in candidates if item["kind"] == "rotation"
        ]
        self.assertEqual(len(translations), 32)
        self.assertEqual(len(rotations), 3)
        radii = sorted({round(math.hypot(item["x"], item["y"]), 1)
                        for item in translations})
        self.assertEqual(radii, [0.5, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
