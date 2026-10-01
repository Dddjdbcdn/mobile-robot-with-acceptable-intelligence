import asyncio
import json
import math
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BIG_BRAIN = ROOT.parent / "BIG_BRAIN" / "src" / "robot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BIG_BRAIN))

from actions.map_navigation_action import (
    MAP_GUIDANCE,
    MODE_GUIDANCE,
    MapNavigationAction,
    NavigationError,
)
from actions.action_result import ActionResult
from actions.approach_action import ApproachAction
from actions.search_action import SearchAction
from actions.track_action import TrackAction
from actions.tracking.stable_seed import StableTargetSeedTracker
from cognition.sequence.find_target_executor import (
    FindLoopConfig,
    FindTargetExecutor,
)
from cognition.sequence.follow_person_executor import FollowPersonExecutor
from cognition.manager.world_state import robot_state
from robot.map.map_logic import LogicConfig, MapLogic
from robot.map.map_renderer import MapRenderer


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


class FindTargetCandidateTests(unittest.TestCase):
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
        overlay.update({"mode": "context", "search_poses": []})
        overlay["observation"]["candidate_type"] = "context"
        return overlay

    @unittest.skip("standalone map-pose goal selection was removed")
    def test_goal_candidates_do_not_require_an_overlay(self):
        selected = self.logic.plan_candidates(self.prepared, self.pose)

        self.assertTrue(any(item["kind"] == "local" for item in selected))
        self.assertEqual(
            len([item for item in selected if item["kind"] == "rotation"]),
            1,
        )
        self.assertEqual(
            [item["id"] for item in selected if item["kind"] == "rotation"],
            ["H"],
        )

    @unittest.skip("standalone map-pose goal selection was removed")
    def test_nearby_candidates_are_deduplicated_with_semantic_priority(self):
        self.prepared["frontiers"] = [{
            "id": "F1", "kind": "frontier",
            "x": 0.6, "y": 0.0, "yaw": 0.0,
        }]

        selected = self.logic.plan_candidates(self.prepared, self.pose)
        ids = {item["id"] for item in selected}

        self.assertIn("F1", ids)
        self.assertNotIn("1", ids)
        self.assertIn("H", ids)

    def test_candidate_deduplication_uses_thirty_centimetres(self):
        candidates = [
            candidate("1", "local", 0.0, 0.0),
            candidate("2", "local", 0.29, 0.0),
            candidate("3", "local", 0.61, 0.0),
        ]

        selected = self.logic._deduplicate_candidates(candidates)

        self.assertEqual([item["id"] for item in selected], ["1", "3"])

    def test_connectivity_uses_nearest_valid_cell_when_robot_cell_is_blocked(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 3:5] = True
        mask[0, 0] = True

        connected = self.logic._connected(mask, robot_col=2, robot_row=3)

        self.assertTrue(connected[3, 3])
        self.assertTrue(connected[4, 4])
        self.assertFalse(connected[0, 0])

    def test_clearance_does_not_erase_robot_egress(self):
        costs = np.zeros_like(self.grid.data, dtype=np.int16)
        robot_col, robot_row = self.grid.cells(0.0, 0.0)
        self.grid.data[int(robot_row), int(robot_col) + 1] = 100

        reachable = self.logic._reachable(
            self.grid, costs, self.pose, clearance=0.15
        )
        endpoint = self.logic._furthest_reachable_on_ray(
            self.grid, reachable, self.pose, self.pose["yaw"],
            max_distance=0.5,
        )

        self.assertTrue(reachable[int(robot_row), int(robot_col)])
        self.assertIsNotNone(endpoint)

    def test_pose_marker_scale_is_relative_to_crop_pixels(self):
        renderer = MapRenderer()
        grid = self.Grid()
        view_4m = renderer._make_view(grid, self.pose, view_size_m=4.0)
        view_6m = renderer._make_view(grid, self.pose, view_size_m=6.0)

        scale_4m = renderer.pose_marker_scale_for_crop(view_4m, 4.0)
        scale_6m = renderer.pose_marker_scale_for_crop(view_6m, 6.0)

        self.assertAlmostEqual(scale_4m, 1024 / 768)
        self.assertAlmostEqual(scale_6m, scale_4m)
        self.assertEqual((view_6m["width"], view_6m["height"]), (1024, 1024))
        self.assertEqual(renderer._pixel(view_6m, 0.0, 0.0), (512, 512))

    def test_context_samples_the_clue_cone_and_furthest_ray(self):
        selected = self.logic.plan_candidates(
            self.prepared, self.pose, self.context_overlay()
        )
        self.assertTrue(selected)
        self.assertIn("CF", {item["id"] for item in selected})
        for item in selected:
            bearing = math.atan2(item["y"], item["x"])
            self.assertLessEqual(abs(bearing), math.radians(30.0))
            self.assertEqual(item["kind"], "context")

    def test_context_sampling_parameters_come_from_config(self):
        logic = MapLogic(LogicConfig(
            context_radius_samples=1,
            context_bearing_fractions=(0.0,),
        ))

        selected = logic.plan_candidates(
            self.prepared, self.pose, self.context_overlay()
        )

        cone = [
            item for item in selected
            if item["id"].startswith("C") and item["id"] != "CF"
        ]
        self.assertEqual(len(cone), 1)
        self.assertAlmostEqual(cone[0]["x"], 0.0)
        self.assertAlmostEqual(cone[0]["y"], 0.0)

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
        selected = self.logic.plan_candidates(
            self.prepared, self.pose, self.destination_overlay()
        )

        self.assertGreater(len(selected), 1)
        self.assertTrue(all(item["kind"] == "context" for item in selected))

    def test_visual_fallback_uses_observation_cone_candidates(self):
        overlay = self.context_overlay()
        overlay["mode"] = "visual"
        overlay["observation"]["candidate_type"] = "found"

        selected = self.logic.plan_candidates(
            self.prepared, self.pose, overlay
        )

        self.assertTrue(selected)
        self.assertTrue(all(item["kind"] == "context" for item in selected))
        self.assertNotIn("F1", {item["id"] for item in selected})
        self.assertGreater(max(item["distance_m"] for item in selected), 2.0)
        sampled = [item for item in selected if item["id"] != "CF"]
        self.assertTrue(sampled)
        self.assertLessEqual(
            max(item["distance_m"] for item in sampled),
            self.logic.cfg.camera_fov_max_range_m + self.grid.resolution,
        )
        self.assertIn("CF", {item["id"] for item in selected})

    def test_local_navigation_is_resolved_without_rendered_candidates(self):
        destination = self.logic.resolve_local_destination(
            self.prepared, self.pose, "nudge_forward"
        )
        self.assertGreater(destination["x"], self.pose["x"])
        self.assertAlmostEqual(destination["angle"], self.pose["yaw"])

        rotation = self.logic.resolve_local_destination(
            self.prepared, self.pose, "rotate_left"
        )
        self.assertAlmostEqual(rotation["x"], self.pose["x"])
        self.assertAlmostEqual(rotation["angle"], math.pi / 2.0)

    def test_local_translation_arrives_facing_fresh_lidar_person_pose(self):
        person = {"x": 1.0, "y": 2.0, "age_seconds": 0.2}

        for command in ("nudge_forward", "furthest_forward"):
            destination = self.logic.resolve_local_destination(
                self.prepared, self.pose, command, person_pose=person
            )
            expected = math.atan2(
                person["y"] - destination["y"],
                person["x"] - destination["x"],
            )
            self.assertAlmostEqual(destination["angle"], expected)

    def test_local_translation_keeps_heading_for_stale_person_pose(self):
        destination = self.logic.resolve_local_destination(
            self.prepared,
            self.pose,
            "nudge_forward",
            person_pose={"x": 0.0, "y": 2.0, "age_seconds": 1.1},
        )

        self.assertAlmostEqual(destination["angle"], self.pose["yaw"])

    def test_open_space_middle_targets_the_current_room_core(self):
        self.grid.data[:] = 100
        reachable = np.zeros_like(self.grid.data, dtype=bool)
        reachable[15:46, 10:41] = True
        self.grid.data[reachable] = 0
        self.prepared["reachable"] = reachable
        start_x, start_y = self.grid.world(13, 30)
        self.pose.update(x=float(start_x), y=float(start_y))

        result = self.logic.resolve_open_space_middle_step(
            self.prepared, self.pose
        )

        labels, areas = self.logic._exit_open_components(
            self.grid, reachable
        )
        destination = result["destination"]
        col, row = (
            int(value)
            for value in self.grid.cells(
                destination["x"], destination["y"]
            )
        )
        self.assertEqual(result["phase"], "move_to_room_core")
        self.assertGreaterEqual(
            areas[int(labels[row, col])],
            self.logic.cfg.exit_component_min_area_m2,
        )

    def test_open_space_middle_arrives_facing_fresh_lidar_person_pose(self):
        self.grid.data[:] = 100
        reachable = np.zeros_like(self.grid.data, dtype=bool)
        reachable[15:46, 10:41] = True
        self.grid.data[reachable] = 0
        self.prepared["reachable"] = reachable
        start_x, start_y = self.grid.world(13, 30)
        self.pose.update(x=float(start_x), y=float(start_y))
        person = {"x": -2.0, "y": 2.0, "age_seconds": 0.1}

        result = self.logic.resolve_open_space_middle_step(
            self.prepared, self.pose, person_pose=person
        )

        destination = result["destination"]
        self.assertAlmostEqual(
            destination["angle"],
            math.atan2(
                person["y"] - destination["y"],
                person["x"] - destination["x"],
            ),
        )

    def test_open_space_middle_explores_when_no_room_core_is_mapped(self):
        self.grid.data[:] = 100
        reachable = np.zeros_like(self.grid.data, dtype=bool)
        reachable[25:36, 5:56] = True
        self.grid.data[reachable] = 0
        self.prepared["reachable"] = reachable
        start_x, start_y = self.grid.world(30, 30)
        self.pose.update(x=float(start_x), y=float(start_y))

        result = self.logic.resolve_open_space_middle_step(
            self.prepared, self.pose
        )

        self.assertEqual(result["phase"], "stabilize_room_map")
        self.assertEqual(result["recovery_reason"], "no_room_core")
        self.assertEqual(result["state"]["recovery_moves"], 1)

    def test_exit_room_explores_frontier_crosses_boundary_then_completes(self):
        self.grid.data[:] = 100
        reachable = np.zeros_like(self.grid.data, dtype=bool)
        reachable[5:56, 3:28] = True
        reachable[27:34, 28:34] = True
        reachable[3:58, 34:59] = True
        self.grid.data[reachable] = 0
        self.prepared["reachable"] = reachable
        self.pose.update(x=-1.5, y=0.0)
        frontier_x, frontier_y = self.grid.world(20, 10)
        self.prepared["frontiers"] = [{
            "id": "F1", "kind": "frontier",
            "x": float(frontier_x), "y": float(frontier_y),
            "yaw": 0.0, "information_gain_m2": 1.0,
        }]

        explore = self.logic.resolve_exit_room_step(
            self.prepared, self.pose
        )
        self.assertEqual(explore["phase"], "explore_room_frontier")
        self.assertFalse(explore["complete"])

        self.prepared["frontiers"] = []
        cross = self.logic.resolve_exit_room_step(
            self.prepared, self.pose, explore["state"]
        )
        self.assertEqual(cross["phase"], "cross_room_boundary")
        self.assertGreater(cross["destination"]["x"], 0.0)

        outside_x, outside_y = self.grid.world(45, 30)
        outside_pose = {
            **self.pose, "x": float(outside_x), "y": float(outside_y)
        }
        complete = self.logic.resolve_exit_room_step(
            self.prepared, outside_pose, cross["state"]
        )
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["phase"], "outside")

    def test_exit_room_accepts_distant_frontier_connected_to_room_core(self):
        self.grid.data[:] = 100
        reachable = np.zeros_like(self.grid.data, dtype=bool)
        # A broad starting chamber has an open core. Its narrow reachable arm
        # extends much farther than the former fixed assignment radius.
        reachable[20:41, 5:26] = True
        reachable[29:32, 26:56] = True
        self.grid.data[reachable] = 0
        self.prepared["reachable"] = reachable
        origin_x, origin_y = self.grid.world(15, 30)
        self.pose.update(x=float(origin_x), y=float(origin_y))
        frontier_x, frontier_y = self.grid.world(54, 30)
        self.prepared["frontiers"] = [{
            "id": "F1", "kind": "frontier",
            "x": float(frontier_x), "y": float(frontier_y),
            "yaw": 0.0, "information_gain_m2": 1.0,
        }]

        result = self.logic.resolve_exit_room_step(
            self.prepared, self.pose
        )

        self.assertEqual(result["phase"], "explore_room_frontier")
        self.assertAlmostEqual(result["destination"]["x"], frontier_x)
        self.assertAlmostEqual(result["destination"]["y"], frontier_y)

    def test_exit_room_prefers_useful_gain_over_nearest_frontier(self):
        self.prepared["frontiers"] = [
            {
                "id": "F-close", "kind": "frontier",
                "x": 0.5, "y": 0.0, "yaw": 0.0,
                "information_gain_m2": 1.0,
            },
            {
                "id": "F-informative", "kind": "frontier",
                "x": 2.0, "y": 0.0, "yaw": 0.0,
                "information_gain_m2": 1.4,
            },
        ]

        result = self.logic.resolve_exit_room_step(
            self.prepared, self.pose
        )

        self.assertAlmostEqual(result["destination"]["x"], 2.0)

    def test_exit_room_prefers_nearer_frontier_when_gain_is_similar(self):
        self.prepared["frontiers"] = [
            {
                "id": "F-close", "kind": "frontier",
                "x": 0.5, "y": 0.0, "yaw": 0.0,
                "information_gain_m2": 1.0,
            },
            {
                "id": "F-slightly-better", "kind": "frontier",
                "x": 2.0, "y": 0.0, "yaw": 0.0,
                "information_gain_m2": 1.1,
            },
        ]

        result = self.logic.resolve_exit_room_step(
            self.prepared, self.pose
        )

        self.assertAlmostEqual(result["destination"]["x"], 0.5)

    def test_exit_room_labels_frontier_outside_clearance_eroded_mask(self):
        self.grid.data[:] = 100
        traversable = np.zeros_like(self.grid.data, dtype=bool)
        traversable[20:41, 5:26] = True
        traversable[29:32, 26:56] = True
        self.grid.data[traversable] = 0
        self.prepared["costs"] = np.zeros_like(
            self.grid.data, dtype=np.int16
        )
        self.prepared["reachable"] = traversable.copy()
        # Model the clearance erosion used by prepare(): the actual frontier
        # remains traversable but is not a valid clearance-safe endpoint cell.
        self.prepared["reachable"][29:32, 53:56] = False
        origin_x, origin_y = self.grid.world(15, 30)
        self.pose.update(x=float(origin_x), y=float(origin_y))
        frontier_x, frontier_y = self.grid.world(54, 30)
        self.prepared["frontiers"] = [{
            "id": "F1", "kind": "frontier",
            "x": float(frontier_x), "y": float(frontier_y),
            "yaw": 0.0, "information_gain_m2": 1.0,
        }]

        result = self.logic.resolve_exit_room_step(
            self.prepared, self.pose
        )

        self.assertEqual(result["phase"], "explore_room_frontier")
        self.assertAlmostEqual(result["destination"]["x"], frontier_x)
        self.assertAlmostEqual(result["destination"]["y"], frontier_y)

    def test_exit_room_does_not_claim_frontier_owned_by_other_room_core(self):
        self.grid.data[:] = 100
        reachable = np.zeros_like(self.grid.data, dtype=bool)
        reachable[5:56, 3:28] = True
        reachable[27:34, 28:34] = True
        reachable[3:58, 34:59] = True
        self.grid.data[reachable] = 0
        self.prepared["reachable"] = reachable
        origin_x, origin_y = self.grid.world(15, 30)
        self.pose.update(x=float(origin_x), y=float(origin_y))
        frontier_x, frontier_y = self.grid.world(50, 30)
        self.prepared["frontiers"] = [{
            "id": "F-other", "kind": "frontier",
            "x": float(frontier_x), "y": float(frontier_y),
            "yaw": 0.0, "information_gain_m2": 1.0,
        }]

        result = self.logic.resolve_exit_room_step(
            self.prepared, self.pose
        )

        self.assertEqual(result["phase"], "cross_room_boundary")
        self.assertNotAlmostEqual(result["destination"]["x"], frontier_x)

    def test_exit_room_recovers_safely_when_room_core_is_missing(self):
        self.grid.data[:] = 100
        reachable = np.zeros_like(self.grid.data, dtype=bool)
        # This strip is navigable but too narrow for the 0.65 m room core.
        reachable[25:36, 5:56] = True
        self.grid.data[reachable] = 0
        self.prepared["reachable"] = reachable
        origin_x, origin_y = self.grid.world(30, 30)
        self.pose.update(x=float(origin_x), y=float(origin_y))
        frontier_x, frontier_y = self.grid.world(50, 30)
        self.prepared["frontiers"] = [{
            "id": "F1", "kind": "frontier",
            "x": float(frontier_x), "y": float(frontier_y),
            "yaw": 0.0, "information_gain_m2": 1.0,
        }]

        result = self.logic.resolve_exit_room_step(
            self.prepared, self.pose
        )

        destination = result["destination"]
        distance = math.hypot(
            destination["x"] - self.pose["x"],
            destination["y"] - self.pose["y"],
        )
        self.assertEqual(result["phase"], "stabilize_room_map")
        self.assertEqual(result["recovery_reason"], "no_room_core")
        self.assertGreaterEqual(
            distance, self.logic.cfg.exit_recovery_min_move_m
        )
        self.assertLessEqual(
            distance, self.logic.cfg.exit_recovery_max_move_m
        )

    def test_exit_room_recovers_when_no_frontier_or_exit_is_available(self):
        self.prepared["frontiers"] = []

        result = self.logic.resolve_exit_room_step(
            self.prepared, self.pose
        )

        self.assertEqual(result["phase"], "stabilize_room_map")
        self.assertEqual(result["recovery_reason"], "no_exit_evidence")

    def test_exit_room_map_recovery_is_bounded(self):
        self.prepared["frontiers"] = []
        state = {
            "origin": {"x": self.pose["x"], "y": self.pose["y"]},
            "visited_frontiers": [],
            "recovery_points": [],
            "recovery_moves": self.logic.cfg.exit_recovery_max_moves,
            "passes": self.logic.cfg.exit_recovery_max_moves,
        }

        with self.assertRaisesRegex(ValueError, "map recovery are exhausted"):
            self.logic.resolve_exit_room_step(
                self.prepared, self.pose, state
            )

    def test_exit_wall_mask_ignores_small_furniture(self):
        self.grid.data[:] = 0
        self.grid.data[8, 5:35] = 100
        self.grid.data[25:28, 25:28] = 100

        walls = self.logic._exit_wall_mask(self.grid)

        self.assertTrue(np.all(walls[8, 5:35]))
        self.assertFalse(np.any(walls[25:28, 25:28]))

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
            self.grid, winner, winner["yaw"], 85.0, 2.0
        )
        opposite = self.logic.visibility_mask(
            self.grid, winner, winner["yaw"] + math.pi, 85.0, 2.0
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
        self.executor = FindTargetExecutor.__new__(FindTargetExecutor)
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
        navigation = MapNavigationAction.__new__(MapNavigationAction)
        navigation.camera = camera
        navigation.request_map_snapshot = request_snapshot
        navigation.action_id = "find-1:explore-1"
        navigation._overlay_action_id = "find-1"
        navigation._overlay_revision = 2
        navigation._mode = "exploration"
        navigation.save_map_snapshot = Mock()
        navigation._result_data = {}
        navigation.see_action = Mock()
        navigation.see_action.move_to_region = AsyncMock(
            return_value=ActionResult("camera", "move_camera", "succeeded")
        )
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
        frame = type("Frame", (), {
            "captured_at": float("inf"),
            "tracking_bgr": np.zeros((4, 4, 3), dtype=np.uint8),
        })()
        camera = Mock()
        camera.snapshot.return_value = frame
        navigation = self._navigation_for_capture(camera, request_snapshot)
        navigation.see_action = Mock()
        navigation.see_action.move_to_region = AsyncMock(
            return_value=ActionResult("camera", "move_camera", "succeeded")
        )

        captured, camera_jpeg = await navigation._capture()

        self.assertEqual(captured["metadata"]["snapshot_id"], "snapshot-1")
        self.assertTrue(camera_jpeg)
        camera.snapshot.assert_called_once_with()
        navigation.save_map_snapshot.assert_called_once_with(captured)

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

    async def test_single_exploration_pose_skips_vision_selection(self):
        navigation = MapNavigationAction.__new__(MapNavigationAction)
        navigation._mode = "exploration"
        navigation._candidates = {
            "E1": {**candidate("E1", "exploration", 0.0, 0.5), "yaw": 1.1}
        }
        navigation._select_pose = AsyncMock()

        selection = await navigation._choose_pose({}, b"unused-camera-jpeg")
        destination = navigation._resolve_destination(
            selection, {"metadata": {"robot_pose": {"yaw": -0.4}}}
        )

        navigation._select_pose.assert_not_awaited()
        self.assertEqual(selection["pose_id"], "E1")
        self.assertEqual(selection["heading"], "map_candidate")
        self.assertAlmostEqual(destination["yaw"], 1.1)

    async def test_two_exploration_poses_use_vision_selection(self):
        navigation = MapNavigationAction.__new__(MapNavigationAction)
        navigation._mode = "exploration"
        navigation._candidates = {
            "E1": candidate("E1", "exploration", 0.5, 0.0),
            "EF": candidate("EF", "exploration", 2.0, 0.0),
        }
        expected = {"decision": "move", "pose_id": "E1", "heading": "forward"}
        navigation._select_pose = AsyncMock(return_value=expected)

        selection = await navigation._choose_pose({"snapshot": True}, b"camera")

        self.assertEqual(selection, expected)
        navigation._select_pose.assert_awaited_once_with(
            {"snapshot": True}, b"camera"
        )

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

        frame = type("Frame", (), {
            "captured_at": float("inf"),
            "tracking_bgr": np.zeros((4, 4, 3), dtype=np.uint8),
        })()
        camera = Mock()
        camera.snapshot.return_value = frame
        navigation = self._navigation_for_capture(camera, request_snapshot)
        navigation._overlay_action_id = None
        navigation.action_id = "navigate-1"
        navigation.request_map_snapshot = request_snapshot
        navigation.see_action = Mock()
        navigation.see_action.move_to_region = AsyncMock(
            return_value=ActionResult("camera", "move_camera", "succeeded")
        )

        captured, camera_jpeg = await navigation._capture()

        self.assertEqual(request["operation"], "snapshot")
        self.assertEqual(request["action_id"], "navigate-1")
        self.assertNotIn("crop_size_m", request)
        self.assertEqual(
            captured["metadata"]["snapshot_request_id"],
            request["request_id"],
        )
        self.assertTrue(camera_jpeg)

    async def test_pending_render_contains_frontiers_and_dispatched_pose(self):
        request = {}
        saved = Mock()

        async def request_snapshot(payload):
            request.update(payload)
            return {
                "jpeg_bytes": b"debug-map",
                "metadata": {"snapshot_request_id": payload["request_id"]},
            }

        navigation = MapNavigationAction.__new__(MapNavigationAction)
        navigation.request_map_snapshot = request_snapshot
        navigation.save_map_snapshot = saved
        navigation.action_id = "find-1:explore-1"
        navigation._overlay_action_id = "find-1"
        navigation._overlay_revision = 3
        navigation._mode = "exploration"
        navigation.map_crop_size_m = 6.0
        navigation._result_data = {}
        destination = {
            "x": 1.0, "y": 2.0, "yaw": 0.5, "frame_id": "map",
        }

        await navigation._save_pending_map_render(destination)

        self.assertTrue(request["render_frontiers"])
        self.assertEqual(request["pending_navigation_pose"], destination)
        self.assertNotIn("crop_size_m", request)
        saved.assert_called_once()

    async def test_local_capture_uses_one_aligned_snapshot_without_overlay_history(self):
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
        navigation._mode = "local"
        navigation._overlay_action_id = None
        navigation._overlay_revision = 0
        navigation.map_crop_size_m = 4.0
        navigation.send_map_overlay = AsyncMock()
        navigation.see_action = Mock()
        navigation.see_action.move_to_region = AsyncMock(
            return_value=ActionResult("camera", "move_camera", "succeeded")
        )

        captured, camera_jpeg = await navigation._capture()

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["overlay_revision"], 0)
        self.assertEqual(captured["metadata"]["snapshot_id"], "snapshot-1")
        self.assertTrue(camera_jpeg)
        navigation.send_map_overlay.assert_not_awaited()

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
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
        executor.action_id = "find-1"
        executor.target = "bottle"
        executor._search_waypoint_count = 0
        executor._completed_steps = []
        executor._publish_map_overlay = AsyncMock(return_value=1)
        executor._remember_search_pose = Mock()
        executor.map_navigation_action = type("Navigation", (), {})()
        executor.map_navigation_action.start_find_target_step = AsyncMock(
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


class SemanticNavigationHeadingTests(unittest.TestCase):
    @unittest.skip("local navigation no longer uses vision-selected map poses")
    def test_local_guidance_is_strictly_one_shot(self):
        guidance = " ".join(MODE_GUIDANCE["local"].split())
        contract = " ".join(MAP_GUIDANCE.split())
        self.assertIn("immediate local movement", guidance)
        self.assertIn("Do not explore", guidance)
        self.assertIn("no later navigation assessment", contract)

    def test_semantic_heading_is_relative_to_the_upright_robot_map(self):
        navigation = MapNavigationAction.__new__(MapNavigationAction)
        navigation._candidates = {"7": candidate("7", "local", 1.0, 2.0)}
        destination = navigation._resolve_destination(
            {"pose_id": "7", "heading": "right"},
            {"metadata": {"robot_pose": {"yaw": math.pi / 2.0}}},
        )
        self.assertAlmostEqual(destination["yaw"], 0.0)
        self.assertEqual(destination["heading"], "right")
        self.assertIn("sampled_yaw", destination)

    def test_selection_tool_allows_only_one_move_or_blocked(self):
        navigation = MapNavigationAction.__new__(MapNavigationAction)
        definitions = json.loads(
            (ROOT / "tools" / "vision_oob_tools.json").read_text()
        )
        navigation.selection_tool_template = next(
            tool for tool in definitions if tool["name"] == "select_navigation_pose"
        )
        navigation._candidates = {"7": candidate("7", "local", 1.0, 0.0)}
        properties = navigation._build_selection_tool()["parameters"]["properties"]
        self.assertEqual(properties["decision"]["enum"], ["move", "blocked"])
        self.assertEqual(properties["pose_id"]["enum"], ["7", None])

    def test_selection_content_has_only_current_map_and_frame(self):
        content = MapNavigationAction._selection_content(
            "prompt", b"map-one", b"vision-one"
        )
        labels = [item["text"] for item in content if item["type"] == "input_text"]
        self.assertEqual(labels, [
            "prompt", "ROBOT-RELATIVE MAP (Image 1)",
            "CAMERA OR GROUNDED CLUE (Image 2)",
        ])
        self.assertEqual(
            len([item for item in content if item["type"] == "input_image"]), 2
        )


class SemanticNavigationResponseTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def navigation():
        navigation = MapNavigationAction.__new__(MapNavigationAction)
        navigation.active = True
        navigation._stop_requested = False
        navigation._mode = "local"
        navigation.action_id = "navigate-1"
        navigation._request_id = "request-1"
        navigation._snapshot_id = "snapshot-1"
        navigation._candidates = {
            "7": candidate("7", "local", 1.0, 0.0)
        }
        navigation._selection_future = asyncio.get_running_loop().create_future()
        return navigation

    async def test_move_accepts_one_shot_motion(self):
        navigation = self.navigation()
        await navigation.select_navigation_pose({
            "decision": "move",
            "pose_id": "7",
            "heading": "forward_left",
            "reasoning": "One short diagonal move fulfills the bounded command.",
        }, {
            "kind": "navigation_selection",
            "action_id": "navigate-1",
            "request_id": "request-1",
            "snapshot_id": "snapshot-1",
        })

        selection = navigation._selection_future.result()
        self.assertEqual(selection["decision"], "move")
        self.assertEqual(selection["heading"], "forward_left")

    async def test_removed_looping_decision_is_rejected(self):
        navigation = self.navigation()
        await navigation.select_navigation_pose({
            "decision": "move_and_finish",
            "pose_id": "7",
            "heading": "forward",
            "reasoning": "Old two-stage decision.",
        }, {
            "kind": "navigation_selection",
            "action_id": "navigate-1",
            "request_id": "request-1",
            "snapshot_id": "snapshot-1",
        })

        with self.assertRaises(NavigationError):
            navigation._selection_future.result()


class ContextualClueContractTests(unittest.TestCase):
    def test_candidate_requires_clue_and_type(self):
        self.assertTrue(SearchAction._contextual_clue_is_valid({
            "result": "candidate",
            "candidate_type": "context",
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

    def test_context_has_one_explicit_budget(self):
        config = FindLoopConfig()
        self.assertEqual(config.max_context_waypoints, 4)

    def test_exploration_prompt_explains_near_and_far_tradeoff(self):
        guidance = MODE_GUIDANCE["exploration"]
        self.assertIn("EF", guidance)
        self.assertIn("skip openings or useful detail", guidance)
        self.assertIn("fresh centered view", guidance)

    def test_vision_selected_modes_share_one_map_guide(self):
        for mode in ("visual", "context", "exploration"):
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


@unittest.skip("semantic destination was folded into find_object(place)")
class SemanticDestinationContractTests(unittest.IsolatedAsyncioTestCase):
    def semantic_search(self):
        search = SearchAction.__new__(SearchAction)
        search.active = True
        search.search_mode = "semantic"
        search.search_id = "semantic-1"
        search.pending_request_id = "request-1"
        search.pan_angle = 95.0
        search.tilt_angle = 90.0
        search.current_initial_frame = {
            "jpeg_bytes": b"center", "pan_angle": 95.0, "tilt_angle": 90.0,
        }
        search.current_candidate = None
        search.sweep_pan_positions = ("leftmost", "rightmost")
        search.capture_sweep_batch = AsyncMock()
        search.move_to_candidate_frame = AsyncMock()
        search.complete_searching = AsyncMock()
        return search

    def test_semantic_schema_has_only_candidate_or_no_clue(self):
        search = self.semantic_search()
        definitions = json.loads(
            (ROOT / "tools" / "vision_oob_tools.json").read_text()
        )
        search_tool = next(
            tool for tool in definitions if tool["name"] == "assess_frame_search"
        )
        with patch("actions.search_action.ASSESS_FRAME_SEARCH_TOOL", search_tool):
            properties = search._assessment_tool(batch=False)["parameters"]["properties"]
        self.assertEqual(properties["result"]["enum"], ["candidate", "no_clue"])
        self.assertEqual(properties["candidate_type"]["enum"], ["destination", None])
        self.assertEqual(properties["target_position"]["enum"], [None])

    async def test_first_actionable_frame_stops_sweep_immediately(self):
        search = self.semantic_search()
        await search.assess_frame_search({
            "result": "candidate",
            "candidate_type": "destination",
            "target_position": None,
            "contextual_clue": "The requested hallway and its entrance are clear ahead.",
        }, {
            "kind": "visual_search_assessment",
            "search_id": "semantic-1",
            "request_id": "request-1",
        })
        search.move_to_candidate_frame.assert_awaited_once()
        search.complete_searching.assert_awaited_once_with(
            status="failed",
            outcome="contextual_clue",
            reason_code="SEARCH_CONTEXTUAL_CLUE",
        )
        search.capture_sweep_batch.assert_not_awaited()

    async def test_goal_executor_candidate_uses_one_final_move_without_approach(self):
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
        executor.goal = "navigate_semantic"
        executor.action_id = "semantic-1"
        executor.target = "the hallway"
        executor._completed_steps = []
        executor._search_poses = []
        executor._search_waypoint_count = 0
        executor._approach_result_data = {}
        executor._search_limit_result = lambda: None
        observation = {
            "jpeg_bytes": b"frame",
            "candidate_type": "destination",
            "contextual_clue": "The hallway itself is clearly visible ahead.",
        }
        executor.search_action = type(
            "Search", (), {"last_observation_frame": observation}
        )()
        executor.track_action = type("Tracker", (), {"active": False})()
        executor.see_action = type("See", (), {})()
        executor.see_action.move_to_region = AsyncMock(return_value=ActionResult(
            "camera", "see_action", "succeeded",
        ))
        executor._search = AsyncMock(return_value=ActionResult(
            "scan", "search_action", "failed",
            reason_code="SEARCH_CONTEXTUAL_CLUE",
        ))
        executor._navigate_search_step = AsyncMock(return_value=ActionResult(
            "nav", "navigate_action", "succeeded",
        ))
        executor._track_approach_and_reacquire = AsyncMock()

        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        executor._navigate_search_step.assert_awaited_once()
        self.assertEqual(
            executor._navigate_search_step.await_args.kwargs["mode"], "destination"
        )
        executor._track_approach_and_reacquire.assert_not_awaited()


class ContextualClueLoopTests(unittest.IsolatedAsyncioTestCase):
    def executor(self, *, config=None):
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
        executor.find_loop = config or FindLoopConfig()
        executor.action_id = "find-1"
        executor.target = "water bottle"
        executor._search_poses = []
        executor._search_waypoint_count = 0
        executor._completed_steps = []
        executor._approach_result_data = {}
        executor.search_action = type(
            "Search", (), {"last_observation_frame": None}
        )()
        executor.track_action = type("Tracker", (), {"active": False})()
        executor.see_action = type("See", (), {})()
        executor.see_action.move_to_region = AsyncMock(return_value=ActionResult(
            "camera", "see_action", "succeeded",
        ))
        executor._search_limit_result = lambda: None
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
             self.clue("context")),
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
        self.assertEqual(navigation_call.args[0]["candidate_type"], "context")
        self.assertEqual(navigation_call.args[0]["movement_limit"], 4)
        executor._track_approach_and_reacquire.assert_awaited_once_with(0)

    async def test_context_candidate_routes_to_unified_context_mode(self):
        executor = self.executor()
        results = [
            (ActionResult("scan-1", "search_action", "failed",
                          reason_code="SEARCH_CONTEXTUAL_CLUE"),
             self.clue("context")),
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
        self.assertEqual(navigation_call.kwargs["mode"], "context")

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

    async def test_visual_candidate_uses_map_fallback_when_tracking_is_unstable(self):
        executor = self.executor()
        visual = self.clue("visual", "The bottle is visible ahead.")
        scans = [
            ActionResult(
                "visual", "search_action", "failed",
                reason_code="SEARCH_CONTEXTUAL_CLUE",
            ),
            ActionResult("found", "search_action", "succeeded"),
        ]

        async def search(*_args, **_kwargs):
            executor.search_action.last_observation_frame = (
                visual if len(scans) == 2 else None
            )
            return scans.pop(0)

        executor._search = AsyncMock(side_effect=search)
        executor._track_approach_and_reacquire = AsyncMock(side_effect=[
            ActionResult(
                "track", "track_action", "failed",
                reason_code="STABLE_SEED_TIMEOUT",
            ),
            ActionResult("verified", "track_action", "succeeded"),
        ])

        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        executor._navigate_search_step.assert_awaited_once_with(
            visual,
            mode="visual",
            step="visual_fallback_1",
        )
        self.assertEqual(executor._search.await_count, 2)
        self.assertEqual(executor._track_approach_and_reacquire.await_count, 2)

    async def test_found_candidate_uses_same_map_fallback(self):
        executor = self.executor()
        found_frame = self.clue("found", "Confirmed bottle in this view.")
        executor.search_action.last_observation_frame = found_frame
        executor._search = AsyncMock(return_value=ActionResult(
            "found", "search_action", "succeeded"
        ))
        executor._track_approach_and_reacquire = AsyncMock(return_value=ActionResult(
            "track", "track_action", "failed",
            reason_code="OBJECT_DETECTION_FAILED",
        ))
        executor._navigate_search_step = AsyncMock(return_value=ActionResult(
            "fallback", "navigate_action", "failed",
            reason_code="NAVIGATION_BLOCKED",
        ))
        executor.find_loop = FindLoopConfig(max_exploration_waypoints=0)

        result = await executor._search_environment()

        self.assertEqual(result.reason_code, "OBJECT_SEARCH_EXHAUSTED")
        executor._navigate_search_step.assert_any_await(
            found_frame,
            mode="visual",
            step="visual_fallback_1",
        )

    async def test_person_acquisition_skips_approach(self):
        executor = self.executor()
        executor.target = "person"
        executor.target_kind = "person"
        executor.approach_target = False
        executor._search = AsyncMock(return_value=ActionResult(
            "scan", "search_action", "succeeded", target="person"
        ))
        executor._acquire_person_without_approach = AsyncMock(
            return_value=ActionResult(
                "acquired", "track_action", "succeeded", target="person"
            )
        )

        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        executor._acquire_person_without_approach.assert_awaited_once_with(0)
        executor._track_approach_and_reacquire.assert_not_awaited()

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

    async def test_context_budget_is_consumed_by_repeated_frames(self):
        executor = self.executor(config=FindLoopConfig(max_context_waypoints=2))

        async def search(*_args, **_kwargs):
            executor.search_action.last_observation_frame = self.clue("context")
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

    async def test_contextual_type_no_longer_resets_the_unified_budget(self):
        executor = self.executor(config=FindLoopConfig(
            max_context_waypoints=1,
        ))
        observations = [self.clue("context"), self.clue("context"), None]

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
            ["context", "exploration"],
        )
        context_calls = [
            call for call in executor._navigate_search_step.await_args_list
            if call.kwargs["mode"] == "context"
        ]
        self.assertEqual(
            [call.args[0]["remaining_waypoints"] for call in context_calls], [1]
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
                executor.search_action.last_observation_frame = self.clue("context")
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
        self.assertEqual(executor.see_action.move_to_region.await_count, 2)
        executor._navigate_search_step.assert_awaited_once()
        self.assertEqual(
            executor._navigate_search_step.await_args.kwargs["mode"], "context"
        )


class FollowPersonExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_visual_failure_preserves_lidar_session_for_follow(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker.active = True
        tracker.target = "torso_center"
        tracker.action_id = "follow-1:track"
        tracker._tracking_task = None
        tracker._continuous_person_reacquisition = True
        tracker.send_robot_command = AsyncMock(return_value={
            "status": "accepted"
        })
        expected = ActionResult(
            "follow-1:track", "track_action", "failed", target="person"
        )
        tracker.complete_tracking = Mock(return_value=expected)

        result = await tracker.stop_tracking(
            reason_code="PERSON_LOST",
            status="failed",
            outcome="target_lost_after_reacquisition",
        )

        self.assertIs(result, expected)
        tracker.send_robot_command.assert_awaited_once_with({
            "command": "stop_tracking",
            "action_id": "follow-1:track",
            "reset_camera": False,
        })

    async def test_acquisition_searches_then_starts_camera_tracking(self):
        tracker = type("Tracker", (), {"active": False})()
        tracker.stop_tracking = AsyncMock()
        goal_executor = type("Goal", (), {
            "active": False,
            "action_id": None,
        })()
        goal_executor.start = AsyncMock(return_value=ActionResult(
            "follow-1:acquire-person", "find_target", "running",
            target="person",
        ))
        goal_executor.wait_until_finished = AsyncMock(return_value=ActionResult(
            "follow-1:acquire-person", "find_target", "succeeded",
            target="person", outcome="person_acquired",
        ))
        send_robot_command = AsyncMock()
        executor = FollowPersonExecutor(
            goal_executor, tracker, send_robot_command
        )
        executor.action_id = "follow-1"

        result = await executor._acquire_target()

        self.assertEqual(result.status, "succeeded")
        goal_executor.start.assert_awaited_once_with(
            target="person",
            action_id="follow-1:acquire-person",
            target_kind="person",
            approach=False,
            continuous_person_reacquisition=True,
        )
        goal_executor.wait_until_finished.assert_awaited_once_with()
        send_robot_command.assert_not_awaited()

    async def test_acquisition_adopts_existing_person_tracking(self):
        tracker = type("Tracker", (), {
            "active": True,
            "target": "person",
            "action_id": "follow-existing:track",
        })()
        tracker.adopt_person_tracking = AsyncMock(return_value=True)
        tracker.wait_for_stable_target_seed = AsyncMock(return_value={
            "target": "person",
            "session_id": "follow-existing:track",
        })
        goal_executor = type("Goal", (), {
            "active": False,
            "action_id": None,
        })()
        goal_executor.start = AsyncMock()
        executor = FollowPersonExecutor(
            goal_executor, tracker, AsyncMock()
        )
        executor.action_id = "follow-existing"

        result = await executor._acquire_target()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.outcome, "person_tracking_reused")
        tracker.adopt_person_tracking.assert_awaited_once_with(
            "follow-existing:track",
            continuous_person_reacquisition=True,
        )
        tracker.wait_for_stable_target_seed.assert_awaited_once_with(
            target="person",
            session_id="follow-existing:track",
            timeout=10.0,
        )
        goal_executor.start.assert_not_awaited()

    async def test_run_requests_one_explicit_bridge_follow(self):
        executor = FollowPersonExecutor.__new__(FollowPersonExecutor)
        executor.active = True
        executor.action_id = "follow-1"
        executor._stop_requested = False
        executor._lidar_lost = False
        executor._acquire_target = AsyncMock(return_value=ActionResult(
            "acquire", "follow_person", "succeeded", target="person"
        ))
        tracking_finished = asyncio.Event()
        executor.track_action = type("Tracker", (), {})()
        executor.track_action.wait_until_finished = AsyncMock(
            side_effect=tracking_finished.wait
        )
        executor.send_robot_command = AsyncMock(return_value={
            "status": "accepted",
            "message": "Waiting for lidar person pose",
        })

        task = asyncio.create_task(executor._run())
        while executor.send_robot_command.await_count == 0:
            await asyncio.sleep(0)

        executor.send_robot_command.assert_awaited_once_with({
            "command": "follow_action",
            "action_id": "follow-1",
        })
        self.assertTrue(executor.active)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_stop_uses_explicit_bridge_stop(self):
        tracker = type("Tracker", (), {"active": True})()
        tracker.stop_tracking = AsyncMock()
        goal_executor = type("Goal", (), {
            "active": False,
            "action_id": None,
        })()
        send_robot_command = AsyncMock(return_value={"status": "accepted"})
        executor = FollowPersonExecutor(
            goal_executor, tracker, send_robot_command
        )
        executor.active = True
        executor.action_id = "follow-1"
        executor.target = "person"
        executor.completion_future = asyncio.get_running_loop().create_future()
        executor._runner = None
        executor._stop_requested = False

        result = await executor.stop()

        self.assertEqual(result.status, "cancelled")
        send_robot_command.assert_awaited_once_with({
            "command": "stop_follow_action",
            "action_id": "follow-1",
        })
        tracker.stop_tracking.assert_not_awaited()

    async def test_camera_tracking_end_restarts_camera_without_stopping_follow(self):
        executor = FollowPersonExecutor.__new__(FollowPersonExecutor)
        executor.active = True
        executor.action_id = "follow-1"
        executor.target = "person"
        executor._stop_requested = False
        executor._lidar_lost = False
        executor.completion_future = asyncio.get_running_loop().create_future()
        executor._acquire_target = AsyncMock(return_value=ActionResult(
            "acquire", "follow_person", "succeeded", target="person"
        ))
        visual_restarted = asyncio.Event()
        executor.track_action = type("Tracker", (), {})()
        wait_calls = 0

        async def wait_until_finished():
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                return ActionResult(
                "track", "track_action", "failed", target="person",
                outcome="target_lost_after_reacquisition",
                reason_code="PERSON_LOST",
                )
            await visual_restarted.wait()

        executor.track_action.wait_until_finished = AsyncMock(
            side_effect=wait_until_finished
        )
        executor.track_action.start_tracking = AsyncMock(return_value=ActionResult(
            "track", "track_action", "running", target="person"
        ))
        executor.send_robot_command = AsyncMock(return_value={
            "status": "accepted"
        })

        task = asyncio.create_task(executor._run())
        while executor.track_action.start_tracking.await_count == 0:
            await asyncio.sleep(0)

        self.assertEqual(
            [call.args[0]["command"] for call in
             executor.send_robot_command.await_args_list],
            ["follow_action"],
        )
        executor.track_action.start_tracking.assert_awaited_once_with(
            target="torso center",
            action_id="follow-1:track",
            allow_grounding_dino=True,
            continuous_person_reacquisition=True,
        )
        self.assertTrue(executor.active)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_lidar_lost_ends_follow_action(self):
        executor = FollowPersonExecutor.__new__(FollowPersonExecutor)
        executor.active = True
        executor.action_id = "follow-1"
        executor.target = "person"
        executor._stop_requested = False
        executor._lidar_lost = False
        executor.completion_future = asyncio.get_running_loop().create_future()
        executor.track_action = type("Tracker", (), {"active": True})()
        executor.track_action.stop_tracking = AsyncMock()
        executor.send_robot_command = AsyncMock(return_value={
            "status": "accepted"
        })

        handled = await executor.handle_navigation_event({
            "type": "event",
            "event": "person_tracker",
            "status": "lost",
            "action_id": "follow-1",
        })

        self.assertTrue(handled)
        executor.send_robot_command.assert_awaited_once_with({
            "command": "stop_follow_action",
            "action_id": "follow-1",
        })
        executor.track_action.stop_tracking.assert_not_awaited()
        result = executor.completion_future.result()
        self.assertEqual(result.outcome, "target_lost")
        self.assertEqual(result.reason_code, "LIDAR_TRACK_LOST")


class PostApproachReacquisitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_acquisition_only_faces_fresh_lidar_pose_and_returns_tracking(self):
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
        executor.action_id = "point-1:acquire-person"
        executor.target_kind = "person"
        executor.approach_target = False
        executor._completed_steps = []
        executor.local_navigation_action = type("Local", (), {})()
        executor.local_navigation_action.start = AsyncMock(return_value=ActionResult(
            "face", "navigate_action", "succeeded", target="face_person"
        ))
        executor._search_environment = AsyncMock(return_value=ActionResult(
            "acquired", "track_action", "succeeded", target="person",
            data={"tracking_active": True},
        ))
        executor._complete = Mock()
        previous_person = robot_state.get("person")
        previous_pose = robot_state.get("pose")
        robot_state["person"] = {
            "x": 1.0, "y": 0.0, "timestamp": time.monotonic()
        }
        robot_state["pose"] = {"x": 0.0, "y": 0.0, "yaw": math.pi}

        try:
            await executor._run()
        finally:
            robot_state["person"] = previous_person
            robot_state["pose"] = previous_pose

        executor.local_navigation_action.start.assert_awaited_once_with(
            "face_person", "point-1:acquire-person:face_known_person"
        )
        self.assertIn("faced_known_person", executor._completed_steps)
        executor._complete.assert_called_once_with(
            status="succeeded",
            outcome="person_acquired",
            data={"tracking_active": True},
        )

    async def test_acquisition_only_skips_lidar_turn_within_sixty_degrees(self):
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
        executor.action_id = "point-2:acquire-person"
        executor.target_kind = "person"
        executor.approach_target = False
        executor._completed_steps = []
        executor.local_navigation_action = type("Local", (), {})()
        executor.local_navigation_action.start = AsyncMock()
        executor._search_environment = AsyncMock(return_value=ActionResult(
            "acquired", "track_action", "succeeded", target="person",
            data={"tracking_active": True},
        ))
        executor._complete = Mock()
        previous_person = robot_state.get("person")
        previous_pose = robot_state.get("pose")
        robot_state["person"] = {
            "x": 1.0, "y": 1.0, "timestamp": time.monotonic()
        }
        robot_state["pose"] = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        try:
            await executor._run()
        finally:
            robot_state["person"] = previous_person
            robot_state["pose"] = previous_pose

        executor.local_navigation_action.start.assert_not_awaited()
        self.assertNotIn("faced_known_person", executor._completed_steps)
        executor._search_environment.assert_awaited_once_with()

    async def test_acquisition_returns_after_tracking_starts(self):
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
        executor.action_id = "follow-1:acquire-person"
        executor.target = "person"
        executor.target_kind = "person"
        executor._completed_steps = []
        executor._track = AsyncMock(return_value=ActionResult(
            "track", "track_action", "succeeded", target="person",
            data={"detection_source": "yolo_pose"},
        ))
        executor.track_action = type("Tracker", (), {})()
        executor.track_action.wait_for_stable_target_seed = AsyncMock(
            side_effect=AssertionError("find must not wait for a seed")
        )

        result = await executor._acquire_person_without_approach(0)

        self.assertEqual(result.status, "succeeded")
        self.assertTrue(result.data["tracking_active"])
        self.assertEqual(result.data["detection_source"], "yolo_pose")
        self.assertEqual(result.outcome, "target_tracking_started")
        executor.track_action.wait_for_stable_target_seed.assert_not_awaited()

    def setUp(self):
        self.saved_pose = robot_state.get("pose")
        self.saved_camera = dict(robot_state.get("camera") or {})

    def tearDown(self):
        robot_state["pose"] = self.saved_pose
        robot_state["camera"].clear()
        robot_state["camera"].update(self.saved_camera)

    def test_person_seed_uses_pose_mask_and_bounded_map_motion(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker.stable_seeds = StableTargetSeedTracker()
        tracker.camera = Mock()
        tracker.camera.snapshot.return_value = type("Frame", (), {
            "tracking_bgr": np.zeros((360, 640, 3), dtype=np.uint8),
        })()
        tracker.target = "person"
        tracker.action_id = "track-person"

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

        self.assertIsNone(tracker._update_person_seed(person))
        for index in range(1, TrackAction.TARGET_SEED_SAMPLES):
            robot_state["camera"].update({
                "object_map_x": 3.0 + 0.02 * index,
                "timestamp": now - 0.2 + 0.02 * index,
            })
            seed = tracker._update_person_seed(person)
        self.assertIsNotNone(seed)
        self.assertEqual(seed["session_id"], "track-person")

        robot_state["camera"].update({
            "object_map_x": 5.0,
            "timestamp": now,
        })
        self.assertIsNone(tracker._update_person_seed(person))

    async def test_stable_tracking_saves_session_map_location(self):
        tracker = TrackAction.__new__(TrackAction)
        tracker._memory_recorded_for_session = False
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

        tracker._remember_stable_target()

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
        location = {
            "x": 4.0,
            "y": 0.5,
            "angle": 0.12,
            "tof_range": 4.1,
            "target": "bottle",
            "session_id": "track-bottle",
        }
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
        approach = ApproachAction(send_robot_command)
        result = await approach.start(
            location, "approach-2", target="bottle"
        )

        self.assertEqual(result.status, "running")
        self.assertEqual(result.data["destination"], resolved)
        self.assertEqual(result.data["approach_location"], location)
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

    async def test_approach_rejects_invalid_location_without_navigation(self):
        send_robot_command = AsyncMock()
        approach = ApproachAction(send_robot_command)

        result = await approach.start(
            {"x": 1.0, "y": 0.0, "angle": 0.0, "tof_range": float("nan")},
            "approach-invalid",
            target="bottle",
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason_code, "INVALID_APPROACH_LOCATION")
        send_robot_command.assert_not_awaited()

    async def test_find_executor_resolves_location_before_approach(self):
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
        executor.action_id = "find-1"
        executor.target = "bottle"
        executor._approach_standoff_m = 0.3
        executor._approach_result_data = {}
        location = {
            "x": 1.2,
            "y": 0.1,
            "angle": 0.05,
            "tof_range": 1.3,
        }
        executor.track_action = type("Tracker", (), {
            "action_id": "find-1:track",
            "wait_for_stable_target_seed": AsyncMock(return_value=location),
        })()
        executor.approach_action = type("Approach", (), {
            "start": AsyncMock(return_value=ActionResult(
                "find-1:approach", "approach_action", "succeeded",
                target="bottle", data={"destination": {"x": 1.0}},
            )),
        })()

        result = await executor._approach()

        self.assertEqual(result.status, "succeeded")
        executor.track_action.wait_for_stable_target_seed.assert_awaited_once_with(
            target="bottle",
            session_id="find-1:track",
            timeout=10.0,
        )
        executor.approach_action.start.assert_awaited_once_with(
            location=location,
            action_id="find-1:approach",
            target="bottle",
            standoff_m=0.3,
        )

    async def test_track_approach_and_reacquire_uses_max_effort(self):
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
        executor.action_id = "find-1"
        executor.target = "bottle"
        executor._completed_steps = []
        executor._approach_result_data = {}
        executor._search_waypoint_count = 3
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
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
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
            "xmin": -2.0, "xmax": 2.0,
            "ymin": -2.0, "ymax": 2.0,
            "ppm": 50.25,
            "origin_x": 0.0, "origin_y": 0.0,
            "heading_yaw": 0.0,
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
    def test_explicit_navigation_render_includes_frontiers_and_pending_pose(self):
        renderer = MapRenderer()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        view = {
            "xmin": -2.0, "xmax": 2.0,
            "ymin": -2.0, "ymax": 2.0,
            "ppm": 50.0, "width": 200, "height": 200,
            "origin_x": 0.0, "origin_y": 0.0,
            "heading_yaw": 0.0,
        }
        frontier = candidate("F1", "frontier", 0.5, 0.5)
        pending = {
            "x": 1.0, "y": 0.0, "yaw": math.pi / 2.0,
            "frame_id": "map",
        }
        renderer._draw_frontier_candidates(image, view, [frontier])
        renderer._draw_pending_navigation_pose(image, view, pending)

        frontier_pixel = renderer._pixel(view, 0.5, 0.5)
        pending_pixel = renderer._pixel(view, 1.0, 0.0)
        self.assertGreater(int(image[frontier_pixel[1], frontier_pixel[0], 1]), 0)
        self.assertGreater(int(image[pending_pixel[1], pending_pixel[0], 1]), 0)

    def test_normal_navigation_uses_full_camera_fov(self):
        live_fov = MapLogic().live_camera_fov(
            CameraCoverageTests.pose, camera_pan_angle=95.0
        )

        self.assertEqual(live_fov["horizontal_fov_deg"], 85.0)

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

    def test_robot_heading_is_always_rendered_up(self):
        grid = FindTargetCandidateTests.Grid()
        renderer = MapRenderer()
        pose = {
            "x": 0.0, "y": 0.0, "yaw": math.pi / 2.0,
            "frame_id": "map",
        }
        view = renderer._make_view(grid, pose)
        robot = renderer._pixel(view, 0.0, 0.0)
        forward = renderer._pixel(view, 0.0, 1.0)
        left = renderer._pixel(view, -1.0, 0.0)

        self.assertLess(forward[1], robot[1])
        self.assertAlmostEqual(forward[0], robot[0], delta=1)
        self.assertLess(left[0], robot[0])
        self.assertAlmostEqual(left[1], robot[1], delta=1)

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
        executor = FindTargetExecutor.__new__(FindTargetExecutor)
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


if __name__ == "__main__":
    unittest.main()
