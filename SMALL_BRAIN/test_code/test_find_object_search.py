import math
from pathlib import Path
import sys
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
from cognition.goal_executor import FindLoopConfig, GoalExecutor
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
            context_min_radius_m=1.25,
            context_radius_samples=1,
            context_bearing_fractions=(0.0,),
            context_backup_max_distance_m=0.25,
            context_backup_range_fraction=1.0,
        ))

        selected = logic.plan_candidates(
            self.prepared, self.pose, self.context_overlay()
        )

        cone = [
            item for item in selected
            if item["id"].startswith("C") and item["id"] not in ("CB", "CF")
        ]
        self.assertEqual(len(cone), 1)
        self.assertAlmostEqual(cone[0]["x"], 1.25)
        self.assertAlmostEqual(cone[0]["y"], 0.0)
        self.assertAlmostEqual(selected[-1]["x"], -0.25)
        self.assertAlmostEqual(selected[-1]["y"], 0.0)

    def test_context_includes_frontier_beyond_reliable_range_in_view(self):
        self.prepared["frontiers"] = [
            candidate("F1", "frontier", 2.8, 0.4),
            candidate("F2", "frontier", 0.0, 2.8),
        ]
        # Frontier extraction accepts connected free endpoints before the
        # extra local-candidate clearance erosion. Context mode must not drop
        # the frontier merely because that stricter mask excludes its cell.
        frontier_col, frontier_row = self.grid.cells(2.8, 0.4)
        self.prepared["reachable"][frontier_row, frontier_col] = False

        selected = self.logic.plan_candidates(
            self.prepared, self.pose, self.context_overlay()
        )

        by_id = {item["id"]: item for item in selected}
        self.assertIn("F1", by_id)
        self.assertEqual(
            by_id["F1"]["selection_reason"],
            "frontier_inside_observation_fov",
        )
        self.assertGreater(by_id["F1"]["distance_m"], 2.0)
        self.assertNotIn("F2", by_id)

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

    def test_exploration_heading_reveals_the_selected_unseen_area(self):
        coverage = np.ones_like(self.grid.data, dtype=bool)
        coverage[30:51, 24:37] = False
        winner = self.logic._exploration_candidate(
            self.prepared, self.pose, {"search_poses": []}, coverage
        )
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

    def test_exploration_uses_frontier_when_coverage_is_complete(self):
        logic = MapLogic(LogicConfig(
            exploration_frontier_distance_floor_m=1.0
        ))
        self.prepared["frontiers"] = [{
            "id": "F1", "kind": "frontier",
            "x": 0.1, "y": 0.0, "yaw": 0.0,
            "information_gain_m2": 2.0,
        }]
        coverage = np.ones_like(self.grid.data, dtype=bool)
        winner = logic._exploration_candidate(
            self.prepared, self.pose, {"search_poses": []}, coverage
        )
        self.assertEqual(winner["id"], "F1")
        self.assertEqual(winner["selection_reason"], "coverage_complete_frontier")
        self.assertEqual(winner["ranking_score"], 2.0)

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
    def _navigation_for_capture(camera):
        navigation = NavigateAction.__new__(NavigateAction)
        navigation.camera = camera
        navigation._overlay_action_id = "find-1"
        navigation._overlay_revision = 2
        navigation._mode = "exploration"
        return navigation

    async def test_capture_retries_a_stale_map_until_fresh(self):
        snapshot = {
            "jpeg_bytes": b"jpeg",
            "metadata": {
                "snapshot_id": "snapshot-1",
                "candidates": [candidate("F1", "frontier", 1.0, 0.0)],
                "search_overlay": {"action_id": "find-1", "revision": 2},
            },
        }
        camera = Mock()
        camera.map_snapshot.side_effect = [
            RuntimeError("Map image is stale (3.0 seconds)"),
            snapshot,
        ]
        navigation = self._navigation_for_capture(camera)

        captured, camera_jpeg = await navigation._capture()

        self.assertIs(captured, snapshot)
        self.assertIsNone(camera_jpeg)
        self.assertEqual(camera.map_snapshot.call_count, 2)

    async def test_capture_reports_persistent_map_stream_failure(self):
        camera = Mock()
        camera.map_snapshot.side_effect = RuntimeError(
            "Map image is stale (9.0 seconds)"
        )
        navigation = self._navigation_for_capture(camera)
        navigation.MAP_OVERLAY_TIMEOUT = 0.01

        with self.assertRaises(NavigationError) as raised:
            await navigation._capture()

        self.assertEqual(raised.exception.code, "MAP_STREAM_UNAVAILABLE")
        self.assertIn("Map image is stale", str(raised.exception))

    async def test_capture_requests_a_matching_standalone_snapshot(self):
        request = {}

        async def request_snapshot(payload):
            request.update(payload)

        camera = Mock()
        camera.map_snapshot.side_effect = lambda: {
            "jpeg_bytes": b"jpeg",
            "metadata": {
                "snapshot_id": "snapshot-1",
                "snapshot_request_id": request["request_id"],
                "candidates": [candidate("F1", "frontier", 1.0, 0.0)],
                "search_overlay": None,
            },
        }
        navigation = self._navigation_for_capture(camera)
        navigation._overlay_action_id = None
        navigation.action_id = "navigate-1"
        navigation.request_map_snapshot = request_snapshot

        captured, camera_jpeg = await navigation._capture()

        self.assertEqual(request["operation"], "snapshot")
        self.assertEqual(request["action_id"], "navigate-1")
        self.assertEqual(
            captured["metadata"]["snapshot_request_id"],
            request["request_id"],
        )
        self.assertIsNone(camera_jpeg)

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
    def test_candidate_requires_clue_and_confidence(self):
        self.assertTrue(SearchAction._contextual_clue_is_valid({
            "result": "candidate",
            "confidence": 0.8,
            "contextual_clue": "A desk surface is visible on the right.",
        }))
        self.assertFalse(SearchAction._contextual_clue_is_valid({
            "result": "candidate",
            "contextual_clue": "A desk surface is visible on the right.",
        }))
        self.assertFalse(SearchAction._contextual_clue_is_valid({
            "result": "not_found",
            "confidence": 0.9,
            "contextual_clue": None,
        }))

    def test_exploration_has_no_llm_prompt(self):
        self.assertNotIn("exploration", MODE_GUIDANCE)

    def test_goal_and_context_share_one_map_guide(self):
        navigation = NavigateAction.__new__(NavigateAction)
        for mode in ("goal", "context"):
            navigation._mode = mode
            prompt = navigation._build_selection_prompt({
                "query": "find bottle",
                "mode": mode,
            })
            self.assertTrue(prompt.startswith(MAP_GUIDANCE))
            self.assertIn(MODE_GUIDANCE[mode], prompt)
            self.assertIn('LIVE CONTEXT\n{"query": "find bottle"', prompt)

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
    async def test_inspection_and_context_navigation_repeat_until_found(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor.find_loop = FindLoopConfig()
        executor.search_action = type(
            "Search", (), {"last_observation_frame": {
                "jpeg_bytes": b"one-frame",
                "contextual_clue": "A desk is visible on the right.",
            }}
        )()
        executor._search_limit_result = lambda: None
        executor._move_camera_for_goal = AsyncMock(return_value=ActionResult(
            "camera", "move_camera_action", "succeeded",
        ))
        executor._try_yolo_current_view = AsyncMock(return_value=None)
        executor._search_and_verify = AsyncMock(side_effect=[
            ActionResult(
                "scan-1", "search_action", "failed",
                reason_code="SEARCH_CONTEXTUAL_CLUE",
            ),
            ActionResult("scan-2", "search_action", "succeeded"),
        ])
        executor._remember_search_pose = Mock()
        executor._navigate_search_step = AsyncMock(return_value=ActionResult(
            "nav", "navigate_action", "succeeded",
        ))
        result = await executor._inspect_area("initial")

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(executor._search_and_verify.await_count, 2)
        executor._navigate_search_step.assert_awaited_once_with(
            executor.search_action.last_observation_frame,
            mode="context",
            step="initial_context_1",
        )

    async def test_exploration_does_not_forward_a_stale_clue_frame(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor.find_loop = FindLoopConfig(max_exploration_waypoints=1)
        executor.search_action = type(
            "Search", (), {"last_observation_frame": {"contextual_clue": "old"}}
        )()
        executor._inspect_area = AsyncMock(side_effect=[
            None,
            ActionResult("found", "search_action", "succeeded"),
        ])
        executor._search_limit_result = lambda: None
        executor._navigate_search_step = AsyncMock(return_value=ActionResult(
            "nav", "navigate_action", "succeeded",
        ))

        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        executor._navigate_search_step.assert_awaited_once_with(
            None,
            mode="exploration",
            step="explore_1",
        )
        self.assertEqual(
            executor._inspect_area.await_args_list[1].args[0], "explore_1"
        )

    async def test_inspection_propagates_operational_scan_failure(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor.find_loop = FindLoopConfig()
        executor._move_camera_for_goal = AsyncMock(return_value=ActionResult(
            "camera", "move_camera_action", "succeeded",
        ))
        failure = ActionResult(
            "scan", "search_action", "failed",
            reason_code="CAMERA_FAILURE",
        )
        executor._search_and_verify = AsyncMock(return_value=failure)
        executor._remember_search_pose = Mock()
        executor._navigate_search_step = AsyncMock()

        result = await executor._inspect_area("initial", ())

        self.assertIs(result, failure)
        executor._remember_search_pose.assert_not_called()
        executor._navigate_search_step.assert_not_awaited()



class PostApproachReacquisitionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saved_pose = robot_state.get("pose")
        self.saved_camera = dict(robot_state.get("camera") or {})

    def tearDown(self):
        robot_state["pose"] = self.saved_pose
        robot_state["camera"].clear()
        robot_state["camera"].update(self.saved_camera)

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

    async def test_saved_location_search_precedes_general_fallback(self):
        approach = ApproachAction.__new__(ApproachAction)
        approach.track_action = type("Tracker", (), {"active": False})()
        approach._aim_at_saved_target_location = AsyncMock(return_value=True)
        approach._run_reacquisition_search = AsyncMock(side_effect=[
            ActionResult("focused", "search_action", "failed", outcome="not_found"),
            ActionResult("fallback", "search_action", "succeeded", outcome="found"),
        ])

        reacquired = await approach._reacquire_target_before_verification()

        self.assertTrue(reacquired)
        self.assertEqual(approach._run_reacquisition_search.await_args_list, [
            call(phase="saved_location", effort="center", initial_view_only=True),
            call(
                phase="general_fallback",
                effort="best_effort",
                initial_view_only=False,
            ),
        ])

    async def test_saved_map_point_aims_camera_from_new_robot_pose(self):
        approach = ApproachAction.__new__(ApproachAction)
        approach._saved_target_location = {
            "map_x": 1.0,
            "map_y": 1.0,
            "camera_tilt_angle": 87.0,
        }
        approach.search_action = type("Search", (), {})()
        approach.search_action.move_camera_angles = AsyncMock()
        robot_state["pose"] = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        robot_state["camera"].update({"pan_angle": 95.0, "tilt_angle": 90.0})

        aimed = await approach._aim_at_saved_target_location()

        self.assertTrue(aimed)
        approach.search_action.move_camera_angles.assert_awaited_once_with(
            140.0, 87.0
        )

    async def test_reacquisition_runs_before_visual_verification(self):
        events = []
        approach = ApproachAction.__new__(ApproachAction)
        approach._verification_task = None
        approach._verification_history = []
        approach._verification_retries = 0
        approach.max_verification_retries = 1

        async def reacquire():
            events.append("reacquire")
            return True

        async def verify():
            events.append("verify")
            return {
                "result": "reached",
                "confidence": 0.9,
                "target_visible": True,
            }

        approach._reacquire_target_before_verification = reacquire
        approach._request_approach_verification = verify
        approach._verification_data = Mock(return_value={})
        approach.complete_approaching = Mock()

        await approach._verify_navigation_result()

        self.assertEqual(events, ["reacquire", "verify"])
        approach.complete_approaching.assert_called_once_with(
            status="succeeded",
            outcome="reached_and_verified",
            data={},
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
                "confidence": 0.7,
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

        result = await executor._search_and_verify(
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
