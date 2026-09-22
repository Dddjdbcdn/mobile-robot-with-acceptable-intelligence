from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import time

from actions.action_result import ActionResult
from cognition.state import robot_state
from actions.track_action import (
    normalize_human_target,
    normalize_object_target,
)


@dataclass(frozen=True, slots=True)
class FindLoopConfig:
    max_local_waypoints: int = 2
    max_destination_waypoints: int = 10
    max_exploration_waypoints: int = 20
    max_total_waypoints: int = 50
    max_duration_seconds: float = 600.0
    camera_center_pan_deg: float = 95.0
    camera_horizontal_fov_deg: float = 60.0
    camera_reliable_range_m: float = 2.0


class GoalExecutor:
    """Own the complete find -> track -> approach object goal."""

    MAP_SNAPSHOT_TIMEOUT = 8.0
    REACQUISITION_MISS_CODES = {
        "TARGET_NOT_VISIBLE",
        "SEARCH_CONTEXTUAL_CLUE",
        "OBJECT_DETECTION_FAILED",
        "PERSON_DETECTION_FAILED",
        "PERSON_KEYPOINT_TIMEOUT",
        "TRACKING_STABILITY_TIMEOUT",
        "POST_APPROACH_TRACKING_UNSTABLE",
        "OBJECT_LOST",
        "PERSON_LOST",
    }

    TURN_COMMANDS = {
        "left": {"angular_velocity": 1.0, "angle": math.pi / 2.0},
        "right": {"angular_velocity": -1.0, "angle": math.pi / 2.0},
        "behind": {"angular_velocity": -3.0, "angle": math.pi},
    }

    def __init__(
        self,
        move_action,
        search_action,
        track_action,
        approach_action,
        see_action,
        semantic_memory=None,
        navigate_action=None,
        send_map_overlay=None,
    ):
        self.move_action = move_action
        self.search_action = search_action
        self.track_action = track_action
        self.approach_action = approach_action
        self.see_action = see_action
        self.semantic_memory = semantic_memory
        self.navigate_action = navigate_action
        self.send_map_overlay = send_map_overlay
        self.find_loop = FindLoopConfig()

        self.active = False
        self.action_id: str | None = None
        self.target: str | None = None
        self.goal: str | None = None
        self.action_type: str | None = None
        self.completion_future: asyncio.Future[ActionResult] | None = None
        self._runner: asyncio.Task | None = None
        self._completed_steps: list[str] = []
        self._started_tracking = False
        self._approach_result_data: dict = {}
        self._search_poses: list[dict] = []
        self._map_overlay_revision = 0
        self._search_waypoint_count = 0
        self._search_started_at = 0.0
        self._approach_standoff_m: float | None = None
        self._approach_follow = False

    async def start(
        self,
        goal,
        target,
        action_id,
        approach_standoff_m=None,
        approach_follow=False,
    ) -> ActionResult:
        """Start the one supported composite goal: find, track, and approach."""
        target = str(target or "").strip()
        if self.active:
            return ActionResult(
                action_id, "find_object", "already_running", target=target or None,
                outcome="already_running", reason_code="GOAL_BUSY", retryable=True,
                data={"active_action_id": self.action_id},
            )
        if str(goal or "").strip().lower() != "find_object":
            return self._invalid_result(
                action_id, target or None, "UNKNOWN_GOAL", "unsupported_goal",
                "find_object",
            )
        if not target:
            return self._invalid_result(
                action_id, None, "GOAL_TARGET_REQUIRED", "invalid_request",
                "find_object",
            )

        self.active = True
        self.action_id = action_id
        self.target = target
        self.goal = "find_object"
        self.action_type = "find_object"
        self._completed_steps = []
        self._started_tracking = False
        self._approach_result_data = {}
        self._search_poses = []
        self._search_waypoint_count = 0
        self._map_overlay_revision = 0
        self._approach_standoff_m = (
            float(approach_standoff_m)
            if approach_standoff_m is not None else None
        )
        self._approach_follow = bool(approach_follow)
        self._remember_search_pose("search_start")
        await self._publish_map_overlay()
        self._search_started_at = time.monotonic()

        self.completion_future = asyncio.get_running_loop().create_future()
        self._runner = asyncio.create_task(
            self._run(), name=f"find-object-{action_id}"
        )
        self._runner.add_done_callback(self._runner_done)
        return ActionResult(
            action_id, "find_object", "running", target=target,
            outcome="finding_tracking_and_approaching",
        )

    async def wait_until_finished(self) -> ActionResult:
        return await self.completion_future

    async def stop(self, reason_code="USER_REQUESTED") -> ActionResult | None:
        if not self.active:
            return None

        runner = self._runner
        if runner is not None and runner is not asyncio.current_task():
            runner.cancel()

        if self.approach_action.active:
            await self.approach_action.stop_approaching(reason_code)
        if self.navigate_action is not None and self.navigate_action.active:
            await self.navigate_action.stop(reason_code)
        if self.search_action.active:
            await self.search_action.stop_searching(reason_code)
        if self.move_action.active:
            await self.move_action.stop_moving(reason_code)
        if self._started_tracking and self.track_action.active:
            await self.track_action.stop_tracking(reason_code)

        result = self._complete(
            status="cancelled",
            outcome="goal_cancelled",
            reason_code=reason_code,
        )
        if runner is not None and runner is not asyncio.current_task():
            await asyncio.gather(runner, return_exceptions=True)
        return result

    async def _run(self) -> None:
        """Repeat find/approach until the target is tracked after arrival."""
        try:
            search_round = 0
            while True:
                limit_result = self._search_limit_result()
                if limit_result is not None:
                    raise GoalStepFailed("find", limit_result)

                search_result = await self._search_environment(search_round)
                if search_result.status != "succeeded":
                    raise GoalStepFailed("find", search_result)
                self._completed_steps.append(
                    "target_found" if search_round == 0
                    else f"target_refound_{search_round}"
                )

                if not self.track_action.active:
                    track_result = await self._track(
                        allow_grounding_dino=True,
                        step=(
                            "track" if search_round == 0
                            else f"track_{search_round}"
                        ),
                    )
                    if track_result.status != "succeeded":
                        raise GoalStepFailed("track", track_result)
                self._completed_steps.append(
                    "target_tracked" if search_round == 0
                    else f"target_retracked_{search_round}"
                )

                approach_result = await self._approach(
                    step=(
                        "approach" if search_round == 0
                        else f"approach_{search_round}"
                    )
                )
                if approach_result.status != "succeeded":
                    raise GoalStepFailed("approach", approach_result)
                if getattr(self, "_approach_follow", False):
                    self._completed_steps.append("target_following")
                    break

                reacquired = await self._post_approach_reacquire(search_round)
                if reacquired.status == "succeeded":
                    self._approach_result_data.update({
                        "verified": True,
                        "post_approach_reacquired": True,
                        "reacquisition": self._result_data(reacquired),
                    })
                    self._completed_steps.extend([
                        "target_reacquired",
                        "target_reached",
                    ])
                    break

                if reacquired.reason_code in self.REACQUISITION_MISS_CODES:
                    self._completed_steps.append(
                        f"approach_destination_rejected_{search_round}"
                    )
                    search_round += 1
                    continue
                raise GoalStepFailed("post_approach_reacquire", reacquired)
        except asyncio.CancelledError:
            return
        except GoalStepFailed as error:
            self._complete(
                status="failed",
                outcome="find_object_failed",
                reason_code=error.result.reason_code or "GOAL_STEP_FAILED",
                data={
                    "failed_step": error.step,
                    "step_result": self._result_data(error.result),
                },
            )
            return
        except Exception as error:
            self._complete(
                status="failed",
                outcome="execution_error",
                reason_code="GOAL_EXECUTION_ERROR",
                data={"error": f"{type(error).__name__}: {error}"},
            )
            return

        self._complete(
            status="succeeded",
            outcome=(
                "found_tracked_and_following"
                if getattr(self, "_approach_follow", False)
                else "found_reached_and_reacquired"
            ),
        )

    async def _turn_body(self, direction: str, step: str) -> ActionResult:
        command = self.TURN_COMMANDS[direction]
        result = await self.move_action.start_moving(
            linear_velocity=0.0,
            distance=0.0,
            angular_velocity=command["angular_velocity"],
            angle=command["angle"],
            action_id=self._step_id(step),
        )
        result = await self._terminal_result(self.move_action, result)
        if result.status == "succeeded":
            self.search_action.last_found_target = None
        return result

    async def _move_camera_for_goal(self, region: str, step: str) -> ActionResult:
        if self.track_action.active:
            await self.track_action.stop_tracking(
                reason_code="REPLACED",
                status="cancelled",
                outcome="replaced_by_new_goal",
            )
        return await self.see_action.move_to_region(
            region=region,
            action_id=self._step_id(step),
        )

    async def _track(
        self,
        allow_grounding_dino=True,
        step="track",
    ) -> ActionResult:
        result = await self.track_action.start_tracking(
            target=self.target,
            action_id=self._step_id(step),
            allow_grounding_dino=allow_grounding_dino,
        )
        if result.status == "running":
            self._started_tracking = True
            return ActionResult(
                action_id=result.action_id,
                action_type=result.action_type,
                status="succeeded",
                target=result.target,
                outcome="tracking_started",
                data=result.data,
            )
        return result

    async def _approach(self, step="approach") -> ActionResult:
        result = await self.approach_action.start_approaching(
            target=self.target,
            action_id=self._step_id(step),
            standoff_m=self._approach_standoff_m,
            follow=getattr(self, "_approach_follow", False),
        )
        if getattr(self, "_approach_follow", False) and result.status == "running":
            self._approach_result_data = dict(result.data)
            return ActionResult(
                action_id=result.action_id,
                action_type=result.action_type,
                status="succeeded",
                target=result.target,
                outcome="follow_navigation_started",
                data=result.data,
            )
        result = await self._terminal_result(self.approach_action, result)
        if result.status == "succeeded":
            self._approach_result_data = dict(result.data)
        return result

    async def _post_approach_reacquire(self, search_round) -> ActionResult:
        """Max-effort search and stable tracking validate the destination."""
        if self.track_action.active:
            await self.track_action.stop_tracking(
                reason_code="APPROACH_NAVIGATION_COMPLETE",
                status="succeeded",
                outcome="ready_for_post_approach_reacquisition",
                reset_camera=False,
            )

        step = f"post_approach_search_{search_round}"
        search_result = await self._search(
            step,
            effort="best_effort",
            initial_view_only=False,
            sweep_directions=None,
        )
        if search_result.status != "succeeded":
            return search_result

        track_result = await self._track(
            allow_grounding_dino=True,
            step=f"post_approach_track_{search_round}",
        )
        if track_result.status != "succeeded":
            return track_result

        return ActionResult(
            action_id=self._step_id(
                f"post_approach_track_{search_round}"
            ),
            action_type="track_action",
            status="succeeded",
            target=self.target,
            outcome="post_approach_target_reacquired",
            data={"tracking_stable": True, "search_effort": "best_effort"},
        )

    async def _navigate_search_step(self, observation, mode, step):
        revision = await self._publish_map_overlay(
            mode=mode,
            observation=observation,
        )
        result = await self.navigate_action.start_find_object_step(
            target=self.target,
            action_id=self._step_id(step),
            mode=mode,
            observation=observation,
            overlay_action_id=self.action_id,
            overlay_revision=revision,
        )
        result = await self._terminal_result(self.navigate_action, result)
        if result.status == "succeeded":
            self._search_waypoint_count += 1
            self._completed_steps.append(step)
        return result

    async def _search_environment(self, search_round=0) -> ActionResult:
        root_step = "initial" if search_round == 0 else f"retry_{search_round}"
        print("🌊 [INSPECTION] INITIAL ATTEMPT")
        result = await self._inspect_area(
            root_step,
        )
        if result is not None:
            return result

        for exploration_index in range(1, self.find_loop.max_exploration_waypoints + 1):
            limit_result = self._search_limit_result()
            if limit_result is not None:
                return limit_result

            print(f"🌿 [EXPLORE] {exploration_index}/{self.find_loop.max_exploration_waypoints}")
            exploration_step = (
                f"explore_{exploration_index}"
                if search_round == 0
                else f"{root_step}_explore_{exploration_index}"
            )
            navigation = await self._navigate_search_step(
                None,
                mode="exploration",
                step=exploration_step,
            )
            if navigation.status != "succeeded":
                if navigation.reason_code in {
                    "NAVIGATION_BLOCKED", "NO_NAVIGATION_CANDIDATES"
                }:
                    return self._search_exhausted_result(navigation)
                return navigation

            print(f"🌊 [INSPECTION]: {exploration_index}/{self.find_loop.max_exploration_waypoints}")
            result = await self._inspect_area(
                exploration_step,
            )
            if result is not None:
                return result

        return self._search_exhausted_result()

    async def _inspect_area(self, prefix, *, candidate_context=None):
        scan = await self._scan_area(
            prefix,
            candidate_context=candidate_context,
            initial=True,
        )
        if scan.status == "succeeded":
            return scan
        if scan.reason_code not in {
            "TARGET_NOT_VISIBLE", "SEARCH_CONTEXTUAL_CLUE",
        }:
            return scan

        observation = self.search_action.last_observation_frame
        if observation is None:
            print("🌊 [SEARCH]: NO CLUE")
            return None
        return await self._route_candidate(prefix, observation)

    async def _scan_area(
        self, step, *, candidate_context=None, initial=False
    ) -> ActionResult:
        centered = await self._move_camera_for_goal(
            region="center", step=f"{step}_center_camera"
        )
        if centered.status != "succeeded":
            return centered
        return await self._search(
            f"{step}_center_scan" if initial else f"{step}_camera_check",
            initial_view_only=False,
            sweep_directions=("left", "right"),
            candidate_context=candidate_context,
        )

    async def _route_candidate(self, prefix, observation):
        """Dispatch a search assessment to its deliberately distinct behavior."""
        candidate_type = str(observation.get("candidate_type") or "")
        print(f"🌊 [SEARCH]: CANDIDATE: {candidate_type or 'none'}")
        handlers = {
            "visual": self._handle_visual_candidate,
            "local": self._handle_contextual_candidate,
            "destination": self._handle_contextual_candidate,
            "speculative": self._handle_speculative_candidate,
        }
        handler = handlers.get(candidate_type)
        if handler is None:
            return None
        return await handler(prefix, observation)

    async def _handle_visual_candidate(self, prefix, observation):
        context = dict(observation)
        context.update({
            "candidate_type": "visual",
            "reassessment_limit": 1,
            "reassessments_used": 0,
        })
        reassessment = await self._search(
            f"{prefix}_visual_reassessment",
            initial_view_only=True,
            sweep_directions=(),
            candidate_context=context,
        )
        if reassessment.status == "succeeded":
            return reassessment
        if reassessment.reason_code not in {
            "TARGET_NOT_VISIBLE", "SEARCH_CONTEXTUAL_CLUE",
        }:
            return reassessment
        if self.search_action.last_observation_frame is None:
            return None

        track_result = await self._track(
            allow_grounding_dino=True,
            step=f"{prefix}_visual_track",
        )
        if track_result.status != "succeeded":
            print("🌊 [SEARCH]: VISUAL CANDIDATE REJECTED")
            return None
        print("🌊 [SEARCH]: VISUAL CANDIDATE ACCEPTED")
        return ActionResult(
            action_id=self._step_id(f"{prefix}_visual_track_confirmed"),
            action_type="search_action",
            status="succeeded",
            target=self.target,
            outcome="visual_candidate_tracked",
            data={"tracker_result": self._result_data(track_result)},
        )

    async def _handle_contextual_candidate(self, prefix, observation):
        """Follow local/destination clues and reset when their type changes."""
        candidate_type = str(observation.get("candidate_type") or "")
        movement_index = 0

        while candidate_type in {"local", "destination"}:
            if candidate_type == "local":
                mode = "context"
                label = "LOCAL CLUE"
                movement_limit = self.find_loop.max_local_waypoints
            else:
                mode = "destination"
                label = "DESTINATION"
                movement_limit = self.find_loop.max_destination_waypoints

            if movement_index >= movement_limit:
                return None

            limit_result = self._search_limit_result()
            if limit_result is not None:
                return limit_result

            payload = dict(observation)
            payload.update({
                "movement_limit": movement_limit,
                "movements_used": movement_index,
                "remaining_waypoints": movement_limit - movement_index,
            })
            print(
                f"🔥 [FOLLOW {label}]: {movement_index}/{movement_limit}"
            )
            step_kind = "context" if mode == "context" else "destination"
            step = f"{prefix}_{step_kind}_{movement_index + 1}"
            navigation = await self._navigate_search_step(
                payload,
                mode=mode,
                step=step,
            )
            if navigation.status != "succeeded":
                if navigation.reason_code in {
                    "NAVIGATION_BLOCKED", "NO_NAVIGATION_CANDIDATES"
                }:
                    return None
                return navigation

            movement_index += 1
            scan_context = dict(payload)
            scan_context.update({
                "movements_used": movement_index,
                "remaining_waypoints": movement_limit - movement_index,
            })
            scan = await self._scan_area(
                step,
                candidate_context=scan_context,
            )
            if scan.status == "succeeded":
                return scan
            if scan.reason_code not in {
                "TARGET_NOT_VISIBLE", "SEARCH_CONTEXTUAL_CLUE",
            }:
                return scan

            next_observation = self.search_action.last_observation_frame
            next_type = str(
                (next_observation or {}).get("candidate_type") or ""
            )
            if next_type not in {"local", "destination"}:
                if next_type in {"visual", "speculative"}:
                    return await self._route_candidate(step, next_observation)
                return None

            if next_type != candidate_type:
                prefix = step
                movement_index = 0
            observation = next_observation
            candidate_type = next_type
        return None

    async def _handle_speculative_candidate(self, _prefix, _observation):
        """Speculation is intentionally acknowledged without causing motion."""
        return None

    async def _search(
        self, step, initial_view_only, sweep_directions=None, effort="center",
        candidate_context=None,
    ):
        """Run one search step; tracking begins after search returns found."""
        if (
            self.search_action.active
            and self._normalize_target(self.search_action.target)
            == self._normalize_target(self.target)
        ):
            search_result = await self.search_action.wait_until_finished()
        else:
            search_result = await self.search_action.start_searching(
                target=self.target,
                action_id=self._step_id(step),
                effort=effort,
                initial_view_only=initial_view_only,
                sweep_directions=sweep_directions,
                candidate_context=candidate_context,
            )
            search_result = await self._terminal_result(
                self.search_action, search_result
            )
        coverage_frames = getattr(
            self.search_action, "last_coverage_frames", None
        ) or ([self.search_action.last_observation_frame]
              if self.search_action.last_observation_frame else [])
        if self._remember_search_pose(step, frames=coverage_frames):
            await self._publish_map_overlay()
        return search_result

    def _remember_search_pose(self, reason, pose=None, frames=None):
        """Upsert robot poses and attach every camera view captured there."""
        entries = []
        if frames is not None:
            for frame in frames:
                if (
                    isinstance(frame, dict)
                    and isinstance(frame.get("robot_pose"), dict)
                ):
                    entries.append((frame["robot_pose"], frame))
        else:
            entries.append((pose or robot_state.get("pose") or {}, None))

        changed = False
        for raw_pose, frame in entries:
            try:
                recorded = {
                    "x": float(raw_pose["x"]),
                    "y": float(raw_pose["y"]),
                    "yaw": float(raw_pose.get("yaw", 0.0)),
                    "frame_id": str(raw_pose.get("frame_id") or "map"),
                    "kind": str(raw_pose.get("kind") or "pose"),
                    "reason": reason,
                    "step": self._search_waypoint_count,
                    "views": [],
                }
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(recorded[key]) for key in ("x", "y", "yaw")):
                continue

            search_pose = next((
                item for item in reversed(self._search_poses)
                if math.hypot(
                    recorded["x"] - item["x"],
                    recorded["y"] - item["y"],
                ) <= 0.03
                and abs(math.atan2(
                    math.sin(recorded["yaw"] - item["yaw"]),
                    math.cos(recorded["yaw"] - item["yaw"]),
                )) <= math.radians(1.0)
            ), None)
            if search_pose is None:
                search_pose = recorded
                self._search_poses.append(search_pose)
                changed = True
            else:
                updates = {
                    "reason": reason,
                    "step": self._search_waypoint_count,
                }
                if raw_pose.get("kind"):
                    updates["kind"] = recorded["kind"]
                if any(search_pose.get(key) != value for key, value in updates.items()):
                    search_pose.update(updates)
                    changed = True

            if frame is None:
                continue
            view = {
                "pan": float(frame.get(
                    "pan_angle", self.find_loop.camera_center_pan_deg
                )),
                "tilt": float(frame.get("tilt_angle", 90.0)),
            }
            stored_view = next((
                item for item in search_pose["views"]
                if item.get("pan") == view["pan"]
                and item.get("tilt") == view["tilt"]
            ), None)
            for key in (
                "candidate_type", "contextual_clue",
                "movement_limit", "movements_used",
                "remaining_waypoints", "reassessment_limit",
                "reassessments_used",
            ):
                if frame.get(key) is not None:
                    view[key] = frame[key]
            if stored_view is None:
                search_pose["views"].append(view)
                changed = True
            elif any(stored_view.get(key) != value for key, value in view.items()):
                stored_view.update(view)
                changed = True
        return changed

    async def _publish_map_overlay(self, mode=None, observation=None):
        if self.send_map_overlay is None or not self.action_id:
            return self._map_overlay_revision
        self._map_overlay_revision += 1
        frame_id = (
            self._search_poses[0]["frame_id"]
            if self._search_poses else "map"
        )
        observation_payload = None
        if isinstance(observation, dict):
            robot_pose = observation.get("robot_pose")
            if isinstance(robot_pose, dict):
                try:
                    observation_payload = {
                        "robot_pose": {
                            "x": float(robot_pose["x"]),
                            "y": float(robot_pose["y"]),
                            "yaw": float(robot_pose["yaw"]),
                            "frame_id": str(robot_pose.get("frame_id") or frame_id),
                        },
                        "pan_angle": float(observation.get(
                            "pan_angle", self.find_loop.camera_center_pan_deg
                        )),
                        "tilt_angle": float(observation.get("tilt_angle", 90.0)),
                        "contextual_clue": str(
                            observation.get("contextual_clue") or ""
                        ),
                        "candidate_type": str(
                            observation.get("candidate_type") or ""
                        ),
                        "movement_limit": int(
                            observation.get("movement_limit") or 0
                        ),
                        "movements_used": int(
                            observation.get("movements_used") or 0
                        ),
                        "remaining_waypoints": int(
                            observation.get("remaining_waypoints") or 0
                        ),
                        "reassessment_limit": int(
                            observation.get("reassessment_limit") or 0
                        ),
                        "reassessments_used": int(
                            observation.get("reassessments_used") or 0
                        ),
                    }
                except (KeyError, TypeError, ValueError):
                    observation_payload = None
        await self.send_map_overlay({
            "schema_version": 1,
            "operation": "set",
            "action_id": self.action_id,
            "revision": self._map_overlay_revision,
            "frame_id": frame_id,
            "mode": str(mode or "goal"),
            "observation": observation_payload,
            "search_poses": list(self._search_poses),
            "camera_horizontal_fov_deg": self.find_loop.camera_horizontal_fov_deg,
            "camera_reliable_range_m": self.find_loop.camera_reliable_range_m,
        })
        return self._map_overlay_revision

    def _discard_map_overlay(self, action_id):
        if self.send_map_overlay is None or not action_id:
            return
        asyncio.create_task(self.send_map_overlay({
            "schema_version": 1,
            "operation": "clear",
            "action_id": action_id,
            "revision": self._map_overlay_revision,
        }))


    def _search_limit_result(self):
        within_duration = (
            time.monotonic() - self._search_started_at
            <= self.find_loop.max_duration_seconds
        )
        within_waypoints = (
            self._search_waypoint_count < self.find_loop.max_total_waypoints
        )
        if within_duration and within_waypoints:
            return None
        reason_code = (
            "OBJECT_SEARCH_WAYPOINT_LIMIT"
            if not within_waypoints
            else "OBJECT_SEARCH_DURATION_LIMIT"
        )
        return ActionResult(
            self._step_id("search_timeout"),
            "search_action",
            "failed",
            target=self.target,
            outcome="search_limit_reached",
            reason_code=reason_code,
            data={"search_poses": list(self._search_poses)},
        )

    def _search_exhausted_result(self, navigation_result=None):
        data = {
            "pose_count": len(self._search_poses),
            "waypoint_count": self._search_waypoint_count,
            "search_poses": list(self._search_poses),
        }
        if navigation_result is not None:
            data["navigation_result"] = self._result_data(navigation_result)
        return ActionResult(
            self._step_id("search_exhausted"),
            "search_action",
            "failed",
            target=self.target,
            outcome="search_space_exhausted",
            reason_code="OBJECT_SEARCH_EXHAUSTED",
            data=data,
        )

    @staticmethod
    async def _terminal_result(action, result: ActionResult) -> ActionResult:
        if result.status == "running":
            return await action.wait_until_finished()
        return result

    def _step_id(self, step: str) -> str:
        return f"{self.action_id}:{step}"

    def _complete(
        self, status, outcome, reason_code=None, data=None
    ) -> ActionResult:
        completed_action_id = self.action_id
        result = ActionResult(
            action_id=self.action_id or "unassigned",
            action_type=self.action_type or "goal_action",
            status=status,
            target=self.target,
            outcome=outcome,
            reason_code=reason_code,
            retryable=status == "failed",
            data={
                "goal": self.goal,
                "completed_steps": list(self._completed_steps),
                "search_waypoint_count": self._search_waypoint_count,
                "search_poses": list(self._search_poses),
                **(
                    {
                        "verified": self._approach_result_data.get(
                            "verified", False
                        ),
                        "approach_result": dict(self._approach_result_data),
                    }
                    if self._approach_result_data
                    else {}
                ),
                **(data or {}),
            },
        )
        completion_future = self.completion_future
        self.active = False
        self.action_id = None
        self.target = None
        self.goal = None
        self.action_type = None
        self._search_poses = []
        self._discard_map_overlay(completed_action_id)
        self._runner = None
        if completion_future is not None and not completion_future.done():
            completion_future.set_result(result)
        return result

    def _runner_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and self.active:
            self._complete(
                status="failed",
                outcome="execution_error",
                reason_code="GOAL_EXECUTION_ERROR",
                data={"error": f"{type(error).__name__}: {error}"},
            )

    @staticmethod
    def _normalize_target(target):
        return normalize_human_target(target) or normalize_object_target(target)

    @staticmethod
    def _invalid_result(
        action_id, target, reason_code, outcome, action_type="goal_action"
    ):
        return ActionResult(
            action_id=action_id,
            action_type=action_type,
            status="failed",
            target=target,
            outcome=outcome,
            reason_code=reason_code,
        )

    @staticmethod
    def _result_data(result: ActionResult) -> dict:
        return {
            "action_id": result.action_id,
            "action_type": result.action_type,
            "status": result.status,
            "target": result.target,
            "outcome": result.outcome,
            "reason_code": result.reason_code,
            "retryable": result.retryable,
            "data": result.data,
        }


class GoalStepFailed(RuntimeError):
    def __init__(self, step: str, result: ActionResult):
        super().__init__(f"{step} failed: {result.reason_code or result.outcome}")
        self.step = step
        self.result = result
