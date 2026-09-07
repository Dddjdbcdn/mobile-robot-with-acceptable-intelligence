from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from typing import Awaitable, Callable

from actions.action_result import ActionResult
from cognition.state import robot_state
from actions.move_camera_action import SEMANTIC_CAMERA_REGIONS
from actions.track_action import (
    human_trackable_parts,
    is_yolo_trackable_target,
    normalize_human_target,
    normalize_object_target,
)


@dataclass(frozen=True, slots=True)
class FactRule:
    requires: tuple[str, ...]
    handler_name: str


class GoalExecutor:
    """Resolve high-level robot goals through reusable action prerequisites."""

    GOAL_FACTS = {
        "search": "target_found",
        "track": "target_tracked",
        "approach": "target_reached",
    }

    GOAL_ACTION_TYPES = {
        "search": "find_target",
        "track": "watch_target",
        "approach": "approach_target",
    }

    FACT_RULES = {
        "oriented": FactRule((), "_orient"),
        "camera_aimed": FactRule((), "_aim_camera"),
        "target_found": FactRule(
            ("oriented", "camera_aimed"), "_search"
        ),
        "target_tracked": FactRule(("target_found",), "_track"),
        "target_reached": FactRule(("target_tracked",), "_approach"),
    }

    TURN_COMMANDS = {
        "left": {"angular_velocity": 1.0, "angle": math.pi / 2.0},
        "right": {"angular_velocity": -1.0, "angle": math.pi / 2.0},
        "behind": {"angular_velocity": -3.0, "angle": math.pi},
    }

    EXPLICIT_TARGET_DIRECTIONS = {"behind"}

    CAMERA_EDGE_BODY_FALLBACKS = {
        "upper_left": ("up", "left"),
        "left": ("center", "left"),
        "lower_left": ("down", "left"),
        "upper_right": ("up", "right"),
        "right": ("center", "right"),
        "lower_right": ("down", "right"),
    }

    def __init__(
        self,
        move_action,
        search_action,
        track_action,
        approach_action,
        move_camera_action,
        semantic_memory=None,
    ):
        self.move_action = move_action
        self.search_action = search_action
        self.track_action = track_action
        self.approach_action = approach_action
        self.move_camera_action = move_camera_action
        self.semantic_memory = semantic_memory

        self.active = False
        self.action_id: str | None = None
        self.target: str | None = None
        self.goal: str | None = None
        self.action_type: str | None = None
        self.direction: str | None = None
        self.camera_region: str | None = None
        self.completion_future: asyncio.Future[ActionResult] | None = None
        self._runner: asyncio.Task | None = None
        self._satisfied_facts: set[str] = set()
        self._completed_steps: list[str] = []
        self._started_tracking = False
        self._memory_recall_attempted = False
        self._memory_search_pose_used = False
        self._body_fallback_attempted = False
        self.user_confirms_visible = False
        self.user_confirmation_requested = False
        self.confirmation_retry_target: str | None = None
        self.effort = "center"

    async def start(
        self,
        goal,
        target,
        action_id,
        direction=None,
        camera_region=None,
        user_confirms_visible=False,
        effort="center",
    ) -> ActionResult:
        normalized_goal = str(goal or "").strip().lower()
        normalized_direction = str(direction or "").strip().lower() or None
        normalized_camera_region = (
            str(camera_region or "").strip().lower().replace("-", "_") or None
        )
        normalized_effort = str(effort or "center").strip().lower().replace("-", "_")
        target = str(target or "").strip()
        normalized_target = self._normalize_target(target)
        confirmation_requested = user_confirms_visible is True
        requested_action_type = self.GOAL_ACTION_TYPES.get(
            normalized_goal, "goal_action"
        )

        if self.active:
            return ActionResult(
                action_id=action_id,
                action_type=requested_action_type,
                status="already_running",
                target=target or None,
                outcome="already_running",
                reason_code="GOAL_BUSY",
                retryable=True,
                data={"active_goal": self.goal, "active_action_id": self.action_id},
            )
        if normalized_goal not in self.GOAL_FACTS:
            return self._invalid_result(
                action_id, target, "UNKNOWN_GOAL", "unsupported_goal",
                requested_action_type,
            )
        if not target:
            return self._invalid_result(
                action_id, None, "GOAL_TARGET_REQUIRED", "invalid_request",
                requested_action_type,
            )
        if normalized_direction not in {None, *self.EXPLICIT_TARGET_DIRECTIONS}:
            return self._invalid_result(
                action_id, target, "UNKNOWN_DIRECTION", "invalid_request",
                requested_action_type,
            )
        if (
            normalized_camera_region is not None
            and normalized_camera_region not in SEMANTIC_CAMERA_REGIONS
        ):
            return self._invalid_result(
                action_id, target, "UNKNOWN_CAMERA_REGION", "invalid_request",
                requested_action_type,
            )
        if normalized_effort not in {"center", "high", "low", "best_effort"}:
            return self._invalid_result(
                action_id, target, "UNKNOWN_SEARCH_EFFORT", "invalid_request",
                requested_action_type,
            )

        if (
            self.confirmation_retry_target is not None
            and normalized_target != self.confirmation_retry_target
        ):
            self.confirmation_retry_target = None

        confirmation_accepted = (
            confirmation_requested
            and normalized_direction is None
            and normalized_camera_region is None
            and normalized_target == self.confirmation_retry_target
        )
        if confirmation_accepted:
            self.confirmation_retry_target = None

        self.active = True
        self.action_id = action_id
        self.target = target
        self.goal = normalized_goal
        self.action_type = self.GOAL_ACTION_TYPES[normalized_goal]
        self.direction = normalized_direction
        self.camera_region = normalized_camera_region
        self._satisfied_facts = set()
        self._completed_steps = []
        self._started_tracking = False
        self._memory_recall_attempted = False
        self._memory_search_pose_used = False
        self._body_fallback_attempted = False
        self.user_confirms_visible = confirmation_accepted
        self.user_confirmation_requested = confirmation_requested
        self.effort = normalized_effort
        if normalized_direction is None:
            self._satisfied_facts.add("oriented")
        if normalized_camera_region is None:
            self._satisfied_facts.add("camera_aimed")

        self.completion_future = asyncio.get_running_loop().create_future()
        self._runner = asyncio.create_task(
            self._run(), name=f"goal-{normalized_goal}-{action_id}"
        )
        self._runner.add_done_callback(self._runner_done)

        return ActionResult(
            action_id=action_id,
            action_type=self.action_type,
            status="running",
            target=target,
            outcome="executing_goal",
            data={
                "goal": normalized_goal,
                "direction": normalized_direction,
                "camera_region": normalized_camera_region,
                "user_confirms_visible": self.user_confirms_visible,
                "user_confirmation_ignored": (
                    self.user_confirmation_requested
                    and not self.user_confirms_visible
                ),
                "effort": self.effort,
            },
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
        try:
            await self._ensure(self.GOAL_FACTS[self.goal])
        except asyncio.CancelledError:
            return
        except GoalStepFailed as error:
            if (
                error.step == "target_found"
                and error.result.reason_code
                in {"SEARCH_GUIDANCE_REQUIRED", "TARGET_NOT_VISIBLE"}
            ):
                self.confirmation_retry_target = self._normalize_target(self.target)
            failure_data = {
                "failed_step": error.step,
                "step_result": self._result_data(error.result),
            }
            if error.result.reason_code == "SEARCH_GUIDANCE_REQUIRED":
                failure_data["continuation_guidance"] = {
                    "unresolved_action_type": self.action_type,
                    "target": self.target,
                    "preserve_action_type": True,
                    "retry_answers": {
                        "high": {"effort": "high"},
                        "low": {"effort": "low"},
                        "best_effort": {"effort": "best_effort"},
                        "behind": {
                            "direction": "behind",
                            "effort": "center",
                        },
                    },
                    "out_of_frame_requires_no_retry": True,
                }
            self._complete(
                status="failed",
                outcome="goal_failed",
                reason_code=error.result.reason_code or "GOAL_STEP_FAILED",
                data=failure_data,
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
            outcome={
                "search": "found",
                "track": "tracking_started",
                "approach": "reached",
            }[self.goal],
        )
        self.confirmation_retry_target = None

    async def _ensure(self, fact: str) -> None:
        if self._fact_is_satisfied(fact):
            self._satisfied_facts.add(fact)
            return

        rule = self.FACT_RULES[fact]

        if fact == "target_found":
            await self._try_recalled_search_pose()

        if fact == "target_tracked" and await self._try_recalled_target():
            return

        if (
            fact == "target_tracked"
            and not self.user_confirms_visible
            and not self._memory_recall_attempted
            and is_yolo_trackable_target(self.target)
        ):
            await self._ensure("oriented")
            await self._ensure("camera_aimed")
            if self._fact_is_satisfied(fact):
                self._satisfied_facts.add(fact)
                return

            direct_result = await self._track(
                require_search=False,
                allow_grounding_dino=False,
            )
            if direct_result.status == "succeeded":
                self._satisfied_facts.update({"target_found", "target_tracked"})
                self._completed_steps.extend(["target_found", "target_tracked"])
                return

        for prerequisite in rule.requires:
            await self._ensure(prerequisite)

        if self._fact_is_satisfied(fact):
            self._satisfied_facts.add(fact)
            return

        handler: Callable[[], Awaitable[ActionResult]] = getattr(
            self, rule.handler_name
        )
        result = await handler()
        if (
            fact == "target_found"
            and result.status != "succeeded"
            and self._memory_search_pose_used
        ):
            self.semantic_memory.mark_miss(self._normalize_target(self.target))
        if fact == "target_found" and result.status != "succeeded":
            fallback_result = await self._try_camera_edge_body_fallback(result)
            if fallback_result is not None:
                result = fallback_result
        if result.status != "succeeded":
            raise GoalStepFailed(fact, result)
        self._satisfied_facts.add(fact)
        self._completed_steps.append(fact)

    async def _try_recalled_search_pose(self) -> bool:
        """Aim a find goal at remembered map coordinates before searching."""
        normalized_target = self._normalize_target(self.target)
        if (
            self.semantic_memory is None
            or self._memory_recall_attempted
            or normalized_target in human_trackable_parts
            or self.direction is not None
            or self.camera_region is not None
            or self.user_confirms_visible
        ):
            return False

        record = self.semantic_memory.recall(normalized_target)
        pose = robot_state.get("pose") or {}
        if not isinstance(record, dict) or not all(
            isinstance(pose.get(key), (int, float)) for key in ("x", "y", "yaw")
        ):
            return False

        self._memory_recall_attempted = True
        bearing = math.atan2(
            float(record["map_y"]) - float(pose["y"]),
            float(record["map_x"]) - float(pose["x"]),
        )
        relative_angle = math.atan2(
            math.sin(bearing - float(pose["yaw"])),
            math.cos(bearing - float(pose["yaw"])),
        )
        if abs(relative_angle) > math.radians(4.0):
            result = await self.move_action.start_moving(
                linear_velocity=0.0,
                distance=0.0,
                angular_velocity=1.0 if relative_angle > 0.0 else -1.0,
                angle=abs(relative_angle),
                action_id=self._step_id("memory_orient"),
            )
            result = await self._terminal_result(self.move_action, result)
            if result.status != "succeeded":
                return False

        camera_result = await self.move_camera_action.move_to_region(
            region="center",
            action_id=self._step_id("memory_camera"),
        )
        if camera_result.status != "succeeded":
            return False

        self.search_action.last_found_target = None
        self._memory_search_pose_used = True
        self._satisfied_facts.update({"oriented", "camera_aimed"})
        self._completed_steps.append("memory_recalled")
        return True

    async def _try_recalled_target(self) -> bool:
        normalized_target = self._normalize_target(self.target)
        if (
            self.semantic_memory is None
            or normalized_target in human_trackable_parts
            or self.direction is not None
            or self.camera_region is not None
            or self.user_confirms_visible
        ):
            return False

        record = self.semantic_memory.recall(normalized_target)
        pose = robot_state.get("pose") or {}
        if not isinstance(record, dict) or not all(
            isinstance(pose.get(key), (int, float)) for key in ("x", "y", "yaw")
        ):
            return False

        self._memory_recall_attempted = True
        bearing = math.atan2(
            float(record["map_y"]) - float(pose["y"]),
            float(record["map_x"]) - float(pose["x"]),
        )
        relative_angle = math.atan2(
            math.sin(bearing - float(pose["yaw"])),
            math.cos(bearing - float(pose["yaw"])),
        )
        if abs(relative_angle) > math.radians(4.0):
            result = await self.move_action.start_moving(
                linear_velocity=0.0,
                distance=0.0,
                angular_velocity=1.0 if relative_angle > 0.0 else -1.0,
                angle=abs(relative_angle),
                action_id=self._step_id("memory_orient"),
            )
            result = await self._terminal_result(self.move_action, result)
            if result.status != "succeeded":
                return False

        camera_result = await self.move_camera_action.move_to_region(
            region="center",
            action_id=self._step_id("memory_camera"),
        )
        if camera_result.status != "succeeded":
            return False

        self.search_action.last_found_target = None
        track_result = await self._track(
            require_search=False,
            allow_grounding_dino=True,
        )
        if track_result.status != "succeeded":
            self.semantic_memory.mark_miss(normalized_target)
            return False

        self._satisfied_facts.update({"target_found", "target_tracked"})
        self._completed_steps.extend(
            ["memory_recalled", "target_found", "target_tracked"]
        )
        return True

    def _fact_is_satisfied(self, fact: str) -> bool:
        normalized_target = self._normalize_target(self.target)
        if fact == "oriented":
            return fact in self._satisfied_facts
        if fact == "camera_aimed":
            return fact in self._satisfied_facts
        if fact == "target_found":
            if (
                "oriented" not in self._satisfied_facts
                or "camera_aimed" not in self._satisfied_facts
            ):
                return False
            if (
                self.user_confirms_visible
                and "oriented" in self._satisfied_facts
                and "camera_aimed" in self._satisfied_facts
            ):
                return True
            found_target = self._normalize_target(self.search_action.last_found_target)
            return not self.search_action.active and found_target == normalized_target
        if fact == "target_tracked":
            return (
                self.track_action.active
                and self._normalize_target(self.track_action.target) == normalized_target
            )
        return False

    async def _orient(self) -> ActionResult:
        return await self._turn_body(self.direction, "orient")

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

    async def _try_camera_edge_body_fallback(
        self, initial_search_result: ActionResult
    ) -> ActionResult | None:
        fallback = self.CAMERA_EDGE_BODY_FALLBACKS.get(self.camera_region)
        if (
            fallback is None
            or self.direction is not None
            or self._body_fallback_attempted
            or initial_search_result.reason_code
            not in {"SEARCH_GUIDANCE_REQUIRED", "TARGET_NOT_VISIBLE"}
        ):
            return None

        self._body_fallback_attempted = True
        centered_region, turn_direction = fallback

        camera_result = await self.move_camera_action.move_to_region(
            region=centered_region,
            action_id=self._step_id("body_fallback_camera"),
        )
        if camera_result.status != "succeeded":
            return camera_result
        self._completed_steps.append("body_fallback_camera_centered")

        turn_result = await self._turn_body(
            turn_direction, "body_fallback_turn"
        )
        if turn_result.status != "succeeded":
            return turn_result
        self._completed_steps.append(f"body_turned_{turn_direction}")

        retry_result = await self._search("body_fallback_search")
        if retry_result.status == "succeeded":
            return retry_result

        return ActionResult(
            action_id=retry_result.action_id,
            action_type=retry_result.action_type,
            status=retry_result.status,
            target=retry_result.target,
            outcome=retry_result.outcome,
            reason_code=retry_result.reason_code,
            retryable=retry_result.retryable,
            data={
                **retry_result.data,
                "automatic_body_fallback": {
                    "initial_camera_region": self.camera_region,
                    "centered_camera_region": centered_region,
                    "body_direction": turn_direction,
                    "initial_reason_code": initial_search_result.reason_code,
                    "attempted_once": True,
                },
            },
        )

    async def _aim_camera(self) -> ActionResult:
        result = await self.move_camera_action.move_to_region(
            region=self.camera_region,
            action_id=self._step_id("aim_camera"),
        )
        if result.status == "succeeded":
            self.search_action.last_found_target = None
        return result

    async def _search(self, step: str = "search") -> ActionResult:
        if (
            self.search_action.active
            and self._normalize_target(self.search_action.target)
            == self._normalize_target(self.target)
        ):
            return await self.search_action.wait_until_finished()

        result = await self.search_action.start_searching(
            target=self.target,
            action_id=self._step_id(step),
            effort=self.effort,
        )
        return await self._terminal_result(self.search_action, result)

    async def _track(
        self,
        require_search=None,
        allow_grounding_dino=True,
    ) -> ActionResult:
        if require_search is None:
            require_search = not self.user_confirms_visible
        result = await self.track_action.start_tracking(
            target=self.target,
            action_id=self._step_id("track"),
            require_search=require_search,
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

    async def _approach(self) -> ActionResult:
        if (
            self.approach_action.active
            and self._normalize_target(self.approach_action.target)
            == self._normalize_target(self.target)
        ):
            return await self.approach_action.wait_until_finished()

        result = await self.approach_action.start_approaching(
            target=self.target,
            action_id=self._step_id("approach"),
        )
        return await self._terminal_result(self.approach_action, result)

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
                "direction": self.direction,
                "camera_region": self.camera_region,
                "user_confirms_visible": self.user_confirms_visible,
                "user_confirmation_ignored": (
                    self.user_confirmation_requested
                    and not self.user_confirms_visible
                ),
                "effort": self.effort,
                "completed_steps": list(self._completed_steps),
                **(data or {}),
            },
        )
        completion_future = self.completion_future
        self.active = False
        self.action_id = None
        self.target = None
        self.goal = None
        self.action_type = None
        self.direction = None
        self.camera_region = None
        self.user_confirms_visible = False
        self.user_confirmation_requested = False
        self.effort = "center"
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
