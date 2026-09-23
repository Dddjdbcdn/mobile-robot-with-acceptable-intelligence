from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import time

from actions.action_result import ActionResult
from actions.track_action import normalize_human_target
from cognition.state import robot_state


@dataclass(frozen=True, slots=True)
class FollowConfig:
    default_radius_m: float = 0.3
    poll_seconds: float = 0.20
    target_max_age_seconds: float = 1.0
    tracking_recovery_seconds: float = 3.0
    approach_standoff_ratio: float = 0.8


class FollowExecutor:
    """Continuously find, track, and safely approach one person."""

    def __init__(self, goal_executor, track_action, approach_action, config=None):
        self.goal_executor = goal_executor
        self.search_action = goal_executor.search_action
        self.track_action = track_action
        self.approach_action = approach_action
        self.config = config or FollowConfig()
        self.active = False
        self.action_id: str | None = None
        self.target: str | None = None
        self.radius_m = self.config.default_radius_m
        self.completion_future: asyncio.Future[ActionResult] | None = None
        self._runner: asyncio.Task | None = None
        self._stop_requested = False
        self._approach_count = 0
        self._recovery_count = 0

    async def start(self, target, action_id, radius_m=None) -> ActionResult:
        if self.active:
            return ActionResult(
                action_id, "follow_action", "already_running", target=self.target,
                outcome="already_running", reason_code="FOLLOW_BUSY", retryable=True,
                data={"active_action_id": self.action_id},
            )
        normalized_target = normalize_human_target(target or "person")
        if normalized_target != "person":
            return ActionResult(
                action_id, "follow_action", "failed", target=target,
                outcome="invalid_request", reason_code="FOLLOW_PERSON_REQUIRED",
            )
        try:
            radius = float(
                self.config.default_radius_m if radius_m is None else radius_m
            )
        except (TypeError, ValueError):
            radius = math.nan
        if not math.isfinite(radius) or not 0.4 <= radius <= 5.0:
            return ActionResult(
                action_id, "follow_action", "failed", target="person",
                outcome="invalid_request", reason_code="FOLLOW_RADIUS_INVALID",
                data={"minimum_m": 0.4, "maximum_m": 5.0},
            )

        self.active = True
        self.action_id = action_id
        self.target = "person"
        self.radius_m = radius
        self._stop_requested = False
        self._approach_count = 0
        self._recovery_count = 0
        self.completion_future = asyncio.get_running_loop().create_future()
        self._runner = asyncio.create_task(
            self._run(), name=f"follow-person-{action_id}"
        )
        return ActionResult(
            action_id, "follow_action", "running", target="person",
            outcome="finding_and_following", data={"radius_m": radius},
        )

    async def _run(self):
        try:
            acquired = await self._acquire_target("initial")
            if acquired.status != "succeeded":
                await self._stop_follow_motion("FOLLOW_ACQUISITION_FAILED")
                self._finish(
                    "failed", "follow_failed",
                    acquired.reason_code or "FOLLOW_ACQUISITION_FAILED",
                    {"failed_step": "initial_acquisition"},
                )
                return

            while self.active and not self._stop_requested:
                distance = self._fresh_distance()
                if not self.track_action.active or distance is None:
                    recovered = await self._recover_tracking()
                    if not recovered:
                        await self._stop_follow_motion(
                            "FOLLOW_REACQUISITION_FAILED"
                        )
                        self._finish(
                            "failed", "target_lost",
                            "FOLLOW_REACQUISITION_FAILED",
                            {"reacquisition_attempts": self._recovery_count},
                        )
                        return
                    continue

                if distance <= self.radius_m:
                    if self.approach_action.active:
                        await self.approach_action.stop_approaching(
                            "FOLLOW_RADIUS_REACHED"
                        )
                    await asyncio.sleep(self.config.poll_seconds)
                    continue

                was_approaching = self.approach_action.active
                result = await self.approach_action.start_approaching(
                    target="person",
                    action_id=f"{self.action_id}:follow",
                    standoff_m=self._approach_standoff_m(),
                    follow=True,
                )
                if result.status != "running":
                    recovered = await self._recover_tracking()
                    if not recovered:
                        await self._stop_follow_motion(
                            "FOLLOW_REACQUISITION_FAILED"
                        )
                        self._finish(
                            "failed", "approach_failed",
                            result.reason_code or "FOLLOW_APPROACH_FAILED",
                            {"failed_step": "follow_update"},
                        )
                        return
                elif not was_approaching:
                    self._approach_count += 1
                await asyncio.sleep(self.config.poll_seconds)
        except asyncio.CancelledError:
            return
        except Exception as error:
            if self.active and not self._stop_requested:
                await self._stop_follow_motion("FOLLOW_EXECUTION_ERROR")
                self._finish(
                    "failed", "execution_error", "FOLLOW_EXECUTION_ERROR",
                    {"error": f"{type(error).__name__}: {error}"},
                )

    async def _acquire_target(self, step):
        """Best-effort search followed by tracking; never invokes find_object."""
        search_result = await self.search_action.start_searching(
            target="person",
            action_id=f"{self.action_id}:{step}:search",
            effort="best_effort",
            initial_view_only=False,
        )
        search_result = await self._terminal_result(
            self.search_action, search_result
        )
        if search_result.status != "succeeded":
            return search_result

        track_result = await self.track_action.start_tracking(
            target="person",
            action_id=f"{self.action_id}:{step}:track",
            allow_grounding_dino=True,
        )
        if track_result.status != "running":
            return track_result

        return ActionResult(
            f"{self.action_id}:{step}", "follow_action", "succeeded",
            target="person", outcome="target_acquired_and_tracking",
            data={"radius_m": self.radius_m},
        )

    async def _recover_tracking(self):
        """Give live tracking time to recover, then reacquire once if needed."""
        if self.approach_action.active:
            await self.approach_action.stop_approaching("TRACKING_LOST")

        deadline = (
            asyncio.get_running_loop().time()
            + self.config.tracking_recovery_seconds
        )
        while self.track_action.active and not self._stop_requested:
            if self._fresh_distance() is not None:
                return True
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(self.config.poll_seconds)

        if self._stop_requested:
            return False
        if self.track_action.active:
            await self.track_action.stop_tracking("TRACKING_RECOVERY_TIMEOUT")

        self._recovery_count += 1
        result = await self._acquire_target(
            f"reacquire:{self._recovery_count}"
        )
        if result.status == "succeeded" and self.track_action.active:
            return True
        return False

    def _approach_standoff_m(self):
        return self.radius_m * self.config.approach_standoff_ratio

    async def _stop_follow_motion(self, reason_code):
        if self.approach_action.active:
            await self.approach_action.stop_approaching(reason_code)
        if self.track_action.active:
            await self.track_action.stop_tracking(reason_code)

    @staticmethod
    async def _terminal_result(action, result):
        if result.status == "running":
            return await action.wait_until_finished()
        return result

    def _fresh_distance(self):
        camera = robot_state.get("camera") or {}
        x = camera.get("object_x")
        y = camera.get("object_y")
        timestamp = camera.get("timestamp")
        if not all(isinstance(value, (int, float)) for value in (x, y)):
            return None
        if not all(math.isfinite(float(value)) for value in (x, y)):
            return None
        if not isinstance(timestamp, (int, float)):
            return None
        if time.monotonic() - float(timestamp) > self.config.target_max_age_seconds:
            return None
        distance = math.hypot(float(x), float(y))
        return distance if distance > 0.05 else None

    async def wait_until_finished(self):
        return await asyncio.shield(self.completion_future)

    async def stop(self, reason_code="USER_REQUESTED"):
        if not self.active:
            return None
        self._stop_requested = True
        runner = self._runner
        if runner is not None and runner is not asyncio.current_task():
            runner.cancel()
        if self.goal_executor.active:
            await self.goal_executor.stop(reason_code)
        if self.search_action.active:
            await self.search_action.stop_searching(reason_code)
        if self.approach_action.active:
            await self.approach_action.stop_approaching(reason_code)
        if self.track_action.active:
            await self.track_action.stop_tracking(reason_code)
        result = self._finish("cancelled", "stopped", reason_code)
        if runner is not None and runner is not asyncio.current_task():
            await asyncio.gather(runner, return_exceptions=True)
        return result

    def _finish(self, status, outcome, reason_code=None, data=None):
        result = ActionResult(
            self.action_id or "unassigned", "follow_action", status,
            target=self.target, outcome=outcome, reason_code=reason_code,
            retryable=status == "failed",
            data={
                "radius_m": self.radius_m,
                "approach_count": self._approach_count,
                "recovery_count": self._recovery_count,
                **(data or {}),
            },
        )
        self.active = False
        future = self.completion_future
        if future is not None and not future.done():
            future.set_result(result)
        return result
