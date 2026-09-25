from __future__ import annotations

import asyncio

from actions.action_result import ActionResult
from actions.track_action import normalize_human_target


class FollowExecutor:
    """Acquire a person, then hand continuous following to the ROS bridge."""

    def __init__(self, goal_executor, track_action, send_robot_command):
        self.goal_executor = goal_executor
        self.search_action = goal_executor.search_action
        self.track_action = track_action
        self.send_robot_command = send_robot_command
        self.active = False
        self.action_id: str | None = None
        self.target: str | None = None
        self.completion_future: asyncio.Future[ActionResult] | None = None
        self._runner: asyncio.Task | None = None
        self._stop_requested = False
        self._lidar_lost = False

    async def start(self, target, action_id) -> ActionResult:
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

        self.active = True
        self.action_id = action_id
        self.target = "person"
        self._stop_requested = False
        self._lidar_lost = False
        self.completion_future = asyncio.get_running_loop().create_future()
        self._runner = asyncio.create_task(
            self._run(), name=f"follow-person-{action_id}"
        )
        return ActionResult(
            action_id, "follow_action", "running", target="person",
            outcome="acquiring_person",
        )

    async def _run(self):
        try:
            acquired = await self._acquire_target()
            if acquired.status != "succeeded":
                await self._stop_tracking("FOLLOW_ACQUISITION_FAILED")
                self._finish(
                    "failed", "follow_failed",
                    acquired.reason_code or "FOLLOW_ACQUISITION_FAILED",
                    {"failed_step": "person_acquisition"},
                )
                return

            feedback = await self.send_robot_command({
                "command": "follow_action",
                "action_id": self.action_id,
            })
            if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
                message = (
                    feedback.get("message", "follow_rejected")
                    if isinstance(feedback, dict)
                    else "invalid_follow_feedback"
                )
                await self._stop_tracking("FOLLOW_REJECTED")
                self._finish(
                    "failed", "follow_rejected", "FOLLOW_REJECTED",
                    {"message": message},
                )
                return

            while self.active and not self._stop_requested:
                await self.track_action.wait_until_finished()
                if not self.active or self._stop_requested or self._lidar_lost:
                    return
                if not await self._reacquire_camera_tracking():
                    return
        except asyncio.CancelledError:
            return
        except Exception as error:
            if self.active and not self._stop_requested:
                await self.send_robot_command({
                    "command": "stop_follow_action",
                    "action_id": self.action_id,
                })
                await self._stop_tracking("FOLLOW_EXECUTION_ERROR")
                self._finish(
                    "failed", "execution_error", "FOLLOW_EXECUTION_ERROR",
                    {"error": f"{type(error).__name__}: {error}"},
                )

    async def _acquire_target(self):
        search_result = await self.search_action.start_searching(
            target="person",
            action_id=f"{self.action_id}:search",
            effort="best_effort",
            initial_view_only=False,
        )
        search_result = await self._terminal_result(
            self.search_action, search_result
        )
        if search_result.status != "succeeded":
            return search_result

        track_result = await self.track_action.start_tracking(
            target="torso center",
            action_id=f"{self.action_id}:track",
            allow_grounding_dino=True,
            continuous_person_reacquisition=True,
        )
        if track_result.status != "running":
            return track_result

        return ActionResult(
            self.action_id, "follow_action", "succeeded", target="person",
            outcome="person_acquired",
        )

    async def _reacquire_camera_tracking(self):
        """Retry vision indefinitely while lidar still owns a valid track."""
        tracking_action_id = f"{self.action_id}:track"
        while self.active and not self._stop_requested and not self._lidar_lost:
            keep_alive = getattr(
                self.track_action, "keep_person_tracker_alive", None
            )
            if keep_alive is not None:
                await keep_alive(tracking_action_id)

            result = await self.track_action.start_tracking(
                target="torso center",
                action_id=tracking_action_id,
                allow_grounding_dino=True,
                continuous_person_reacquisition=True,
            )
            if result.status == "running":
                return True
            await asyncio.sleep(0.25)
        return False

    @staticmethod
    async def _terminal_result(action, result):
        if result.status == "running":
            return await action.wait_until_finished()
        return result

    async def _stop_tracking(self, reason_code):
        if self.track_action.active:
            await self.track_action.stop_tracking(reason_code)

    async def handle_navigation_event(self, payload):
        if not self.active:
            return False
        if payload.get("action_id") not in {None, self.action_id}:
            return False

        if (
            payload.get("event") == "person_tracker"
            and payload.get("status") == "lost"
        ):
            self._lidar_lost = True
            await self.send_robot_command({
                "command": "stop_follow_action",
                "action_id": self.action_id,
            })
            await self._stop_tracking("LIDAR_TRACK_LOST")
            self._finish(
                "failed", "target_lost", "LIDAR_TRACK_LOST",
                {"tracker_state": "lost"},
            )
            return True

        if payload.get("event") != "navigation":
            return False

        status = str(payload.get("status", ""))
        await self._stop_tracking("FOLLOW_NAVIGATION_ENDED")
        self._finish(
            "failed", "navigation_ended", "FOLLOW_NAVIGATION_ENDED",
            {"robot_status": status or "unknown"},
        )
        return True

    async def wait_until_finished(self):
        return await asyncio.shield(self.completion_future)

    async def stop(self, reason_code="USER_REQUESTED"):
        if not self.active:
            return None

        self._stop_requested = True
        runner = self._runner
        if runner is not None and runner is not asyncio.current_task():
            runner.cancel()
        if self.search_action.active:
            await self.search_action.stop_searching(reason_code)
        await self.send_robot_command({
            "command": "stop_follow_action",
            "action_id": self.action_id,
        })
        await self._stop_tracking(reason_code)
        result = self._finish("cancelled", "stopped", reason_code)
        if runner is not None and runner is not asyncio.current_task():
            await asyncio.gather(runner, return_exceptions=True)
        return result

    def _finish(self, status, outcome, reason_code=None, data=None):
        result = ActionResult(
            self.action_id or "unassigned", "follow_action", status,
            target=self.target, outcome=outcome, reason_code=reason_code,
            retryable=status == "failed", data=data or {},
        )
        self.active = False
        future = self.completion_future
        if future is not None and not future.done():
            future.set_result(result)
        return result
