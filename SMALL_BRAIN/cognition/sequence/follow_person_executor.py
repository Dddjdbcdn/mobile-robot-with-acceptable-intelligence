from __future__ import annotations

import asyncio

from actions.action_result import ActionResult
from actions.tracking.target_catalog import normalize_human_target


class FollowPersonExecutor:
    """Acquire a person, then hand continuous following to the ROS bridge."""

    def __init__(self, find_target_executor, track_action, send_robot_command):
        self.find_target_executor = find_target_executor
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
                action_id, "follow_person", "already_running", target=self.target,
                outcome="already_running", reason_code="FOLLOW_BUSY", retryable=True,
                data={"active_action_id": self.action_id},
            )

        normalized_target = normalize_human_target(target or "person")
        if normalized_target != "person":
            return ActionResult(
                action_id, "follow_person", "failed", target=target,
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
            action_id, "follow_person", "running", target="person",
            outcome="acquiring_person",
        )

    async def _run(self):
        try:
            acquired = await self._acquire_target()
            if acquired.status != "succeeded":
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
                self._finish(
                    "failed", "execution_error", "FOLLOW_EXECUTION_ERROR",
                    {"error": f"{type(error).__name__}: {error}"},
                )

    async def _acquire_target(self):
        acquisition_id = f"{self.action_id}:acquire-person"
        tracking_action_id = f"{self.action_id}:track"
        person_tracking_active = bool(
            self.track_action.active
            and normalize_human_target(self.track_action.target) is not None
        )
        if person_tracking_active:
            shared_tracking = await self.track_action.start_tracking(
                target=self.track_action.target,
                action_id=tracking_action_id,
                allow_grounding_dino=False,
                continuous_person_reacquisition=True,
            )
            if shared_tracking.status == "running":
                tracking_session_id = self.track_action.session_id
                seed = await self.track_action.wait_for_stable_target_seed(
                    target=self.track_action.target,
                    session_id=tracking_session_id,
                    timeout=10.0,
                )
                if seed is not None:
                    return ActionResult(
                        self.action_id, "follow_person", "succeeded",
                        target="person", outcome="person_tracking_reused",
                        data={
                            "reused_person_tracking": True,
                            "stable_seed": dict(seed),
                        },
                    )
                return ActionResult(
                    self.action_id, "follow_person", "failed", target="person",
                    outcome="stable_seed_timeout",
                    reason_code="STABLE_SEED_TIMEOUT",
                    retryable=True,
                )

        result = await self.find_target_executor.start(
            target="person",
            action_id=acquisition_id,
            target_kind="person",
            approach=False,
            continuous_person_reacquisition=True,
        )
        if result.status == "running":
            result = await asyncio.shield(
                self.find_target_executor.wait_until_finished()
            )
        if result.status != "succeeded":
            return result

        return ActionResult(
            self.action_id, "follow_person", "succeeded", target="person",
            outcome="person_acquired",
        )

    async def _reacquire_camera_tracking(self):
        """Retry vision indefinitely while lidar still owns a valid track."""
        tracking_action_id = f"{self.action_id}:track"
        while self.active and not self._stop_requested and not self._lidar_lost:
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
            self._finish(
                "failed", "target_lost", "LIDAR_TRACK_LOST",
                {"tracker_state": "lost"},
            )
            return True

        if payload.get("event") != "navigation":
            return False

        status = str(payload.get("status", ""))
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
        acquisition_id = f"{self.action_id}:acquire-person"
        if (
            self.find_target_executor.active
            and self.find_target_executor.action_id == acquisition_id
        ):
            await self.find_target_executor.stop(reason_code)
        await self.send_robot_command({
            "command": "stop_follow_action",
            "action_id": self.action_id,
        })
        result = self._finish("cancelled", "stopped", reason_code)
        if runner is not None and runner is not asyncio.current_task():
            await asyncio.gather(runner, return_exceptions=True)
        return result

    def _finish(self, status, outcome, reason_code=None, data=None):
        result = ActionResult(
            self.action_id or "unassigned", "follow_person", status,
            target=self.target, outcome=outcome, reason_code=reason_code,
            retryable=status == "failed", data=data or {},
        )
        self.active = False
        future = self.completion_future
        if future is not None and not future.done():
            future.set_result(result)
        return result
