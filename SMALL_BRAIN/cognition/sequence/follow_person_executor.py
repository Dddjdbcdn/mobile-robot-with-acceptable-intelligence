from __future__ import annotations

import asyncio
import math

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
        self._hand_target_override_active = False

    def _debug(self, event, **fields):
        action_id = self.action_id or "unassigned"
        details = " ".join(
            f"{name}={value!r}" for name, value in fields.items()
        )
        suffix = f" {details}" if details else ""
        print(f"[FOLLOW_PERSON][{action_id}] {event}{suffix}")

    async def start(self, target, action_id) -> ActionResult:
        if self.active:
            self._debug(
                "start_rejected_busy",
                requested_action_id=action_id,
                requested_target=target,
            )
            return ActionResult(
                action_id, "follow_person", "already_running", target=self.target,
                outcome="already_running", reason_code="FOLLOW_BUSY", retryable=True,
                data={"active_action_id": self.action_id},
            )

        normalized_target = normalize_human_target(target or "person")
        if normalized_target != "person":
            print(
                f"[FOLLOW_PERSON][{action_id}] start_rejected_invalid_target "
                f"target={target!r} normalized_target={normalized_target!r}"
            )
            return ActionResult(
                action_id, "follow_person", "failed", target=target,
                outcome="invalid_request", reason_code="FOLLOW_PERSON_REQUIRED",
            )

        self.active = True
        self.action_id = action_id
        self.target = "person"
        self._stop_requested = False
        self._hand_target_override_active = False
        self.track_action.set_follow_person_active(True)
        self.completion_future = asyncio.get_running_loop().create_future()
        self._runner = asyncio.create_task(
            self._supervise_run(action_id), name=f"follow-person-{action_id}"
        )
        self._debug("started", target=self.target)
        return ActionResult(
            action_id, "follow_person", "running", target="person",
            outcome="acquiring_person",
        )

    async def _supervise_run(self, action_id):
        try:
            await self._run()
        except asyncio.CancelledError:
            self._debug("runner_cancelled")
            raise
        except Exception as error:
            if (
                self.active
                and self.action_id == action_id
                and not self._stop_requested
            ):
                await self._fail_execution(error)

    async def _run(self):
        self._debug("acquisition_started")
        acquired = await self._acquire_target()
        if acquired.status != "succeeded":
            self._debug(
                "acquisition_failed",
                status=acquired.status,
                outcome=acquired.outcome,
                reason_code=acquired.reason_code,
            )
            self._finish(
                "failed", "follow_failed",
                acquired.reason_code or "FOLLOW_ACQUISITION_FAILED",
                {"failed_step": "person_acquisition"},
            )
            return

        self._debug("acquisition_succeeded", outcome=acquired.outcome)
        self._debug("sending_follow_command")
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
            self._debug(
                "follow_command_rejected",
                feedback=feedback,
                message=message,
            )
            self._finish(
                "failed", "follow_rejected", "FOLLOW_REJECTED",
                {"message": message},
            )
            return

        self._debug("follow_command_accepted", feedback=feedback)

    async def _request_follow_stop(self):
        try:
            await self.send_robot_command({
                "command": "stop_follow_action",
                "action_id": self.action_id,
            })
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            self._debug("stop_command_failed", error=error_text)
            return error_text
        return None

    async def _fail_execution(self, error):
        error_text = f"{type(error).__name__}: {error}"
        self._debug("execution_failed", error=error_text)
        cleanup_error = await self._request_follow_stop()
        data = {"error": error_text}
        if cleanup_error is not None:
            data["cleanup_error"] = cleanup_error
        self._finish(
            "failed", "execution_error", "FOLLOW_EXECUTION_ERROR", data
        )

    async def _acquire_target(self):
        acquisition_id = f"{self.action_id}:acquire-person"
        result = await self.find_target_executor.start(
            target="person",
            action_id=acquisition_id,
            target_kind="person",
            approach=False,
        )
        if result.status == "running":
            result = await asyncio.shield(
                self.find_target_executor.wait_until_finished()
            )
        if result.status != "succeeded":
            return result

        tracking_session_id = self.track_action.session_id
        seed = await self.track_action.wait_for_stable_target_seed(
            target="person",
            session_id=tracking_session_id,
            timeout=10.0,
        )
        if seed is None:
            return ActionResult(
                self.action_id, "follow_person", "failed", target="person",
                outcome="stable_seed_timeout",
                reason_code="STABLE_SEED_TIMEOUT",
                retryable=True,
            )

        return ActionResult(
            self.action_id, "follow_person", "succeeded", target="person",
            outcome="person_acquired",
            data={"stable_seed": dict(seed)},
        )

    async def handle_navigation_event(self, payload):
        if not self.active:
            return False
        event_action_id = payload.get("action_id")
        if event_action_id not in {None, self.action_id}:
            return False

        if payload.get("event") == "person_tracker":
            tracker_status = str(payload.get("status") or "")
            if tracker_status != "lost":
                return False

            self._debug("lidar_track_lost", payload=payload)
            cleanup_error = await self._request_follow_stop()
            data = {"tracker_state": "lost"}
            if cleanup_error is not None:
                data["cleanup_error"] = cleanup_error
            self._finish(
                "failed", "target_lost", "LIDAR_TRACK_LOST",
                data,
            )
            return True

        if payload.get("event") != "navigation":
            return False

        status = str(payload.get("status", ""))
        self._debug("navigation_ended", status=status, payload=payload)
        navigation_data = payload.get("data")
        self._finish(
            "failed",
            str(payload.get("outcome") or "navigation_ended"),
            str(payload.get("reason_code") or "FOLLOW_NAVIGATION_ENDED"),
            {
                "robot_status": status or "unknown",
                "navigation": (
                    dict(navigation_data)
                    if isinstance(navigation_data, dict) else {}
                ),
            },
        )
        return True

    async def update_hand_target_override(self, seed, gesture="welcome"):
        """Use a validated welcome/push hand seed for the follow goal."""
        if not self.active or not isinstance(seed, dict):
            return False
        try:
            map_x = float(seed["map_x"])
            map_y = float(seed["map_y"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(map_x) or not math.isfinite(map_y):
            return False
        if gesture not in {"welcome", "push"}:
            return False

        feedback = await self.send_robot_command({
            "command": "set_follow_target_override",
            "action_id": self.action_id,
            "target": "hand",
            "override_type": gesture,
            "frame_id": "map",
            "x": map_x,
            "y": map_y,
        })
        accepted = (
            isinstance(feedback, dict)
            and feedback.get("status") == "accepted"
        )
        if accepted:
            self._hand_target_override_active = True
        return accepted

    async def clear_hand_target_override(self):
        """Return following to the lidar person pose without stopping it."""
        if not self._hand_target_override_active:
            return False
        self._hand_target_override_active = False
        if not self.active:
            return False
        feedback = await self.send_robot_command({
            "command": "clear_follow_target_override",
            "action_id": self.action_id,
        })
        return (
            isinstance(feedback, dict)
            and feedback.get("status") == "accepted"
        )

    async def wait_until_finished(self):
        return await asyncio.shield(self.completion_future)

    async def stop(self, reason_code="USER_REQUESTED"):
        if not self.active:
            return None

        self._debug("stop_requested", reason_code=reason_code)
        self._stop_requested = True
        runner = self._runner
        if runner is not None and runner is not asyncio.current_task():
            runner.cancel()
        acquisition_id = f"{self.action_id}:acquire-person"
        cleanup_errors = {}
        if (
            self.find_target_executor is not None
            and self.find_target_executor.active
            and self.find_target_executor.action_id == acquisition_id
        ):
            try:
                await self.find_target_executor.stop(reason_code)
            except Exception as error:
                cleanup_errors["acquisition_cleanup_error"] = (
                    f"{type(error).__name__}: {error}"
                )
        follow_cleanup_error = await self._request_follow_stop()
        if follow_cleanup_error is not None:
            cleanup_errors["cleanup_error"] = follow_cleanup_error
        self._hand_target_override_active = False
        result = self._finish(
            "cancelled", "stopped", reason_code, cleanup_errors
        )
        if runner is not None and runner is not asyncio.current_task():
            await asyncio.gather(runner, return_exceptions=True)
        return result

    def _finish(self, status, outcome, reason_code=None, data=None):
        result_data = dict(data or {})
        self._debug(
            "finished",
            status=status,
            outcome=outcome,
            reason_code=reason_code,
            data=result_data,
        )
        self.active = False
        self._hand_target_override_active = False
        if self.track_action is not None:
            try:
                self.track_action.set_follow_person_active(False)
            except Exception as error:
                result_data["tracking_cleanup_error"] = (
                    f"{type(error).__name__}: {error}"
                )
        result = ActionResult(
            self.action_id or "unassigned", "follow_person", status,
            target=self.target, outcome=outcome, reason_code=reason_code,
            retryable=status == "failed", data=result_data,
        )
        future = self.completion_future
        if future is not None and not future.done():
            future.set_result(result)
        return result
