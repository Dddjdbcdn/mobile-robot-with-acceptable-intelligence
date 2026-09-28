from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
import json
from statistics import median
import time
import uuid

from actions.action_result import ActionResult
from actions.tracking.target_catalog import normalize_human_target

@dataclass(frozen=True, slots=True)
class ToolRequest:
    function_name: str
    arguments: dict
    call_id: str
    action_id: str
    response_metadata: dict


@dataclass(frozen=True, slots=True)
class ActionExecuted:
    request: ToolRequest
    result: object = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class ActionLifecycleFinished:
    request: ToolRequest
    result: object = None
    error: BaseException | None = None

@dataclass(frozen=True, slots=True)
class WorldState:
    payload: dict


@dataclass(frozen=True, slots=True)
class Shutdown:
    pass


class CognitionManager:
    OOB_TOOLS = {
        "assess_frame_search",
        "assess_batch_search",
        "select_navigation_pose",
    }
    STOP_TOOLS = {
        "stop_goal",
        "stop_navigation",
        "stop_follow",
        "stop_watching_target",
    }

    def __init__(
        self,
        approach_action,
        search_action,
        see_action,
        track_action,
        find_target_executor,
        follow_person_executor,
        response_manager,
        proximity_cooldown=8.0,
        explicit_navigation_action=None,
        map_navigation_action=None,
        hand_guided_navigation=None,
    ):
        self.approach_action = approach_action
        self.search_action = search_action
        self.see_action = see_action
        self.track_action = track_action
        self.explicit_navigation_action = explicit_navigation_action
        self.map_navigation_action = map_navigation_action
        self.hand_guided_navigation = hand_guided_navigation
        self.find_target_executor = find_target_executor
        self.follow_person_executor = follow_person_executor
        self.response_manager = response_manager

        self.events = asyncio.Queue()
        self.action_results = []
        self.current_action: ToolRequest | None = None
        self.current_action_task: asyncio.Task | None = None
        self._running = False

        self.world_state = {}
        self._pending_world_state: dict | None = None
        self._world_state_queued = False
        self._last_proximity_reaction = 0.0
        self._acquire_person_after_startup = True
        self._startup_person_acquisition_pending = True
        self._autonomy_task: asyncio.Task | None = None
        self._tof_history = deque(maxlen=5)
        self._proximity_candidate_count = 0

        self.proximity_cooldown = proximity_cooldown

    async def handle_tool_call(
        self,
        function_name,
        arguments,
        call_id,
        response_metadata=None,
    ):
        args = json.loads(arguments) if arguments else {}
        if function_name not in self.OOB_TOOLS:
            self._cancel_startup_person_acquisition()

        request = ToolRequest(
            function_name=function_name,
            arguments=args,
            call_id=call_id,
            action_id=uuid.uuid4().hex,
            response_metadata=response_metadata or {},
        )

        print(f"\n🌼 [FUNCTION CALLED]: NAME: {function_name}. ARGS: {args}\n")
        await self.events.put(request)

    async def handle_tool_msg(self, message):
        self.approach_action.handle_navigation_event(message)
        await self.follow_person_executor.handle_navigation_event(message)
        if self.explicit_navigation_action is not None:
            self.explicit_navigation_action.handle_navigation_event(message)
        map_navigation = getattr(self, "map_navigation_action", None)
        hand_navigation = getattr(self, "hand_guided_navigation", None)
        if map_navigation is not None:
            map_navigation.handle_navigation_event(message)
        if hand_navigation is not None:
            hand_navigation.handle_navigation_event(message)

    async def publish_world_state(self, payload):
        self._pending_world_state = dict(payload)
        if self._world_state_queued:
            return
        self._world_state_queued = True
        await self.events.put(WorldState({}))

    def note_user_activity(self):
        self._cancel_startup_person_acquisition()

    def _cancel_startup_person_acquisition(self):
        task = self._autonomy_task
        if (
            task is not None
            and not task.done()
            and task.get_name() == "startup-person-acquisition"
        ):
            self._startup_person_acquisition_pending = True
            task.cancel()

    async def shutdown(self):
        if self.follow_person_executor.active:
            await self.follow_person_executor.stop("SHUTDOWN")
        if self.hand_guided_navigation is not None:
            await self.hand_guided_navigation.stop("SHUTDOWN")
        if self.find_target_executor.active:
            await self.find_target_executor.stop("SHUTDOWN")
        elif self.explicit_navigation_action is not None and self.explicit_navigation_action.active:
            await self.explicit_navigation_action.stop("SHUTDOWN")
        autonomy_task = self._autonomy_task
        if autonomy_task is not None and not autonomy_task.done():
            autonomy_task.cancel()
            await asyncio.gather(autonomy_task, return_exceptions=True)
        await self.events.put(Shutdown())

    async def cognition_loop(self):
        self._running = True

        while self._running:
            event = await self.events.get()

            if isinstance(event, ToolRequest):
                await self.handle_tool_request(event)
            elif isinstance(event, ActionExecuted):
                await self.handle_action_executed(event)
            elif isinstance(event, ActionLifecycleFinished):
                await self.handle_lifecycle_finished(event)
            elif isinstance(event, WorldState):
                payload = self._pending_world_state or event.payload
                self._pending_world_state = None
                self._world_state_queued = False
                await self._handle_world_state(payload)
            elif isinstance(event, Shutdown):
                self._running = False

    async def handle_tool_request(self, request):
        if request.function_name in self.OOB_TOOLS:
            await self.oob_assessment(request)
            return

        if request.function_name in self.STOP_TOOLS:
            await self.execute_stop(request)
            return

        if self.current_action_task is not None:
            result = ActionResult(
                action_id=request.action_id,
                action_type=request.function_name,
                status="failed",
                target=request.arguments.get("target"),
                outcome="rejected",
                reason_code="ACTION_START_BUSY",
                retryable=True,
                data={
                    "active_action": self.current_action.function_name,
                    "active_action_id": self.current_action.action_id,
                },
            )
            await self.response_manager.send_function_output(
                request.call_id,
                self._action_envelope("command_result", result),
            )
            await self.response_manager.create_voice_response()
            return

        self.current_action = request
        self.current_action_task = asyncio.create_task(
            self.execute_request(request),
            name=f"cognition-start-{request.function_name}",
        )

        def action_executed(task):
            try:
                event = ActionExecuted(request, result=task.result())
            except BaseException as error:
                event = ActionExecuted(request, error=error)
            self.events.put_nowait(event)

        self.current_action_task.add_done_callback(action_executed)

    async def execute_request(self, request):
        name = request.function_name
        args = request.arguments

        camera_tools = {
            "find_target", "follow_person", "see_action",
            "watch_target", "explicit_navigation",
        }
        if name in camera_tools:
            camera_owner = self._camera_owner()
            if camera_owner == "target_tracking":
                can_reuse_tracking = (
                    name == "follow_person"
                    and self._passive_person_tracking_active()
                )
                if can_reuse_tracking:
                    camera_owner = None
                else:
                    await self.track_action.stop_tracking(
                        reason_code="REPLACED",
                        status="cancelled",
                        outcome=f"replaced_by_{name}",
                        reset_camera=name == "explicit_navigation",
                    )
                    camera_owner = self._camera_owner()
            if camera_owner is not None:
                return ActionResult(
                    action_id=request.action_id,
                    action_type=name,
                    status="failed",
                    target=args.get("target") or args.get("region"),
                    outcome="precondition_failed",
                    reason_code="CAMERA_IN_USE",
                    retryable=True,
                    data={"camera_owner": camera_owner},
                )

        if name in {
            "explicit_navigation", "watch_target", "find_target",
            "follow_person",
        }:
            active_actions = self._decision_state()["active_actions"]
            if name == "follow_person" and self._passive_person_tracking_active():
                active_actions = [
                    action for action in active_actions
                    if not (
                        action.get("action_type") == "track_action"
                    )
                ]
            if active_actions:
                return ActionResult(
                    request.action_id, name, "failed",
                    reason_code="ROBOT_BUSY", retryable=True,
                )

        if name == "explicit_navigation":
            if self.explicit_navigation_action is None:
                return ActionResult(
                    request.action_id, name, "failed",
                    reason_code="NAVIGATION_UNAVAILABLE",
                )
            local_command = args.get("local_command")
            if local_command == "go_to_room":
                return await self.explicit_navigation_action.start(
                    local_command, request.action_id,
                    room_id=args.get("room_id"),
                )
            return await self.explicit_navigation_action.start(
                local_command, request.action_id
            )

        if name == "watch_target":
            return await self.find_target_executor.start(
                target=args.get("target"),
                action_id=request.action_id,
                target_kind=args.get("target_kind"),
                approach=False,
                continuous_person_reacquisition=(
                    args.get("target_kind") == "person"
                ),
                action_type="watch_target",
            )

        if name == "find_target":
            if args.get("target_kind") not in {"object", "place"}:
                return ActionResult(
                    request.action_id, name, "failed",
                    target=args.get("target"), outcome="invalid_request",
                    reason_code="PERSON_TARGET_REQUIRES_PERSON_ACTION",
                )
            return await self.find_target_executor.start(
                target=args.get("target"),
                action_id=request.action_id,
                target_kind=args.get("target_kind"),
            )

        if name == "follow_person":
            return await self.follow_person_executor.start(
                target=args.get("target"),
                action_id=request.action_id,
            )

        if name == "see_action":
            return await self.see_action.see(
                query=args.get("query"),
                region=args.get("region"),
                action_id=request.action_id,
            )

        return ActionResult(
            action_id=request.action_id,
            action_type=name,
            status="failed",
            outcome="unsupported_action",
            reason_code="UNKNOWN_ACTION",
        )

    async def execute_stop(self, request):
        if request.function_name == "stop_watching_target":
            camera_owner = self._camera_owner()
            if camera_owner not in {None, "target_tracking", "watch_target"}:
                result = ActionResult(
                    action_id=request.action_id,
                    action_type=request.function_name,
                    status="failed",
                    outcome="precondition_failed",
                    reason_code="CAMERA_IN_USE",
                    retryable=False,
                    data={"camera_owner": camera_owner},
                )
                await self.response_manager.send_function_output(
                    request.call_id,
                    self._action_envelope("command_result", result),
                )
                await self.response_manager.create_voice_response()
                return

        action, stop_method_name = {
            "stop_goal": (self.find_target_executor, "stop"),
            "stop_navigation": (self.explicit_navigation_action, "stop"),
            "stop_follow": (self.follow_person_executor, "stop"),
            "stop_watching_target": (self.track_action, "stop_tracking"),
        }[request.function_name]
        if (
            request.function_name == "stop_watching_target"
            and self.find_target_executor.active
            and self.find_target_executor.action_type == "watch_target"
        ):
            action = self.find_target_executor
            stop_method_name = "stop"
        if (
            request.function_name == "stop_navigation"
            and self.hand_guided_navigation is not None
            and self.hand_guided_navigation.active
        ):
            action = self.hand_guided_navigation
        was_active = bool(getattr(action, "active", False))
        if was_active:
            stop_result = await getattr(action, stop_method_name)()
            if isinstance(stop_result, ActionResult) and stop_result.status == "failed":
                await self.response_manager.send_function_output(
                    request.call_id, self._action_envelope("command_result", stop_result)
                )
                await self.response_manager.create_voice_response()
                return

        await self.response_manager.send_function_output(
            request.call_id,
            self._action_envelope(
                "command_result",
                {
                    "command": request.function_name,
                    "status": "completed" if was_active else "skipped",
                    "outcome": "stopped" if was_active else "not_active",
                },
            ),
        )
        if request.function_name == "stop_watching_target" or not was_active:
            await self.response_manager.create_voice_response()

    async def oob_assessment(self, request):
        if request.function_name == "select_navigation_pose":
            if self.map_navigation_action is not None:
                await self.map_navigation_action.select_navigation_pose(
                    request.arguments, request.response_metadata
                )
            return
        if request.function_name == "assess_frame_search":
            assessment = self.search_action.assess_frame_search(
                request.arguments,
                request.response_metadata,
            )
        elif request.function_name == "assess_batch_search":
            assessment = self.search_action.assess_batch_search(
                request.arguments,
                request.response_metadata,
            )
        asyncio.create_task(assessment, name=request.function_name)

    async def handle_action_executed(self, event):
        if event.request is not self.current_action:
            return

        self.current_action_task = None
        self.current_action = None
        result = event.result
        if event.error is not None:
            result = ActionResult(
                action_id=event.request.action_id,
                action_type=event.request.function_name,
                status="failed",
                target=event.request.arguments.get("target"),
                outcome="execution_error",
                reason_code="ACTION_EXECUTION_ERROR",
                retryable=True,
                data={
                    "error": (
                        f"{type(event.error).__name__}: {event.error}"
                    )
                },
            )
        self.action_results.append(result)

        await self.response_manager.send_function_output(
            event.request.call_id,
            self._action_envelope("command_result", result),
        )

        print(f"\n🌸 [ACTION RESULT]: {result}\n")

        if result.status == "running":
            action = {
                "explicit_navigation": self.explicit_navigation_action,
                "watch_target": self.find_target_executor,
                "find_target": self.find_target_executor,
                "follow_person": self.follow_person_executor,
            }.get(event.request.function_name)

            # Capture this run's future before another navigation can start.
            completion = (
                action.completion_future
                if event.request.function_name in {
                    "explicit_navigation",
                }
                else None
            )

            async def wait_for_completion(action):
                try:
                    result = (await asyncio.shield(completion) if completion is not None
                              else await action.wait_until_finished())
                    lifecycle_event = ActionLifecycleFinished(
                        event.request,
                        result=result,
                    )
                except BaseException as error:
                    lifecycle_event = ActionLifecycleFinished(
                        event.request,
                        error=error,
                    )
                await self.events.put(lifecycle_event)

            if action is not None:
                asyncio.create_task(
                    wait_for_completion(action),
                    name=f"cognition-finish-{event.request.action_id}",
                )
        else:
            await self.response_manager.create_voice_response()

    async def handle_lifecycle_finished(self, event):
        result = event.result
        if event.error is not None:
            result = ActionResult(
                action_id=event.request.action_id,
                action_type=event.request.function_name,
                status="failed",
                target=event.request.arguments.get("target"),
                outcome="lifecycle_error",
                reason_code="ACTION_LIFECYCLE_ERROR",
                retryable=True,
                data={
                    "error": (
                        f"{type(event.error).__name__}: {event.error}"
                    )
                },
            )
        self.action_results.append(result)

        summary = (
            "🌳 [ACTION LIFECYCLE FINISHED] "
            + json.dumps(
                self._action_envelope(
                    "action_finished",
                    result,
                ),
                separators=(",", ":"),
            )
        )
        print(f"\n{summary}\n")
        await self.response_manager.send_system_context(summary)
        await self.response_manager.create_voice_response()

    def _action_envelope(
        self, event_type, result, decision_instruction=None
    ):
        envelope = {
            "event_type": event_type,
            "result": self._compact_result(result),
            "decision_instruction": (
                decision_instruction
                if decision_instruction is not None
                else self._decision_instruction(result)
            ),
        }
        decision_state = self._decision_state()
        if (
            decision_state["active_actions"]
            or decision_state["autonomy_active"]
        ):
            envelope["decision_state"] = decision_state
        return envelope

    @classmethod
    def _compact_result(cls, result):
        """Keep model context actionable; retain full results only internally."""
        if not isinstance(result, ActionResult):
            return cls._serialize(result)

        compact = {
            "action_type": result.action_type,
            "status": result.status,
        }
        for key, value in (
            ("target", result.target),
            ("reason_code", result.reason_code),
        ):
            if value is not None:
                compact[key] = value
        if result.outcome is not None and (
            result.status == "succeeded" or result.reason_code is None
        ):
            compact["outcome"] = result.outcome
        if result.retryable:
            compact["retryable"] = True

        data = result.data if isinstance(result.data, dict) else {}
        for key in ("goal", "direction", "failed_step", "camera_owner"):
            value = data.get(key)
            if value is not None:
                compact[key] = value

        step_result = data.get("step_result")
        if isinstance(step_result, dict):
            step_data = step_result.get("data")
            if isinstance(step_data, dict):
                if step_data.get("message"):
                    compact["message"] = step_data["message"]
                if step_data.get("error"):
                    compact["error"] = step_data["error"]
                if step_data.get("error_type"):
                    compact["error_type"] = step_data["error_type"]
                if step_data.get("error_stage"):
                    compact["error_stage"] = step_data["error_stage"]
            step_reason = step_result.get("reason_code")
            if step_reason and step_reason != result.reason_code:
                compact["step_reason_code"] = step_reason
        elif data.get("message"):
            compact["message"] = data["message"]
        elif data.get("error"):
            compact["error"] = data["error"]

        guidance = data.get("continuation_guidance")
        if isinstance(guidance, dict):
            retry_answers = guidance.get("retry_answers")
            compact_guidance = {}
            if guidance.get("unresolved_action_type"):
                compact_guidance["action"] = guidance["unresolved_action_type"]
            if isinstance(retry_answers, dict):
                compact_guidance["retry"] = retry_answers
            if guidance.get("out_of_frame_requires_no_retry"):
                compact_guidance["out_of_frame_stops"] = True
            if compact_guidance:
                compact["guidance"] = compact_guidance

        if "verified" in data:
            compact["verified"] = bool(data["verified"])
        return compact

    def _decision_instruction(self, result):
        if (
            isinstance(result, ActionResult)
            and result.reason_code in {
                "APPROACH_VERIFICATION_REQUIRED",
                "APPROACH_VERIFICATION_TIMEOUT",
                "APPROACH_RANGE_INVALID",
                "STABLE_SEED_TIMEOUT",
                "TARGET_NOT_TRACKED_AFTER_NAVIGATION",
            }
        ):
            return (
                "Briefly report that approach was not safely confirmed. Ask whether "
                "to continue carefully or reposition the target; retry only if asked."
            )
        if (
            isinstance(result, ActionResult)
            and result.reason_code
            in {"PERSON_DETECTION_FAILED", "OBJECT_DETECTION_FAILED"}
            and result.data.get("user_confirms_visible") is True
        ):
            return (
                "Report that direct detection failed. Do not retry or move without "
                "new user guidance."
            )
        if (
            isinstance(result, ActionResult)
            and result.reason_code == "SEARCH_GUIDANCE_REQUIRED"
        ):
            action_type = result.data.get(
                "continuation_guidance", {}
            ).get("unresolved_action_type", result.action_type)
            target = result.target
            if result.data.get("direction") == "behind":
                choices = "high, low, elsewhere out of frame, or best effort"
                retry_rule = "Do not retry behind or elsewhere out of frame."
            else:
                choices = (
                    "high, low, behind DJ, elsewhere out of frame, or best effort"
                )
                retry_rule = "Do not retry if it is elsewhere out of frame."
            return (
                f"Ask one short question: is {target!r} {choices}? Retry "
                f"{action_type} using the matching guidance. {retry_rule}"
            )
        if (
            isinstance(result, ActionResult)
            and result.action_type == "watch_target"
        ):
            if result.status == "succeeded":
                return (
                    "The robot is now continuously watching the target. Confirm this "
                    "briefly and do not call another tool."
                )
            return "Report the watch-target failure briefly."
        if (
            isinstance(result, ActionResult)
            and result.action_type in {
                "navigate_action", "explicit_navigation",
            }
        ):
            if result.status == "succeeded":
                return (
                    "Navigation completed. Report completion briefly. Do not call "
                    "explicit_navigation "
                    "again unless the user gives a new movement goal."
                )
            return (
                "Report the navigation reason briefly. Do not substitute blind motion. Retry only "
                "if new sensor evidence or user guidance materially changes the situation."
            )
        if (
            isinstance(result, ActionResult)
            and result.action_type == "find_target"
        ):
            if result.status != "succeeded":
                return (
                    "The autonomous object/place search did not complete. "
                    "Report the reason and checked-space summary briefly. Do not retry "
                    "unless the user supplies new information or explicitly asks."
                )
            return (
                "The robot completed the requested find goal. An object was approached "
                "and reacquired, or a place was reached with its one-shot contextual move. "
                "Report completion briefly and do not call another tool."
            )
        if (
            isinstance(result, ActionResult)
            and result.action_type == "follow_person"
        ):
            if result.status == "cancelled":
                return "Briefly confirm that following has stopped."
            return (
                "Report why continuous following ended. Do not restart follow_person "
                "unless the user explicitly asks."
            )
        return (
            "Continue the unresolved goal if needed; do not repeat the finished action."
        )

    def _camera_owner(self):
        if self.follow_person_executor.active:
            return "follow_person"
        hand_navigation = getattr(self, "hand_guided_navigation", None)
        if hand_navigation is not None and hand_navigation.active:
            return "hand_guided_navigation"
        if self.find_target_executor.active:
            return self.find_target_executor.action_type
        if self.explicit_navigation_action is not None and self.explicit_navigation_action.active:
            return "explicit_navigation"
        if self.search_action.active:
            return "search_action"
        if self.track_action.active:
            return "target_tracking"
        if self.approach_action.active:
            return "approach_action"
        return None

    def _passive_person_tracking_active(self):
        return bool(
            self.track_action.active
            and normalize_human_target(self.track_action.target) is not None
        )

    def _decision_state(self):
        active_actions = []
        for action_type, action, active_attribute in (
            ("search_action", self.search_action, "active"),
            ("track_action", self.track_action, "active"),
            ("approach_action", self.approach_action, "active"),
            ("explicit_navigation", self.explicit_navigation_action, "active"),
            (
                "hand_guided_navigation",
                getattr(self, "hand_guided_navigation", None),
                "active",
            ),
            ("follow_person", self.follow_person_executor, "active"),
        ):
            if action is not None and getattr(action, active_attribute, False):
                active_actions.append({
                    "action_id": getattr(action, "action_id", None),
                    "action_type": action_type,
                    "target": getattr(action, "target", None),
                })

        if self.find_target_executor.active:
            active_actions.append({
                "action_id": self.find_target_executor.action_id,
                "action_type": self.find_target_executor.action_type,
                "target": self.find_target_executor.target,
            })

        state = {
            "active_actions": active_actions,
            "autonomy_active": (
                self._autonomy_task is not None
                and not self._autonomy_task.done()
            ),
        }

        return state

    async def _handle_world_state(self, payload):
        self.world_state = dict(payload)
        now = time.monotonic()

        if self._robot_action_in_progress():
            self._tof_history.clear()
            self._proximity_candidate_count = 0
            return

        proximity = self._observe_abrupt_approach(self.world_state)
        if (
            proximity is not None
            and now - self._last_proximity_reaction >= self.proximity_cooldown
        ):
            self._last_proximity_reaction = now
            self._replace_autonomy_task(
                self._run_proximity_retreat(),
                "proximity-retreat",
            )
            return

        if self._autonomy_task is not None and not self._autonomy_task.done():
            return

        if self._startup_person_acquisition_pending and self._acquire_person_after_startup:
            self._startup_person_acquisition_pending = False
            self._replace_autonomy_task(
                self._run_startup_person_acquisition(),
                "startup-person-acquisition",
            )

    def _robot_action_in_progress(self):
        active = self._decision_state()["active_actions"]
        only_tracking = (
            len(active) == 1 and active[0]["action_type"] == "track_action"
        )
        return self.current_action_task is not None or bool(active) and not only_tracking

    def _replace_autonomy_task(self, coroutine, name):
        old_task = self._autonomy_task
        if old_task is not None and not old_task.done():
            old_task.cancel()
        task = asyncio.create_task(coroutine, name=name)
        self._autonomy_task = task

        def autonomy_done(done_task):
            if self._autonomy_task is done_task:
                self._autonomy_task = None
            if done_task.cancelled():
                return
            try:
                done_task.result()
            except Exception as error:
                print(
                    f"[Autonomy error: {done_task.get_name()}] "
                    f"{type(error).__name__}: {error}"
                )

        task.add_done_callback(autonomy_done)

    def _observe_abrupt_approach(self, current):
        distance = current.get("camera_tof_range")
        if (
            not isinstance(distance, (int, float))
            or not 0.01 <= float(distance) <= 1.0
        ):
            self._tof_history.clear()
            self._proximity_candidate_count = 0
            return None

        distance = float(distance)
        if len(self._tof_history) < 4:
            self._tof_history.append(distance)
            return None

        baseline = float(median(self._tof_history))
        drop = baseline - distance
        approaching = (
            (distance <= 0.5
            and drop >= 0.20)
            or distance <= 0.05
        )
        self._proximity_candidate_count = (
            self._proximity_candidate_count + 1 if approaching else 0
        )
        self._tof_history.append(distance)
        if self._proximity_candidate_count < 2:
            return None

        self._proximity_candidate_count = 0
        self._tof_history.clear()
        return True

    async def _run_proximity_retreat(self):
        if self._robot_action_in_progress():
            return

        move_result = await self.explicit_navigation_action.start(
            "nudge_backward",
            action_id=f"autonomy-proximity-{uuid.uuid4().hex}",
        )
        if move_result.status == "running":
            try:
                move_result = await asyncio.wait_for(
                    asyncio.shield(self.explicit_navigation_action.wait_until_finished()),
                    timeout=10.0,
                )
            except TimeoutError:
                move_result = await self.explicit_navigation_action.stop("AUTONOMY_TIMEOUT")
        self.action_results.append(move_result)

    async def _run_startup_person_acquisition(self):
        if self._robot_action_in_progress():
            self._startup_person_acquisition_pending = True
            return

        action_id = f"startup-person-{uuid.uuid4().hex}"
        try:
            result = await self.find_target_executor.start(
                target="person",
                action_id=action_id,
                target_kind="person",
                approach=False,
                continuous_person_reacquisition=True,
            )
            if result.status == "running":
                result = await asyncio.shield(
                    self.find_target_executor.wait_until_finished()
                )
            self.action_results.append(result)
        except asyncio.CancelledError:
            if (
                self.find_target_executor.active
                and self.find_target_executor.action_id == action_id
            ):
                await self.find_target_executor.stop("AUTONOMY_INTERRUPTED")
            raise
        except BaseException as error:
            print(
                "[Startup person acquisition error] "
                f"{type(error).__name__}: {error}"
            )

    @classmethod
    def _serialize(cls, value):
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {key: cls._serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._serialize(item) for item in value]
        if hasattr(value, "__dict__"):
            return {
                key: cls._serialize(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        return value
