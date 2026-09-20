from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
import json
from statistics import median
import time
import uuid

from actions.action_result import ActionResult

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
        "stop_searching",
        "stop_tracking",
        "stop_approaching",
        "stop_goal",
        "stop_navigation",
    }

    def __init__(
        self,
        approach_action,
        search_action,
        see_action,
        track_action,
        move_action,
        move_camera_action,
        goal_executor,
        response_manager,
        astra_action=None,
        idle_prompt_seconds=6000.0,
        boot_observation_delay=3.0,
        long_idle_person_seek_seconds=12000.0,
        proximity_cooldown=8.0,
        navigate_action=None,
    ):
        self.approach_action = approach_action
        self.search_action = search_action
        self.see_action = see_action
        self.track_action = track_action
        self.move_action = move_action
        self.move_camera_action = move_camera_action
        self.navigate_action = navigate_action
        self.goal_executor = goal_executor
        self.astra_action = astra_action
        self.response_manager = response_manager

        self.events = asyncio.Queue()
        self.action_results = []
        self.current_action: ToolRequest | None = None
        self.current_action_task: asyncio.Task | None = None
        self._running = False

        self.world_state = {}
        self._pending_world_state: dict | None = None
        self._world_state_queued = False
        self._last_idle_wakeup = 0.0
        self._last_active_context = 0.0
        self._started_at = time.monotonic()
        self._last_activity_at = self._started_at
        self._last_user_activity_at = self._started_at
        self._last_person_seek_at = 0.0
        self._last_proximity_reaction = 0.0
        self._boot_observation_pending = True
        self._settled = False
        self._autonomy_task: asyncio.Task | None = None
        self._tof_history = deque(maxlen=5)
        self._proximity_candidate_count = 0

        self.idle_prompt_seconds = idle_prompt_seconds
        self.boot_observation_delay = boot_observation_delay
        self.long_idle_person_seek_seconds = long_idle_person_seek_seconds
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
            self._last_activity_at = time.monotonic()
            if function_name != "go_idle":
                self._settled = False
                self._cancel_passive_autonomy()

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
        if self.astra_action is not None:
            self.astra_action.handle_navigation_event(message)
        self.approach_action.handle_navigation_event(message)
        if self.navigate_action is not None:
            self.navigate_action.handle_navigation_event(message)
        self.move_action.handle_moving_event(message)

    async def publish_world_state(self, payload):
        self._pending_world_state = dict(payload)
        if self._world_state_queued:
            return
        self._world_state_queued = True
        await self.events.put(WorldState({}))

    def note_user_activity(self):
        now = time.monotonic()
        self._last_user_activity_at = now
        self._last_activity_at = now
        self._boot_observation_pending = False
        self._settled = False
        self._cancel_passive_autonomy()

    def _cancel_passive_autonomy(self):
        task = self._autonomy_task
        if (
            task is not None
            and not task.done()
            and task.get_name() != "proximity-reaction"
        ):
            task.cancel()

    async def shutdown(self):
        if self.goal_executor.active:
            await self.goal_executor.stop("SHUTDOWN")
        elif self.navigate_action is not None and self.navigate_action.active:
            await self.navigate_action.stop("SHUTDOWN")
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

        if name == "go_idle":
            return await self._enter_idle_mode(request.action_id)

        # Navigation owns motion and the camera throughout selection and travel.
        motion_tools = {"navigate_action", "move_camera", "find_object"}
        if (self.navigate_action is not None and self.navigate_action.active
                and name in motion_tools):
            return ActionResult(request.action_id, name, "failed",
                                reason_code="NAVIGATION_BUSY", retryable=True)

        if name == "navigate_action":
            if self.navigate_action is None:
                return ActionResult(request.action_id, name, "failed",
                                    reason_code="NAVIGATION_UNAVAILABLE")
            if self._decision_state()["active_actions"]:
                return ActionResult(request.action_id, name, "failed",
                                    reason_code="ROBOT_BUSY", retryable=True)
            return await self.navigate_action.start(args.get("query"), request.action_id)

        if name == "find_object":
            return await self.goal_executor.start(
                goal="find_object",
                target=args.get("target"),
                action_id=request.action_id,
            )

        if name == "watch_target":
            return await self.track_action.start_tracking(
                target=args.get("target"),
                action_id=request.action_id)

        if name == "move_camera":
            camera_owner = None
            if self.goal_executor.active:
                camera_owner = self.goal_executor.action_type
            elif self.search_action.active:
                camera_owner = "search_action"
            elif self.track_action.active:
                camera_owner = "track_action"
            elif self.approach_action.active:
                camera_owner = "approach_action"
            elif self.astra_action is not None and self.astra_action.active:
                camera_owner = "astra_approach_action"

            if camera_owner is not None:
                return ActionResult(
                    action_id=request.action_id,
                    action_type="move_camera",
                    status="failed",
                    target=args.get("region"),
                    outcome="precondition_failed",
                    reason_code="CAMERA_IN_USE",
                    retryable=True,
                    data={"camera_owner": camera_owner},
                )

            return await self.move_camera_action.move_to_region(
                region=args.get("region"),
                action_id=request.action_id,
            )
        if name == "see_action":
            return await self.see_action.see(
                query=args.get("query"),
                action_id=request.action_id,
            )

        return ActionResult(
            action_id=request.action_id,
            action_type=name,
            status="failed",
            outcome="unsupported_action",
            reason_code="UNKNOWN_ACTION",
        )

    async def _enter_idle_mode(self, action_id):
        self._settled = True
        autonomy_task = self._autonomy_task
        if (
            autonomy_task is not None
            and autonomy_task is not asyncio.current_task()
            and not autonomy_task.done()
        ):
            autonomy_task.cancel()
            await asyncio.gather(autonomy_task, return_exceptions=True)

        stopped = []
        if self.navigate_action is not None and self.navigate_action.active:
            await self.navigate_action.stop("USER_REQUESTED_IDLE")
            stopped.append("visual_navigation")
        if self.goal_executor.active:
            await self.goal_executor.stop("USER_REQUESTED_IDLE")
            stopped.append("goal")
        if self.astra_action is not None and self.astra_action.approaching_active:
            await self.astra_action.stop_approaching("USER_REQUESTED_IDLE")
            stopped.append("astra_approach")
        if self.approach_action.active:
            await self.approach_action.stop_approaching("USER_REQUESTED_IDLE")
            stopped.append("approach")
        if self.move_action.active:
            await self.move_action.stop_moving("USER_REQUESTED_IDLE")
            stopped.append("movement")
        if self.astra_action is not None and self.astra_action.tracking_active:
            await self.astra_action.stop_tracking("USER_REQUESTED_IDLE")
            stopped.append("astra_tracking")
        if self.track_action.active:
            await self.track_action.stop_tracking("USER_REQUESTED_IDLE")
            stopped.append("tracking")

        camera_result = None
        if not self.move_camera_action.active:
            camera_result = await self.move_camera_action.move_to_region(
                region="center",
                action_id=f"{action_id}:settle_camera",
            )
        self._tof_history.clear()
        self._proximity_candidate_count = 0
        return ActionResult(
            action_id=action_id,
            action_type="go_idle",
            status="succeeded",
            outcome="settled",
            data={
                "settled": True,
                "stopped": stopped,
                "camera_centered": (
                    camera_result is not None
                    and camera_result.status == "succeeded"
                ),
            },
        )

    async def execute_stop(self, request):
        if (request.function_name == "stop_navigation"
                and self.navigate_action is not None and self.navigate_action.active):
            action, stop_method_name = self.navigate_action, "stop"
        elif (
            request.function_name == "stop_tracking"
            and self.astra_action is not None
            and self.astra_action.tracking_active
        ):
            action, stop_method_name = self.astra_action, "stop_tracking"
        elif (
            request.function_name == "stop_approaching"
            and self.astra_action is not None
            and self.astra_action.approaching_active
        ):
            action, stop_method_name = self.astra_action, "stop_approaching"
        else:
            action, stop_method_name = {
                "stop_searching": (
                    self.search_action,
                    "stop_searching",
                ),
                "stop_tracking": (
                    self.track_action,
                    "stop_tracking",
                ),
                "stop_approaching": (
                    self.approach_action,
                    "stop_approaching",
                ),
                "stop_goal": (
                    self.goal_executor,
                    "stop",
                ),
                "stop_navigation": (
                    self.navigate_action,
                    "stop",
                ),
            }[request.function_name]
        was_active = bool(getattr(action, "active", False))
        if was_active:
            stop_result = await getattr(action, stop_method_name)()
            if isinstance(stop_result, ActionResult) and stop_result.status == "failed":
                await self.response_manager.send_function_output(
                    request.call_id, self._action_envelope("command_result", stop_result)
                )
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
        if not was_active:
            await self.response_manager.create_voice_response()

    async def oob_assessment(self, request):
        if request.function_name == "select_navigation_pose":
            if self.navigate_action is not None:
                await self.navigate_action.select_navigation_pose(
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
        self._last_activity_at = time.monotonic()

        await self.response_manager.send_function_output(
            event.request.call_id,
            self._action_envelope("command_result", result),
        )

        print(f"\n🌸 [ACTION RESULT]: {result}\n")

        if result.status == "running":
            action = {
                "navigate_action": self.navigate_action,
                "find_object": self.goal_executor,
            }.get(event.request.function_name)

            # Capture this run's future before another navigation can start.
            completion = (action.completion_future
                          if action is self.navigate_action and action is not None else None)

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
            if not self._settled or result.action_type == "go_idle":
                await self.response_manager.create_voice_response()

    async def handle_lifecycle_finished(self, event):
        result = event.result
        autonomous_person_seek = (
            event.request.response_metadata.get("origin") == "autonomy"
            and event.request.response_metadata.get("kind") == "person_seek"
        )
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
        self._last_activity_at = time.monotonic()

        should_respond = not self._settled
        decision_instruction = None
        if autonomous_person_seek:
            person_found = (
                isinstance(result, ActionResult)
                and result.action_type == "find_object"
                and result.status == "succeeded"
            )
            if person_found:
                decision_instruction = (
                    "DJ autonomously chose to look for a person and has now found "
                    "one and approached them. Greet the visible person once, "
                    "briefly and naturally, in first person. Do not say the user "
                    "asked DJ to look, and do not call another tool."
                )
            else:
                should_respond = False
                decision_instruction = (
                    "This was an autonomous background person-seek lifecycle event. "
                    "No greeting is needed because a new person was not just found. "
                    "Remain silent and do not retry or ask for search guidance."
                )

        summary = (
            "🌳 [ACTION LIFECYCLE FINISHED] "
            + json.dumps(
                self._action_envelope(
                    "action_finished",
                    result,
                    decision_instruction=decision_instruction,
                ),
                separators=(",", ":"),
            )
        )
        print(f"\n{summary}\n")
        await self.response_manager.send_system_context(summary)
        if should_respond:
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
            or decision_state["settled"]
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
        for key in ("goal", "direction", "failed_step"):
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
            and result.action_type == "go_idle"
            and result.status == "succeeded"
        ):
            return (
                "Briefly acknowledge that DJ is settled, then remain silent."
            )
        if (
            isinstance(result, ActionResult)
            and result.reason_code in {
                "APPROACH_VERIFICATION_REQUIRED",
                "APPROACH_VERIFICATION_TIMEOUT",
                "APPROACH_RANGE_INVALID",
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
            and result.action_type == "navigate_action"
        ):
            if result.status == "succeeded":
                return (
                    "Navigation finished only after fresh map/camera assessment confirmed the "
                    "complete movement objective. Report the completed result briefly. Do not call "
                    "navigate_action again unless the user gives a new movement goal."
                )
            return (
                "Report the navigation reason briefly. Do not substitute blind motion. Retry only "
                "if new sensor evidence or user guidance materially changes the situation."
            )
        if (
            isinstance(result, ActionResult)
            and result.action_type == "find_object"
        ):
            if result.status != "succeeded":
                return (
                    "The autonomous object search ended without stable post-approach "
                    "reacquisition. "
                    "Report the reason and checked-space summary briefly. Do not retry "
                    "unless the user supplies new information or explicitly asks."
                )
            return (
                "The robot found and approached the requested object, then reacquired "
                "stable tracking at the destination. Report completion briefly and do "
                "not call another tool."
            )
        return (
            "Continue the unresolved goal if needed; do not repeat the finished action."
        )

    def _active_tracking_action(self):
        if self.astra_action is not None and self.astra_action.tracking_active:
            return self.astra_action
        if self.track_action.active:
            return self.track_action
        return None

    def _decision_state(self):
        active_actions = []
        for action_type, action, active_attribute in (
            ("search_action", self.search_action, "active"),
            ("track_action", self.track_action, "active"),
            ("approach_action", self.approach_action, "active"),
            ("astra_approach_action", self.astra_action, "active"),
            ("move_action_internal", self.move_action, "active"),
            ("move_camera", self.move_camera_action, "active"),
            ("navigate_action", self.navigate_action, "active"),
        ):
            if action is not None and getattr(action, active_attribute, False):
                active_actions.append({
                    "action_id": getattr(action, "action_id", None),
                    "action_type": action_type,
                    "target": getattr(action, "target", None),
                })

        if self.goal_executor.active:
            active_actions.append({
                "action_id": self.goal_executor.action_id,
                "action_type": self.goal_executor.action_type,
                "target": self.goal_executor.target,
            })

        state = {
            "active_actions": active_actions,
            "settled": self._settled,
            "autonomy_active": (
                self._autonomy_task is not None
                and not self._autonomy_task.done()
            ),
        }

        return state

    async def _handle_world_state(self, payload):
        previous = self.world_state
        self.world_state = dict(payload)
        now = time.monotonic()

        if self._settled or self._robot_action_in_progress():
            self._tof_history.clear()
            self._proximity_candidate_count = 0
            if not self._settled:
                self._last_activity_at = now
            return

        proximity = self._observe_abrupt_approach(previous, self.world_state)
        if (
            proximity is not None
            and now - self._last_proximity_reaction >= self.proximity_cooldown
        ):
            self._last_proximity_reaction = now
            self._last_activity_at = now
            self._replace_autonomy_task(
                self._run_proximity_reaction(proximity),
                "proximity-reaction",
            )
            return

        if self.response_manager.busy or (self._autonomy_task is not None and not self._autonomy_task.done()):
            return

        if (
            self._boot_observation_pending
            and now - self._started_at >= self.boot_observation_delay
        ):
            self._boot_observation_pending = False
            self._last_person_seek_at = now
            self._last_activity_at = now
            self._replace_autonomy_task(
                self._run_person_seek("boot_person_seek"),
                "boot-person-seek",
            )
            return

        if (
            now - self._last_user_activity_at >= self.long_idle_person_seek_seconds
            and now - self._last_person_seek_at >= self.long_idle_person_seek_seconds
        ):
            self._last_person_seek_at = now
            self._last_activity_at = now
            self._replace_autonomy_task(
                self._run_person_seek("long_idle_person_seek"),
                "idle-person-seek",
            )
            return

        latest_activity = max(
            self._last_activity_at,
            getattr(self.response_manager, "last_activity_at", 0.0),
        )
        if now - latest_activity >= self.idle_prompt_seconds:
            self._last_idle_wakeup = now
            self._last_activity_at = now
            self._replace_autonomy_task(
                self._run_visual_wakeup("idle_conversation"),
                "idle-conversation",
            )

    def _robot_action_in_progress(self):
        only_tracking_in_progress = len(self._decision_state()["active_actions"]) == 1 and self._decision_state()["active_actions"][0]["action_type"] == "track_action"
        action_in_progress = bool(self._decision_state()["active_actions"]) and not only_tracking_in_progress
        return (
            self.current_action_task is not None
            or action_in_progress
        )

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

    def _observe_abrupt_approach(self, previous, current):
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
        return {
            "previous_distance": baseline,
            "current_distance": distance,
            "drop": drop,
        }

    async def _run_proximity_reaction(self, proximity):
        if self._settled or self._robot_action_in_progress():
            return

        move_result = await self.move_action.start_moving(
            linear_velocity=-0.15,
            distance=0.20,
            angular_velocity=0.0,
            angle=0.0,
            action_id=f"autonomy-proximity-{uuid.uuid4().hex}",
        )
        vision_result = await self.see_action.see(
            query=(
                "[AUTONOMOUS PROXIMITY OBSERVATION] Something may have rapidly "
                f"approached DJ: ToF changed from {proximity['previous_distance']:.2f} "
                f"m to {proximity['current_distance']:.2f} m. A short safety backup "
                f"was requested with status {move_result.status}."
            ),
            action_id=f"autonomy-see-proximity-{uuid.uuid4().hex}",
            autonomous=True,
        )
        self.action_results.append(vision_result)
        if vision_result.status == "succeeded":
            await self.response_manager.create_voice_response(
                system_msg=(
                    "This perception and backup were initiated autonomously by DJ, "
                    "not requested by the user. Briefly react in first person based "
                    "on the supplied frame. Do not call a movement or vision tool; "
                    "both actions have already been handled."
                )
            )

        if move_result.status == "running":
            try:
                await asyncio.wait_for(
                    asyncio.shield(self.move_action.wait_until_finished()),
                    timeout=5.0,
                )
            except TimeoutError:
                if self.move_action.active:
                    await self.move_action.stop_moving("AUTONOMY_TIMEOUT")

    async def _run_visual_wakeup(self, kind):
        if self._settled or self._robot_action_in_progress():
            return
        instruction = (
            "DJ chose to observe the current scene after a quiet idle period."
        )
        vision_result = await self.see_action.see(
            query=instruction,
            action_id=f"autonomy-see-{kind}-{uuid.uuid4().hex}",
            autonomous=True,
        )
        self.action_results.append(vision_result)
        if vision_result.status != "succeeded":
            return

        response_instruction = (
            "DJ initiated the preceding visual observation autonomously after a "
            "quiet period; the user did not request it. Make one brief, natural, "
            "first-person scene-grounded comment or question. Do not call "
            "see_action again or imply that the user asked DJ to look."
        )
        await self.response_manager.create_voice_response(
            system_msg=response_instruction
        )

    async def _run_person_seek(self, trigger):
        if self._settled or self._robot_action_in_progress():
            return

        action_id = f"autonomy-watch-person-{uuid.uuid4().hex}"
        request = ToolRequest(
            function_name="find_object",
            arguments={"target": "person"},
            call_id="",
            action_id=action_id,
            response_metadata={
                "origin": "autonomy",
                "kind": "person_seek",
                "trigger": trigger,
            },
        )
        try:
            result = await self.goal_executor.start(
                goal="find_object",
                target="person",
                action_id=action_id,
            )
            if result.status == "running":
                result = await asyncio.shield(
                    self.goal_executor.wait_until_finished()
                )
            await self.events.put(ActionLifecycleFinished(request, result=result))
        except asyncio.CancelledError:
            if (
                self.goal_executor.active
                and self.goal_executor.action_id == action_id
            ):
                await self.goal_executor.stop("AUTONOMY_INTERRUPTED")
            raise
        except BaseException as error:
            await self.events.put(ActionLifecycleFinished(request, error=error))

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
