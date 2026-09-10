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
        "assess_approach_verification",
    }
    STOP_TOOLS = {
        "stop_searching",
        "stop_tracking",
        "stop_approaching",
        "stop_moving",
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
        semantic_navigation_action,
        idle_prompt_seconds=60.0,
        boot_observation_delay=6.0,
        long_idle_person_seek_seconds=120.0,
        proximity_cooldown=8.0,
    ):
        self.approach_action = approach_action
        self.search_action = search_action
        self.see_action = see_action
        self.track_action = track_action
        self.move_action = move_action
        self.move_camera_action = move_camera_action
        self.semantic_navigation_action = semantic_navigation_action
        self.goal_executor = goal_executor
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
        self.approach_action.handle_navigation_event(message)
        self.semantic_navigation_action.handle_navigation_event(message)
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

        semantic_goals = {
            "find_target": "search",
            "watch_target": "track",
            "approach_target": "approach",
        }
        if name in semantic_goals:
            return await self.goal_executor.start(
                goal=semantic_goals[name],
                target=args.get("target"),
                direction=args.get("direction"),
                camera_region=args.get("camera_region"),
                user_confirms_visible=args.get(
                    "user_confirms_visible", False
                ),
                effort=args.get("effort", "center"),
                action_id=request.action_id,
            )

        if name == "navigate_semantic":
            if self.semantic_navigation_action is None:
                return ActionResult(
                    action_id=request.action_id,
                    action_type=name,
                    status="failed",
                    outcome="unsupported_action",
                    reason_code="SEMANTIC_NAVIGATION_UNAVAILABLE",
                )
            return await self.semantic_navigation_action.start(
                destination=args.get("destination"),
                object_a=args.get("object_a"),
                object_b=args.get("object_b"),
                action_id=request.action_id,
            )

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

        if name == "move_action":
            return await self.move_action.start_moving(
                action_id=request.action_id,
                **args,
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
        if self.goal_executor.active:
            await self.goal_executor.stop("USER_REQUESTED_IDLE")
            stopped.append("goal")
        if (
            self.semantic_navigation_action is not None
            and self.semantic_navigation_action.active
        ):
            await self.semantic_navigation_action.stop("USER_REQUESTED_IDLE")
            stopped.append("navigation")
        if self.approach_action.active:
            await self.approach_action.stop_approaching("USER_REQUESTED_IDLE")
            stopped.append("approach")
        if self.move_action.active:
            await self.move_action.stop_moving("USER_REQUESTED_IDLE")
            stopped.append("movement")
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
            "stop_moving": (
                self.move_action,
                "stop_moving",
            ),
            "stop_goal": (
                self.goal_executor,
                "stop",
            ),
            "stop_navigation": (
                self.semantic_navigation_action,
                "stop",
            ),
        }[request.function_name]
        was_active = bool(getattr(action, "active", False))
        if was_active:
            await getattr(action, stop_method_name)()

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
        elif request.function_name == "assess_approach_verification":
            assessment = self.approach_action.assess_approach_verification(
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
                "move_action": self.move_action,
                "navigate_semantic": self.semantic_navigation_action,
                "find_target": self.goal_executor,
                "watch_target": self.goal_executor,
                "approach_target": self.goal_executor,
            }.get(event.request.function_name)

            async def wait_for_completion(action):
                try:
                    lifecycle_event = ActionLifecycleFinished(
                        event.request,
                        result=await action.wait_until_finished(),
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

        if (
            isinstance(result, ActionResult)
            and result.action_type == "watch_target"
            and result.status == "succeeded"
            and self.track_action.active
        ):
            async def wait_for_tracking_end():
                try:
                    tracking_result = await self.track_action.wait_until_finished()
                    tracking_event = ActionLifecycleFinished(
                        event.request, result=tracking_result
                    )
                except BaseException as error:
                    tracking_event = ActionLifecycleFinished(
                        event.request, error=error
                    )
                await self.events.put(tracking_event)

            asyncio.create_task(
                wait_for_tracking_end(),
                name=f"watch-finish-{event.request.action_id}",
            )

        should_respond = not self._settled
        decision_instruction = None
        if autonomous_person_seek:
            person_found = (
                isinstance(result, ActionResult)
                and result.action_type == "watch_target"
                and result.status == "succeeded"
                and self.track_action.active
            )
            if person_found:
                decision_instruction = (
                    "DJ autonomously chose to look for a person and has now found "
                    "one and started tracking them. Greet the visible person once, "
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
        return {
            "event_type": event_type,
            "result": self._serialize(result),
            "decision_state": self._decision_state(),
            "decision_instruction": (
                decision_instruction
                if decision_instruction is not None
                else self._decision_instruction(result)
            ),
        }

    def _decision_instruction(self, result):
        if (
            isinstance(result, ActionResult)
            and result.action_type == "go_idle"
            and result.status == "succeeded"
        ):
            return (
                "Acknowledge briefly that DJ is settled, then remain silent and do "
                "not call another tool until the user speaks again."
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
                "The approach could not safely confirm or continue toward the "
                "target. Explain the visibility, range, or occlusion "
                "evidence briefly and ask whether DJ should continue carefully or "
                "whether the user can uncover or reposition the target. Do not claim "
                "success. Retry approach_target only if the user explicitly asks to "
                "continue."
            )
        if (
            isinstance(result, ActionResult)
            and result.reason_code
            in {"PERSON_DETECTION_FAILED", "OBJECT_DETECTION_FAILED"}
            and result.data.get("user_confirms_visible") is True
        ):
            return (
                "The user's visibility confirmation has already been honored "
                "with one direct detector attempt. Report that detection failed. "
                "Do not retry, search, rotate, or move unless the user gives new guidance."
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
                behind_rule = (
                    "The behind turn was already performed, so do not offer or "
                    "send direction=behind again."
                )
            else:
                choices = (
                    "high, low, behind DJ, elsewhere out of frame, or best effort"
                )
                behind_rule = (
                    f"If the user says behind, immediately retry {action_type} "
                    f"for {target!r} with direction=behind and effort=center."
                )
            return (
                f"The unresolved user goal remains {action_type} for {target!r}; "
                "the failed search was only its internal prerequisite. Ask one "
                f"concise question whether the target is {choices}. "
                f"{behind_rule} For high, low, or best effort, immediately retry "
                f"the same {action_type} goal with the corresponding effort. "
                "Preserve any compatible guidance the user combines in one answer. "
                "Do not switch to find_target unless that was the failed top-level "
                "goal, and do not ask for confirmation after a valid answer. If the "
                "target is elsewhere out of frame, do not retry."
            )
        if (
            isinstance(result, ActionResult)
            and result.action_type == "find_target"
            and result.status == "succeeded"
        ):
            return (
                "The target is now found and the location hint has already been "
                "consumed. Re-evaluate the unresolved user goal. If the user also "
                "asked DJ to watch or approach this same target, call the matching "
                "high-level tool without direction or camera_region so it reuses "
                "the current target view. Do not turn or aim toward the same hint "
                "again."
            )
        return (
            "Re-evaluate the unresolved user goal using this result. "
            "If you have not satisfied the user goal, you must continue choosing action based on action guides. "
            "Do not repeat the action you just finished"
        )

    def _decision_state(self):
        active_actions = []
        for action_type, action, active_attribute in (
            ("search_action", self.search_action, "active"),
            ("track_action", self.track_action, "active"),
            ("approach_action", self.approach_action, "active"),
            ("move_action", self.move_action, "active"),
            ("move_camera", self.move_camera_action, "active"),
            ("navigate_semantic", self.semantic_navigation_action, "active"),
        ):
            if getattr(action, active_attribute, False):
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
            function_name="watch_target",
            arguments={"target": "person", "effort": "best_effort"},
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
                goal="track",
                target="person",
                action_id=action_id,
                effort="best_effort",
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
