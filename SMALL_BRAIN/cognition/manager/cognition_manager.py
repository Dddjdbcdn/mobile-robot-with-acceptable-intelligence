from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
from statistics import median
import time
import uuid

from actions.action_result import ActionResult
from cognition.manager.action_reporting import action_envelope
from cognition.manager.action_state import ActionState

@dataclass(frozen=True, slots=True)
class ToolRequest:
    function_name: str
    arguments: dict
    call_id: str
    action_id: str
    response_metadata: dict

    @property
    def silent(self):
        return bool(self.response_metadata.get("silent"))


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
    HAND_APPROACH_STANDOFF_M = 0.05
    HAND_APPROACH_TIMEOUT = 90.0
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
        hand_gesture_interface=None,
    ):
        self.approach_action = approach_action
        self.search_action = search_action
        self.see_action = see_action
        self.track_action = track_action
        self.explicit_navigation_action = explicit_navigation_action
        self.map_navigation_action = map_navigation_action
        self.hand_gesture_interface = hand_gesture_interface
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
        self._proximity_check = False
        self._acquire_person_after_startup = False
        self._startup_person_acquisition_pending = True
        self._autonomy_task: asyncio.Task | None = None
        self._tof_history = deque(maxlen=5)
        self._proximity_candidate_count = 0
        self._hand_location = None
        self._hand_approach_started_at = None

        self.action_state = ActionState(
            search=self.search_action,
            tracking=self.track_action,
            approach=self.approach_action,
            explicit_navigation=self.explicit_navigation_action,
            find_target=self.find_target_executor,
            follow_person=self.follow_person_executor,
        )

        self.proximity_cooldown = proximity_cooldown
        if self.hand_gesture_interface is not None:
            self.hand_gesture_interface.set_dispatcher(
                self.handle_hand_command
            )

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
        if map_navigation is not None:
            map_navigation.handle_navigation_event(message)

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
        if self.hand_gesture_interface is not None:
            await self.hand_gesture_interface.stop("SHUTDOWN")
        if self.find_target_executor.active:
            await self.find_target_executor.stop("SHUTDOWN")
        elif self.explicit_navigation_action is not None and self.explicit_navigation_action.active:
            await self.explicit_navigation_action.stop("SHUTDOWN")
        autonomy_task = self._autonomy_task
        if autonomy_task is not None and not autonomy_task.done():
            autonomy_task.cancel()
            await asyncio.gather(autonomy_task, return_exceptions=True)
        await self.events.put(Shutdown())

    async def handle_hand_command(self, command, data):
        """Resolve gesture input against the same global actions used by tools."""

        if command == "observe_gesture":
            if self.action_state.person_tracking:
                seed = self.track_action.stable_seeds.get(
                    target="hand", session_id=self.track_action.action_id
                )
                if data.get("gesture") in {"welcome", "push"}:
                    palm_center = data.get("palm_center")
                    if palm_center is not None:
                        self.track_action.set_visual_target_override(*palm_center)
                    latest_seed = self.track_action.stable_seeds.get(
                        target="hand", session_id=self.track_action.action_id
                    )
                    if isinstance(latest_seed, dict):
                        seed = latest_seed
                if isinstance(seed, dict):
                    self._hand_location = dict(seed)

            approach_started_at = getattr(
                self, "_hand_approach_started_at", None
            )
            if (
                self.approach_action.active
                and approach_started_at is not None
                and time.monotonic() - approach_started_at
                > self.HAND_APPROACH_TIMEOUT
            ):
                await self.approach_action.stop_approaching(
                    "HAND_APPROACH_TIMEOUT"
                )
            if not self.approach_action.active:
                self._hand_approach_started_at = None
            return self.action_state.gesture_context

        arguments = dict(data)
        if command == "approach_target":
            location = self._hand_location
            if location is None:
                return False
            arguments.update({
                "location": dict(location),
                "standoff_m": self.HAND_APPROACH_STANDOFF_M,
            })

        supported_tools = (
            self.STOP_TOOLS
            | {
                "watch_target",
                "follow_person",
                "approach_target",
                "explicit_navigation",
                "navigation_sequence",
            }
        )
        if command not in supported_tools:
            raise ValueError(f"Unknown hand gesture tool {command!r}")

        self._hand_location = None
        self.track_action.clear_visual_target_override()

        request_id = uuid.uuid4().hex
        await self.events.put(ToolRequest(
            function_name=command,
            arguments=arguments,
            call_id=f"hand-gesture-{request_id}",
            action_id=f"hand-{request_id}",
            response_metadata={
                "source": "hand_gesture",
                "silent": True,
            },
        ))
        return True

    async def _run_navigation_sequence(self, commands, action_id):
        if self.explicit_navigation_action is None:
            return ActionResult(
                action_id, "navigation_sequence", "failed",
                reason_code="NAVIGATION_UNAVAILABLE",
            )
        try:
            for index, command in enumerate(commands):
                result = await self.execute_request(ToolRequest(
                    function_name="explicit_navigation",
                    arguments={"local_command": command, "room_id": None},
                    call_id="hand-gesture",
                    action_id=f"{action_id}:{index}:{command}",
                    response_metadata={},
                ))
                if getattr(result, "status", None) == "running":
                    result = await self.explicit_navigation_action.wait_until_finished()
                if getattr(result, "status", None) != "succeeded":
                    return result
            return ActionResult(
                action_id, "navigation_sequence", "succeeded",
                outcome="sequence_completed",
                data={"commands": list(commands)},
            )
        except asyncio.CancelledError:
            if self.explicit_navigation_action.active:
                await self.explicit_navigation_action.stop("COMMAND_CANCELLED")
            raise

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
            if not request.silent:
                await self.response_manager.send_function_output(
                    request.call_id,
                    action_envelope(
                        "command_result", result,
                        self.action_state.snapshot(self._autonomy_task),
                    ),
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
            "watch_target",
        }
        if name in camera_tools:
            camera_owner = self.action_state.camera_owner
            if camera_owner == "target_tracking":
                can_reuse_tracking = (
                    name == "follow_person"
                    and self.action_state.person_tracking
                )
                if can_reuse_tracking:
                    camera_owner = None
                else:
                    await self.track_action.stop_tracking(
                        reason_code="REPLACED",
                        status="cancelled",
                        outcome=f"replaced_by_{name}",
                        reset_camera=name in {
                            "explicit_navigation", "navigation_sequence",
                        },
                    )
                    camera_owner = self.action_state.camera_owner
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
            "follow_person", "navigation_sequence",
        }:
            active_actions = self.action_state.snapshot(
                self._autonomy_task
            )["active_actions"]
            can_run_with_tracking = (
                name in {
                    "explicit_navigation",
                    "navigation_sequence",
                    "follow_person",
                }
                and self.track_action.active
            )
            if can_run_with_tracking:
                active_actions = [
                    action for action in active_actions
                    if action.get("action_type") != "track_action"
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

        if name == "navigation_sequence":
            commands = args.get("commands")
            if not isinstance(commands, list) or not commands:
                return ActionResult(
                    request.action_id, name, "failed",
                    outcome="invalid_request",
                    reason_code="NAVIGATION_SEQUENCE_REQUIRED",
                )
            return await self._run_navigation_sequence(
                commands, request.action_id
            )

        if name == "approach_target":
            active_actions = [
                action
                for action in self.action_state.snapshot(
                    self._autonomy_task
                )["active_actions"]
                if action.get("action_type") != "track_action"
            ]
            if active_actions:
                return ActionResult(
                    request.action_id, name, "failed",
                    target=args.get("target"),
                    outcome="rejected",
                    reason_code="ROBOT_BUSY",
                    retryable=True,
                )
            result = await self.approach_action.start(
                location=args.get("location"),
                action_id=request.action_id,
                target=args.get("target"),
                standoff_m=args.get("standoff_m"),
            )
            if getattr(result, "status", None) in {
                "running", "already_running",
            } and args.get("target") == "hand":
                self._hand_approach_started_at = time.monotonic()
            return result

        if name == "watch_target":
            direct_tracking = await self.track_action.start_tracking(
                target=args.get("target"),
                action_id=request.action_id,
                allow_grounding_dino=False,
                continuous_person_reacquisition=(
                    args.get("target_kind") == "person"
                ),
            )
            if direct_tracking.status == "running":
                return self._as_watch_target_result(
                    direct_tracking,
                    direct_tracker=True,
                )
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

    @staticmethod
    def _as_watch_target_result(result, *, direct_tracker=False):
        """Keep direct tracking behind the public watch-target lifecycle."""
        if not isinstance(result, ActionResult):
            return result
        data = dict(result.data or {})
        if direct_tracker:
            data["watch_source"] = "direct_tracker"
        return ActionResult(
            action_id=result.action_id,
            action_type="watch_target",
            status=result.status,
            target=result.target,
            outcome=result.outcome,
            reason_code=result.reason_code,
            retryable=result.retryable,
            data=data,
        )

    async def execute_stop(self, request):
        if request.function_name == "stop_watching_target":
            camera_owner = self.action_state.camera_owner
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
                if not request.silent:
                    await self.response_manager.send_function_output(
                        request.call_id,
                        action_envelope(
                            "command_result", result,
                            self.action_state.snapshot(self._autonomy_task),
                        ),
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
            and self.approach_action.active
        ):
            action = self.approach_action
            stop_method_name = "stop_approaching"
        was_active = bool(getattr(action, "active", False))
        if was_active:
            reason = request.arguments.get("reason")
            stop_method = getattr(action, stop_method_name)
            stop_result = await (
                stop_method(reason) if reason is not None else stop_method()
            )
            if isinstance(stop_result, ActionResult) and stop_result.status == "failed":
                if not request.silent:
                    await self.response_manager.send_function_output(
                        request.call_id,
                        action_envelope(
                            "command_result", stop_result,
                            self.action_state.snapshot(self._autonomy_task),
                        ),
                    )
                    await self.response_manager.create_voice_response()
                return

        if not request.silent:
            await self.response_manager.send_function_output(
                request.call_id,
                action_envelope(
                    "command_result",
                    {
                        "command": request.function_name,
                        "status": "completed" if was_active else "skipped",
                        "outcome": "stopped" if was_active else "not_active",
                    },
                    self.action_state.snapshot(self._autonomy_task),
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

        if not event.request.silent:
            await self.response_manager.send_function_output(
                event.request.call_id,
                action_envelope(
                    "command_result", result,
                    self.action_state.snapshot(self._autonomy_task),
                ),
            )

        print(f"\n🌸 [ACTION RESULT]: {result}\n")

        if result.status == "running":
            direct_watch = (
                event.request.function_name == "watch_target"
                and isinstance(result, ActionResult)
                and result.data.get("watch_source") == "direct_tracker"
            )
            action = {
                "explicit_navigation": self.explicit_navigation_action,
                "watch_target": (
                    self.track_action
                    if direct_watch else self.find_target_executor
                ),
                "find_target": self.find_target_executor,
                "follow_person": self.follow_person_executor,
                "approach_target": self.approach_action,
            }.get(event.request.function_name)

            # Capture this run's future before another navigation can start.
            completion = (
                action.completion_future
                if event.request.function_name in {
                    "explicit_navigation", "approach_target",
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
        elif not event.request.silent:
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
        elif (
            event.request.function_name == "watch_target"
            and isinstance(result, ActionResult)
            and result.action_type == "track_action"
        ):
            result = self._as_watch_target_result(
                result,
                direct_tracker=True,
            )
        self.action_results.append(result)

        if not event.request.silent:
            summary = (
                "🌳 [ACTION LIFECYCLE FINISHED] "
                + json.dumps(
                    action_envelope(
                        "action_finished",
                        result,
                        self.action_state.snapshot(self._autonomy_task),
                    ),
                    separators=(",", ":"),
                )
            )
            print(f"\n{summary}\n")
            await self.response_manager.send_system_context(summary)
            await self.response_manager.create_voice_response()


    async def _handle_world_state(self, payload):
        self.world_state = dict(payload)
        now = time.monotonic()

        if self._robot_action_in_progress():
            self._tof_history.clear()
            self._proximity_candidate_count = 0
            return

        if self._proximity_check:
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
        active = self.action_state.snapshot(self._autonomy_task)["active_actions"]
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
