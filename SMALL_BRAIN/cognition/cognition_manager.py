from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
import json
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
    OOB_TOOLS = {"assess_frame_search", "assess_batch_search"}
    STOP_TOOLS = {
        "stop_searching",
        "stop_tracking",
        "stop_approaching",
        "stop_moving",
    }

    def __init__(
        self,
        approach_action,
        search_action,
        see_action,
        track_action,
        move_action,
        response_manager,
        idle_salience_threshold=0.75,
        idle_wakeup_cooldown=15.0,
        active_context_interval=3.0,
    ):
        self.approach_action = approach_action
        self.search_action = search_action
        self.see_action = see_action
        self.track_action = track_action
        self.move_action = move_action
        self.response_manager = response_manager

        self.events = asyncio.Queue()
        self.action_results = []
        self.current_action: ToolRequest | None = None
        self.current_action_task: asyncio.Task | None = None
        self._running = False

        self.world_state = {}
        self._last_idle_wakeup = 0.0
        self._last_active_context = 0.0
        self.idle_salience_threshold = idle_salience_threshold
        self.idle_wakeup_cooldown = idle_wakeup_cooldown
        self.active_context_interval = active_context_interval

    async def handle_tool_call(
        self,
        function_name,
        arguments,
        call_id,
        response_metadata=None,
    ):
        args = json.loads(arguments) if arguments else {}

        request = ToolRequest(
            function_name=function_name,
            arguments=args,
            call_id=call_id,
            action_id=uuid.uuid4().hex,
            response_metadata=response_metadata or {},
        )

        print(f"[{function_name} function is called]. Args: {args}")
        await self.events.put(request)

    async def handle_tool_msg(self, message):
        self.approach_action.handle_navigation_event(message)
        self.move_action.handle_moving_event(message)

    async def publish_world_state(self, payload):
        await self.events.put(WorldState(payload))

    async def shutdown(self):
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
                await self._handle_world_state(event.payload)
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

        if name == "see_action":
            return await self.see_action.see(
                query=args.get("query"),
                action_id=request.action_id,
            )

        if name == "search_action":
            return await self.search_action.start_searching(
                target=args.get("target"),
                action_id=request.action_id,
            )

        if name == "track_action":
            return await self.track_action.start_tracking(
                target=args.get("target"),
                action_id=request.action_id,
            )

        if name == "approach_action":
            return await self.approach_action.start_approaching(
                target=args.get("target"),
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

        asyncio.create_task(assessment, name=request.function_name)
        # These calls come from responses created with conversation="none".
        # Their arguments are the complete one-shot assessment, and their
        # call IDs cannot be answered in the default Realtime conversation.

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

        if isinstance(result, ActionResult) and result.status == "running":
            action = {
                "search_action": self.search_action,
                "track_action": self.track_action,
                "approach_action": self.approach_action,
                "move_action": self.move_action,
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
            "[ROBOT ACTION EVENT]\n"
            + json.dumps(
                self._action_envelope("action_finished", result),
                separators=(",", ":"),
            )
        )
        await self.response_manager.send_system_context(summary)
        await self.response_manager.create_voice_response()

    def _action_envelope(self, event_type, result):
        return {
            "schema": "robot.action_event.v1",
            "event_type": event_type,
            "result": self._serialize(result),
            "robot_state": self._decision_state(),
            "decision_instruction": (
                "Re-evaluate the current user goal using this result. "
                "Choose the next action only if useful; do not assume success "
                "from a running action."
            ),
        }

    def _decision_state(self):
        active_actions = []
        for action_type, action, active_attribute in (
            ("search_action", self.search_action, "active"),
            ("track_action", self.track_action, "active"),
            ("approach_action", self.approach_action, "active"),
            ("move_action", self.move_action, "active"),
        ):
            if getattr(action, active_attribute, False):
                active_actions.append({
                    "action_id": getattr(action, "action_id", None),
                    "action_type": action_type,
                    "target": getattr(action, "target", None),
                })

        track_active = bool(getattr(self.track_action, "active", False))
        tracking_stable = bool(getattr(self.track_action, "stable", False))
        tracked_target = getattr(self.track_action, "target", None)
        return {
            "active_actions": active_actions,
            "track_action_active": track_active,
            "tracking_stable": tracking_stable,
            "tracked_target": tracked_target,
            "camera_tof_range": self.world_state.get("camera_tof_range"),
            "servo_pan_angle": self.world_state.get("servo_pan_angle"),
            "servo_tilt_angle": self.world_state.get("servo_tilt_angle"),
            "constraints": {
                "can_approach": track_active and tracking_stable,
                "approach_requires_target": tracked_target,
            },
        }

    async def _handle_world_state(self, payload):
        previous = self.world_state
        self.world_state = dict(payload)
        salience, reasons = self._score_salience(previous, self.world_state)
        now = time.monotonic()

        if self._decision_state()["active_actions"]:
            interval_elapsed = now - self._last_active_context
            if salience >= 0.5 or interval_elapsed >= self.active_context_interval:
                self._last_active_context = now
                await self.response_manager.send_system_context(
                    self._state_summary(salience, reasons)
                )

            if salience >= self.idle_salience_threshold:
                asyncio.create_task(
                    self.response_manager.create_voice_response(),
                    name="active-state-wakeup",
                )
            return

        if (
            salience >= self.idle_salience_threshold
            and now - self._last_idle_wakeup >= self.idle_wakeup_cooldown
        ):
            self._last_idle_wakeup = now
            summary = self._state_summary(salience, reasons)
            await self.response_manager.send_system_context(summary)
            asyncio.create_task(
                self.response_manager.create_voice_response(
                    system_msg=(
                        "[SALIENT WORLD EVENT] Re-evaluate the current goal. "
                        "Speak or act only when this event requires intervention."
                    )
                ),
                name="salience-wakeup",
            )

    @staticmethod
    def _score_salience(previous, current):
        score = 0.0
        reasons = []

        distance = current.get("camera_tof_range")
        old_distance = previous.get("camera_tof_range")
        if isinstance(distance, (int, float)) and distance > 0.0:
            old_is_near = (
                isinstance(old_distance, (int, float))
                and 0.0 < old_distance <= 0.45
            )
            if distance <= 0.35 and not old_is_near:
                score = 1.0
                reasons.append(f"very close obstacle at {distance:.2f} m")
            elif distance <= 0.75 and not (
                isinstance(old_distance, (int, float))
                and 0.0 < old_distance <= 0.85
            ):
                score = max(score, 0.75)
                reasons.append(f"nearby obstacle at {distance:.2f} m")
            elif isinstance(old_distance, (int, float)) and old_distance > 0.0:
                if abs(distance - old_distance) >= 0.5:
                    score = max(score, 0.55)
                    reasons.append("large distance change")

        if (
            previous
            and current.get("track_action_active")
            != previous.get("track_action_active")
        ):
            score = max(score, 0.8)
            reasons.append(
                "track action active changed to "
                f"{current.get('track_action_active')}"
            )
        if previous and (
            current.get("tracking_stable") != previous.get("tracking_stable")
        ):
            score = max(score, 0.8)
            reasons.append(
                f"tracking stability changed to {current.get('tracking_stable')}"
            )
        return score, reasons

    def _state_summary(self, salience, reasons):
        state = {
            **self._decision_state(),
            "salience": round(salience, 2),
            "reasons": reasons,
        }
        return "[WORLD STATE] " + json.dumps(state, separators=(",", ":"))

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
