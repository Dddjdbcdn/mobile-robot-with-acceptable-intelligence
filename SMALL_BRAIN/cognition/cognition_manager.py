from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
import json
import time


@dataclass(slots=True)
class ToolRequest:
    function_name: str
    arguments: dict
    call_id: str
    response_metadata: dict
    accepted: asyncio.Future


@dataclass(slots=True)
class ActionFinished:
    request: ToolRequest
    result: object = None
    error: BaseException | None = None


@dataclass(slots=True)
class RosEvent:
    payload: dict


@dataclass(slots=True)
class WorldState:
    payload: dict


async def _ignore_context(_message):
    return None


class Shutdown:
    pass


class CognitionManager:
    """Continuous, event-driven owner of high-level robot actions."""

    OOB_TOOLS = {"assess_frame_search", "assess_batch_search"}
    STOP_TOOLS = {"stop_visual_search", "stop_object_tracking", "stop_motors"}

    def __init__(
        self,
        *,
        approach_action,
        search_action,
        see_action,
        track_action,
        send_tool_output,
        request_voice,
        send_robot_command,
        inform_llm=None,
        idle_salience_threshold=0.75,
        idle_wakeup_cooldown=15.0,
        active_context_interval=3.0,
    ):
        self.approach_action = approach_action
        self.search_action = search_action
        self.see_action = see_action
        self.track_action = track_action
        self.send_tool_output = send_tool_output
        self.request_voice = request_voice
        self.send_robot_command = send_robot_command
        self.inform_llm = inform_llm or _ignore_context
        self.idle_salience_threshold = idle_salience_threshold
        self.idle_wakeup_cooldown = idle_wakeup_cooldown
        self.active_context_interval = active_context_interval

        self.events = asyncio.Queue()
        self.pending_requests = deque()
        self.action_results = []
        self.current_action = None
        self.current_request = None
        self.world_state = {}
        self._last_idle_wakeup = 0.0
        self._last_active_context = 0.0
        self._action_task = None
        self._running = False

    async def handle_tool_call(
        self,
        ws=None,
        *,
        function_name,
        arguments,
        call_id,
        response_metadata=None,
    ):
        """Validate a Realtime call and enqueue it as cognition input."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            args = {} if args is None else args
            if not isinstance(args, dict):
                raise ValueError("tool arguments must be an object")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            await self.send_tool_output(
                call_id,
                {"status": "error", "message": f"invalid arguments: {error}"},
            )
            return

        accepted = asyncio.get_running_loop().create_future()
        await self.events.put(
            ToolRequest(
                function_name,
                args,
                call_id,
                response_metadata or {},
                accepted,
            )
        )
        await accepted

    async def publish_ros_event(self, payload):
        await self.events.put(RosEvent(payload))

    async def publish_world_state(self, payload):
        await self.events.put(WorldState(payload))

    async def cognition_loop(self):
        """Sleep on the event queue and react without polling."""
        self._running = True
        try:
            while self._running:
                event = await self.events.get()
                try:
                    if isinstance(event, ToolRequest):
                        await self._handle_tool_request(event)
                    elif isinstance(event, ActionFinished):
                        await self._handle_action_finished(event)
                    elif isinstance(event, RosEvent):
                        await self._handle_ros_event(event.payload)
                    elif isinstance(event, WorldState):
                        await self._handle_world_state(event.payload)
                    elif isinstance(event, Shutdown):
                        self._running = False
                finally:
                    self.events.task_done()
        finally:
            await self._cancel_current_action()

    async def shutdown(self):
        await self.events.put(Shutdown())

    async def _handle_tool_request(self, request):
        try:
            if request.function_name in self.OOB_TOOLS:
                await self._route_assessment(request)
            elif request.function_name in self.STOP_TOOLS:
                await self._execute_stop(request)
            elif self._action_task is not None:
                self.pending_requests.append(request)
                await self.send_tool_output(
                    request.call_id,
                    {
                        "status": "queued",
                        "position": len(self.pending_requests),
                        "active_action": self.current_action,
                    },
                )
            else:
                await self.send_tool_output(
                    request.call_id,
                    {"status": "accepted", "action": request.function_name},
                )
                self._start_request(request)
        except Exception as error:
            await self.send_tool_output(
                request.call_id,
                {"status": "error", "message": f"{type(error).__name__}: {error}"},
            )
        finally:
            if not request.accepted.done():
                request.accepted.set_result(None)

    def _start_request(self, request):
        self.current_request = request
        self.current_action = request.function_name
        self._action_task = asyncio.create_task(
            self._execute_request(request),
            name=f"cognition-{request.function_name}",
        )
        self._action_task.add_done_callback(
            lambda task, request=request: self._action_done(request, task)
        )

    async def _execute_request(self, request):
        name = request.function_name
        args = request.arguments

        if name in {"get_vision", "vision_reasoning"}:
            mode = "reasoning" if name == "vision_reasoning" else args.get("action")
            if mode == "reasoning":
                return await self.see_action.start_seeing(args.get("query"))
            if mode == "find_object":
                return await self.search_action.start_visual_search(args.get("target"))
            if mode == "look_at_object":
                return await self._start_tracking(args.get("target"))
            raise ValueError(f"unsupported vision action: {mode}")

        if name == "find_and_approach":
            return await self._find_and_approach(args.get("target"))
        if name == "look_at":
            return await self._start_tracking(args.get("target"))
        if name == "approach_object":
            return await self.approach_action.start_approaching()
        if name in {"blind_move", "navigate_to_pose"}:
            if self.track_action.tracking:
                await self.track_action.stop_tracking("motion_command_started")
            return await self.send_robot_command({"command": name, **args})
        if name == "capture_depth":
            raise NotImplementedError("capture_depth has no action adapter")
        raise ValueError(f"unknown function: {name}")

    async def _start_tracking(self, target):
        if not target:
            raise ValueError("tracking target is required")
        camera = self.see_action.camera
        jpeg_bytes = await asyncio.to_thread(
            camera.jpeg_bytes_snapshot,
            70,
            False,
            "results/search_results/tracking_input.jpg",
        )
        return await self.track_action.start_tracking(
            jpeg_bytes,
            target,
            camera.snapshot(),
        )

    async def _find_and_approach(self, target):
        if not target:
            raise ValueError("search target is required")
        result = await self.search_action.start_visual_search(target)
        if not self._succeeded(result) or (
            isinstance(result.msg, dict) and result.msg.get("result") != "found"
        ):
            return result
        result = await self._start_tracking(target)
        if not self._succeeded(result):
            return result
        if not await self.track_action.wait_until_stable():
            await self.track_action.stop_tracking("stability_timeout")
            raise TimeoutError("tracker did not become stable")
        return await self.approach_action.start_approaching()

    async def _route_assessment(self, request):
        if request.function_name == "assess_frame_search":
            assessment = self.search_action.assess_frame_search(
                request.arguments,
                request.response_metadata,
            )
        else:
            assessment = self.search_action.assess_batch_search(
                request.arguments,
                request.response_metadata,
            )
        task = asyncio.create_task(assessment, name=request.function_name)
        task.add_done_callback(self._auxiliary_done)
        await self.send_tool_output(request.call_id, {"status": "received"})

    def _auxiliary_done(self, task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self.action_results.append({
                "status": "failure",
                "message": f"{type(error).__name__}: {error}",
            })

    async def _handle_world_state(self, payload):
        previous = self.world_state
        self.world_state = dict(payload)
        salience, reasons = self._score_salience(previous, self.world_state)
        now = time.monotonic()

        if self._action_task is not None:
            interval_elapsed = now - self._last_active_context
            if salience >= 0.5 or interval_elapsed >= self.active_context_interval:
                self._last_active_context = now
                await self.inform_llm(self._state_summary(salience, reasons))
            return

        if (
            salience >= self.idle_salience_threshold
            and now - self._last_idle_wakeup >= self.idle_wakeup_cooldown
        ):
            self._last_idle_wakeup = now
            summary = self._state_summary(salience, reasons)
            await self.inform_llm(summary)
            task = asyncio.create_task(
                self.request_voice(
                    system_msg=(
                        f"[SALIENT WORLD EVENT]: {summary}. "
                        "Decide whether to speak or take an appropriate action. "
                        "Do nothing if intervention is unnecessary."
                    )
                ),
                name="salience-wakeup",
            )
            task.add_done_callback(self._auxiliary_done)

    async def _handle_ros_event(self, payload):
        self.action_results.append(payload)
        summary = f"[ROS EVENT]: {json.dumps(payload)}"
        await self.inform_llm(summary)
        if self._action_task is None:
            task = asyncio.create_task(
                self.request_voice(system_msg=summary),
                name="ros-event-wakeup",
            )
            task.add_done_callback(self._auxiliary_done)

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

        if previous and current.get("tracking") != previous.get("tracking"):
            score = max(score, 0.8)
            reasons.append(f"tracking changed to {current.get('tracking')}")
        if previous and (
            current.get("tracking_stable") != previous.get("tracking_stable")
        ):
            score = max(score, 0.65)
            reasons.append(
                f"tracking stability changed to {current.get('tracking_stable')}"
            )
        return score, reasons

    def _state_summary(self, salience, reasons):
        state = {
            "active_action": self.current_action,
            "salience": round(salience, 2),
            "reasons": reasons,
            "camera_tof_range": self.world_state.get("camera_tof_range"),
            "tracking": self.world_state.get("tracking"),
            "tracking_stable": self.world_state.get("tracking_stable"),
            "servo_pan_angle": self.world_state.get("servo_pan_angle"),
            "servo_tilt_angle": self.world_state.get("servo_tilt_angle"),
        }
        return f"[WORLD STATE]: {json.dumps(state)}"

    async def _execute_stop(self, request):
        if request.function_name == "stop_visual_search":
            result = await self.search_action.stop_visual_search()
        elif request.function_name == "stop_object_tracking":
            result = await self.track_action.stop_tracking("user_requested")
        else:
            await self._cancel_current_action()
            if self.track_action.tracking:
                await self.track_action.stop_tracking("emergency_stop")
            result = await self.send_robot_command({"command": "stop_motors"})
        await self.send_tool_output(request.call_id, self._serialize(result))

    async def _handle_action_finished(self, event):
        if event.request is not self.current_request:
            return
        self._action_task = None
        self.current_request = None
        self.current_action = None

        if event.error is None:
            result = event.result
        elif isinstance(event.error, asyncio.CancelledError):
            result = {"status": "cancelled"}
        else:
            result = {
                "status": "failure",
                "message": f"{type(event.error).__name__}: {event.error}",
            }
        self.action_results.append(result)
        await self.request_voice(
            system_msg=f"[ACTION RESULT]: {json.dumps(self._serialize(result))}"
        )
        if self.pending_requests:
            self._start_request(self.pending_requests.popleft())

    async def _cancel_current_action(self):
        if self._action_task is None:
            return
        self._action_task.cancel()
        try:
            await self._action_task
        except asyncio.CancelledError:
            pass
        self._action_task = None
        self.current_request = None
        self.current_action = None

    def _action_done(self, request, task):
        try:
            event = ActionFinished(request, result=task.result())
        except BaseException as error:
            event = ActionFinished(request, error=error)
        self.events.put_nowait(event)

    @staticmethod
    def _succeeded(result):
        if isinstance(result, dict):
            return result.get("status") in {"success", "accepted", "succeeded"}
        return getattr(result, "status", None) == "success"

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
