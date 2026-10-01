import asyncio
import base64
import json
import os
from pathlib import Path
import queue
import sys
import threading

import cv2
import pyaudio
import websockets
import zmq
import zmq.asyncio
from typing import Any

from utilities.camera_sampler import display_camera_loop

from actions.approach_action import ApproachAction
from actions.search_action import SearchAction
from actions.see_action import SeeAction
from actions.track_action import TrackAction
from actions.explicit_navigation_action import ExplicitNavigationAction
from actions.map_navigation_action import MapNavigationAction

from services.audio_stream import AudioApp, send_mic_audio
from services.camera_stream import CameraStream
from services.map_client import MapClient
from services.response_manager import ResponseManager

from services.groundingdino_service import GroundingDINOService
from services.csrt_tracker import CSRTTrackingManager
from services.depthanything_service import DepthAnythingService # unused
from services.hand_landmark_service import HandLandmarkService
from services.sam2_service import SAM2OpenVINOService # unused
from services.yolo_service import YoloService

from cognition.manager.cognition_manager import CognitionManager
from cognition.sequence.follow_person_executor import FollowPersonExecutor
from cognition.sequence.find_target_executor import FindTargetExecutor
from cognition.hand.interface import HandGestureInterface
from cognition.manager.world_state import update_state

context = zmq.asyncio.Context()

zmq_req_socket = context.socket(zmq.REQ)
zmq_req_socket.connect("tcp://localhost:5555")

zmq_sub_socket = context.socket(zmq.SUB)
zmq_sub_socket.connect("tcp://localhost:5556")
zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

zmq_pub_socket = context.socket(zmq.PUB)
zmq_pub_socket.connect("tcp://localhost:5557")

person_tracker_pub_socket = context.socket(zmq.PUB)
person_tracker_pub_socket.setsockopt(zmq.SNDHWM, 10)
person_tracker_pub_socket.connect(
    os.environ.get("PERSON_TRACKER_ENDPOINT", "tcp://localhost:5560")
)


zmq_req_lock = asyncio.Lock()

SMALL_BRAIN_ROOT = Path(__file__).resolve().parent
IDENTITY_PATH = SMALL_BRAIN_ROOT / "database" / "identity.json"
MEMORY_PATH = SMALL_BRAIN_ROOT / "database" / "memory.json"
TOOL_ROUTING_PATH = SMALL_BRAIN_ROOT / "tools" / "tool_routing.json"
TOOLS_PATHS = [
    SMALL_BRAIN_ROOT / "tools" / "action_tools.json",
    SMALL_BRAIN_ROOT / "tools" / "stop_tools.json",
]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

MODEL = "gpt-realtime-2.1-mini"
URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
DEBUG_MODE = False

GROUNDING_DINO_REPO = Path("vision_models/groundingdino_tools/GroundingDINO")
GROUNDING_DINO_CONFIG = Path("vision_models/groundingdino_tools/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_MODEL = Path("vision_models/groundingdino_tools/models/groundingdino_swint_512x768_onnx.xml")

tool_tasks: set[asyncio.Task[Any]] = set()

def load_json(path):
    with Path(path).open(encoding="utf-8") as json_file:
        return json.load(json_file)

def handle_task_done(task):
    tool_tasks.discard(task)

    if task.cancelled(): return

    try: task.result()
    except Exception as error:
        print(
            f"[Background task error: {task.get_name()}] "
            f"{type(error).__name__}: {error}"
        )

def create_tool_task(coroutine,name) :
    task = asyncio.create_task(coroutine, name=name)
    tool_tasks.add(task)
    task.add_done_callback(handle_task_done)
    return task

async def background_status_monitor(cognitive_manager):
    print("[System: Background Monitor Listening for ROS 2 feedback...]")
    while True:
        try:
            message = await zmq_sub_socket.recv_json()
            if message.get("type") == "event":
                await cognitive_manager.handle_tool_msg(message)

            elif message.get("type") == "state":
                update_state(message)
                active_tracker = cognitive_manager.track_action
                stable_seed = (
                    active_tracker.stable_seeds.get(
                        target=active_tracker.target,
                        session_id=active_tracker.action_id,
                    )
                    if active_tracker is not None else None
                )
                await cognitive_manager.publish_world_state({
                    **message,
                    "track_action_active": active_tracker is not None,
                    "stable_seed_ready": stable_seed is not None,
                    "tracked_target": (
                        active_tracker.target if active_tracker is not None else None
                    ),
                })

        except Exception as e:
            print(f"[System Error in Monitor]: {e}")
            await asyncio.sleep(1)

async def _read_stdin_line(prompt):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    stdin_fd = sys.stdin.fileno()

    def stdin_ready():
        loop.remove_reader(stdin_fd)
        try:
            line = sys.stdin.readline()
        except BaseException as error:
            if not future.done():
                future.set_exception(error)
            return
        if not future.done():
            future.set_result(line)

    print(prompt, end="", flush=True)
    loop.add_reader(stdin_fd, stdin_ready)
    try:
        return await future
    finally:
        loop.remove_reader(stdin_fd)


async def send_typed_messages(response_manager, cognitive_manager=None):
    print("[System: Typed chat ready. Type a message and press Enter.]")

    while True:
        message = await _read_stdin_line("\nYou: ")
        if message == "":
            return
        if not message.strip():
            continue

        if cognitive_manager is not None:
            cognitive_manager.note_user_activity()

        await response_manager.send_user_text(message.rstrip("\n"))

async def receive_events(ws,app,response_manager,camera,cognitive_manager):
    human_speaking = False

    async for message in ws:
        event = json.loads(message)
        event_type = event.get("type")

        if event_type == "error":
            error = event.get("error", {})

            print(
                "\n\n❌ OPENAI ERROR"
                f"\ntype: {error.get('type')}"
                f"\ncode: {error.get('code')}"
                f"\nmessage: {error.get('message')}"
                f"\nparam: {error.get('param')}"
                f"\nevent_id: {error.get('event_id')}"
                "\n"
            )

        elif event_type == "response.created":
            response = event.get("response", {})
            response_manager.handle_response_created(response)


        elif event_type == "response.done":
            response = event.get("response", {})

            response_manager.handle_response_done(response)

            usage = response.get("usage") or {}
            metadata = response.get("metadata") or {}

            if response.get("status") != "completed":
                print(
                    "\n\n[RESPONSE DONE]",
                    {
                        "id": response.get("id"),
                        "conversation_id": response.get("conversation_id"),
                        "kind": metadata.get("kind"),
                        "status": response.get("status"),
                        "status_details": response.get("status_details"),
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "input_details": usage.get("input_token_details"),
                        "output_details": usage.get("output_token_details"),
                        "output_items": len(response.get("output", [])),
                    },
                    "\n",
                )


            if response.get("status") != "completed":
                continue

            for output_item in response.get("output", []):
                if output_item.get("type") != "function_call":
                    continue

                response_metadata = response.get("metadata") or {}

                create_tool_task(
                    cognitive_manager.handle_tool_call(
                        function_name=output_item.get("name"),
                        arguments=output_item.get("arguments"),
                        call_id=output_item.get("call_id"),
                        response_metadata=response_metadata,
                    ),
                    name=f"tool-{output_item.get('name', 'unknown')}",
                )

        elif event_type == "conversation.item.input_audio_transcription.delta":
            transcript = event.get("delta", "")

            def has_meaningful_speech(text):
                text = text.strip()
                if not text: return False
                if text.startswith(("[", "(")): return False
                return sum(char.isalnum() for char in text) >= 2

            if not human_speaking and has_meaningful_speech(transcript):
                app.clear_queue()
                human_speaking = True
                cognitive_manager.note_user_activity()

                print("\n[SPEECH STARTED]")

        elif event_type == "input_audio_buffer.speech_started":
            print("\n[VAD TRIGGERED]")

        elif event_type == "input_audio_buffer.speech_stopped":
            if human_speaking:
                human_speaking = False
                create_tool_task(response_manager.create_voice_response(),name="response_task")

        elif event_type == "response.output_audio.delta":
            audio_base64 = event.get("delta")

            if audio_base64:
                app.play_queue.put(base64.b64decode(audio_base64))

        elif event_type == "response.output_audio_transcript.delta":
            print(event.get("delta", ""), end="", flush=True)

def build_system_prompt(identity, memory, tool_routing=None):
    def format_value(value):
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)

        if isinstance(value, dict):
            parts = []
            for name, nested_value in value.items():
                prefix, separator, label = name.partition("_")
                heading = label if separator and prefix.isdigit() else name
                heading = heading.replace("_", " ").upper()
                parts.append(f"### {heading} ###\n{format_value(nested_value)}")
            return "\n".join(parts)

        return str(value)

    def section(heading, value):
        return f"=== {heading} ===\n{format_value(value)}"

    sections = [
        section("WHO YOU ARE", identity.get("identity", [])),
        section("YOUR CREATOR", identity.get("owner", [])),
        section("HOW YOU ARE", identity.get("character", [])),
        section("HARD RULES (NEVER BREAK THESE)", identity["rules"]),
    ]

    if tool_routing:
        sections.append(section("TOOL ROUTING", tool_routing))

    remembered = [
        f"{key}: {'; '.join(value)}"
        for key, value in memory.items()
        if isinstance(value, list) and value
    ]
    if remembered:
        sections.append("=== WHAT YOU REMEMBER ===\n" + "\n".join(remembered))

    return "\n\n".join(sections)

async def main():
    print("\n🤖 DJ STARTING TO CONNECT")
    cognitive_manager = None

    identity_file = load_json(IDENTITY_PATH)
    memory_file = load_json(MEMORY_PATH)
    tool_routing_file = load_json(TOOL_ROUTING_PATH)
    tools_file = []
    for path in TOOLS_PATHS:
        tools_file.extend(load_json(path))

    system_prompt = build_system_prompt(
        identity_file,
        memory_file,
        tool_routing_file,
    )

    app = AudioApp()
    print("\n✅ AUDIO IS READY")

    map_client = MapClient(
        context,
        endpoint=os.environ.get(
            "MAP_STREAM_ENDPOINT", "tcp://127.0.0.1:5559"
        ),
    )

    camera = CameraStream(
            camera_index=0,
            capture_width=1280,
            capture_height=720,
            tracking_width=640,
            tracking_height=360,
            history_frames=60,
            fps=30,
            usb_controls=CameraStream.usb_controls_from_env(),
            usb_device=os.environ.get(
                "USB_CAMERA_DEVICE",
                "/dev/v4l/by-id/usb-HBVCAM_Camera_"
                "USB_Camera_HB202400001-video-index0",
            ),
        )
    camera.start()
    print("\n✅ CAMERA IS READY")

    csrt_tracker = CSRTTrackingManager(
        camera=camera,
        max_initial_replay_frames=15
    )

    csrt_tracker.start_worker()
    print("\n✅ CSRT TRACKER IS READY")

    grounding_dino = GroundingDINOService(
        repo=GROUNDING_DINO_REPO,
        config=GROUNDING_DINO_CONFIG,
        model=GROUNDING_DINO_MODEL,
        device="GPU",
    )
    yolo = YoloService(
        camera=camera
    )
    hand_landmarks = HandLandmarkService(camera)

    grounding_dino.start_background()
    yolo.start_background()

    async def wait_for_dino():
        await grounding_dino.wait_until_ready()
        print("\n✅ GROUNDING DINO IS READY")
    async def wait_for_yolo():
        await yolo.wait_until_ready()
        print("\n✅ YOLO IS READY")
    async def send_robot_command(payload):
        async with zmq_req_lock:
            await zmq_req_socket.send_json(payload)
            return await asyncio.wait_for(
                zmq_req_socket.recv_json(),
                timeout=5.0,
            )

    async def send_map_overlay(payload):
        return await map_client.command(payload)

    headers = {
        "Authorization": "Bearer " + OPENAI_API_KEY,
        "OpenAI-Safety-Identifier": "hashed-user-id",
    }
    try:
        async with websockets.connect(URL, additional_headers=headers) as ws:
            print("\n✅ CONNECTED TO GPT REALTIME AGENT.\n")

            response_manager = ResponseManager(ws=ws,app=app)

            search_action = SearchAction(
                ws=ws,
                send_robot_command=send_robot_command,
                camera=camera,
                yolo=yolo,
            )
            track_action = TrackAction(
                csrt_tracker=csrt_tracker,
                grounding_dino=grounding_dino,
                yolo=yolo,
                camera=camera,
                zmq_pub_socket=zmq_pub_socket,
                person_tracker_pub_socket=person_tracker_pub_socket,
                send_robot_command=send_robot_command,
            )
            approach_action = ApproachAction(
                send_robot_command=send_robot_command,
            )
            see_action = SeeAction(
                ws=ws,
                camera=camera,
                send_robot_command=send_robot_command,
            )
            explicit_navigation_action = ExplicitNavigationAction(
                send_robot_command,
                request_map_snapshot=map_client.request_snapshot,
                save_map_snapshot=map_client.save_snapshot,
            )
            map_navigation_action = MapNavigationAction(
                ws, camera, send_robot_command,
                request_map_snapshot=map_client.request_snapshot,
                save_map_snapshot=map_client.save_snapshot,
                see_action=see_action,
                send_map_overlay=send_map_overlay,
            )
            find_target_executor = FindTargetExecutor(
                search_action=search_action,
                track_action=track_action,
                approach_action=approach_action,
                see_action=see_action,
                map_navigation_action=map_navigation_action,
                local_navigation_action=explicit_navigation_action,
                send_map_overlay=send_map_overlay,
            )
            hand_gesture_interface = HandGestureInterface(hand_landmarks)
            follow_person_executor = FollowPersonExecutor(
                find_target_executor=find_target_executor,
                track_action=track_action,
                send_robot_command=send_robot_command,
            )

            cognitive_manager = CognitionManager(
                approach_action=approach_action,
                search_action=search_action,
                see_action=see_action,
                track_action=track_action,
                explicit_navigation_action=explicit_navigation_action,
                map_navigation_action=map_navigation_action,
                hand_gesture_interface=hand_gesture_interface,
                find_target_executor=find_target_executor,
                follow_person_executor=follow_person_executor,
                response_manager=response_manager,
            )
            session_update = {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": MODEL,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": 24000,
                            },
                            "transcription": {
                                "model": "gpt-realtime-whisper",
                                "language": "en",
                                "delay": "minimal",
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.9,
                                "prefix_padding_ms": 250,
                                "silence_duration_ms": 600,
                                "create_response": False,
                                "interrupt_response": False
                            },
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": 24000,
                            },
                            "voice": "shimmer",
                        }
                    },

                    "instructions": system_prompt,
                    "tools": tools_file
                }
            }
            await ws.send(json.dumps(session_update))

            await asyncio.gather(
                send_mic_audio(ws, app),
                send_typed_messages(response_manager, cognitive_manager),
                receive_events(ws,app,response_manager,camera,cognitive_manager),
                background_status_monitor(cognitive_manager),
                cognitive_manager.cognition_loop(),
                display_camera_loop(
                    camera, csrt_tracker, yolo, track_action,
                    hand_gesture_interface,
                ),
                hand_gesture_interface.run(yolo),
                wait_for_dino(),
                wait_for_yolo()
            )
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by server.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if cognitive_manager is not None:
            await cognitive_manager.shutdown()
        print("\nCleaning up audio hardware...")
        app.stop()
        camera.stop()
        cv2.destroyAllWindows()
        csrt_tracker.stop_worker()
        await grounding_dino.close()
        await yolo.close()
        await hand_landmarks.close()
        map_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
