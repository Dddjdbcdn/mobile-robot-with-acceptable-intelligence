import asyncio
import base64
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np
import pyaudio
import websockets
import zmq
import zmq.asyncio
from typing import Any

from database.state import robot_state,update_camera_state

from utilities.database_functions import load_json, update_memory, build_system_prompt
from utilities.camera_sampler import draw_tof_overlay,draw_yolo_overlay,draw_csrt_overlay

from actions.search_action import ObjectSearchingManager
from actions.track_action import ObjectTrackingManager,human_trackable_parts
from actions.approach_actions import ObjectApproachingManager

from services.audio_stream import AudioApp, send_mic_audio
from services.response_manager import ResponseManager
from services.groundingdino_service import GroundingDINOService
from services.camera_stream import CameraStream, send_camera_image

from services.csrt_tracker import CSRTTrackingManager
from services.depthanything_service import DepthAnythingService
from services.sam2_service import SAM2OpenVINOService # unused


context = zmq.asyncio.Context()

zmq_req_socket = context.socket(zmq.REQ)
zmq_req_socket.connect("tcp://localhost:5555")

zmq_sub_socket = context.socket(zmq.SUB)
zmq_sub_socket.connect("tcp://localhost:5556")
zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

zmq_pub_socket = context.socket(zmq.PUB)
zmq_pub_socket.connect("tcp://localhost:5557")

zmq_req_lock = asyncio.Lock()

IDENTITY_PATH = "database/identity.json"
MEMORY_PATH = "database/memory.json"
TOOLS_PATHS = [
    "tools/database_tools.json", 
    "tools/navigate_tools.json", 
    "tools/vision_tools.json"
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

DEPTH_ANYTHING_MODEL = Path("vision_models/depthanything_tools/openvino_models/dav2_metric_indoor_vitb_896x504_fp16.xml")

tool_tasks: set[asyncio.Task[Any]] = set()

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

async def handle_tool_call(ws, function_name, arguments, call_id, response_metadata,response_manager,camera,object_searching_manager,object_tracking_manager,object_approaching_manager,session_state):
    try:
        args = json.loads(arguments) if arguments else {}

        if function_name == "update_memory":
            print("[System: Saving to disk -> "f"{args.get('category')}: {args.get('new_info')}]")
            feedback = update_memory(MEMORY_PATH,args.get("category"),args.get("new_info"),)
            await response_manager.send_function_output(call_id, feedback)
            return

        elif function_name == "get_vision":
            jpeg_bytes = await asyncio.to_thread(camera.jpeg_bytes_snapshot, 70, False, "results/search_results/get_vision_snapshot.jpg")
            snapshot = camera.snapshot()
            feedback = {"status": "image_captured"}
            target = ensure_string(args.get("target"))
            action = args.get("action")

            print(f"[GET VISION]: ACTION: {action}. TARGET: {target}")

            await response_manager.send_function_output(call_id, feedback)

            if action == "reasoning":
                query = args.get("query", "").strip()
                await send_camera_image(ws,jpeg_bytes,instruction=query)
                await response_manager.create_voice_response()

            elif action == "find_object":
                await send_camera_image(ws,jpeg_bytes,instruction=f"Find {target} and explain the result")
                session_state["jpeg_bytes"] = jpeg_bytes
                session_state["target"] = target
                session_state["snapshot"] = snapshot
                await response_manager.create_tool_response(tool="INBAND_VISUAL_SEARCH")

            elif action == "look_at_object":
                tracking_result = await object_tracking_manager.start_tracking(jpeg_bytes,target,snapshot)
                await response_manager.send_function_output(call_id, {"status": tracking_result})
                if tracking_result != "tracking": await response_manager.create_voice_response()

            return

        elif function_name == "inband_visual_search":
            await response_manager.send_function_output(call_id,{"status": "success"})

            target = session_state["target"]
            jpeg_bytes = session_state["jpeg_bytes"]
            snapshot = session_state["snapshot"]
            result = args.get("result")

            print(f"TARGET: {target}")

            if result == "found":
                await response_manager.create_voice_response(system_msg=f"There is {target} in the current camera view")
                
            else:
                print("[System: Initiating wide visual search...]")

                if object_tracking_manager.tracking: 
                    await object_tracking_manager.stop_tracking()

                await object_searching_manager.start_visual_search(target,call_id)
            return

        elif function_name == "stop_visual_search":
            print("[System: Stopping enviroment search...]")
            await object_searching_manager.stop_visual_search()
            await response_manager.send_function_output(call_id, {"status": "success"})
            return
        
        elif function_name == "assess_batch_search":
            await object_searching_manager.assess_batch_search_action(args,response_metadata)
            return

        elif function_name == "approach_object":
            print("[System: Initiating object approaching...]")
            await object_approaching_manager.start_approaching(call_id)
            return

        elif function_name in {"blind_move","navigate_to_pose","stop_motors","move_camera"}:
            if object_tracking_manager.tracking: 
                await object_tracking_manager.stop_tracking()

            payload = {"command": function_name, **args}

            print(f"[System: Dispatching {function_name} command to ROS 2...]")
            print(f"Payload: {payload}")

            async with zmq_req_lock:
                await zmq_req_socket.send_json(payload)
                feedback = await zmq_req_socket.recv_json()

            print(f"[System: ROS 2 Acknowledged: {feedback.get('status', 'unknown')}]")

            await response_manager.send_function_output(call_id, feedback)
            return

        elif function_name == 'stop_object_tracking':
            print("[System: Stopping object tracking...]")
            await object_tracking_manager.stop_tracking()
            return

        elif function_name == 'capture_depth':
            jpeg_bytes = await asyncio.to_thread(camera.jpeg_bytes_snapshot, 70, False, "results/search_results/depth_input_snapshot.jpg")
    
            depth_result = await object_approaching_manager.depth_anything.detect(jpeg_bytes,output_root=Path("results/depth_results"))

            print(f"DEPTH ANYTHING FINISHED. LATENCY: {depth_result.total_latency_ms}")

        else:
            await response_manager.send_function_output(call_id,{"status": "error","message": f"Unknown function: {function_name}"})
            await response_manager.create_voice_response()

    except Exception as error:
        print(f"[Tool Error: {function_name}] {error}")

        await response_manager.send_function_output(call_id,{"status": "error","message": str(error)})
        await response_manager.create_voice_response()

async def background_status_monitor(response_manager,object_tracking_manager,object_approaching_manager):
    print("[System: Background Monitor Listening for ROS 2 feedback...]")
    while True:
        try:
            message = await zmq_sub_socket.recv_json()
            if message.get("type") == "event":
                alert = f"[SYSTEM NOTIFICATION]: {message.get('event')} - {message['status']}"
                await response_manager.create_voice_response(system_msg=alert)

                if object_approaching_manager.approaching and message.get('event') == "navigation":
                    object_approaching_manager.approaching = False

            elif message.get("type") == "state":
                update_camera_state(message,object_tracking_manager)

            elif message.get("type") == "yolo":
                robot_state["camera"]["yolo_detections"] = message["detections"]

        except Exception as e:
            print(f"[System Error in Monitor]: {e}")
            await asyncio.sleep(1) 

async def receive_events(ws,app,response_manager,camera,object_tracking_manager,object_searching_manager,object_approaching_manager):
    human_speaking = False
    session_state = {}

    async for message in ws:
        event = json.loads(message)
        event_type = event.get("type")

        if event_type == "error":
            error = event.get("error", {})

            print(
                "\n❌ OPENAI ERROR"
                f"\ntype: {error.get('type')}"
                f"\ncode: {error.get('code')}"
                f"\nmessage: {error.get('message')}"
                f"\nparam: {error.get('param')}"
                f"\nevent_id: {error.get('event_id')}"
            )

        elif event_type == "response.created":
            response = event.get("response", {})
            response_manager.handle_response_created(response)
            
      
        elif event_type == "response.done":
            response = event.get("response", {})
            response_manager.handle_response_done(response)

            usage = response.get("usage") or {}
            used = usage.get("input_tokens", 0)

            MAX_TOKENS = 32000
            print(f"\n🪙 Tokens left: {MAX_TOKENS - used:,} / {MAX_TOKENS:,}")


            if response.get("status") != "completed":
                continue

            for output_item in response.get("output", []):
                if output_item.get("type") != "function_call":
                    continue

                response_metadata = response.get("metadata") or {}

                create_tool_task(
                    handle_tool_call(
                        ws,
                        function_name=output_item.get("name"),
                        arguments=output_item.get("arguments"),
                        call_id=output_item.get("call_id"),
                        response_metadata=response_metadata,
                        response_manager=response_manager,
                        camera=camera,
                        object_searching_manager=object_searching_manager,
                        object_tracking_manager=object_tracking_manager,
                        object_approaching_manager=object_approaching_manager,
                        session_state=session_state
                    ),
                    name=f"tool-{output_item.get('name', 'unknown')}",
                )

        elif event_type == "conversation.item.input_audio_transcription.delta":
            transcript = event.get("delta", "")

            if not human_speaking and has_meaningful_speech(transcript):
                app.clear_queue()
                human_speaking = True

                print("[SPEECH STARTED]")

        elif event_type == "input_audio_buffer.speech_started":
            print("[VAD TRIGGERED]")

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

async def display_camera_loop(camera, csrt_tracker):
    print("[System: Starting live camera display...]")

    window_name = "Robot Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fullscreen_requested = False
    last_fullscreen_attempt = 0.0

    while True:
        try:
            snapshot = camera.snapshot()
            display_frame = cv2.flip(snapshot.full_bgr.copy(), 1)

            display_frame = draw_tof_overlay(display_frame)

            tracking_update = csrt_tracker.tracking_update

            if robot_state["camera"].get("yolo_detections"):
                display_frame = draw_yolo_overlay(display_frame)
            if (tracking_update is not None and tracking_update.is_tracking and tracking_update.success):
                display_frame = draw_csrt_overlay(display_frame,tracking_update.target,tracking_update.bbox_xywh)

            cv2.imshow(window_name, display_frame)
            cv2.waitKey(1)

            now = time.monotonic()
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) != cv2.WINDOW_FULLSCREEN:
                if not fullscreen_requested or now - last_fullscreen_attempt >= 0.5:
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                    fullscreen_requested = True
                    last_fullscreen_attempt = now

        except RuntimeError:
            await asyncio.sleep(0.05)
            continue

        await asyncio.sleep(0.03)

async def main():
    print("🤖 DJ STARTING TO CONNECT")
    headers = {
        "Authorization": "Bearer " + OPENAI_API_KEY,
        "OpenAI-Safety-Identifier": "hashed-user-id",
    }
    
    app = AudioApp()
    print("✅ AUDIO IS READY")

    response_manager = ResponseManager(ws=None,app=app)
    camera = CameraStream(
            camera_index=0,
            capture_width=1280,
            capture_height=720,
            tracking_width=640,
            tracking_height=360,
            history_frames=60,
            fps=30,
        )

    grounding_dino = GroundingDINOService(
        repo=GROUNDING_DINO_REPO,
        config=GROUNDING_DINO_CONFIG,
        model=GROUNDING_DINO_MODEL,
        device="GPU",
    )

    depth_anything = DepthAnythingService(
        model=DEPTH_ANYTHING_MODEL,
        max_depth_m=20.0,
        device="GPU",
        resize_mode="stretch",
        depth_scale=0.508
    )

    csrt_tracker = CSRTTrackingManager(
        camera=camera,
        max_initial_replay_frames=15
    )

    object_tracking_manager = ObjectTrackingManager(
        csrt_tracker=csrt_tracker,
        grounding_dino=grounding_dino,
        zmq_req_lock=zmq_req_lock,
        zmq_req_socket=zmq_req_socket,
        zmq_pub_socket=zmq_pub_socket,
    )

    object_searching_manager = ObjectSearchingManager(
            ws=None,
            zmq_req_lock=zmq_req_lock,
            zmq_req_socket=zmq_req_socket,
            camera=camera,
            response_manager = response_manager,
            object_tracking_manager=object_tracking_manager
        )
    
    object_approaching_manager = ObjectApproachingManager(
        camera=camera, 
        csrt_tracker=csrt_tracker,
        grounding_dino=grounding_dino,
        depth_anything=depth_anything,
        object_tracking_manager=object_tracking_manager,
        response_manager=response_manager,
        zmq_req_lock=zmq_req_lock,
        zmq_req_socket=zmq_req_socket,
    )

    camera.start()
    csrt_tracker.start_worker()
    print("✅ CSRT TRACKER IS READY")

    grounding_dino.start_background()
    depth_anything.start_background()

    async def wait_for_dino():
        await grounding_dino.wait_until_ready()
        print("✅ GROUNDING DINO IS READY")
    async def wait_for_depth():
        await depth_anything.wait_until_ready()
        print("✅ DEPTH ANYTHING IS READY")

    identity_file = load_json(IDENTITY_PATH)
    memory_file = load_json(MEMORY_PATH)
    
    tools_file = []
    for path in TOOLS_PATHS:
        tools_file.extend(load_json(path))

    system_prompt = build_system_prompt(identity_file, memory_file)
    try:
        async with websockets.connect(URL, additional_headers=headers) as ws:
            print("\n✅ CONNECTED TO GPT REALTIME AGENT.\n")

            response_manager.ws = ws
            object_searching_manager.ws = ws
            
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
                receive_events(ws,app,response_manager,camera,object_tracking_manager,object_searching_manager,object_approaching_manager),
                background_status_monitor(response_manager,object_tracking_manager,object_approaching_manager),
                display_camera_loop(camera,csrt_tracker),
                wait_for_dino(),
                wait_for_depth(),
            )
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by server.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up audio hardware...")
        app.stop()
        camera.stop()            
        cv2.destroyAllWindows()   
        csrt_tracker.stop_worker()
        await grounding_dino.close()
        await depth_anything.close()

# HELPER FUNCTIONS

def ensure_string(raw_target) -> str | None:
    if not raw_target or raw_target == "":
        return None
        
    if isinstance(raw_target, str): return raw_target.strip()
        
    if isinstance(raw_target, dict):
        for key in ['type', 'target', 'name', 'object', 'value']:
            if key in raw_target and isinstance(raw_target[key], str):
                return raw_target[key].strip()
        
        return json.dumps(raw_target)
        
    if isinstance(raw_target, list):
        return ", ".join(str(item) for item in raw_target if item)
    
    return str(raw_target)
def has_meaningful_speech(text: str) -> bool:
    text = text.strip()
    if not text: return False
    if text.startswith(("[", "(")): return False
    return sum(char.isalnum() for char in text) >= 2

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...")