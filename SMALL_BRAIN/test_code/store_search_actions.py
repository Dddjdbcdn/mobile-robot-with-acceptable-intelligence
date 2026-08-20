from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

import asyncio
import base64
import json
from pathlib import Path
from typing import Any
import uuid

from utilities.database_functions import load_json

from services.camera_stream import clear_images_folder

REPO_ROOT = Path(__file__).resolve().parent.parent
VISION_OOB_TOOLS_PATH = str(REPO_ROOT / "tools" / "vision_oob_tools.json")

tools = load_json(VISION_OOB_TOOLS_PATH)
vision_oob_tools = {tool["name"]: tool for tool in tools}

ASSESS_BATCH_SEARCH_TOOL = vision_oob_tools.get("assess_batch_search")
VERIFY_SEARCH = vision_oob_tools.get("verify_search")

# Scan/verification policy.
CANDIDATE_CONFIDENCE_THRESHOLD = 0.5
FOUND_CONFIDENCE_THRESHOLD = 0.7

# Approximate camera field of view. Calibrate these for the actual camera.
CAMERA_HORIZONTAL_FOV_DEG = 85
CAMERA_VERTICAL_FOV_DEG = 52
AIM_GAIN = 1.0

CAMERA_MOVE_TIMEOUT_SECONDS = 2.0
CAMERA_SETTLE_SECONDS = 0.5

PAN_POSITION_ANGLE = {
    "center": 95.0,
    "leftmost": 140.0,
    "rightmost": 50.0,
}

TILT_POSITION_ANGLE = {
    "center": 90.0,
    "upmost": 50.0,
    "downmost": 120.0,
}

PAN_SWEEP_ORDER = (
    "leftmost",
    "center",
    "rightmost",
)

SWEEP_TILT_ORDER = (
    "center",
    "upmost",
    "downmost",
)


def make_empty_search_state() -> dict[str, Any]:
    return {
        "active": False,
        "phase": "idle",
        "search_id": None,
        "main_call_id": None,
        "target": None,
        "sweep_index": 0,
        "pan_angle": PAN_POSITION_ANGLE["center"],
        "tilt_angle": TILT_POSITION_ANGLE["center"],
        "current_batch_frames": [],
        "current_candidate": None,
        "verification_frame": None,
        "batch_results": [],
        "pending_request_id": None,
    }


active_state = make_empty_search_state()


def reset_visual_search_state() -> None:
    active_state.clear()
    active_state.update(make_empty_search_state())

def target_to_servo_angles(
    center_x, center_y,
    base_pan_angle: float,
    base_tilt_angle: float,
) -> tuple[float, float]:

    image_x_error = center_x - 0.5
    image_y_error = center_y - 0.5

    pan_correction = -image_x_error * CAMERA_HORIZONTAL_FOV_DEG * AIM_GAIN
    tilt_correction = image_y_error * CAMERA_VERTICAL_FOV_DEG * AIM_GAIN

    target_pan = base_pan_angle + pan_correction
    target_tilt = base_tilt_angle + tilt_correction

    return target_pan, target_tilt


async def move_camera_angles(
    target_pan: float,
    target_tilt: float,
    zmq_req_lock,
    zmq_req_socket,
) -> None:

    delta_pan = target_pan - float(active_state["pan_angle"])
    delta_tilt = target_tilt - float(active_state["tilt_angle"])

    payload = {
        "command": "move_camera",
        "delta_pan_angle": delta_pan,
        "delta_tilt_angle": delta_tilt,
    }

    async with zmq_req_lock:
        await zmq_req_socket.send_json(payload)
        feedback = await asyncio.wait_for(
            zmq_req_socket.recv_json(),
            timeout=CAMERA_MOVE_TIMEOUT_SECONDS,
        )

    if feedback.get("status") != "accepted":
        raise RuntimeError(f"Camera movement rejected: {feedback}")

    active_state["pan_angle"] = float(feedback.get("pan_angle", target_pan))
    active_state["tilt_angle"] = float(feedback.get("tilt_angle", target_tilt))

    await asyncio.sleep(CAMERA_SETTLE_SECONDS)

async def capture_sweep_batch(
    ws,
    zmq_req_lock,
    zmq_req_socket,
    camera
) -> None:
    if not active_state["active"]:
        return

    sweep_index = int(active_state["sweep_index"])
    if sweep_index >= len(SWEEP_TILT_ORDER):
        raise RuntimeError("Sweep index exceeded configured tilt rows")

    tilt_position = SWEEP_TILT_ORDER[sweep_index]
    frames: list[dict[str, Any]] = []

    for image_number, pan_position in enumerate(PAN_SWEEP_ORDER, start=1):
        await move_camera_angles(
            PAN_POSITION_ANGLE[pan_position],
            TILT_POSITION_ANGLE[tilt_position],
            zmq_req_lock,
            zmq_req_socket,
        )

        print(f"CAMERA MOVE: PAN: {PAN_POSITION_ANGLE[pan_position]}, TILT: {TILT_POSITION_ANGLE[tilt_position]}")

        jpeg_bytes = await asyncio.to_thread(
            camera.jpeg_bytes_snapshot,
            tracking_bgr=True,
            save_path=f"results/search_results/row_{sweep_index}_{image_number}.jpg",
        )

        frames.append(
            {
                "image_id": f"image_{image_number}",
                "pan_position": pan_position,
                "tilt_position": tilt_position,
                "pan_angle": float(active_state["pan_angle"]),
                "tilt_angle": float(active_state["tilt_angle"]),
                "jpeg_bytes": jpeg_bytes,
            }
        )

    active_state["current_batch_frames"] = frames
    await request_batch_assessment(ws, frames)


async def request_batch_assessment(ws, frames: list[dict[str, Any]]) -> None:
    request_id = uuid.uuid4().hex
    active_state["phase"] = "scan"
    active_state["pending_request_id"] = request_id

    tilt_position = SWEEP_TILT_ORDER[int(active_state["sweep_index"])]

    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"Search for this object: {active_state['target']}\n\n"
                "These three images form one horizontal camera sweep.\n"
                f"Current tilt: {tilt_position}\n"
                "image_1: leftmost\n"
                "image_2: center\n"
                "image_3: rightmost\n\n"
                "Examine all three images.\n"
                "Use candidate when an object is visually clear or plausibly resembles the target.\n"
                "Use not_found when the target is not visible.\n"
                "For candidate, return the best image and a normalized "
                "center position of the target. For other results, return "
                "candidate_image=none and candidate_position=null.\n"
                "Always call assess_batch_search exactly once."
            ),
        }
    ]

    for frame in frames:
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"{frame['image_id']}: "
                        f"pan={frame['pan_position']}, "
                        f"tilt={frame['tilt_position']}"
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": jpeg_to_data_url(frame["jpeg_bytes"]),
                },
            ]
        )

    event = {
        "event_id": f"visual_search_scan_{request_id}",
        "type": "response.create",
        "response": {
            "conversation": "none",
            "metadata": {
                "kind": "visual_search_assessment",
                "search_id": str(active_state["search_id"]),
                "request_id": request_id,
                "phase": "scan",
                "sweep_index": str(active_state["sweep_index"]),
            },
            "output_modalities": ["text"],
            "tools": [ASSESS_BATCH_SEARCH_TOOL],
            "tool_choice": "required",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": content,
                }
            ],
        },
    }
    await ws.send(json.dumps(event))

async def assess_batch_search_action(
    ws,
    args: dict[str, Any] | str,
    response_metadata: dict[str, Any],
    zmq_req_lock,
    zmq_req_socket,
    response_manager,
    camera
) -> None:
    if not active_state["active"]:
        return

    if response_metadata.get("kind") != "visual_search_assessment": return
    if str(response_metadata.get("search_id")) != str(active_state["search_id"]): return
    if response_metadata.get("request_id") != active_state["pending_request_id"]: return

    phase = response_metadata.get("phase")
    if phase != active_state["phase"]:
        return

    active_state["pending_request_id"] = None

    batch_result = {
        "sweep_index": int(active_state["sweep_index"]),
        "tilt_position": SWEEP_TILT_ORDER[
            int(active_state["sweep_index"])
        ],
        "assessment": args,
    }
    active_state["batch_results"].append(batch_result)

    result = args.get("result")
    confidence = float(args.get("confidence", 0.0))

    if (
    result == "candidate"
    and confidence >= CANDIDATE_CONFIDENCE_THRESHOLD
    ):
        position_validated = get_valid_position(args.get("candidate_position"))
        frame_validated = get_valid_frame(args.get("candidate_image"),active_state["current_batch_frames"])

        if position_validated is not None and frame_validated is not None:
            candidate_position = position_validated
            candidate_frame = frame_validated

            save_candidate_debug_image(
                jpeg_bytes=candidate_frame["jpeg_bytes"],
                candidate_position=candidate_position,
                save_path=(REPO_ROOT / "results"/ "search_results" / "candidate.jpg"),)

            active_state["current_candidate"] = {
                "assessment": {
                    **args,
                    "candidate_position": candidate_position,
                },
                "frame": candidate_frame,
                "sweep_index": int(active_state["sweep_index"]),
            }

            await capture_verification_image(
                ws,
                zmq_req_lock,
                zmq_req_socket,
                camera,
                response_manager
            )
            return

    active_state["sweep_index"] += 1
    if active_state["sweep_index"] < len(SWEEP_TILT_ORDER):
        await capture_sweep_batch(
            ws,
            zmq_req_lock,
            zmq_req_socket,
            camera
        )
        return
    else:
        await move_camera_angles(
            PAN_POSITION_ANGLE["center"],
            TILT_POSITION_ANGLE["center"],
            zmq_req_lock,
            zmq_req_socket,
        )
        await complete_visual_search(
            ws,
            result="not_found",
            final_assessment=args,
            message=(
                f"I could not find {active_state['target']} after scanning "
                "all rows."
            ),
            response_manager=response_manager
        )
    return

async def capture_verification_image(
    ws,
    zmq_req_lock,
    zmq_req_socket,
    camera,
    response_manager
) -> None:
    candidate = active_state.get("current_candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError("Verification requested without a candidate")

    candidate_image = str(candidate["assessment"]["candidate_image"])
    candidate_x_position = candidate["assessment"].get("candidate_position").get("x")
    candidate_y_position = candidate["assessment"].get("candidate_position").get("y")
    candidate_frame = candidate.get("frame")

    await move_camera_angles(
        candidate_frame["pan_angle"],
        candidate_frame["tilt_angle"],
        zmq_req_lock,
        zmq_req_socket,
    )
    target_pan, target_tilt = target_to_servo_angles(
        candidate_x_position,
        candidate_y_position,
        base_pan_angle=float(candidate_frame["pan_angle"]),
        base_tilt_angle=float(candidate_frame["tilt_angle"]),
    )

    await move_camera_angles(
        target_pan,
        target_tilt,
        zmq_req_lock,
        zmq_req_socket,
    )

    jpeg_bytes = await asyncio.to_thread(
        camera.jpeg_bytes_snapshot, save_path=f"results/search_results/verification.jpg",
    )

    verification_frame = {
        "source_candidate_image": candidate_image,
        "pan_angle": float(active_state["pan_angle"]),
        "tilt_angle": float(active_state["tilt_angle"]),
        "jpeg_bytes": jpeg_bytes,
    }
    active_state["verification_frame"] = verification_frame

    # await request_verification(
    #     ws,
    #     verification_frame,
    #     candidate["assessment"],
    # )

    await complete_visual_search(
        ws,
        result="found",
        final_assessment=active_state["current_candidate"]["assessment"],
        message=f"I found {active_state['target']}.",
        response_manager=response_manager
    )


async def request_verification(
    ws,
    verification_frame: dict[str, Any],
    original_assessment: dict[str, Any],
) -> None:
    request_id = uuid.uuid4().hex
    active_state["phase"] = "verify"
    active_state["pending_request_id"] = request_id

    content = [
        {
            "type": "input_text",
            "text": (
                f"Verify whether this close-up image clearly shows: "
                f"{active_state['target']}\n\n"
                "Judge only the new verification image, which is supposed to aim roughly at the center of the target.\n"
                "Return found only if the requested object is visually clear.\n"
                "Return not_found if it is absent.\n"
                "For found, give the target position in this new image. "
                "Otherwise, use target_position=null.\n"
                "Always call verify_search exactly once.\n\n"
                f"Original scan evidence: "
                f"{original_assessment.get('evidence', '')}"
            ),
        },
        {
            "type": "input_image",
            "image_url": jpeg_to_data_url(
                verification_frame["jpeg_bytes"]
            ),
        },
    ]

    event = {
        "event_id": f"visual_search_verify_{request_id}",
        "type": "response.create",
        "response": {
            "conversation": "none",
            "metadata": {
                "kind": "visual_search_assessment",
                "search_id": str(active_state["search_id"]),
                "request_id": request_id,
                "phase": "verify",
                "sweep_index": str(active_state["sweep_index"]),
            },
            "output_modalities": ["text"],
            "tools": [VERIFY_SEARCH],
            "tool_choice": "required",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": content,
                }
            ],
        },
    }
    await ws.send(json.dumps(event))

async def verify_search_action(
    ws,
    args: dict[str, Any] | str,
    response_metadata: dict[str, Any],
    zmq_req_lock,
    zmq_req_socket,
    response_manager,
    camera
) -> None:
    if not active_state["active"]:
        return

    print(args)

    if response_metadata.get("kind") != "visual_search_assessment": return
    if str(response_metadata.get("search_id")) != str(active_state["search_id"]): return
    if response_metadata.get("request_id") != active_state["pending_request_id"]: return

    phase = response_metadata.get("phase")
    if phase != active_state["phase"]: return

    active_state["pending_request_id"] = None

    result = args.get("result")
    confidence = float(args.get("confidence", 0.0))

    if result == "found" and confidence >= FOUND_CONFIDENCE_THRESHOLD:
        await complete_visual_search(
            ws,
            result="found",
            final_assessment=args,
            message=f"I found {active_state['target']}.",
            response_manager=response_manager
        )
    else:
        await complete_visual_search(
            ws,
            result="candidate_unverified",
            final_assessment=args,
            message=(
                "I feel like it is here, but the verification image "
                "was not clear enough to be sure."
            ),
            response_manager=response_manager
        )
    return

async def start_visual_search(
    ws,
    args: dict[str, Any],
    main_call_id: str,
    zmq_req_lock,
    zmq_req_socket,
    response_manager,
    camera
) -> None:
    if active_state["active"]:
        await response_manager.send_function_output(
            main_call_id,
            {
                "status": "error",
                "message": "Another visual search is already active.",
                "active_target": active_state["target"],
            },
        )
        await response_manager.create_voice_response()
        return

    target = str(args.get("target", "")).strip()

    reset_visual_search_state()

    async with zmq_req_lock:
        await zmq_req_socket.send_json({"command": "get_state"})

        feedback = await asyncio.wait_for(
            zmq_req_socket.recv_json(),
            timeout=5.0,
        )
    
    initial_pan_angle = feedback.get("servo_pan_angle", 95)
    initial_tilt_angle = feedback.get("servo_tilt_angle", 85)

    active_state.update(
        {
            "active": True,
            "phase": "starting",
            "search_id": uuid.uuid4().hex,
            "main_call_id": main_call_id,
            "target": target,
            "pan_angle": initial_pan_angle,
            "tilt_angle": initial_tilt_angle
        }
    )

    print(
        "[Visual search started] "
        f"id={active_state['search_id']}, target={target}"
    )

    clear_images_folder()

    await capture_sweep_batch(
        ws,
        zmq_req_lock,
        zmq_req_socket,
        camera
    )

async def stop_visual_search(
    ws,
    args: dict[str, Any],
    main_call_id: str,
    zmq_req_lock,
    zmq_req_socket,
    response_manager
) -> None:
    if not active_state["active"]:
        return

    await move_camera_angles(
        PAN_POSITION_ANGLE["center"],
        TILT_POSITION_ANGLE["center"],
        zmq_req_lock,
        zmq_req_socket,
    )

    await complete_visual_search(
        ws,
        result="search_cancelled",
        final_assessment=args,
        message=(
            f"The visual search was cancelled by the user. Report the current progress"
        ),
        response_manager=response_manager
    )
    

async def complete_visual_search(
    ws,
    result: str,
    final_assessment: dict[str, Any],
    message: str,
    response_manager
) -> None:
    if not active_state["active"]:
        return

    main_call_id = active_state["main_call_id"]
    target = str(active_state["target"])

    final_result = {
        "status": "completed",
        "target": target,
        "result": result,
        "message": message,
        "final_assessment": final_assessment,
        "scan_results": active_state["batch_results"],
        "final_camera_pose": {
            "pan_angle": active_state["pan_angle"],
            "tilt_angle": active_state["tilt_angle"],
        },
    }

    print(f"[Visual search completed] {final_result}")

    await response_manager.send_function_output(main_call_id, final_result)
    await response_manager.create_voice_response()

    reset_visual_search_state()


def jpeg_to_data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"

def get_valid_position(position):
    if isinstance(position, dict):
        x = position.get("x")
        y = position.get("y")
    elif isinstance(position, (list, tuple)) and len(position) == 2:
        x, y = position
    else: return None
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0: return None
    return {"x": float(x),"y": float(y)}


def get_valid_frame(image_id,frames):
    if image_id not in {"image_1", "image_2", "image_3"}:
        return None
    return next((frame for frame in frames if frame.get("image_id") == image_id),None)

def get_valid_box(box):
    if isinstance(box, dict):
        x_min = box.get("x_min")
        x_max = box.get("x_max")
        y_min = box.get("y_min")
        y_max = box.get("y_max")
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        x_min,x_max,y_min,y_max = box
    else:
        return None

    if (
        not 0.0 <= x_min <= 1.0 or not 0.0 <= x_max <= 1.0
        or not 0.0 <= y_min <= 1.0 or not 0.0 <= y_max <= 1.0
        or not y_min < y_max or not x_min < x_max
    ):
        return None

    return {"x_min": float(x_min),"x_max": float(x_max),"y_min": float(y_min),"y_max": float(y_max)}

def save_candidate_debug_image(
    jpeg_bytes: bytes,
    candidate_position: dict[str, float],
    save_path: str | Path,
    *,
    radius: int = 8,
):
    x_normalized = float(candidate_position["x"])
    y_normalized = float(candidate_position["y"])

    with Image.open(BytesIO(jpeg_bytes)) as image:
        image = image.convert("RGB")
        width, height = image.size

        # Convert normalized coordinates into pixel coordinates.
        x_pixel = round(x_normalized * (width - 1))
        y_pixel = round(y_normalized * (height - 1))

        draw = ImageDraw.Draw(image)

        draw.ellipse(
            (
                x_pixel - radius,
                y_pixel - radius,
                x_pixel + radius,
                y_pixel + radius,
            ),
            fill="red",
            outline="white",
            width=2,
        )

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image.save(output_path, format="JPEG", quality=95)