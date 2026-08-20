from __future__ import annotations
import asyncio
import base64
import inspect
import io
import json
import math
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from PIL import Image, ImageDraw, ImageFont

from utilities.database_functions import load_json

REPO_ROOT = Path(__file__).resolve().parent.parent
VISION_OOB_TOOLS_PATH = str(REPO_ROOT / "tools" / "vision_oob_tools.json")
BOX_OUTPUT_DIR = REPO_ROOT / "images" / "box_refinements"

_tools = load_json(VISION_OOB_TOOLS_PATH)
_vision_oob_tools = {tool["name"]: tool for tool in _tools}
REFINE_BOX_TOOL = _vision_oob_tools.get("create_bounding_box")

ALIGNMENT_THRESHOLD = 0.80
MAX_ITERATIONS = 3
OUTSIDE_ALPHA = 0

def make_empty_box_state() -> dict[str, Any]:
    return {
        "active": False,
        "refinement_id": None,
        "pending_request_id": None,
        "target": None,
        "iteration": 0,
        "original_image_jpeg_bytes": None,
        "annotated_image": None,
        "current_box": None,
        "best_box": None,
        "best_alignment": -1.0,
        "history": [],
        "output_dir": None,
        "result": None,
    }


active_box_refinement = make_empty_box_state()


def reset_box_creation_state() -> None:
    active_box_refinement.clear()
    active_box_refinement.update(make_empty_box_state())


async def start_box_creation(
    ws,
    target,
    jpeg_bytes,
    initial_box,
) -> None:
    """Start an OOB box-refinement loop from an initial candidate box."""

    if active_box_refinement["active"]:
        raise RuntimeError("Another bounding-box refinement is already active")

    validated_box = get_valid_box(initial_box)
    if validated_box is None:
        raise ValueError(
            f"Invalid initial bounding box: {initial_box!r}"
        )

    reset_box_creation_state()

    annotated_image = draw_box_overlay(
        jpeg_bytes,
        initial_box,
        target=target,
        iteration=1,
        outside_alpha=OUTSIDE_ALPHA,
    )

    active_box_refinement.update(
        {
            "active": True,
            "refinement_id": uuid.uuid4().hex,
            "target": target.strip(),
            "iteration": 1,
            "original_image_jpeg_bytes": jpeg_bytes,
            "annotated_image": annotated_image,
            "current_box": initial_box,
            "best_box": initial_box,
        }
    )

    annotated_image = await asyncio.to_thread(
        draw_box_overlay,
        jpeg_bytes,
        validated_box,
        target,
        1,
        OUTSIDE_ALPHA,
    )

    active_box_refinement["annotated_image"] = annotated_image

    await asyncio.gather(
        save_jpeg(
            BOX_OUTPUT_DIR / "original.jpg",
            jpeg_bytes,
        ),
        save_jpeg(
            BOX_OUTPUT_DIR / "proposal_01.jpg",
            annotated_image,
        ),
    )

    await request_box_creation(ws)

async def request_box_creation(ws) -> None:
    if not active_box_refinement["active"]: return

    original = active_box_refinement.get("original_image_jpeg_bytes")
    current_box = get_valid_box(active_box_refinement.get("current_box"))
    iteration = int(active_box_refinement["iteration"])
    annotated_image = active_box_refinement.get("annotated_image")

    request_id = uuid.uuid4().hex
    active_box_refinement["pending_request_id"] = request_id

    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"Refine the tracking box for: {active_box_refinement['target']}\n\n"
                "The first image is the unmodified source image. The second image "
                "shows the last proposed bounding box in red.\n"
                "The proposed box may be completely wrong and may not overlap the target."
                "Locate the target independently in the unmodified image, and use the proposed box for reference only"
                "Alignment scores the proposed box in the second input image, not your current attempt"
                "Return a normalized box that tightly encloses the visible pixels "
                "of the requested object, with only a small tracking margin.\n"
                "target pixels are enclosed and unrelated background is minimal.\n"
                "Always call create_bounding_box exactly once."
            ),
        },
        {
            "type": "input_text",
            "text": "Original image:",
        },
        {
            "type": "input_image",
            "image_url": jpeg_to_data_url(original),
        },
        {
            "type": "input_text",
            "text": f"Current proposal bounding box at iteration: {iteration}. Normalized position of the box: {current_box}",
        },
        {
            "type": "input_image",
            "image_url": jpeg_to_data_url(annotated_image),
        },
    ]

    event = {
        "event_id": f"box_refinement_{request_id}",
        "type": "response.create",
        "response": {
            "conversation": "none",
            "metadata": {
                "kind": "bounding_box_refinement",
                "refinement_id": str(active_box_refinement["refinement_id"]),
                "request_id": request_id,
                "iteration": str(iteration),
            },
            "output_modalities": ["text"],
            "tools": [REFINE_BOX_TOOL],
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


async def create_box_action(
    ws,
    args: dict[str, Any] | str,
    response_metadata: dict[str, Any],
) -> None:
    if not active_box_refinement["active"]: return

    if response_metadata.get("kind") != "bounding_box_refinement": return
    if str(response_metadata.get("refinement_id")) != str(active_box_refinement["refinement_id"]): return
    if response_metadata.get("request_id") != active_box_refinement["pending_request_id"]: return

    active_box_refinement["pending_request_id"] = None

    if isinstance(args, str): 
        args = json.loads(args)

    target_box = get_valid_box(args.get("target_box"))
    alignment = get_valid_alignment(args.get("alignment"))

    if target_box is None or alignment is None:
        await complete_box_creation(
            status="invalid_model_output",
            message="The box-refinement model returned invalid coordinates or alignment.",
        )
        return

    iteration = int(active_box_refinement["iteration"])
    active_box_refinement["current_box"] = target_box

    if alignment > float(active_box_refinement["best_alignment"]):
        active_box_refinement["best_alignment"] = alignment
        active_box_refinement["best_box"] = target_box

    print(f"ALIGNMENT: {alignment}. ITER: {iteration}")

    if alignment >= ALIGNMENT_THRESHOLD:
        await complete_box_creation(
            status="completed",
            message="Bounding box accepted.",
        )
        return

    if iteration >= MAX_ITERATIONS:
        await complete_box_creation(
            status="max_iterations",
            message=(
                "Bounding-box refinement reached the iteration limit; "
                "the highest-scoring valid box was retained."
            ),
        )
        return


    next_iteration = iteration + 1
    original_image = active_box_refinement["original_image_jpeg_bytes"]

    annotated_image = await asyncio.to_thread(
        draw_box_overlay,
        original_image,
        target_box,
        active_box_refinement["target"],
        next_iteration,
        OUTSIDE_ALPHA,
    )
    active_box_refinement["iteration"] = next_iteration
    active_box_refinement["annotated_image"] = annotated_image

    await save_jpeg(
        BOX_OUTPUT_DIR / f"proposal_{next_iteration:02d}.jpg",
        annotated_image,
    )

    await request_box_creation(ws)


async def complete_box_creation(
    status: str,
    message: str,
) -> None:
    if not active_box_refinement["active"]:
        return

    final_box = get_valid_box(active_box_refinement.get("best_box"))
    if final_box is None:
        final_box = get_valid_box(active_box_refinement.get("current_box"))

    result = {
        "status": status,
        "message": message,
    }

    active_box_refinement["result"] = result

    print(f"[Bounding-box refinement completed] {result}")

    reset_box_creation_state()

def jpeg_to_data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def get_valid_alignment(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def get_valid_box(box: Any) -> dict[str, float] | None:
    if isinstance(box, dict):
        values = (
            box.get("x_min"),
            box.get("x_max"),
            box.get("y_min"),
            box.get("y_max"),
        )
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        values = tuple(box)
    else:
        return None

    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        return None

    x_min, x_max, y_min, y_max = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
        return None
    if not (
        0.0 <= x_min < x_max <= 1.0
        and 0.0 <= y_min < y_max <= 1.0
    ):
        return None

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

async def save_jpeg(path: Path, jpeg_bytes: bytes) -> None:
    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg_bytes)

    await asyncio.to_thread(_write)
    
def to_pixels(
    box: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    target_box = get_valid_box(box)
    if target_box is None:
        raise ValueError(f"Invalid normalized bounding box: {box!r}")

    left = max(0, min(image_width - 1, round(target_box["x_min"] * image_width)))
    top = max(0, min(image_height - 1, round(target_box["y_min"] * image_height)))
    right = max(left + 1, min(image_width, round(target_box["x_max"] * image_width)))
    bottom = max(top + 1, min(image_height, round(target_box["y_max"] * image_height)))
    return left, top, right, bottom


def draw_box_overlay(
    image_bytes: bytes,
    box: dict[str, Any],
    target: str,
    iteration: int,
    outside_alpha: int = OUTSIDE_ALPHA,
) -> bytes:
    if not 0 <= outside_alpha <= 255:
        raise ValueError("outside_alpha must be between 0 and 255")

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGBA")

    width, height = image.size
    left, top, right, bottom = to_pixels(box, width, height)

    if outside_alpha:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        shade = (0, 0, 0, outside_alpha)
        overlay_draw.rectangle((0, 0, width, top), fill=shade)
        overlay_draw.rectangle((0, bottom, width, height), fill=shade)
        overlay_draw.rectangle((0, top, left, bottom), fill=shade)
        overlay_draw.rectangle((right, top, width, bottom), fill=shade)
        image = Image.alpha_composite(image, overlay)

    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(width, height) * 0.006))
    outline = (255, 50, 50, 255)
    accent = (255, 255, 255, 255)

    draw.rectangle(
        (left, top, right - 1, bottom - 1),
        outline=accent,
        width=line_width + 2,
    )
    draw.rectangle(
        (left, top, right - 1, bottom - 1),
        outline=outline,
        width=line_width,
    )

    handle = max(5, line_width * 2)
    for x, y in (
        (left, top),
        (right - 1, top),
        (left, bottom - 1),
        (right - 1, bottom - 1),
    ):
        draw.rectangle(
            (x - handle, y - handle, x + handle, y + handle),
            fill=outline,
            outline=accent,
            width=max(1, line_width // 2),
        )

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    cross = max(8, line_width * 3)
    draw.line(
        (center_x - cross, center_y, center_x + cross, center_y),
        fill=outline,
        width=line_width,
    )
    draw.line(
        (center_x, center_y - cross, center_x, center_y + cross),
        fill=outline,
        width=line_width,
    )

    normalized_box = get_valid_box(box)
    assert normalized_box is not None

    font = ImageFont.load_default()
    safe_target = target.encode("ascii", "replace").decode("ascii")
    label = (
        f"{safe_target} | check {iteration} | "
        f"({normalized_box['x_min']:.3f},{normalized_box['y_min']:.3f})-"
        f"({normalized_box['x_max']:.3f},{normalized_box['y_max']:.3f})"
    )
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_x = max(0, min(left, width - text_width - 8))
    label_y = max(0, top - text_height - 10)
    draw.rectangle(
        (label_x, label_y, label_x + text_width + 8, label_y + text_height + 6),
        fill=(0, 0, 0, 210),
    )
    draw.text((label_x + 4, label_y + 3), label, fill=accent, font=font)

    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()