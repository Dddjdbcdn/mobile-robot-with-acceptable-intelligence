"""Fast object localization with an interior-point guarantee relative to a mask.

One GPT-5.6 Luna request returns:
  1. a tight visible bounding box, and
  2. a seed pixel clearly inside the object.

OpenCV GrabCut then builds a foreground mask locally. The final point is the
pixel with maximum distance from that mask's boundary, so it is guaranteed to
lie inside the predicted foreground mask. This does not guarantee that the
model selected the semantically correct object; no vision-language model can
provide that absolute guarantee for arbitrary images.

python object_point_locator.py \
  results/search_results/latest_snapshot.jpg \
  "door" \
  --draw-box \
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.6-luna")


class ObjectHint(BaseModel):
    """Compact structured result for one API call."""

    model_config = ConfigDict(extra="forbid")

    found: bool = Field(description="Whether one unambiguous target is visible.")
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)
    seed_x: float = Field(
        ge=0.0,
        le=1.0,
        description="A point clearly inside visible pixels of the target.",
    )
    seed_y: float = Field(
        ge=0.0,
        le=1.0,
        description="A point clearly inside visible pixels of the target.",
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_image(image_path: Path) -> Any:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return image


def resize_for_upload(image: Any, max_side: int) -> Any:
    """Resize without cropping."""
    if max_side <= 0:
        return image

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image

    scale = max_side / longest
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def image_to_data_url(image: Any, jpeg_quality: int) -> str:
    quality = max(30, min(95, int(jpeg_quality)))
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("Could not encode upload image as JPEG.")

    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def request_object_hint(
    *,
    client: OpenAI,
    image_data_url: str,
    target: str,
    model: str,
    detail: Literal["low", "high", "auto", "original"],
    reasoning_effort: Literal["none", "low"],
) -> tuple[ObjectHint, float]:
    prompt = (
        f"Locate exactly one visible object matching {target!r}. "
        "Return a tight box around only its visible pixels and one seed point "
        "that is clearly inside the same object's visible material. "
        "Choose the seed in the thickest solid interior region, far from the "
        "silhouette, holes, gaps between parts, transparent areas, reflections, "
        "and occluding objects. Do not mechanically use the box center. "
        "Coordinates are normalized: x=0 left, y=0 top, x=1 right, y=1 bottom. "
        "The seed must be inside the returned box. If the object is ambiguous, "
        "too small, too thin, or no safe interior seed exists, set found=false "
        "and set every coordinate to 0."
    )

    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort, "context": "current_turn"},
        max_output_tokens=120,
        store=False,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": detail,
                    },
                ],
            }
        ],
        text_format=ObjectHint,
    )
    api_seconds = time.perf_counter() - started

    hint = response.output_parsed
    if hint is None:
        raise RuntimeError(
            "The model returned no parsed localization result. "
            f"Raw output: {response.output_text!r}"
        )
    return hint, api_seconds


def to_pixel(value: float, size: int) -> int:
    return round(clamp01(value) * max(0, size - 1))


def normalized_box_to_pixels(
    hint: ObjectHint,
    width: int,
    height: int,
) -> tuple[int, int, int, int, int, int]:
    x1, x2 = sorted(
        (to_pixel(hint.x_min, width), to_pixel(hint.x_max, width))
    )
    y1, y2 = sorted(
        (to_pixel(hint.y_min, height), to_pixel(hint.y_max, height))
    )
    sx = to_pixel(hint.seed_x, width)
    sy = to_pixel(hint.seed_y, height)

    if x2 - x1 < 2 or y2 - y1 < 2:
        raise RuntimeError("Returned bounding box is too small to refine safely.")
    if not (x1 <= sx <= x2 and y1 <= sy <= y2):
        raise RuntimeError("Returned seed point is outside the bounding box.")

    return x1, y1, x2, y2, sx, sy


def refine_interior_point(
    *,
    image: Any,
    hint: ObjectHint,
    iterations: int,
    context_ratio: float,
    seed_ratio: float,
    min_clearance: float,
) -> dict[str, Any]:
    """Return the deepest point inside the GrabCut component containing the seed."""
    height, width = image.shape[:2]
    x1, y1, x2, y2, sx, sy = normalized_box_to_pixels(
        hint,
        width,
        height,
    )

    box_width = x2 - x1 + 1
    box_height = y2 - y1 + 1
    pad_x = max(2, round(box_width * max(0.0, context_ratio)))
    pad_y = max(2, round(box_height * max(0.0, context_ratio)))
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(width - 1, x2 + pad_x)
    py2 = min(height - 1, y2 + pad_y)

    # Probable background everywhere, probable foreground in the model box,
    # definite foreground around the model's interior seed, and definite
    # background outside the padded context region.
    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = cv2.GC_PR_FGD

    if py1 > 0:
        mask[:py1, :] = cv2.GC_BGD
    if py2 < height - 1:
        mask[py2 + 1 :, :] = cv2.GC_BGD
    if px1 > 0:
        mask[:, :px1] = cv2.GC_BGD
    if px2 < width - 1:
        mask[:, px2 + 1 :] = cv2.GC_BGD

    seed_radius = max(
        1,
        round(min(box_width, box_height) * max(0.005, seed_ratio)),
    )
    cv2.circle(mask, (sx, sy), seed_radius, cv2.GC_FGD, thickness=-1)

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image,
            mask,
            None,
            bg_model,
            fg_model,
            iterCount=max(1, iterations),
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        raise RuntimeError(f"GrabCut refinement failed: {exc}") from exc

    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)

    # Keep only the connected foreground component containing the trusted seed.
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    if count <= 1:
        raise RuntimeError("GrabCut produced no usable foreground component.")

    component_id = int(labels[sy, sx])
    if component_id == 0:
        raise RuntimeError("The seed was rejected by the foreground mask.")

    component = (labels == component_id).astype(np.uint8)

    # Prevent any leaked component from pulling the final point outside the
    # model's tight semantic box.
    box_gate = np.zeros_like(component)
    box_gate[y1 : y2 + 1, x1 : x2 + 1] = 1
    component &= box_gate

    if component[sy, sx] == 0:
        raise RuntimeError("Foreground component no longer contains the seed.")

    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    _, max_distance, _, _ = cv2.minMaxLoc(distance)

    # Distance transforms often have a plateau. Among equally deep pixels, use
    # the one nearest the model's trusted seed rather than an arbitrary edge
    # of the plateau.
    plateau_y, plateau_x = np.where(distance >= max_distance - 0.25)
    if plateau_x.size == 0:
        raise RuntimeError("Distance transform produced no interior candidate.")
    nearest_index = int(
        np.argmin((plateau_x - sx) ** 2 + (plateau_y - sy) ** 2)
    )
    point_x = int(plateau_x[nearest_index])
    point_y = int(plateau_y[nearest_index])

    if max_distance < max(0.0, min_clearance):
        raise RuntimeError(
            "No sufficiently deep interior point was found "
            f"(clearance={max_distance:.2f}px)."
        )
    if component[point_y, point_x] == 0:
        raise RuntimeError("Internal error: selected point is outside the mask.")

    return {
        "found": True,
        "x": point_x / max(1, width - 1),
        "y": point_y / max(1, height - 1),
        "upload_pixel_x": point_x,
        "upload_pixel_y": point_y,
        "mask_clearance_px": float(max_distance),
        "box": {
            "x_min": x1 / max(1, width - 1),
            "y_min": y1 / max(1, height - 1),
            "x_max": x2 / max(1, width - 1),
            "y_max": y2 / max(1, height - 1),
        },
        "seed": {
            "x": sx / max(1, width - 1),
            "y": sy / max(1, height - 1),
        },
        "mask": component * 255,
    }


def map_upload_point_to_original(
    *,
    upload_x: int,
    upload_y: int,
    upload_width: int,
    upload_height: int,
    original_width: int,
    original_height: int,
) -> tuple[int, int]:
    if upload_width <= 1:
        original_x = 0
    else:
        original_x = round(upload_x * (original_width - 1) / (upload_width - 1))

    if upload_height <= 1:
        original_y = 0
    else:
        original_y = round(upload_y * (original_height - 1) / (upload_height - 1))

    return original_x, original_y


def draw_result(
    *,
    image: Any,
    output_path: Path,
    pixel_x: int,
    pixel_y: int,
    radius: int,
    box: dict[str, float],
    draw_box: bool,
) -> None:
    height, width = image.shape[:2]
    annotated = image.copy()
    radius = max(1, radius)

    if draw_box:
        x1 = to_pixel(box["x_min"], width)
        y1 = to_pixel(box["y_min"], height)
        x2 = to_pixel(box["x_max"], width)
        y2 = to_pixel(box["y_max"], height)
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=max(2, radius // 4),
            lineType=cv2.LINE_AA,
        )

    cv2.circle(
        annotated,
        (pixel_x, pixel_y),
        radius,
        (0, 255, 0),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not save image: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Locate an object and place a point inside its predicted mask."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("target", help='Example: "my visible hand"')
    parser.add_argument("--output", type=Path)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--draw-box", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--detail",
        choices=("low", "high", "auto", "original"),
        default="original",
        help="Use original for coordinate-sensitive localization.",
    )
    parser.add_argument(
        "--reasoning",
        choices=("none", "low"),
        default="none",
        help="none is the latency-first setting.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=768,
        help="Resize upload so its longest side is this many pixels; 0 disables.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=82,
        help="Upload JPEG quality from 30 to 95.",
    )
    parser.add_argument(
        "--grabcut-iterations",
        type=int,
        default=2,
        help="Local mask refinement iterations; 2 is usually enough.",
    )
    parser.add_argument(
        "--context-ratio",
        type=float,
        default=0.08,
        help="Extra background context around the model box.",
    )
    parser.add_argument(
        "--seed-ratio",
        type=float,
        default=0.02,
        help="Definite-foreground seed radius relative to the shorter box side.",
    )
    parser.add_argument(
        "--min-clearance",
        type=float,
        default=3.0,
        help="Reject points closer than this many upload pixels to the mask edge.",
    )
    parser.add_argument(
        "--save-mask",
        type=Path,
        help="Optional path for the binary upload-resolution foreground mask.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="API timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Retries improve reliability but increase worst-case latency.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        return 2

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    output_path = args.output or args.image.with_name(
        f"{args.image.stem}_point.png"
    )

    total_started = time.perf_counter()

    try:
        original = load_image(args.image)
        upload = resize_for_upload(original, args.max_side)
        data_url = image_to_data_url(upload, args.jpeg_quality)

        original_height, original_width = original.shape[:2]
        upload_height, upload_width = upload.shape[:2]

        client = OpenAI(
            api_key=api_key,
            timeout=max(1.0, args.timeout),
            max_retries=max(0, args.retries),
        )

        hint, api_seconds = request_object_hint(
            client=client,
            image_data_url=data_url,
            target=args.target,
            model=args.model,
            detail=args.detail,
            reasoning_effort=args.reasoning,
        )

        if not hint.found:
            print(f'Object not safely localizable: "{args.target}"')
            print(f"API time: {api_seconds:.3f}s")
            return 1

        refine_started = time.perf_counter()
        result = refine_interior_point(
            image=upload,
            hint=hint,
            iterations=args.grabcut_iterations,
            context_ratio=args.context_ratio,
            seed_ratio=args.seed_ratio,
            min_clearance=args.min_clearance,
        )
        refine_seconds = time.perf_counter() - refine_started

        pixel_x, pixel_y = map_upload_point_to_original(
            upload_x=result["upload_pixel_x"],
            upload_y=result["upload_pixel_y"],
            upload_width=upload_width,
            upload_height=upload_height,
            original_width=original_width,
            original_height=original_height,
        )

        draw_result(
            image=original,
            output_path=output_path,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            radius=args.radius,
            box=result["box"],
            draw_box=args.draw_box,
        )

        if args.save_mask:
            args.save_mask.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save_mask), result["mask"]):
                raise RuntimeError(f"Could not save mask: {args.save_mask}")

        total_seconds = time.perf_counter() - total_started
        public_result = {
            key: value for key, value in result.items() if key != "mask"
        }
        public_result.update(
            {
                "original_pixel_x": pixel_x,
                "original_pixel_y": pixel_y,
                "api_seconds": api_seconds,
                "refine_seconds": refine_seconds,
                "total_seconds": total_seconds,
                "guarantee": "point is inside the predicted GrabCut foreground mask",
            }
        )

        print(json.dumps(public_result, indent=2))
        print(
            f"Image sent: {upload_width}x{upload_height} "
            f"(original {original_width}x{original_height})"
        )
        print(f"Saved: {output_path}")
        if args.save_mask:
            print(f"Mask saved: {args.save_mask}")
        return 0

    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())