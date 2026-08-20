"""Fast object localization with GPT-5.6 Luna.

The script downsizes the image before upload, asks only for a bounding box,
and derives the green dot from the box center. Normalized coordinates remain
valid for the original image because the upload resize preserves aspect ratio.
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
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.6-luna")


class BoundingBox(BaseModel):
    """Small schema means less generation and parsing work."""

    model_config = ConfigDict(extra="forbid")

    found: bool = Field(description="Whether the requested object is visible.")
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_original(image_path: Path) -> Any:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return image


def resize_for_upload(image: Any, max_side: int) -> Any:
    """Resize without cropping so normalized coordinates still line up."""
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


def locate_object(
    *,
    client: OpenAI,
    image_data_url: str,
    target: str,
    model: str,
    detail: Literal["low", "auto", "original"],
    reasoning_effort: Literal["none", "low"],
) -> dict[str, Any]:
    prompt = (
        f"Locate one visible object matching {target!r}. "
        "Return its tight visible bounding box in normalized coordinates. "
        "Use x=0 at left, y=0 at top, x=1 at right, y=1 at bottom. "
        "Do not crop or change the coordinate frame. "
        "If no clear match exists, set found=false and all coordinates to 0."
    )

    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=250,
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
        text_format=BoundingBox,
    )
    api_seconds = time.perf_counter() - started

    box = response.output_parsed
    if box is None:
        raise RuntimeError(
            "The model returned no parsed bounding box. "
            f"Raw output: {response.output_text!r}"
        )

    if not box.found:
        return {
            "found": False,
            "x": 0.0,
            "y": 0.0,
            "box": {
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": 0.0,
                "y_max": 0.0,
            },
            "api_seconds": api_seconds,
        }

    x_min, x_max = sorted((clamp01(box.x_min), clamp01(box.x_max)))
    y_min, y_max = sorted((clamp01(box.y_min), clamp01(box.y_max)))

    # A deterministic center is faster and often more stable than asking the
    # model for a second localization output.
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0

    return {
        "found": True,
        "x": x,
        "y": y,
        "box": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "api_seconds": api_seconds,
    }


def to_pixel(value: float, size: int) -> int:
    return round(clamp01(value) * max(0, size - 1))


def draw_result(
    *,
    image: Any,
    output_path: Path,
    result: dict[str, Any],
    radius: int,
    draw_box: bool,
) -> tuple[int, int]:
    height, width = image.shape[:2]
    pixel_x = to_pixel(result["x"], width)
    pixel_y = to_pixel(result["y"], height)
    radius = max(1, radius)

    annotated = image.copy()

    if draw_box:
        box = result["box"]
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

    return pixel_x, pixel_y


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quickly locate an object and draw a green dot."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("target", help='Example: "my visible hand"')
    parser.add_argument("--output", type=Path)
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--draw-box", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--detail",
        choices=("low", "auto", "original"),
        default="low",
        help="low is fastest; original is more accurate for small objects.",
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
        default=80,
        help="Upload JPEG quality from 30 to 95.",
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
        help="Retries add reliability but can increase worst-case latency.",
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
        f"{args.image.stem}_dotted.png"
    )

    total_started = time.perf_counter()

    try:
        original = load_original(args.image)
        upload = resize_for_upload(original, args.max_side)
        data_url = image_to_data_url(upload, args.jpeg_quality)

        original_height, original_width = original.shape[:2]
        upload_height, upload_width = upload.shape[:2]

        client = OpenAI(
            api_key=api_key,
            timeout=max(1.0, args.timeout),
            max_retries=max(0, args.retries),
        )

        result = locate_object(
            client=client,
            image_data_url=data_url,
            target=args.target,
            model=args.model,
            detail=args.detail,
            reasoning_effort=args.reasoning,
        )

        print("Model result:")
        print(json.dumps(result, indent=2))
        print(
            f"Image sent: {upload_width}x{upload_height} "
            f"(original {original_width}x{original_height})"
        )

        if not result["found"]:
            print(f'Object not found: "{args.target}"')
            print(f"API time: {result['api_seconds']:.3f}s")
            return 1

        pixel_x, pixel_y = draw_result(
            image=original,
            output_path=output_path,
            result=result,
            radius=args.radius,
            draw_box=args.draw_box,
        )

        total_seconds = time.perf_counter() - total_started
        print(f"Dot location: ({pixel_x}, {pixel_y})")
        print(f"Normalized: ({result['x']:.6f}, {result['y']:.6f})")
        print(f"API time: {result['api_seconds']:.3f}s")
        print(f"Total time: {total_seconds:.3f}s")
        print(f"Saved: {output_path}")
        return 0

    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())