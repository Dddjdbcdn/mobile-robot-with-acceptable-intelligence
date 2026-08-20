import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.6")


class Localization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool = Field(description="Whether the target is clearly visible.")
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)
    point_x: float = Field(
        ge=0.0,
        le=1.0,
        description="A point on the target itself, inside the bounding box.",
    )
    point_y: float = Field(
        ge=0.0,
        le=1.0,
        description="A point on the target itself, inside the bounding box.",
    )
    confidence: float = Field(ge=0.0, le=1.0)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_image(image_path: Path) -> tuple[Any, str]:
    """Load once, then send the exact decoded pixels that will be annotated."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode image as PNG.")

    encoded = base64.b64encode(png.tobytes()).decode("ascii")
    return image, f"data:image/png;base64,{encoded}"


def locate_object(
    image: Any,
    image_data_url: str,
    target: str,
    model: str,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    height, width = image.shape[:2]

    prompt = f"""
Locate exactly one visible object matching: {target!r}

The supplied image is exactly {width} pixels wide and {height} pixels high.
Inspect the entire image carefully.

Return a tight bounding box and one point on the target itself.

Rules:
- All coordinates are normalized from 0.0 to 1.0.
- Top-left is x=0, y=0; bottom-right is x=1, y=1.
- Coordinates refer to the full supplied image without cropping.
- Bound only the visible part of the requested target.
- point_x and point_y must be inside the box and on the target.
- Prefer the visual center of the target's visible region.
- For a hand, put the point near the center of the visible palm/hand region
  and exclude the forearm from the bounding box.
- If several objects match, choose the clearest, most prominent instance.
- If the target is not clearly visible, set found=false and all coordinates
  to 0.0.
""".strip()

    client = OpenAI(
        api_key=api_key,
        timeout=120.0,
        max_retries=2,
    )

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a precise visual grounding system. Localize the "
                    "requested visible object carefully and do not guess."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "original",
                    },
                ],
            },
        ],
        text_format=Localization,
    )

    raw = response.output_parsed
    if raw is None:
        raise RuntimeError(
            "No structured localization was returned. "
            f"Raw output: {response.output_text!r}"
        )

    confidence = clamp01(raw.confidence)
    if not raw.found:
        return {
            "found": False,
            "x": 0.0,
            "y": 0.0,
            "confidence": confidence,
            "box": {
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": 0.0,
                "y_max": 0.0,
            },
            "point_source": "not_found",
        }

    x_min, x_max = sorted((clamp01(raw.x_min), clamp01(raw.x_max)))
    y_min, y_max = sorted((clamp01(raw.y_min), clamp01(raw.y_max)))
    point_x = clamp01(raw.point_x)
    point_y = clamp01(raw.point_y)

    if x_min <= point_x <= x_max and y_min <= point_y <= y_max:
        x, y = point_x, point_y
        point_source = "model_point"
    else:
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        point_source = "bounding_box_center"

    return {
        "found": True,
        "x": x,
        "y": y,
        "confidence": confidence,
        "box": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "point_source": point_source,
    }


def to_pixel(value: float, size: int) -> int:
    return round(clamp01(value) * max(0, size - 1))


def draw_result(
    image: Any,
    output_path: Path,
    result: dict[str, Any],
    radius: int,
    draw_box: bool,
) -> tuple[int, int]:
    height, width = image.shape[:2]
    x = to_pixel(result["x"], width)
    y = to_pixel(result["y"], height)
    radius = max(1, radius)

    annotated = image.copy()

    cv2.circle(
        annotated,
        (x, y),
        radius,
        (0, 255, 0),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        annotated,
        (x, y),
        radius,
        (0, 80, 0),
        thickness=max(1, radius // 5),
        lineType=cv2.LINE_AA,
    )

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not save image: {output_path}")

    return x, y


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Locate an object with GPT vision and draw a green dot."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("target", help='Example: "my visible hand"')
    parser.add_argument("--output", type=Path)
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draw-box", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        return 2

    if not 0.0 <= args.min_confidence <= 1.0:
        print("Error: --min-confidence must be from 0 to 1.", file=sys.stderr)
        return 2

    output_path = args.output or args.image.with_name(
        f"{args.image.stem}_dotted.png"
    )

    try:
        image, data_url = load_image(args.image)
        result = locate_object(image, data_url, args.target, args.model)

        print("Model result:")
        print(json.dumps(result, indent=2))

        if not result["found"]:
            print(f'Object not found: "{args.target}"')
            return 1

        if result["confidence"] < args.min_confidence:
            print(
                f"Confidence {result['confidence']:.2f} is below "
                f"{args.min_confidence:.2f}; no image was written."
            )
            return 1

        pixel_x, pixel_y = draw_result(
            image,
            output_path,
            result,
            max(1, args.radius),
            args.draw_box,
        )

        print(f"Dot location: ({pixel_x}, {pixel_y})")
        print(f"Normalized: ({result['x']:.6f}, {result['y']:.6f})")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Point source: {result['point_source']}")
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