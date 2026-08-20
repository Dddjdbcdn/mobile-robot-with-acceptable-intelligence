

from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Sequence
import uuid

import cv2
import numpy as np
import websockets


DEFAULT_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
REALTIME_URL = "wss://api.openai.com/v1/realtime"
NORM_SIZE = 1000.0

SYSTEM_INSTRUCTIONS = """# Role and objective
You are a high-precision visual localization engine.

# Task
For each request, inspect the supplied image and locate exactly one best
instance of the requested target. Return a tight axis-aligned bounding box by
calling the supplied function exactly once.

# Localization rules
- Use only visible image evidence.
- Distinguish the target from visually similar objects.
- Enclose all visible parts of the chosen instance, with minimal background.
- Coordinates use a 0..1000 system relative to the exact supplied image.
- Top-left is (0, 0); bottom-right is (1000, 1000).
- x1 < x2 and y1 < y2 when found=true.
- If the target is absent or too ambiguous to localize, return found=false.
- Never emit prose instead of the function call.
"""

BOX_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "submit_bounding_box",
    "description": (
        "Submit one tight axis-aligned bounding box for the requested visible "
        "object, using integer coordinates from 0 to 1000 relative to the "
        "supplied image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "found": {
                "type": "boolean",
                "description": "True only when the requested object is visibly localizable.",
            },
            "x1": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Left edge. Use 0 when found=false.",
            },
            "y1": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Top edge. Use 0 when found=false.",
            },
            "x2": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Right edge. Use 0 when found=false.",
            },
            "y2": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Bottom edge. Use 0 when found=false.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Calibrated confidence that this is the requested object.",
            },
            "touches_image_edge": {
                "type": "boolean",
                "description": "True when the visible target is cut off by an image edge.",
            },
        },
        "required": [
            "found",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "touches_image_edge",
        ],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class PixelBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    source: str
    touches_image_edge: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clamp(self, width: int, height: int) -> "PixelBox":
        x1 = float(np.clip(self.x1, 0, max(0, width - 1)))
        y1 = float(np.clip(self.y1, 0, max(0, height - 1)))
        x2 = float(np.clip(self.x2, 1, max(1, width)))
        y2 = float(np.clip(self.y2, 1, max(1, height)))
        if x2 <= x1:
            x2 = min(float(width), x1 + 1.0)
        if y2 <= y1:
            y2 = min(float(height), y1 + 1.0)
        return PixelBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=float(np.clip(self.confidence, 0.0, 1.0)),
            source=self.source,
            touches_image_edge=self.touches_image_edge,
        )

    def to_int_xyxy(self, width: int, height: int) -> tuple[int, int, int, int]:
        box = self.clamp(width, height)
        return (
            int(round(box.x1)),
            int(round(box.y1)),
            int(round(box.x2)),
            int(round(box.y2)),
        )

    def to_normalized(self, width: int, height: int) -> dict[str, float]:
        return {
            "x1": self.x1 / max(width, 1),
            "y1": self.y1 / max(height, 1),
            "x2": self.x2 / max(width, 1),
            "y2": self.y2 / max(height, 1),
        }


@dataclass(frozen=True)
class CropRegion:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class RealtimeAPIError(RuntimeError):
    pass


class RealtimeLocalizer:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        reasoning_effort: str,
        timeout_seconds: float,
        debug: bool,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.debug = debug
        self.ws: Any = None

    async def __aenter__(self) -> "RealtimeLocalizer":
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Safety-Identifier": os.getenv(
                "OPENAI_SAFETY_IDENTIFIER", "local-image-localization-cli"
            ),
        }
        url = f"{REALTIME_URL}?model={self.model}"

        connect_kwargs = {
            "max_size": None,
            "open_timeout": 30,
            "ping_interval": 20,
            "ping_timeout": 60,
            "close_timeout": 10,
        }

        # websockets >= 14 uses additional_headers; older versions use extra_headers.
        try:
            self.ws = await websockets.connect(
                url, additional_headers=headers, **connect_kwargs
            )
        except TypeError:
            self.ws = await websockets.connect(
                url, extra_headers=headers, **connect_kwargs
            )

        await self._wait_for_event("session.created")

        event_id = f"session_{uuid.uuid4().hex}"
        await self._send(
            {
                "event_id": event_id,
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "output_modalities": ["text"],
                    "instructions": SYSTEM_INSTRUCTIONS,
                    "reasoning": {"effort": self.reasoning_effort},
                },
            }
        )
        await self._wait_for_event("session.updated", related_event_id=event_id)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.ws is not None:
            await self.ws.close()

    async def locate(
        self,
        image: np.ndarray,
        *,
        target: str,
        stage: str,
        extra_instruction: str = "",
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        image_url = encode_image_data_url(image)
        height, width = image.shape[:2]

        prompt = build_localization_prompt(
            target=target,
            width=width,
            height=height,
            stage=stage,
            extra_instruction=extra_instruction,
        )

        event = {
            "event_id": f"request_{request_id}",
            "type": "response.create",
            "response": {
                "conversation": "none",
                "metadata": {
                    "request_id": request_id,
                    "stage": stage,
                },
                "output_modalities": ["text"],
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ],
                "tools": [BOX_TOOL],
                "tool_choice": "required",
            },
        }
        await self._send(event)

        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for Realtime localization stage {stage!r}."
                )

            raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            server_event = json.loads(raw)
            event_type = server_event.get("type")

            if self.debug:
                self._debug_event(server_event)

            if event_type == "error":
                self._raise_api_error(server_event)

            if event_type != "response.done":
                continue

            response = server_event.get("response", {})
            metadata = response.get("metadata") or {}
            if metadata.get("request_id") != request_id:
                continue

            if response.get("status") != "completed":
                raise RealtimeAPIError(
                    "Realtime response did not complete: "
                    + json.dumps(response.get("status_details"), ensure_ascii=False)
                )

            function_call = next(
                (
                    item
                    for item in response.get("output", [])
                    if item.get("type") == "function_call"
                    and item.get("name") == "submit_bounding_box"
                ),
                None,
            )
            if function_call is None:
                raise RealtimeAPIError(
                    "The model completed without calling submit_bounding_box."
                )

            try:
                arguments = json.loads(function_call.get("arguments", "{}"))
            except json.JSONDecodeError as error:
                raise RealtimeAPIError(
                    f"Invalid function arguments: {function_call.get('arguments')!r}"
                ) from error

            return normalize_tool_result(arguments)

    async def _send(self, event: dict[str, Any]) -> None:
        if self.debug:
            print(
                "[send]",
                json.dumps(
                    {k: v for k, v in event.items() if k != "response"},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        await self.ws.send(json.dumps(event))

    async def _wait_for_event(
        self, event_type: str, *, related_event_id: str | None = None
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {event_type}.")
            raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            event = json.loads(raw)
            if self.debug:
                self._debug_event(event)
            if event.get("type") == "error":
                self._raise_api_error(event)
            if event.get("type") != event_type:
                continue
            if related_event_id is not None:
                # session.updated may expose the triggering event ID as event_id or
                # may omit it. Accept omission, reject an explicit mismatch.
                observed = event.get("event_id")
                if observed is not None and observed == related_event_id:
                    return event
            return event

    @staticmethod
    def _raise_api_error(event: dict[str, Any]) -> None:
        error = event.get("error") or event
        message = error.get("message", "Unknown Realtime API error")
        code = error.get("code")
        param = error.get("param")
        details = f"{message}"
        if code:
            details += f" [code={code}]"
        if param:
            details += f" [param={param}]"
        raise RealtimeAPIError(details)

    @staticmethod
    def _debug_event(event: dict[str, Any]) -> None:
        summary: dict[str, Any] = {"type": event.get("type")}
        if event.get("type") == "error":
            summary["error"] = event.get("error")
        if event.get("type") == "response.done":
            response = event.get("response", {})
            summary["status"] = response.get("status")
            summary["metadata"] = response.get("metadata")
            summary["output_types"] = [
                item.get("type") for item in response.get("output", [])
            ]
        print("[recv]", json.dumps(summary, ensure_ascii=False), file=sys.stderr)


def build_localization_prompt(
    *,
    target: str,
    width: int,
    height: int,
    stage: str,
    extra_instruction: str,
) -> str:
    return f"""# Target
{target}

# Supplied image
- Pixel size: {width} x {height}
- Stage: {stage}

# Required action
Inspect the image carefully and call submit_bounding_box exactly once.
Select the single clearest instance of the target. Return a TIGHT box around
all visible parts of that same instance, not around nearby supports, shadows,
people, furniture, or empty space unless those are part of the target.

# Coordinate contract
- Coordinates are integers from 0 to 1000 relative to this supplied image.
- x1/y1 are the top-left edge; x2/y2 are the bottom-right edge.
- The box should be as tight as practical while containing the visible target.
- If absent or not reliably distinguishable, set found=false and all edges to 0.

{extra_instruction.strip()}
"""


def normalize_tool_result(arguments: dict[str, Any]) -> dict[str, Any]:
    found = bool(arguments.get("found", False))
    confidence = float(np.clip(safe_float(arguments.get("confidence"), 0.0), 0, 1))
    touches = bool(arguments.get("touches_image_edge", False))

    if not found:
        return {
            "found": False,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 0.0,
            "y2": 0.0,
            "confidence": confidence,
            "touches_image_edge": touches,
        }

    x1 = float(np.clip(safe_float(arguments.get("x1"), 0), 0, NORM_SIZE))
    y1 = float(np.clip(safe_float(arguments.get("y1"), 0), 0, NORM_SIZE))
    x2 = float(np.clip(safe_float(arguments.get("x2"), 0), 0, NORM_SIZE))
    y2 = float(np.clip(safe_float(arguments.get("y2"), 0), 0, NORM_SIZE))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 - x1 < 1 or y2 - y1 < 1:
        return {
            "found": False,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 0.0,
            "y2": 0.0,
            "confidence": 0.0,
            "touches_image_edge": touches,
        }

    return {
        "found": True,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": confidence,
        "touches_image_edge": touches,
    }


def safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def prepare_for_api(
    image: np.ndarray,
    *,
    max_side: int = 2048,
    min_side_for_crop: int = 900,
) -> np.ndarray:
    """Resize predictably before upload while preserving aspect ratio."""
    height, width = image.shape[:2]
    longest = max(width, height)

    scale = 1.0
    if longest > max_side:
        scale = max_side / longest
    elif longest < min_side_for_crop:
        scale = min_side_for_crop / max(longest, 1)

    if abs(scale - 1.0) < 1e-6:
        return image

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=interpolation,
    )


def encode_image_data_url(image: np.ndarray, jpeg_quality: int = 95) -> str:
    prepared = prepare_for_api(image)
    ok, encoded = cv2.imencode(
        ".jpg", prepared, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode an image for the API.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def result_to_pixel_box(
    result: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    source: str,
) -> PixelBox | None:
    if not result.get("found"):
        return None

    x1 = offset_x + result["x1"] / NORM_SIZE * frame_width
    y1 = offset_y + result["y1"] / NORM_SIZE * frame_height
    x2 = offset_x + result["x2"] / NORM_SIZE * frame_width
    y2 = offset_y + result["y2"] / NORM_SIZE * frame_height

    box = PixelBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=result["confidence"],
        source=source,
        touches_image_edge=result.get("touches_image_edge", False),
    )
    if box.width < 2 or box.height < 2:
        return None
    return box


def generate_tiles(
    image: np.ndarray, *, grid: int, overlap: float
) -> list[tuple[CropRegion, np.ndarray, str]]:
    if grid <= 0:
        return []

    height, width = image.shape[:2]
    overlap = float(np.clip(overlap, 0.0, 0.8))
    denominator = grid - (grid - 1) * overlap
    tile_width = min(width, max(1, int(math.ceil(width / denominator))))
    tile_height = min(height, max(1, int(math.ceil(height / denominator))))

    xs = [int(round(value)) for value in np.linspace(0, width - tile_width, grid)]
    ys = [int(round(value)) for value in np.linspace(0, height - tile_height, grid)]

    tiles: list[tuple[CropRegion, np.ndarray, str]] = []
    for row, y1 in enumerate(ys):
        for col, x1 in enumerate(xs):
            region = CropRegion(x1, y1, x1 + tile_width, y1 + tile_height)
            tile = image[region.y1 : region.y2, region.x1 : region.x2].copy()
            tiles.append((region, tile, f"tile-r{row + 1}-c{col + 1}"))
    return tiles


def box_iou(a: PixelBox, b: PixelBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def boxes_related(a: PixelBox, b: PixelBox, image_width: int, image_height: int) -> bool:
    if box_iou(a, b) >= 0.08:
        return True

    acx, acy = a.center
    bcx, bcy = b.center
    diagonal = math.hypot(image_width, image_height)
    center_distance = math.hypot(acx - bcx, acy - bcy) / max(diagonal, 1.0)

    a_contains_b_center = a.x1 <= bcx <= a.x2 and a.y1 <= bcy <= a.y2
    b_contains_a_center = b.x1 <= acx <= b.x2 and b.y1 <= acy <= b.y2
    return center_distance <= 0.12 or a_contains_b_center or b_contains_a_center


def candidate_weight(box: PixelBox) -> float:
    source_multiplier = 0.80 if box.source == "full-frame" else 1.0
    if box.source.startswith("verification"):
        source_multiplier = 1.20
    elif box.source.startswith("refine"):
        source_multiplier = 1.15
    edge_multiplier = 0.75 if box.touches_image_edge else 1.0
    return max(0.03, box.confidence) ** 2 * source_multiplier * edge_multiplier


def cluster_candidates(
    boxes: Sequence[PixelBox], image_width: int, image_height: int
) -> list[list[PixelBox]]:
    clusters: list[list[PixelBox]] = []
    for box in sorted(boxes, key=candidate_weight, reverse=True):
        best_index: int | None = None
        best_affinity = -1.0
        for index, cluster in enumerate(clusters):
            affinity = max(
                box_iou(box, member)
                + (0.2 if boxes_related(box, member, image_width, image_height) else 0.0)
                for member in cluster
            )
            if affinity > best_affinity and any(
                boxes_related(box, member, image_width, image_height)
                for member in cluster
            ):
                best_affinity = affinity
                best_index = index
        if best_index is None:
            clusters.append([box])
        else:
            clusters[best_index].append(box)
    return clusters


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return float(np.median(values))
    threshold = total / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def fuse_boxes(boxes: Sequence[PixelBox], *, source: str) -> PixelBox:
    if not boxes:
        raise ValueError("Cannot fuse an empty list of boxes.")
    weights = [candidate_weight(box) for box in boxes]
    total_weight = sum(weights)
    confidence = (
        sum(box.confidence * weight for box, weight in zip(boxes, weights))
        / max(total_weight, 1e-9)
    )
    # Small agreement bonus, capped to preserve calibration.
    confidence = min(0.99, confidence + 0.025 * max(0, len(boxes) - 1))
    return PixelBox(
        x1=weighted_median([box.x1 for box in boxes], weights),
        y1=weighted_median([box.y1 for box in boxes], weights),
        x2=weighted_median([box.x2 for box in boxes], weights),
        y2=weighted_median([box.y2 for box in boxes], weights),
        confidence=confidence,
        source=source,
        touches_image_edge=any(box.touches_image_edge for box in boxes),
    )


def choose_and_fuse_candidates(
    boxes: Sequence[PixelBox], image_width: int, image_height: int
) -> tuple[PixelBox, list[PixelBox]]:
    if not boxes:
        raise LookupError("No valid localization candidates were returned.")

    clusters = cluster_candidates(boxes, image_width, image_height)

    def cluster_score(cluster: Sequence[PixelBox]) -> float:
        support_bonus = 0.12 * max(0, len(cluster) - 1)
        return sum(candidate_weight(box) for box in cluster) + support_bonus

    winner = max(clusters, key=cluster_score)
    return fuse_boxes(winner, source="search-fusion"), list(winner)


def expand_box_to_crop(
    box: PixelBox,
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    minimum_padding: int = 24,
) -> CropRegion:
    padding_x = max(minimum_padding, int(round(box.width * padding_ratio)))
    padding_y = max(minimum_padding, int(round(box.height * padding_ratio)))

    x1 = max(0, int(math.floor(box.x1 - padding_x)))
    y1 = max(0, int(math.floor(box.y1 - padding_y)))
    x2 = min(image_width, int(math.ceil(box.x2 + padding_x)))
    y2 = min(image_height, int(math.ceil(box.y2 + padding_y)))

    if x2 - x1 < 8 or y2 - y1 < 8:
        raise LookupError("Candidate crop is too small to refine.")
    return CropRegion(x1, y1, x2, y2)


def crop_image(image: np.ndarray, region: CropRegion) -> np.ndarray:
    return image[region.y1 : region.y2, region.x1 : region.x2].copy()


def map_box_from_crop(box: PixelBox, region: CropRegion, source: str) -> PixelBox:
    return PixelBox(
        x1=region.x1 + box.x1,
        y1=region.y1 + box.y1,
        x2=region.x1 + box.x2,
        y2=region.y1 + box.y2,
        confidence=box.confidence,
        source=source,
        touches_image_edge=box.touches_image_edge,
    )


def draw_candidate_overlay(image: np.ndarray, box: PixelBox) -> np.ndarray:
    overlay = image.copy()
    height, width = overlay.shape[:2]
    x1, y1, x2, y2 = box.to_int_xyxy(width, height)
    thickness = max(2, round(max(width, height) / 400))
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), thickness)
    cv2.putText(
        overlay,
        "CANDIDATE",
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        max(1, thickness // 2),
        cv2.LINE_AA,
    )
    return overlay


async def locate_object(
    client: RealtimeLocalizer,
    image: np.ndarray,
    *,
    target: str,
    grid: int,
    overlap: float,
    min_confidence: float,
    refine_attempts: int,
    verify: bool,
    debug: bool,
) -> tuple[PixelBox, dict[str, Any]]:
    height, width = image.shape[:2]
    search_candidates: list[PixelBox] = []
    trace: dict[str, Any] = {"search": [], "refine": [], "verification": None}

    full_result = await client.locate(
        image,
        target=target,
        stage="full-frame-search",
        extra_instruction=(
            "Search the entire image before deciding. The target may be small. "
            "Do not place a box on empty space near the target."
        ),
    )
    trace["search"].append({"source": "full-frame", "result": full_result})
    full_box = result_to_pixel_box(
        full_result,
        frame_width=width,
        frame_height=height,
        source="full-frame",
    )
    if full_box is not None and full_box.confidence >= min_confidence:
        search_candidates.append(full_box)

    for region, tile, tile_name in generate_tiles(image, grid=grid, overlap=overlap):
        tile_result = await client.locate(
            tile,
            target=target,
            stage=tile_name,
            extra_instruction=(
                "This image is an overlapping crop from a larger frame. The target "
                "may be absent or cut by the crop boundary. Do not hallucinate it. "
                "If present, localize the visible instance tightly in THIS crop."
            ),
        )
        trace["search"].append(
            {
                "source": tile_name,
                "region": asdict(region),
                "result": tile_result,
            }
        )
        tile_box = result_to_pixel_box(
            tile_result,
            frame_width=region.width,
            frame_height=region.height,
            offset_x=region.x1,
            offset_y=region.y1,
            source=tile_name,
        )
        if tile_box is not None and tile_box.confidence >= min_confidence:
            search_candidates.append(tile_box)

    coarse_box, winning_cluster = choose_and_fuse_candidates(
        search_candidates, width, height
    )
    coarse_box = coarse_box.clamp(width, height)

    refine_region = expand_box_to_crop(
        coarse_box,
        image_width=width,
        image_height=height,
        padding_ratio=0.80 if coarse_box.touches_image_edge else 0.60,
    )
    refine_crop = crop_image(image, refine_region)

    refined_candidates: list[PixelBox] = []
    for attempt in range(max(1, refine_attempts)):
        result = await client.locate(
            refine_crop,
            target=target,
            stage=f"refine-{attempt + 1}",
            extra_instruction=(
                "A coarse search says the target is likely in this crop. Re-detect "
                "it from visual evidence rather than copying the coarse estimate. "
                "Return the tight visible-object bounds in THIS crop."
            ),
        )
        trace["refine"].append(
            {
                "attempt": attempt + 1,
                "region": asdict(refine_region),
                "result": result,
            }
        )
        local_box = result_to_pixel_box(
            result,
            frame_width=refine_region.width,
            frame_height=refine_region.height,
            source=f"refine-local-{attempt + 1}",
        )
        if local_box is not None and local_box.confidence >= min_confidence:
            refined_candidates.append(
                map_box_from_crop(
                    local_box, refine_region, source=f"refine-{attempt + 1}"
                )
            )

    if refined_candidates:
        # Keep only refinements agreeing with at least the coarse region or with
        # another refinement. This resists a single high-confidence hallucination.
        compatible = [
            box
            for box in refined_candidates
            if boxes_related(box, coarse_box, width, height)
            or any(
                other is not box and boxes_related(box, other, width, height)
                for other in refined_candidates
            )
        ]
        if not compatible:
            compatible = refined_candidates
        final_box = fuse_boxes(compatible, source="refine-fusion").clamp(width, height)
    else:
        final_box = coarse_box

    if verify:
        verify_region = expand_box_to_crop(
            final_box,
            image_width=width,
            image_height=height,
            padding_ratio=0.55,
        )
        verify_crop = crop_image(image, verify_region)
        local_candidate = PixelBox(
            x1=final_box.x1 - verify_region.x1,
            y1=final_box.y1 - verify_region.y1,
            x2=final_box.x2 - verify_region.x1,
            y2=final_box.y2 - verify_region.y1,
            confidence=final_box.confidence,
            source="verification-candidate",
        )
        verify_overlay = draw_candidate_overlay(verify_crop, local_candidate)
        verify_result = await client.locate(
            verify_overlay,
            target=target,
            stage="verification-and-correction",
            extra_instruction=(
                "The magenta rectangle marked CANDIDATE is only a proposal. Inspect "
                "the underlying image independently. Return the corrected tight box "
                "for the target in THIS image. If the proposal is already tight, "
                "return approximately the same bounds. Do not include the magenta "
                "stroke itself as part of the object."
            ),
        )
        trace["verification"] = {
            "region": asdict(verify_region),
            "result": verify_result,
        }
        verified_local = result_to_pixel_box(
            verify_result,
            frame_width=verify_region.width,
            frame_height=verify_region.height,
            source="verification-local",
        )
        if verified_local is not None and verified_local.confidence >= min_confidence:
            verified = map_box_from_crop(
                verified_local, verify_region, source="verification"
            ).clamp(width, height)
            # Accept correction when it still points near the prior target. If it
            # jumps far away, keep the multi-pass fused result.
            if boxes_related(verified, final_box, width, height):
                final_box = fuse_boxes(
                    [final_box, verified, verified], source="verified-fusion"
                ).clamp(width, height)

    trace["winning_search_cluster"] = [asdict(box) for box in winning_cluster]
    trace["coarse_box"] = asdict(coarse_box)
    trace["final_box"] = asdict(final_box)

    if debug:
        print(
            f"[debug] candidates={len(search_candidates)}, "
            f"winning_cluster={len(winning_cluster)}, "
            f"refinements={len(refined_candidates)}",
            file=sys.stderr,
        )

    return final_box, trace


def draw_annotation(
    image: np.ndarray,
    *,
    box: PixelBox,
    target: str,
    draw_box: bool,
    draw_dot: bool,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    x1, y1, x2, y2 = box.to_int_xyxy(width, height)
    thickness = max(2, round(max(width, height) / 350))

    if draw_box:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        label = f"{target} {box.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(1.0, max(width, height) / 1400.0))
        text_thickness = max(1, thickness // 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        label_y1 = max(0, y1 - text_height - baseline - 8)
        label_y2 = min(height, label_y1 + text_height + baseline + 8)
        label_x2 = min(width, x1 + text_width + 10)
        cv2.rectangle(output, (x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)
        cv2.putText(
            output,
            label,
            (x1 + 5, label_y2 - baseline - 4),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    if draw_dot:
        center_x = int(round((x1 + x2) / 2))
        center_y = int(round((y1 + y2) / 2))
        radius = max(5, round(max(width, height) / 100))
        cv2.circle(output, (center_x, center_y), radius, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(
            output,
            (center_x, center_y),
            radius,
            (255, 255, 255),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )

    return output


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError(f"Failed to save image: {path}")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug[:48] or "object"


def default_output_path(image_path: Path, target: str) -> Path:
    suffix = image_path.suffix if image_path.suffix else ".jpg"
    return image_path.with_name(
        f"{image_path.stem}_{slugify(target)}_located{suffix}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate a text-described object with GPT Realtime, then optionally "
            "draw and save its bounding box or center dot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("target", help='Object description, for example "tripod".')
    parser.add_argument("--draw-box", action="store_true", help="Draw a green bounding box.")
    parser.add_argument("--draw-dot", action="store_true", help="Draw a green center dot.")
    parser.add_argument("-o", "--output", type=Path, help="Annotated output image path.")
    parser.add_argument("--json-output", type=Path, help="Optional detailed JSON result path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Realtime model name.")
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default="xhigh",
        help="Realtime reasoning effort. xhigh favors accuracy over latency/cost.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        choices=[0, 2, 3],
        default=2,
        help="Overlapping search grid. 0 disables tiled search; 3 is most thorough.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.20,
        help="Fractional overlap between adjacent search tiles.",
    )
    parser.add_argument(
        "--refine-attempts",
        type=int,
        default=2,
        help="Independent crop-refinement passes.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the final annotated-box verification/correction pass.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.15,
        help="Discard model candidates below this confidence.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each Realtime response.",
    )
    parser.add_argument("--debug", action="store_true", help="Print event diagnostics.")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    if not args.image.is_file():
        print(f"Error: image does not exist: {args.image}", file=sys.stderr)
        return 2

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: OpenCV could not decode: {args.image}", file=sys.stderr)
        return 2

    if args.refine_attempts < 1:
        print("Error: --refine-attempts must be at least 1.", file=sys.stderr)
        return 2
    if not 0 <= args.min_confidence <= 1:
        print("Error: --min-confidence must be between 0 and 1.", file=sys.stderr)
        return 2
    if not 0 <= args.overlap < 0.8:
        print("Error: --overlap must be in [0, 0.8).", file=sys.stderr)
        return 2

    started = time.perf_counter()
    try:
        async with RealtimeLocalizer(
            api_key=api_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout,
            debug=args.debug,
        ) as client:
            box, trace = await locate_object(
                client,
                image,
                target=args.target,
                grid=args.grid,
                overlap=args.overlap,
                min_confidence=args.min_confidence,
                refine_attempts=args.refine_attempts,
                verify=not args.no_verify,
                debug=args.debug,
            )
    except LookupError as error:
        print(f"Target not found: {error}", file=sys.stderr)
        return 3
    except (RealtimeAPIError, TimeoutError, OSError) as error:
        print(f"Realtime localization failed: {error}", file=sys.stderr)
        return 4

    height, width = image.shape[:2]
    pixel_xyxy = box.to_int_xyxy(width, height)
    elapsed = time.perf_counter() - started

    result: dict[str, Any] = {
        "found": True,
        "target": args.target,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "image": str(args.image),
        "image_width": width,
        "image_height": height,
        "box_pixels_xyxy": {
            "x1": pixel_xyxy[0],
            "y1": pixel_xyxy[1],
            "x2": pixel_xyxy[2],
            "y2": pixel_xyxy[3],
        },
        "box_normalized_0_to_1": box.to_normalized(width, height),
        "center_pixels": {
            "x": int(round((pixel_xyxy[0] + pixel_xyxy[2]) / 2)),
            "y": int(round((pixel_xyxy[1] + pixel_xyxy[3]) / 2)),
        },
        "confidence": round(box.confidence, 4),
        "elapsed_seconds": round(elapsed, 3),
    }

    if args.draw_box or args.draw_dot:
        output_path = args.output or default_output_path(args.image, args.target)
        annotated = draw_annotation(
            image,
            box=box,
            target=args.target,
            draw_box=args.draw_box,
            draw_dot=args.draw_dot,
        )
        save_image(output_path, annotated)
        result["output_image"] = str(output_path)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        detailed_result = {**result, "trace": trace}
        args.json_output.write_text(
            json.dumps(detailed_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result["json_output"] = str(args.json_output)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    args = parse_args()
    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        exit_code = 130
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()