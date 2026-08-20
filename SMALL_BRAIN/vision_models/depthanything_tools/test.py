"""Interactive and command-line tester for Depth Anything V2 metric checkpoints.

Run this script from Depth-Anything-V2/metric_depth, where the local
``depth_anything_v2`` package is importable.

Examples
--------
Interactive sampling::

    python test.py \
      --image depth_results/input.jpg \
      --encoder vitb \
      --checkpoint checkpoints/depth_anything_v2_metric_hypersim_vitb.pth \
      --max-depth 20 \
      --input-size 518 \
      --interactive

Command-line samples::

    python test.py \
      --image latest_snapshot.jpg \
      --encoder vitb \
      --checkpoint checkpoints/depth_anything_v2_metric_hypersim_vitb.pth \
      --point 550,360 \
      --box 300,180,900,650

Interactive controls
--------------------
- Left click: add a point sample.
- Left drag: add a bounding-box sample.
- U: undo the last sample.
- C: clear all samples.
- S: save outputs.
- Q or Escape: save outputs and exit.

TO TEST THE RESULT INDEPENDENT FROM THE SCRIPT

python run.py \
  --encoder vitb \
  --load-from checkpoints/depth_anything_v2_metric_hypersim_vitb.pth \
  --max-depth 20 \
  --img-path latest_snapshot.jpg \
  --outdir test_results \
  --input-size 518 \
  --save-numpy \
  --pred-only

"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import matplotlib

from depth_anything_v2.dpt import DepthAnythingV2


# ---------------------------------------------------------
# DEPTH CALIBRATION
# ---------------------------------------------------------
# Calibrated depth = raw model depth * DEPTH_SCALE + DEPTH_OFFSET_M
#
# Example: model predicts 2.0 m for an object actually at 0.5 m:
# DEPTH_SCALE = 0.5 / 2.0 = 0.25
#
# Keep DEPTH_OFFSET_M at 0.0 unless measurements across several known
# distances show that a constant offset improves accuracy.
DEPTH_SCALE = 0.508
DEPTH_OFFSET_M = 0.0


MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


@dataclass
class PointSample:
    kind: str
    index: int
    x: int
    y: int
    exact_depth_m: float
    local_median_depth_m: float | None
    local_radius_px: int


@dataclass
class BoxSample:
    kind: str
    index: int
    box_xyxy: tuple[int, int, int, int]
    sampled_box_xyxy: tuple[int, int, int, int]
    pixel_count: int
    median_depth_m: float
    mean_depth_m: float
    min_depth_m: float
    max_depth_m: float
    percentile_10_m: float
    percentile_25_m: float
    percentile_75_m: float
    percentile_90_m: float


Sample = PointSample | BoxSample


def parse_csv_ints(value: str, expected: int, name: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must contain comma-separated integers: {value!r}"
        ) from exc

    if len(parts) != expected:
        raise argparse.ArgumentTypeError(
            f"{name} expects {expected} values, received {len(parts)}: {value!r}"
        )
    return parts


def parse_point(value: str) -> tuple[int, int]:
    x, y = parse_csv_ints(value, 2, "--point")
    return x, y


def parse_box(value: str) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = parse_csv_ints(value, 4, "--box")
    return x1, y1, x2, y2


def choose_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("--device mps requested, but MPS is unavailable.")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_state_dict_safely(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions lacking weights_only.
        return torch.load(path, map_location="cpu")


def load_model(
    encoder: str,
    checkpoint: Path,
    max_depth_m: float,
    device: str,
) -> DepthAnythingV2:
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported encoder: {encoder}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = DepthAnythingV2(
        **MODEL_CONFIGS[encoder],
        max_depth=max_depth_m,
    )
    model.load_state_dict(load_state_dict_safely(checkpoint))
    model = model.to(device).eval()
    return model


def valid_depth_values(values: np.ndarray, max_depth_m: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values[
        np.isfinite(values)
        & (values > 0.0)
        & (values <= max_depth_m)
    ]


def clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return (
        int(np.clip(x, 0, width - 1)),
        int(np.clip(y, 0, height - 1)),
    )


def normalize_box(
    box: Sequence[int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in box)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    left = int(np.clip(left, 0, width - 1))
    top = int(np.clip(top, 0, height - 1))
    right = int(np.clip(right, left + 1, width))
    bottom = int(np.clip(bottom, top + 1, height))
    return left, top, right, bottom


def inset_box(
    box: tuple[int, int, int, int],
    inset_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if inset_ratio <= 0:
        return box

    dx = int(round((x2 - x1) * inset_ratio))
    dy = int(round((y2 - y1) * inset_ratio))

    inner_x1 = min(x2 - 1, x1 + dx)
    inner_y1 = min(y2 - 1, y1 + dy)
    inner_x2 = max(inner_x1 + 1, x2 - dx)
    inner_y2 = max(inner_y1 + 1, y2 - dy)
    return inner_x1, inner_y1, inner_x2, inner_y2


def sample_point(
    depth_m: np.ndarray,
    x: int,
    y: int,
    radius: int,
    max_depth_m: float,
    index: int,
) -> PointSample:
    height, width = depth_m.shape
    x, y = clamp_point(x, y, width, height)
    exact = float(depth_m[y, x])

    local_median: float | None = None
    if radius > 0:
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(width, x + radius + 1)
        y2 = min(height, y + radius + 1)
        valid = valid_depth_values(depth_m[y1:y2, x1:x2], max_depth_m)
        if valid.size:
            local_median = float(np.median(valid))

    return PointSample(
        kind="point",
        index=index,
        x=x,
        y=y,
        exact_depth_m=exact,
        local_median_depth_m=local_median,
        local_radius_px=radius,
    )


def sample_box(
    depth_m: np.ndarray,
    box: Sequence[int],
    inset_ratio: float,
    max_depth_m: float,
    index: int,
) -> BoxSample:
    height, width = depth_m.shape
    normalized = normalize_box(box, width, height)
    sampled = inset_box(normalized, inset_ratio)
    x1, y1, x2, y2 = sampled
    valid = valid_depth_values(depth_m[y1:y2, x1:x2], max_depth_m)

    if valid.size == 0:
        raise ValueError(f"Box {normalized} contains no valid depth values.")

    return BoxSample(
        kind="box",
        index=index,
        box_xyxy=normalized,
        sampled_box_xyxy=sampled,
        pixel_count=int(valid.size),
        median_depth_m=float(np.median(valid)),
        mean_depth_m=float(np.mean(valid)),
        min_depth_m=float(np.min(valid)),
        max_depth_m=float(np.max(valid)),
        percentile_10_m=float(np.percentile(valid, 10)),
        percentile_25_m=float(np.percentile(valid, 25)),
        percentile_75_m=float(np.percentile(valid, 75)),
        percentile_90_m=float(np.percentile(valid, 90)),
    )


def next_index(samples: Iterable[Sample], kind: str) -> int:
    return 1 + sum(1 for sample in samples if sample.kind == kind)


def point_label(sample: PointSample) -> str:
    if sample.local_median_depth_m is None:
        return f"P{sample.index}: {sample.exact_depth_m:.3f} m"
    return (
        f"P{sample.index}: {sample.exact_depth_m:.3f} m "
        f"(med {sample.local_median_depth_m:.3f})"
    )


def box_label(sample: BoxSample) -> str:
    return f"B{sample.index}: median {sample.median_depth_m:.3f} m"


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )

    x = int(np.clip(origin[0], 0, max(0, image.shape[1] - text_width - 8)))
    y = int(np.clip(origin[1], text_height + 8, image.shape[0] - baseline - 4))

    cv2.rectangle(
        image,
        (x - 4, y - text_height - 6),
        (x + text_width + 4, y + baseline + 4),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_annotations(
    original_bgr: np.ndarray,
    samples: Sequence[Sample],
    preview_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    annotated = original_bgr.copy()

    for sample in samples:
        if isinstance(sample, PointSample):
            color = (0, 255, 255)
            cv2.drawMarker(
                annotated,
                (sample.x, sample.y),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(annotated, (sample.x, sample.y), 5, color, 2, cv2.LINE_AA)
            draw_label(
                annotated,
                point_label(sample),
                (sample.x + 10, sample.y - 12),
                color,
            )
        else:
            color = (0, 255, 0)
            x1, y1, x2, y2 = sample.box_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            sx1, sy1, sx2, sy2 = sample.sampled_box_xyxy
            if sample.sampled_box_xyxy != sample.box_xyxy:
                cv2.rectangle(
                    annotated,
                    (sx1, sy1),
                    (sx2, sy2),
                    (255, 255, 0),
                    1,
                )

            draw_label(
                annotated,
                box_label(sample),
                (x1 + 4, max(20, y1 - 8)),
                color,
            )

    if preview_box is not None:
        x1, y1, x2, y2 = preview_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return annotated


def make_depth_preview(
    depth_m: np.ndarray,
    max_depth_m: float,
) -> np.ndarray:
    """Render depth using the same style as official metric_depth/run.py."""

    depth = np.asarray(depth_m, dtype=np.float32).copy()

    finite = np.isfinite(depth)
    if not finite.any():
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        depth_min = float(depth[finite].min())
        depth_max = float(depth[finite].max())

        # Prevent invalid values from affecting visualization.
        depth[~finite] = depth_min

        if depth_max <= depth_min:
            normalized = np.zeros(depth.shape, dtype=np.uint8)
        else:
            normalized = (
                (depth - depth_min)
                / (depth_max - depth_min)
                * 255.0
            ).astype(np.uint8)

    # Official run.py uses Matplotlib's Spectral colormap.
    cmap = matplotlib.colormaps.get_cmap("Spectral")
    preview_rgb = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

    # OpenCV expects BGR.
    return cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)


def json_safe_sample(sample: Sample) -> dict:
    result = asdict(sample)
    # JSON has no tuple type; make coordinates explicit arrays.
    if "box_xyxy" in result:
        result["box_xyxy"] = list(result["box_xyxy"])
    if "sampled_box_xyxy" in result:
        result["sampled_box_xyxy"] = list(result["sampled_box_xyxy"])
    return result


def save_outputs(
    output_dir: Path,
    image_path: Path,
    original_bgr: np.ndarray,
    depth_m: np.ndarray,
    samples: Sequence[Sample],
    metadata: dict,
    max_depth_m: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = render_annotations(original_bgr, samples)
    depth_preview = make_depth_preview(depth_m, max_depth_m)
    side_by_side = np.hstack((annotated, depth_preview))

    np.save(output_dir / "depth_meters.npy", depth_m.astype(np.float32))
    cv2.imwrite(str(output_dir / "input.jpg"), original_bgr)
    cv2.imwrite(str(output_dir / "annotated_samples.jpg"), annotated)
    cv2.imwrite(str(output_dir / "depth_preview_fixed_scale.png"), depth_preview)
    cv2.imwrite(str(output_dir / "annotated_and_depth.jpg"), side_by_side)

    depth_mm = np.clip(
        np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0,
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    cv2.imwrite(str(output_dir / "depth_millimeters_u16.png"), depth_mm)

    payload = {
        **metadata,
        "image_path": str(image_path.resolve()),
        "image_width": int(original_bgr.shape[1]),
        "image_height": int(original_bgr.shape[0]),
        "depth_shape": list(depth_m.shape),
        "depth_min_m": float(np.nanmin(depth_m)),
        "depth_median_m": float(np.nanmedian(depth_m)),
        "depth_max_m": float(np.nanmax(depth_m)),
        "samples": [json_safe_sample(sample) for sample in samples],
    }
    (output_dir / "samples.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Saved outputs to: {output_dir.resolve()}")


def print_sample(sample: Sample) -> None:
    if isinstance(sample, PointSample):
        message = (
            f"Point P{sample.index} at ({sample.x}, {sample.y}): "
            f"exact={sample.exact_depth_m:.4f} m"
        )
        if sample.local_median_depth_m is not None:
            message += (
                f", median(radius={sample.local_radius_px})="
                f"{sample.local_median_depth_m:.4f} m"
            )
        print(message)
    else:
        print(
            f"Box B{sample.index} {sample.box_xyxy}: "
            f"median={sample.median_depth_m:.4f} m, "
            f"p10={sample.percentile_10_m:.4f} m, "
            f"p25={sample.percentile_25_m:.4f} m, "
            f"mean={sample.mean_depth_m:.4f} m, "
            f"pixels={sample.pixel_count}"
        )


def interactive_sampling(
    original_bgr: np.ndarray,
    depth_m: np.ndarray,
    samples: list[Sample],
    max_depth_m: float,
    point_radius: int,
    box_inset_ratio: float,
    save_callback,
) -> None:
    height, width = original_bgr.shape[:2]
    max_display_width = 1500
    max_display_height = 850
    display_scale = min(
        1.0,
        max_display_width / width,
        max_display_height / height,
    )
    display_width = max(1, int(round(width * display_scale)))
    display_height = max(1, int(round(height * display_scale)))

    state = {
        "drag_start": None,
        "drag_current": None,
    }

    def display_to_image(x: int, y: int) -> tuple[int, int]:
        image_x = int(round(x / display_scale))
        image_y = int(round(y / display_scale))
        return clamp_point(image_x, image_y, width, height)

    def on_mouse(event, x, y, _flags, _param) -> None:
        image_x, image_y = display_to_image(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = (image_x, image_y)
            state["drag_current"] = (image_x, image_y)

        elif event == cv2.EVENT_MOUSEMOVE and state["drag_start"] is not None:
            state["drag_current"] = (image_x, image_y)

        elif event == cv2.EVENT_LBUTTONUP and state["drag_start"] is not None:
            start_x, start_y = state["drag_start"]
            end_x, end_y = image_x, image_y
            state["drag_start"] = None
            state["drag_current"] = None

            movement = math.hypot(end_x - start_x, end_y - start_y)
            if movement < 6:
                sample = sample_point(
                    depth_m,
                    end_x,
                    end_y,
                    point_radius,
                    max_depth_m,
                    next_index(samples, "point"),
                )
            else:
                try:
                    sample = sample_box(
                        depth_m,
                        (start_x, start_y, end_x, end_y),
                        box_inset_ratio,
                        max_depth_m,
                        next_index(samples, "box"),
                    )
                except ValueError as exc:
                    print(f"Could not sample box: {exc}")
                    return

            samples.append(sample)
            print_sample(sample)
            save_callback()

    window_name = "Depth Anything V2 metric sampler"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.setMouseCallback(window_name, on_mouse)

    print()
    print("Interactive controls:")
    print("  Left click  -> point sample")
    print("  Left drag   -> bounding-box median")
    print("  U           -> undo")
    print("  C           -> clear")
    print("  S           -> save")
    print("  Q / Escape  -> save and quit")

    while True:
        preview_box = None
        if state["drag_start"] is not None and state["drag_current"] is not None:
            x1, y1 = state["drag_start"]
            x2, y2 = state["drag_current"]
            preview_box = normalize_box((x1, y1, x2, y2), width, height)

        canvas = render_annotations(original_bgr, samples, preview_box)
        if display_scale != 1.0:
            canvas = cv2.resize(
                canvas,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            save_callback()
            break
        if key == ord("s"):
            save_callback()
        elif key == ord("u"):
            if samples:
                removed = samples.pop()
                print(f"Removed {removed.kind} sample {removed.index}")
                save_callback()
        elif key == ord("c"):
            samples.clear()
            print("Cleared all samples")
            save_callback()

    cv2.destroyWindow(window_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official Depth Anything V2 metric PyTorch model, sample "
            "points/boxes, and save annotated validation outputs."
        )
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--encoder",
        choices=tuple(MODEL_CONFIGS),
        default="vitb",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results/depth_sampler"),
    )
    parser.add_argument(
        "--point",
        type=parse_point,
        action="append",
        default=[],
        metavar="X,Y",
        help="Point to sample. May be supplied multiple times.",
    )
    parser.add_argument(
        "--box",
        type=parse_box,
        action="append",
        default=[],
        metavar="X1,Y1,X2,Y2",
        help="Bounding box whose valid depth median is sampled. Repeatable.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=5,
        help=(
            "Also calculate a local median in this pixel radius around each "
            "point. Set to 0 for exact-pixel only."
        ),
    )
    parser.add_argument(
        "--box-inset-ratio",
        type=float,
        default=0.05,
        help=(
            "Ignore this fraction of each box edge to reduce background "
            "contamination. Use 0 for the entire box."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a window for click/drag sampling.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.input_size <= 0:
        raise ValueError("--input-size must be positive.")
    if args.max_depth <= 0:
        raise ValueError("--max-depth must be positive.")
    if args.point_radius < 0:
        raise ValueError("--point-radius must be non-negative.")
    if not (0.0 <= args.box_inset_ratio < 0.5):
        raise ValueError("--box-inset-ratio must be in [0, 0.5).")

    image_path = args.image.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise RuntimeError(f"OpenCV could not decode image: {image_path}")

    device = choose_device(args.device)
    print(f"Loading {args.encoder} checkpoint on {device}: {checkpoint}")
    model = load_model(args.encoder, checkpoint, args.max_depth, device)

    print(
        f"Running official infer_image(input_size={args.input_size}) on "
        f"{original_bgr.shape[1]}x{original_bgr.shape[0]} image..."
    )
    started = time.perf_counter()
    with torch.inference_mode():
        depth_m = model.infer_image(original_bgr, args.input_size)
    inference_latency_ms = (time.perf_counter() - started) * 1000.0

    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        depth_m = np.squeeze(depth_m)
    if depth_m.ndim != 2:
        raise RuntimeError(f"Unexpected depth output shape: {depth_m.shape}")
    if depth_m.shape != original_bgr.shape[:2]:
        raise RuntimeError(
            "Expected infer_image() to restore original resolution, but got "
            f"depth={depth_m.shape}, image={original_bgr.shape[:2]}"
        )

    # Apply camera-specific metric calibration once. Everything below this
    # point—sampling, visualization, statistics, and saved files—uses the
    # calibrated depth map.
    depth_m = (
        depth_m * np.float32(DEPTH_SCALE)
        + np.float32(DEPTH_OFFSET_M)
    )
    depth_m = np.maximum(depth_m, 0.0).astype(np.float32, copy=False)

    print(
        "Calibration: "
        f"depth = raw * {DEPTH_SCALE:.6f} "
        f"+ {DEPTH_OFFSET_M:.6f} m"
    )
    print(f"Inference latency: {inference_latency_ms:.1f} ms")
    print(
        "Depth stats: "
        f"min={float(np.nanmin(depth_m)):.3f} m, "
        f"median={float(np.nanmedian(depth_m)):.3f} m, "
        f"max={float(np.nanmax(depth_m)):.3f} m"
    )

    samples: list[Sample] = []

    for x, y in args.point:
        sample = sample_point(
            depth_m,
            x,
            y,
            args.point_radius,
            args.max_depth,
            next_index(samples, "point"),
        )
        samples.append(sample)
        print_sample(sample)

    for box in args.box:
        sample = sample_box(
            depth_m,
            box,
            args.box_inset_ratio,
            args.max_depth,
            next_index(samples, "box"),
        )
        samples.append(sample)
        print_sample(sample)

    metadata = {
        "encoder": args.encoder,
        "checkpoint": str(checkpoint),
        "device": device,
        "input_size": args.input_size,
        "max_depth_m": args.max_depth,
        "depth_scale": DEPTH_SCALE,
        "depth_offset_m": DEPTH_OFFSET_M,
        "inference_latency_ms": inference_latency_ms,
        "point_radius_px": args.point_radius,
        "box_inset_ratio": args.box_inset_ratio,
    }

    def save() -> None:
        save_outputs(
            output_dir,
            image_path,
            original_bgr,
            depth_m,
            samples,
            metadata,
            args.max_depth,
        )

    save()

    if args.interactive:
        interactive_sampling(
            original_bgr,
            depth_m,
            samples,
            args.max_depth,
            args.point_radius,
            args.box_inset_ratio,
            save,
        )
    elif not samples:
        print(
            "No samples requested. Add --interactive, --point X,Y, or "
            "--box X1,Y1,X2,Y2."
        )


if __name__ == "__main__":
    main()