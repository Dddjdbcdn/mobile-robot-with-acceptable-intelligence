from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np
import openvino as ov
import matplotlib

try:
    from PIL import Image
except ImportError:  # PIL input remains optional.
    Image = None  # type: ignore[assignment]


ResizeMode = Literal["stretch", "letterbox"]
BoxXYXY = Sequence[float]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DepthStatistics:
    minimum_m: float | None
    maximum_m: float | None
    mean_m: float | None
    median_m: float | None
    percentile_05_m: float | None
    percentile_95_m: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "minimum_m": self.minimum_m,
            "maximum_m": self.maximum_m,
            "mean_m": self.mean_m,
            "median_m": self.median_m,
            "percentile_05_m": self.percentile_05_m,
            "percentile_95_m": self.percentile_95_m,
        }


class DepthResult:
    """One metric-depth inference result.

    ``depth_m`` is a HxW float32 array aligned with the original input image.
    NumPy/OpenCV inputs are interpreted as BGR/BGRA, matching cv2 conventions.
    """

    def __init__(
        self,
        *,
        request_id: str,
        image_width: int,
        image_height: int,
        model_width: int,
        model_height: int,
        max_depth_m: float,
        resize_mode: ResizeMode,
        preprocessing_latency_ms: float,
        inference_latency_ms: float,
        postprocessing_latency_ms: float,
        total_latency_ms: float,
        depth_m: np.ndarray,
        output_directory: str | None = None,
    ) -> None:
        if depth_m.ndim != 2:
            raise ValueError(f"depth_m must be HxW, got {depth_m.shape}.")

        self.request_id = request_id
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.model_width = int(model_width)
        self.model_height = int(model_height)
        self.max_depth_m = float(max_depth_m)
        self.resize_mode = resize_mode
        self.preprocessing_latency_ms = float(preprocessing_latency_ms)
        self.inference_latency_ms = float(inference_latency_ms)
        self.postprocessing_latency_ms = float(postprocessing_latency_ms)
        self.total_latency_ms = float(total_latency_ms)
        self.depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)
        self.output_directory = output_directory

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth_m.shape

    @property
    def center_depth_m(self) -> float | None:
        return self.depth_at(self.image_width // 2, self.image_height // 2, radius=5)

    @property
    def statistics(self) -> DepthStatistics:
        values = self._valid_values(self.depth_m)
        if values.size == 0:
            return DepthStatistics(None, None, None, None, None, None)

        return DepthStatistics(
            minimum_m=float(np.min(values)),
            maximum_m=float(np.max(values)),
            mean_m=float(np.mean(values)),
            median_m=float(np.median(values)),
            percentile_05_m=float(np.percentile(values, 5.0)),
            percentile_95_m=float(np.percentile(values, 95.0)),
        )

    def depth_at(
        self,
        x: int,
        y: int,
        *,
        radius: int = 3,
        percentile: float = 50.0,
    ) -> float | None:
        """Return robust depth around an original-image pixel.

        radius=0 samples one pixel. A small patch median is usually more stable.
        """
        if not 0 <= x < self.image_width or not 0 <= y < self.image_height:
            raise ValueError(
                f"Point ({x}, {y}) is outside image bounds "
                f"0..{self.image_width - 1}, 0..{self.image_height - 1}."
            )
        if radius < 0:
            raise ValueError("radius must be >= 0.")
        self._validate_percentile(percentile)

        x1 = max(0, int(x) - radius)
        y1 = max(0, int(y) - radius)
        x2 = min(self.image_width, int(x) + radius + 1)
        y2 = min(self.image_height, int(y) + radius + 1)

        values = self._valid_values(self.depth_m[y1:y2, x1:x2])
        if values.size == 0:
            return None
        return float(np.percentile(values, percentile))

    def depth_in_box(
        self,
        box_xyxy: BoxXYXY | None = None,
        *,
        box_xywh: BoxXYWH | None = None,
        percentile: float = 50.0,
        inset_ratio: float = 0.10,
    ) -> float | None:
        """Return depth within a detection box in original-image coordinates.

        Must provide exactly one of `box_xyxy` or `box_xywh`.
        
        ``percentile=50`` gives the median. Values around 20-35 can be useful
        when the closest visible portion of an obstacle matters more.
        ``inset_ratio`` removes noisy box edges/background.
        """
        if (box_xyxy is None) == (box_xywh is None):
            raise ValueError("You must provide exactly one of 'box_xyxy' or 'box_xywh'.")

        self._validate_percentile(percentile)
        if not 0.0 <= inset_ratio < 0.5:
            raise ValueError("inset_ratio must be in [0.0, 0.5).")

        # Standardize to x1, y1, x2, y2 coordinates
        if box_xyxy is not None:
            if len(box_xyxy) != 4:
                raise ValueError("box_xyxy must contain exactly four values.")
            x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        else:
            if len(box_xywh) != 4:
                raise ValueError("box_xywh must contain exactly four values.")
            x, y, w, h = (float(v) for v in box_xywh)
            x1, y1, x2, y2 = x, y, x + w, y + h

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate box coordinates (x1, y1, x2, y2): {(x1, y1, x2, y2)}")

        box_w = x2 - x1
        box_h = y2 - y1
        x1 += box_w * inset_ratio
        x2 -= box_w * inset_ratio
        y1 += box_h * inset_ratio
        y2 -= box_h * inset_ratio

        left = max(0, min(self.image_width - 1, math.floor(x1)))
        top = max(0, min(self.image_height - 1, math.floor(y1)))
        right = max(left + 1, min(self.image_width, math.ceil(x2)))
        bottom = max(top + 1, min(self.image_height, math.ceil(y2)))

        values = self._valid_values(self.depth_m[top:bottom, left:right])
        if values.size == 0:
            return None
            
        return float(np.percentile(values, percentile))

    def depth_in_box_centered(
    self,
    box_xyxy: BoxXYXY,
    *,
    inset_ratio: float = 0.10,
    center_ratio: float = 0.35,
    relative_tolerance: float = 0.20,
    absolute_tolerance_m: float = 0.10,
    min_pixels: int = 30,
    ) -> float | None:
        x1, y1, x2, y2 = map(float, box_xyxy)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid box: {box_xyxy}")

        box_w, box_h = x2 - x1, y2 - y1
        x1 += box_w * inset_ratio
        x2 -= box_w * inset_ratio
        y1 += box_h * inset_ratio
        y2 -= box_h * inset_ratio

        left = max(0, min(self.image_width - 1, math.floor(x1)))
        top = max(0, min(self.image_height - 1, math.floor(y1)))
        right = max(left + 1, min(self.image_width, math.ceil(x2)))
        bottom = max(top + 1, min(self.image_height, math.ceil(y2)))

        roi = self.depth_m[top:bottom, left:right]
        h, w = roi.shape

        seed_w = max(3, int(w * center_ratio))
        seed_h = max(3, int(h * center_ratio))
        sx1 = max(0, (w - seed_w) // 2)
        sy1 = max(0, (h - seed_h) // 2)
        sx2 = min(w, sx1 + seed_w)
        sy2 = min(h, sy1 + seed_h)

        seed_values = self._valid_values(roi[sy1:sy2, sx1:sx2])
        if seed_values.size < min_pixels:
            return None

        seed_depth = float(np.median(seed_values))
        tolerance = max(
            absolute_tolerance_m,
            seed_depth * relative_tolerance,
        )

        valid = np.isfinite(roi) & (roi > 0.0)
        similar = valid & (np.abs(roi - seed_depth) <= tolerance)

        mask = similar.astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        if count <= 1:
            return None

        center_labels = labels[sy1:sy2, sx1:sx2]
        center_labels = center_labels[center_labels > 0]
        if center_labels.size == 0:
            return None

        frequencies = np.bincount(center_labels)
        selected_label = int(np.argmax(frequencies[1:]) + 1)

        if stats[selected_label, cv2.CC_STAT_AREA] < min_pixels:
            return None

        object_values = self._valid_values(
            roi[labels == selected_label]
        )
        if object_values.size < min_pixels:
            return None

        return float(np.median(object_values))

    def to_dict(self, *, include_depth: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "request_id": self.request_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_width": self.model_width,
            "model_height": self.model_height,
            "max_depth_m": self.max_depth_m,
            "resize_mode": self.resize_mode,
            "preprocessing_latency_ms": self.preprocessing_latency_ms,
            "inference_latency_ms": self.inference_latency_ms,
            "postprocessing_latency_ms": self.postprocessing_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "center_depth_m": self.center_depth_m,
            "statistics": self.statistics.to_dict(),
            "depth_shape": list(self.depth_m.shape),
            "depth_dtype": str(self.depth_m.dtype),
            "output_directory": self.output_directory,
        }
        if include_depth:
            result["depth_m"] = self.depth_m.tolist()
        return result

    @staticmethod
    def _valid_values(array: np.ndarray) -> np.ndarray:
        return array[np.isfinite(array) & (array > 0.0)]

    @staticmethod
    def _validate_percentile(percentile: float) -> None:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be in [0, 100].")


# ---------------------------------------------------------
# EXTERNAL SERVICE API (what other files call)
# ---------------------------------------------------------


class DepthAnythingService:
    """Async interface around one persistent DepthAnythingRuntime.

    The runtime and InferRequest live on one dedicated worker thread. Calls are
    serialized because one InferRequest cannot be used concurrently.
    """

    def __init__(
        self,
        *,
        model: Path,
        max_depth_m: float = 20.0,
        device: str = "GPU",
        cache_dir: Path = Path("__pycache__/.ov_cache_depth_anything"),
        performance_hint: str = "LATENCY",
        warmup: int = 0,
        resize_mode: ResizeMode = "letterbox",
        depth_scale: float = 1.0,
        depth_offset_m: float = 0.0,
        clip_depth: bool = True,
    ) -> None:
        self._runtime_args = {
            "model": model,
            "max_depth_m": max_depth_m,
            "device": device,
            "cache_dir": cache_dir,
            "performance_hint": performance_hint,
            "warmup": warmup,
            "resize_mode": resize_mode,
            "depth_scale": depth_scale,
            "depth_offset_m": depth_offset_m,
            "clip_depth": clip_depth,
        }
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="depth-anything",
        )
        self._runtime: DepthAnythingRuntime | None = None
        self._startup_task: asyncio.Task[None] | None = None
        self._inference_lock = asyncio.Lock()
        self._closed = False

    def start_background(self) -> asyncio.Task[None]:
        if self._closed:
            raise RuntimeError("Depth Anything service is closed.")
        if self._startup_task is None:
            self._startup_task = asyncio.create_task(
                self._initialize(),
                name="depth-anything-initialize",
            )
        return self._startup_task

    async def _initialize(self) -> None:
        loop = asyncio.get_running_loop()
        self._runtime = await loop.run_in_executor(
            self._executor,
            partial(DepthAnythingRuntime, **self._runtime_args),
        )

    async def wait_until_ready(self) -> None:
        if self._closed:
            raise RuntimeError("Depth Anything service is closed.")
        await self.start_background()
        if self._runtime is None:
            raise RuntimeError("Depth Anything initialized without a runtime.")

    async def detect(
        self,
        image_source: Any,
        *,
        output_root: Path | str | None = None,
    ) -> DepthResult:
        """Enqueue one metric-depth inference and return an aligned depth map."""
        await self.wait_until_ready()

        async with self._inference_lock:
            if self._closed or self._runtime is None:
                raise RuntimeError("Depth Anything service is not available.")

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                partial(
                    self._runtime.detect,
                    image_source,
                    output_root=output_root,
                ),
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._startup_task is not None:
            try:
                await self._startup_task
            except Exception:
                pass

        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
        self._runtime = None


# ---------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ResizeTransform:
    mode: ResizeMode
    source_width: int
    source_height: int
    model_width: int
    model_height: int
    content_x: int
    content_y: int
    content_width: int
    content_height: int

    def restore_depth(self, model_depth: np.ndarray) -> np.ndarray:
        if model_depth.shape != (self.model_height, self.model_width):
            raise ValueError(
                f"Model depth shape {model_depth.shape} does not match "
                f"{(self.model_height, self.model_width)}."
            )

        if self.mode == "letterbox":
            x1 = self.content_x
            y1 = self.content_y
            x2 = x1 + self.content_width
            y2 = y1 + self.content_height
            model_depth = model_depth[y1:y2, x1:x2]

        restored = cv2.resize(
            model_depth,
            (self.source_width, self.source_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.ascontiguousarray(restored, dtype=np.float32)


class DepthAnythingRuntime:
    def __init__(
        self,
        *,
        model: Path,
        max_depth_m: float = 20.0,
        device: str = "GPU",
        cache_dir: Path = Path(".ov_cache_depth_anything"),
        performance_hint: str = "LATENCY",
        warmup: int = 1,
        resize_mode: ResizeMode = "letterbox",
        depth_scale: float = 1.0,
        depth_offset_m: float = 0.0,
        clip_depth: bool = True,
    ) -> None:
        self.model_path = Path(model).expanduser().resolve()
        self.device = device.upper()
        self.max_depth_m = float(max_depth_m)
        self.resize_mode = resize_mode
        self.depth_scale = float(depth_scale)
        self.depth_offset_m = float(depth_offset_m)
        self.clip_depth = bool(clip_depth)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"OpenVINO model not found: {self.model_path}")
        if self.model_path.suffix.lower() != ".xml":
            raise ValueError("model must point to an OpenVINO .xml IR file.")
        if self.max_depth_m <= 0.0:
            raise ValueError("max_depth_m must be positive.")
        if resize_mode not in ("stretch", "letterbox"):
            raise ValueError("resize_mode must be 'stretch' or 'letterbox'.")
        if warmup < 0:
            raise ValueError("warmup must be >= 0.")

        self.core = ov.Core()
        self._validate_device()

        resolved_cache = Path(cache_dir).expanduser().resolve()
        resolved_cache.mkdir(parents=True, exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(resolved_cache)})

        ov_model = self.core.read_model(str(self.model_path))
        self.model_height, self.model_width = self._read_input_shape(ov_model)

        self.compiled_model = self.core.compile_model(
            ov_model,
            self.device,
            {"PERFORMANCE_HINT": performance_hint.upper()},
        )
        self.input_port = self.compiled_model.input(0)
        self.output_port = self.compiled_model.output(0)
        self.request = self.compiled_model.create_infer_request()

        if warmup:
            warmup_input = np.zeros(
                (1, 3, self.model_height, self.model_width),
                dtype=np.float32,
            )
            for _ in range(warmup):
                self.request.infer(
                    {self.input_port: warmup_input},
                    share_inputs=False,
                )

    def detect(
        self,
        image_source: Any,
        *,
        output_root: Path | str | None = None,
    ) -> DepthResult:
        total_started = time.perf_counter()

        preprocess_started = time.perf_counter()
        image_bgr = self._load_image(image_source)
        image_height, image_width = image_bgr.shape[:2]
        model_input, resize_transform = self._prepare_input(image_bgr)
        preprocessing_latency_ms = (
            time.perf_counter() - preprocess_started
        ) * 1000.0

        inference_started = time.perf_counter()
        self.request.infer(
            {self.input_port: model_input},
            share_inputs=False,
        )
        inference_latency_ms = (
            time.perf_counter() - inference_started
        ) * 1000.0

        postprocess_started = time.perf_counter()
        raw_output = np.array(
            self.request.get_tensor(self.output_port).data,
            dtype=np.float32,
            copy=True,
        )
        model_depth = np.squeeze(raw_output)
        if model_depth.ndim != 2:
            raise RuntimeError(
                f"Unexpected OpenVINO depth output shape: {raw_output.shape}"
            )
        if model_depth.shape != (self.model_height, self.model_width):
            raise RuntimeError(
                f"Depth output shape {model_depth.shape} does not match model "
                f"input spatial shape {(self.model_height, self.model_width)}."
            )
        if not np.isfinite(model_depth).all():
            raise RuntimeError("Depth Anything returned NaN or infinite values.")

        depth_m = resize_transform.restore_depth(model_depth)
        depth_m = depth_m * self.depth_scale + self.depth_offset_m
        if self.clip_depth:
            depth_m = np.clip(depth_m, 0.0, self.max_depth_m)
        depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)

        postprocessing_latency_ms = (
            time.perf_counter() - postprocess_started
        ) * 1000.0
        total_latency_ms = (time.perf_counter() - total_started) * 1000.0

        request_id = uuid.uuid4().hex
        run_directory: Path | None = None
        if output_root is not None:
            run_directory = Path(output_root)

        result = DepthResult(
            request_id=request_id,
            image_width=image_width,
            image_height=image_height,
            model_width=self.model_width,
            model_height=self.model_height,
            max_depth_m=self.max_depth_m,
            resize_mode=self.resize_mode,
            preprocessing_latency_ms=preprocessing_latency_ms,
            inference_latency_ms=inference_latency_ms,
            postprocessing_latency_ms=postprocessing_latency_ms,
            total_latency_ms=total_latency_ms,
            depth_m=depth_m,
            output_directory=str(run_directory) if run_directory else None,
        )

        if run_directory is not None:
            self._store_result(image_bgr, result, run_directory)

        return result

    def _validate_device(self) -> None:
        available = set(self.core.available_devices)
        device_root = self.device.split(".", maxsplit=1)[0]
        virtual_device = device_root in {"AUTO", "MULTI", "HETERO"}
        if (
            not virtual_device
            and self.device not in available
            and device_root not in available
        ):
            raise RuntimeError(
                f"OpenVINO device {self.device!r} is unavailable. "
                f"Available devices: {sorted(available)}"
            )

    @staticmethod
    def _read_input_shape(ov_model: ov.Model) -> tuple[int, int]:
        if len(ov_model.inputs) != 1:
            raise ValueError(
                f"Expected a one-input Depth Anything model, got "
                f"{len(ov_model.inputs)} inputs."
            )

        partial_shape = ov_model.input(0).partial_shape
        if not partial_shape.is_static:
            raise ValueError(
                "This service expects a static OpenVINO model input shape."
            )

        shape = list(partial_shape.to_shape())
        if len(shape) != 4:
            raise ValueError(f"Expected NCHW input, got shape {shape}.")
        batch, channels, height, width = (int(v) for v in shape)
        if batch != 1 or channels != 3:
            raise ValueError(
                f"Expected input [1, 3, H, W], got {shape}."
            )
        return height, width

    @staticmethod
    def _load_image(image_source: Any) -> np.ndarray:
        if isinstance(image_source, (bytes, bytearray, memoryview)):
            encoded = np.frombuffer(bytes(image_source), dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise ValueError("Could not decode image bytes.")
            return DepthAnythingRuntime._to_bgr(decoded)

        if Image is not None and isinstance(image_source, Image.Image):
            rgb = np.asarray(image_source.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image_source, np.ndarray):
            return DepthAnythingRuntime._to_bgr(image_source)

        image_path = Path(image_source).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        decoded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"Could not decode image: {image_path}")
        return DepthAnythingRuntime._to_bgr(decoded)

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.size == 0:
            raise ValueError("Image is empty.")

        if array.dtype != np.uint8:
            array = array.astype(np.float32, copy=False)
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                raise ValueError("Image contains no finite values.")
            if float(np.max(finite)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)

        if array.ndim == 2:
            result = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        elif array.ndim == 3 and array.shape[2] == 1:
            result = cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif array.ndim == 3 and array.shape[2] == 3:
            # NumPy input follows OpenCV convention: it is already BGR.
            result = array
        elif array.ndim == 3 and array.shape[2] == 4:
            result = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(
                "NumPy image must be HxW, HxWx1, HxWx3 BGR, or HxWx4 BGRA."
            )

        return np.ascontiguousarray(result)

    def _prepare_input(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[np.ndarray, _ResizeTransform]:
        source_height, source_width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0

        if self.resize_mode == "stretch":
            network_rgb = cv2.resize(
                rgb,
                (self.model_width, self.model_height),
                interpolation=cv2.INTER_CUBIC,
            )
            transform = _ResizeTransform(
                mode="stretch",
                source_width=source_width,
                source_height=source_height,
                model_width=self.model_width,
                model_height=self.model_height,
                content_x=0,
                content_y=0,
                content_width=self.model_width,
                content_height=self.model_height,
            )
        else:
            scale = min(
                self.model_width / source_width,
                self.model_height / source_height,
            )
            content_width = max(
                1,
                min(self.model_width, int(round(source_width * scale))),
            )
            content_height = max(
                1,
                min(self.model_height, int(round(source_height * scale))),
            )
            content_x = (self.model_width - content_width) // 2
            content_y = (self.model_height - content_height) // 2

            resized = cv2.resize(
                rgb,
                (content_width, content_height),
                interpolation=cv2.INTER_CUBIC,
            )
            # Mean-colored padding becomes zero after normalization.
            network_rgb = np.empty(
                (self.model_height, self.model_width, 3),
                dtype=np.float32,
            )
            network_rgb[...] = _IMAGENET_MEAN
            network_rgb[
                content_y : content_y + content_height,
                content_x : content_x + content_width,
            ] = resized

            transform = _ResizeTransform(
                mode="letterbox",
                source_width=source_width,
                source_height=source_height,
                model_width=self.model_width,
                model_height=self.model_height,
                content_x=content_x,
                content_y=content_y,
                content_width=content_width,
                content_height=content_height,
            )

        normalized = (network_rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        nchw = np.transpose(normalized, (2, 0, 1))[None]
        return np.ascontiguousarray(nchw, dtype=np.float32), transform

    def _store_result(
        self,
        image_bgr: np.ndarray,
        result: DepthResult,
        run_directory: Path,
    ) -> None:
        run_directory.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(run_directory / "input.jpg"), image_bgr):
            raise OSError(f"Could not write input image to {run_directory}")

        np.save(run_directory / "depth_meters.npy", result.depth_m)

        depth = np.asarray(result.depth_m, dtype=np.float32)
        finite = np.isfinite(depth)
        depth_min = float(depth[finite].min()) if finite.any() else 0.0
        depth_max = float(depth[finite].max()) if finite.any() else 0.0
        preview_u8 = np.zeros(depth.shape, dtype=np.uint8)
        if depth_max > depth_min:
            preview_u8[finite] = ((depth[finite] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        preview_rgb = (matplotlib.colormaps["Spectral"](preview_u8)[:, :, :3] * 255).astype(np.uint8)
        preview_color = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(run_directory / "depth_preview.png"), preview_color):
            raise OSError(f"Could not write depth preview to {run_directory}")

        # Millimetre PNG is lossless and convenient for indoor models <=65.535m.
        if self.max_depth_m <= 65.535:
            depth_mm = np.rint(result.depth_m * 1000.0)
            depth_mm = np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)
            if not cv2.imwrite(
                str(run_directory / "depth_millimeters_u16.png"),
                depth_mm,
            ):
                raise OSError(
                    f"Could not write millimetre depth to {run_directory}"
                )

        (run_directory / "result.json").write_text(
            json.dumps(result.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )
