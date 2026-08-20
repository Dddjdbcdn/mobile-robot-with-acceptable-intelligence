"""Persistent asynchronous OpenVINO SAM2 image segmentation service.

This v3 service expects the v2 export format produced by the accompanying
exporter. It selects the best candidate by predicted IoU, resizes low-resolution
logits outside OpenVINO, and makes NumPy input color order explicit.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import openvino as ov
from PIL import Image


@dataclass(frozen=True)
class SAM2EncodedImage:
    request_id: str
    image_width: int
    image_height: int
    encoder_latency_ms: float
    image_embeddings: np.ndarray = field(repr=False)
    high_res_feats_256: np.ndarray = field(repr=False)
    high_res_feats_128: np.ndarray = field(repr=False)
    rgb_image: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class SAM2SegmentationResult:
    request_id: str
    encoded_request_id: str
    image_width: int
    image_height: int
    box_xyxy: tuple[float, float, float, float]
    score: float
    candidate_index: int
    candidate_scores: tuple[float, ...]
    area_pixels: int
    mask_bbox_xyxy: tuple[int, int, int, int] | None
    encoder_latency_ms: float
    decoder_latency_ms: float
    mask: np.ndarray = field(repr=False)
    mask_logits: np.ndarray = field(repr=False)
    output_directory: str | None = None

    @property
    def mask_uint8(self) -> np.ndarray:
        return self.mask.astype(np.uint8) * 255

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "encoded_request_id": self.encoded_request_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "box_xyxy": list(self.box_xyxy),
            "score": self.score,
            "candidate_index": self.candidate_index,
            "candidate_scores": list(self.candidate_scores),
            "area_pixels": self.area_pixels,
            "mask_bbox_xyxy": list(self.mask_bbox_xyxy)
            if self.mask_bbox_xyxy is not None
            else None,
            "encoder_latency_ms": self.encoder_latency_ms,
            "decoder_latency_ms": self.decoder_latency_ms,
            "logit_min": float(self.mask_logits.min()),
            "logit_max": float(self.mask_logits.max()),
            "logit_mean": float(self.mask_logits.mean()),
            "output_directory": self.output_directory,
        }


class SAM2OpenVINOService:
    def __init__(
        self,
        *,
        model_dir: Path,
        device: str = "GPU",
        cache_dir: Path = Path(".ov_cache_sam2_v3"),
        performance_hint: str = "LATENCY",
        warmup: int = 0,
        numpy_color_format: str = "BGR",
    ) -> None:
        self._runtime_args = {
            "model_dir": model_dir,
            "device": device,
            "cache_dir": cache_dir,
            "performance_hint": performance_hint,
            "warmup": warmup,
            "numpy_color_format": numpy_color_format,
        }
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam2-ov")
        self._runtime: SAM2OpenVINORuntime | None = None
        self._startup_task: asyncio.Task | None = None
        self._inference_lock = asyncio.Lock()
        self._closed = False

    def start_background(self):
        if self._closed:
            raise RuntimeError("SAM2 service is closed")
        if self._startup_task is None:
            self._startup_task = asyncio.create_task(self._initialize())
        return self._startup_task

    async def _initialize(self) -> None:
        loop = asyncio.get_running_loop()
        self._runtime = await loop.run_in_executor(
            self._executor, partial(SAM2OpenVINORuntime, **self._runtime_args)
        )

    async def wait_until_ready(self) -> None:
        await self.start_background()
        if self._runtime is None:
            raise RuntimeError("SAM2 initialized without a runtime")

    async def encode(self, image_source: Any) -> SAM2EncodedImage:
        await self.wait_until_ready()
        async with self._inference_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, partial(self._runtime.encode, image_source)
            )

    async def segment_encoded_box(
        self,
        encoded: SAM2EncodedImage,
        box_xyxy: Sequence[float],
        *,
        output_root: Path | None = None,
    ) -> SAM2SegmentationResult:
        await self.wait_until_ready()
        async with self._inference_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                partial(
                    self._runtime.segment_encoded_box,
                    encoded,
                    box_xyxy,
                    output_root=output_root,
                ),
            )

    async def segment_box(
        self,
        image_source: Any,
        box_xyxy: Sequence[float],
        *,
        output_root: Path | None = None,
    ) -> SAM2SegmentationResult:
        encoded = await self.encode(image_source)
        return await self.segment_encoded_box(encoded, box_xyxy, output_root=output_root)

    async def segment_boxes(
        self,
        image_source: Any,
        boxes_xyxy: Iterable[Sequence[float]],
        *,
        output_root: Path | None = None,
    ) -> tuple[SAM2SegmentationResult, ...]:
        encoded = await self.encode(image_source)
        results = []
        for index, box in enumerate(boxes_xyxy):
            item_root = None if output_root is None else Path(output_root) / f"mask_{index:03d}"
            results.append(
                await self.segment_encoded_box(encoded, box, output_root=item_root)
            )
        return tuple(results)

    async def segment_tracker_box(
        self,
        image_source: Any,
        tracker_box_xywh: Sequence[float],
        *,
        output_root: Path | None = None,
    ) -> SAM2SegmentationResult:
        return await self.segment_box(
            image_source,
            tracker_xywh_to_xyxy(tracker_box_xywh),
            output_root=output_root,
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
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        self._runtime = None


class SAM2OpenVINORuntime:
    def __init__(
        self,
        model_dir: Path,
        device: str = "GPU",
        cache_dir: Path = Path(".ov_cache_sam2_v3"),
        performance_hint: str = "LATENCY",
        warmup: int = 0,
        numpy_color_format: str = "BGR",
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        metadata_path = self.model_dir / "sam2_openvino.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"SAM2 metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != "sam2-openvino-box-v2":
            raise RuntimeError(
                "This service requires a v2 re-export. The older decoder bypasses "
                "the native prompt encoder and is not compatible with this fix."
            )
        self.image_size = int(self.metadata["image_size"])
        self.mask_threshold = float(self.metadata.get("mask_threshold", 0.0))
        self.numpy_color_format = numpy_color_format.upper()
        if self.numpy_color_format not in {"BGR", "RGB"}:
            raise ValueError("numpy_color_format must be BGR or RGB")

        core = ov.Core()
        cache_dir = Path(cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
        properties = {"PERFORMANCE_HINT": performance_hint.upper()}
        self.encoder_model = core.compile_model(
            core.read_model(self.model_dir / self.metadata["encoder"]),
            device.upper(),
            properties,
        )
        self.decoder_model = core.compile_model(
            core.read_model(self.model_dir / self.metadata["decoder"]),
            device.upper(),
            properties,
        )
        self.encoder_request = self.encoder_model.create_infer_request()
        self.decoder_request = self.decoder_model.create_infer_request()

        if warmup:
            blank = np.zeros((self.image_size, self.image_size, 3), np.uint8)
            for _ in range(warmup):
                enc = self.encode(blank)
                self.segment_encoded_box(enc, (1, 1, self.image_size - 1, self.image_size - 1))

    def encode(self, image_source: Any) -> SAM2EncodedImage:
        rgb = load_rgb_image(image_source, self.numpy_color_format)
        h, w = rgb.shape[:2]
        tensor = preprocess_image(rgb, self.image_size)
        started = time.perf_counter()
        self.encoder_request.infer({"image": tensor}, share_inputs=False)
        latency = (time.perf_counter() - started) * 1000.0
        return SAM2EncodedImage(
            request_id=uuid.uuid4().hex,
            image_width=w,
            image_height=h,
            encoder_latency_ms=latency,
            image_embeddings=np.array(self.encoder_request.get_tensor("image_embeddings").data, copy=True),
            high_res_feats_256=np.array(self.encoder_request.get_tensor("high_res_feats_256").data, copy=True),
            high_res_feats_128=np.array(self.encoder_request.get_tensor("high_res_feats_128").data, copy=True),
            rgb_image=rgb,
        )

    def segment_box(
        self, image_source: Any, box_xyxy: Sequence[float], *, output_root: Path | None = None
    ) -> SAM2SegmentationResult:
        return self.segment_encoded_box(self.encode(image_source), box_xyxy, output_root=output_root)

    def segment_encoded_box(
        self,
        encoded: SAM2EncodedImage,
        box_xyxy: Sequence[float],
        *,
        output_root: Path | None = None,
    ) -> SAM2SegmentationResult:
        box = clip_box_xyxy(box_xyxy, encoded.image_width, encoded.image_height)
        coords = transform_box_to_model(
            box,
            original_width=encoded.image_width,
            original_height=encoded.image_height,
            image_size=self.image_size,
        )
        inputs = {
            "image_embeddings": encoded.image_embeddings,
            "high_res_feats_256": encoded.high_res_feats_256,
            "high_res_feats_128": encoded.high_res_feats_128,
            "point_coords": coords,
            "point_labels": np.array([[2, 3]], dtype=np.int64),
        }
        started = time.perf_counter()
        self.decoder_request.infer(inputs, share_inputs=False)
        latency = (time.perf_counter() - started) * 1000.0

        low_res = np.array(self.decoder_request.get_tensor("low_res_masks").data, copy=True)
        scores = np.array(self.decoder_request.get_tensor("iou_predictions").data, copy=True)
        low_res = ensure_bchw(low_res)
        score_row = scores.reshape(scores.shape[0], -1)[0]
        candidate_index = int(np.argmax(score_row))
        selected_low_res = low_res[0, candidate_index]
        logits = cv2.resize(
            selected_low_res,
            (encoded.image_width, encoded.image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = logits > self.mask_threshold
        result = SAM2SegmentationResult(
            request_id=uuid.uuid4().hex,
            encoded_request_id=encoded.request_id,
            image_width=encoded.image_width,
            image_height=encoded.image_height,
            box_xyxy=box,
            score=float(score_row[candidate_index]),
            candidate_index=candidate_index,
            candidate_scores=tuple(float(v) for v in score_row),
            area_pixels=int(mask.sum()),
            mask_bbox_xyxy=binary_mask_bbox(mask),
            encoder_latency_ms=encoded.encoder_latency_ms,
            decoder_latency_ms=latency,
            mask=mask,
            mask_logits=logits,
            output_directory=str(output_root) if output_root else None,
        )
        if output_root is not None:
            store_segmentation(encoded.rgb_image, result, Path(output_root))
        return result


def load_rgb_image(image_source: Any, numpy_color_format: str = "BGR") -> np.ndarray:
    if isinstance(image_source, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image_source))) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image_source, Image.Image):
        return np.asarray(image_source.convert("RGB"), dtype=np.uint8).copy()
    if isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            return cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
        if image_source.ndim != 3 or image_source.shape[2] not in (3, 4):
            raise ValueError("NumPy image must be HxW, HxWx3, or HxWx4")
        if image_source.shape[2] == 4:
            code = cv2.COLOR_BGRA2RGB if numpy_color_format == "BGR" else cv2.COLOR_RGBA2RGB
            return cv2.cvtColor(image_source, code)
        if numpy_color_format == "BGR":
            return cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(image_source)
    path = Path(image_source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def preprocess_image(rgb: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    resized /= 255.0
    resized = (resized - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.transpose(resized, (2, 0, 1))[None].copy()


def transform_box_to_model(
    box_xyxy: Sequence[float], *, original_width: int, original_height: int, image_size: int
) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    return np.array(
        [[
            [x1 / original_width * image_size, y1 / original_height * image_size],
            [x2 / original_width * image_size, y2 / original_height * image_size],
        ]],
        dtype=np.float32,
    )


def clip_box_xyxy(
    box_xyxy: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must contain x1,y1,x2,y2")
    x1, y1, x2, y2 = map(float, box_xyxy)
    x1 = np.clip(x1, 0, image_width - 1)
    y1 = np.clip(y1, 0, image_height - 1)
    x2 = np.clip(x2, x1 + 1, image_width)
    y2 = np.clip(y2, y1 + 1, image_height)
    return float(x1), float(y1), float(x2), float(y2)


def ensure_bchw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:
        return array
    if array.ndim == 3:
        return array[:, None, :, :]
    if array.ndim == 2:
        return array[None, None, :, :]
    raise RuntimeError(f"Unexpected mask shape: {array.shape}")


def binary_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def tracker_xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("tracker box must contain x,y,width,height")
    x, y, w, h = map(float, box)
    return x, y, x + w, y + h


def store_segmentation(rgb: np.ndarray, result: SAM2SegmentationResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output / "input.jpg"), bgr)
    cv2.imwrite(str(output / "mask.png"), result.mask_uint8)
    np.save(output / "mask_logits.npy", result.mask_logits)
    overlay = bgr.copy()
    overlay[result.mask] = (
        0.55 * overlay[result.mask] + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    x1, y1, x2, y2 = map(lambda v: int(round(v)), result.box_xyxy)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imwrite(str(output / "overlay.jpg"), overlay)
    (output / "segmentation.json").write_text(json.dumps(result.to_dict(), indent=2))