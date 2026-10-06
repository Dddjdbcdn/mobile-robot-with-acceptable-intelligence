"""OpenVINO person re-identification inference, independent of tracking policy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
import threading

import cv2
import numpy as np


DEFAULT_MODEL_NAME = "person-reidentification-retail-0288"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "vision_models"
    / "reid_tools"
    / DEFAULT_MODEL_NAME
    / "FP16"
    / f"{DEFAULT_MODEL_NAME}.xml"
)


class PersonReIDService:
    """Extract normalized appearance embeddings from BGR person crops.

    The service deliberately knows nothing about track IDs, enrollment views,
    or match thresholds. Those policies live in
    ``actions.tracking.person_embeddings``.

    ``inference_callable`` is primarily a lightweight test/benchmark seam. It
    receives the preprocessed NCHW tensor and must return one embedding.
    """

    INPUT_HEIGHT = 256
    INPUT_WIDTH = 128

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "GPU",
        cache_dir: str | Path | None = None,
        inference_callable: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.model_name = DEFAULT_MODEL_NAME
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH).expanduser()
        self.device = str(device).upper()
        self._inference_callable = inference_callable
        self._request = None
        self._input_port = None
        self._output_port = None
        self._inference_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        if inference_callable is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Person ReID model is missing: {self.model_path}. "
                "Run `SMALL_BRAIN/venv/bin/python "
                "SMALL_BRAIN/vision_models/reid_tools/download_model.py` "
                "from the repository root, or pass model_path."
            )

        import openvino as ov

        core = ov.Core()
        if cache_dir is not None:
            resolved_cache = Path(cache_dir).expanduser().resolve()
            resolved_cache.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(resolved_cache)})
        model = core.read_model(str(self.model_path))
        compiled_model = core.compile_model(
            model,
            self.device,
            {"PERFORMANCE_HINT": "LATENCY"},
        )
        self._request = compiled_model.create_infer_request()
        self._input_port = compiled_model.input(0)
        self._output_port = compiled_model.output(0)

    @staticmethod
    def _box_values(
        bbox: Mapping[str, object] | Sequence[float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float, float]:
        value = bbox.get("bbox", bbox) if isinstance(bbox, Mapping) else bbox
        if isinstance(value, Mapping):
            normalized_names = (
                "normalized_x1",
                "normalized_y1",
                "normalized_x2",
                "normalized_y2",
            )
            pixel_names = ("x1", "y1", "x2", "y2")
            if all(name in value for name in normalized_names):
                x1, y1, x2, y2 = (
                    float(value[name]) for name in normalized_names
                )
                return (
                    x1 * frame_width,
                    y1 * frame_height,
                    x2 * frame_width,
                    y2 * frame_height,
                )
            if all(name in value for name in pixel_names):
                return tuple(float(value[name]) for name in pixel_names)
            raise ValueError("bbox mapping must contain normalized or pixel xyxy")
        if len(value) != 4:
            raise ValueError("bbox sequence must contain x1, y1, x2, y2")
        return tuple(float(item) for item in value)

    @classmethod
    def crop_person(
        cls,
        frame_bgr: np.ndarray,
        bbox: Mapping[str, object] | Sequence[float] | None = None,
        padding: float = 0.0,
    ) -> np.ndarray:
        """Return a clipped person crop; normalized detection boxes are valid."""
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a NumPy array")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape HxWx3")
        height, width = frame_bgr.shape[:2]
        if height < 1 or width < 1:
            raise ValueError("frame_bgr cannot be empty")
        if bbox is None:
            return np.ascontiguousarray(frame_bgr)

        x1, y1, x2, y2 = cls._box_values(bbox, width, height)
        if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must be finite and have positive area")
        padding = float(padding)
        if not np.isfinite(padding) or padding < 0.0:
            raise ValueError("padding must be a finite non-negative fraction")
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        left = max(0, int(np.floor(x1 - pad_x)))
        top = max(0, int(np.floor(y1 - pad_y)))
        right = min(width, int(np.ceil(x2 + pad_x)))
        bottom = min(height, int(np.ceil(y2 + pad_y)))
        if right <= left or bottom <= top:
            raise ValueError("bbox does not intersect the frame")
        return np.ascontiguousarray(frame_bgr[top:bottom, left:right])

    @classmethod
    def preprocess(cls, person_bgr: np.ndarray) -> np.ndarray:
        """Prepare the model's raw BGR, NCHW, 1x3x256x128 input."""
        crop = cls.crop_person(person_bgr)
        resized = cv2.resize(
            crop,
            (cls.INPUT_WIDTH, cls.INPUT_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.ascontiguousarray(
            resized.transpose(2, 0, 1)[None], dtype=np.float32
        )

    @staticmethod
    def normalize_embedding(embedding: np.ndarray | Sequence[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("embedding must contain finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("embedding norm must be greater than zero")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        if self._inference_callable is not None:
            return np.asarray(self._inference_callable(tensor))
        result = self._request.infer({self._input_port: tensor})
        return np.asarray(result[self._output_port])

    def extract_embedding(
        self,
        frame_bgr: np.ndarray,
        bbox: Mapping[str, object] | Sequence[float] | None = None,
        padding: float = 0.0,
    ) -> np.ndarray:
        """Synchronously extract one L2-normalized appearance embedding."""
        crop = self.crop_person(frame_bgr, bbox=bbox, padding=padding)
        tensor = self.preprocess(crop)
        with self._inference_lock:
            output = self._infer(tensor)
        return self.normalize_embedding(output)

    def extract_embeddings(
        self,
        frame_bgr: np.ndarray,
        bboxes: Iterable[Mapping[str, object] | Sequence[float]],
        padding: float = 0.0,
    ) -> list[np.ndarray]:
        """Extract embeddings for multiple detections from the same frame."""
        return [
            self.extract_embedding(frame_bgr, bbox, padding)
            for bbox in bboxes
        ]

    async def extract_embedding_async(
        self,
        frame_bgr: np.ndarray,
        bbox: Mapping[str, object] | Sequence[float] | None = None,
        padding: float = 0.0,
    ) -> np.ndarray:
        """Run inference without blocking the caller's asyncio event loop."""
        async with self._async_lock:
            return await asyncio.to_thread(
                self.extract_embedding,
                frame_bgr,
                bbox,
                padding,
            )
