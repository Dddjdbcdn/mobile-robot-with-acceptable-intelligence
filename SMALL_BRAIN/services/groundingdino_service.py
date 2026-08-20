import asyncio
import io
import json
import math
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov
import torch
from PIL import Image
from torchvision.ops import nms
from torchvision.transforms.functional import InterpolationMode, resize

# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------

class GroundingDetection:
    def __init__(self, label, score, box_xyxy, tracker_box_xywh):
        self.label = label
        self.score = score
        self.box_xyxy = box_xyxy
        self.tracker_box_xywh = tracker_box_xywh

    def to_dict(self):
        return {
            "label": self.label, "score": self.score,
            "box_xyxy": list(self.box_xyxy), "tracker_box_xywh": list(self.tracker_box_xywh),
        }

class GroundingResult:
    def __init__(self, request_id, target, normalized_caption, image_width, image_height, inference_latency_ms, detections, output_directory=None):
        self.request_id = request_id
        self.target = target
        self.normalized_caption = normalized_caption
        self.image_width = image_width
        self.image_height = image_height
        self.inference_latency_ms = inference_latency_ms
        self.detections = detections
        self.output_directory = output_directory

    @property
    def best(self):
        return self.detections[0] if self.detections else None

    def to_dict(self):
        return {
            "request_id": self.request_id, "target": self.target, "normalized_caption": self.normalized_caption,
            "image_width": self.image_width, "image_height": self.image_height, "inference_latency_ms": self.inference_latency_ms,
            "detections": [d.to_dict() for d in self.detections],
            "best": self.best.to_dict() if self.best else None, "output_directory": self.output_directory,
        }

# ---------------------------------------------------------
# EXTERNAL SERVICE API (What other files call)
# ---------------------------------------------------------

class GroundingDINOService:
    """Async interface around one persistent GroundingDINORuntime."""

    def __init__(self, *, repo, config, model, device="GPU", cache_dir=Path("__pycache__/.ov_cache_onnx"), performance_hint="LATENCY", warmup=1):
        self._runtime_args = {"repo": repo, "config": config, "model": model, "device": device, "cache_dir": cache_dir, "performance_hint": performance_hint, "warmup": warmup}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grounding-dino")
        self._runtime = None
        self._startup_task = None
        self._inference_lock = asyncio.Lock()
        self._closed = False

    def start_background(self):
        if self._closed: raise RuntimeError("GroundingDINO service is closed.")
        if self._startup_task is None:
            self._startup_task = asyncio.create_task(self._initialize(), name="grounding-dino-initialize")
        return self._startup_task

    async def _initialize(self):
        loop = asyncio.get_running_loop()
        self._runtime = await loop.run_in_executor(self._executor, partial(GroundingDINORuntime, **self._runtime_args))

    async def wait_until_ready(self):
        if self._closed: raise RuntimeError("GroundingDINO service is closed.")
        await self.start_background()
        if self._runtime is None: raise RuntimeError("GroundingDINO initialized without a runtime.")

    async def detect(self, image_source, target, *, box_threshold=0.30, text_threshold=0.25, nms_threshold=0.80, output_root=None):
        """Wait for startup if necessary, enqueue one inference, and return."""
        await self.wait_until_ready()
        async with self._inference_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                partial(self._runtime.detect, image_source, target, box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold, output_root=output_root)
            )

    async def close(self):
        if self._closed: return
        self._closed = True
        if self._startup_task is not None:
            try: await self._startup_task
            except Exception: pass
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        self._runtime = None

# ---------------------------------------------------------
# CORE LOGIC (The detection math)
# ---------------------------------------------------------

class GroundingDINORuntime:
    def detect(self, image_source, target, *, box_threshold=0.30, text_threshold=0.25, nms_threshold=0.80, output_root=None):
        target = target.strip()
        if not target: raise ValueError("GroundingDINO target cannot be empty.")

        image = self._load_image(image_source)
        inputs, (caption, (orig_w, orig_h)) = self._prepare_inputs(image, target)

        started = time.perf_counter()
        self.request.infer(inputs, share_inputs=False)
        latency_ms = (time.perf_counter() - started) * 1000.0

        # Retrieve and process outputs
        pred_logits = np.array(self.request.get_tensor("pred_logits").data, copy=True)[0]
        pred_boxes = np.array(self.request.get_tensor("pred_boxes").data, copy=True)[0]

        probabilities = 1.0 / (1.0 + np.exp(-np.clip(pred_logits, -50.0, 50.0)))
        scores = probabilities.max(axis=1)
        
        # Apply confidence thresholds
        mask = scores > box_threshold
        probabilities, pred_boxes, scores = probabilities[mask], pred_boxes[mask], scores[mask]
        labels = []

        if len(pred_boxes):
            # Resolve phrases from tokens
            phrase_tokens = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.text_len)
            for row in probabilities:
                phrase = self.get_phrases_from_posmap(torch.from_numpy(row > text_threshold), phrase_tokens, self.tokenizer)
                labels.append(phrase or target)

            # Convert to xyxy and clip bounds
            boxes = cxcywh_to_xyxy(pred_boxes, orig_w, orig_h).astype(np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

            # Filter degenerate boxes
            valid_mask = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
            boxes, scores = boxes[valid_mask], scores[valid_mask]
            labels = [lbl for lbl, valid in zip(labels, valid_mask) if valid]

            if len(boxes):
                # Non-Maximum Suppression (NMS)
                keep_idx = nms(torch.from_numpy(boxes), torch.from_numpy(scores.astype(np.float32)), nms_threshold).cpu().numpy()
                boxes, scores = boxes[keep_idx], scores[keep_idx]
                labels = [labels[int(i)] for i in keep_idx]

                # Sort descending by score
                sort_idx = np.argsort(-scores)
                boxes, scores = boxes[sort_idx], scores[sort_idx]
                labels = [labels[int(i)] for i in sort_idx]
        else:
            boxes = np.empty((0, 4), dtype=np.float32)

        # Build detection objects
        detections = [
            GroundingDetection(
                label=lbl, score=float(scr), box_xyxy=tuple(float(v) for v in box),
                tracker_box_xywh=self._make_tracker_box(box, orig_w, orig_h)
            ) for lbl, scr, box in zip(labels, scores, boxes)
        ]

        result = GroundingResult(
            request_id=uuid.uuid4().hex, target=target, normalized_caption=caption,
            image_width=orig_w, image_height=orig_h, inference_latency_ms=latency_ms,
            detections=tuple(detections), output_directory=str(output_root) if output_root else None,
        )

        if output_root:
            self._store_result(image, result, output_root)

        return result

    # ---------------------------------------------------------
    # 5. BOILERPLATE & HELPERS
    # ---------------------------------------------------------

    def __init__(self, repo, config, model, device="GPU", cache_dir=Path(".ov_cache_onnx"), performance_hint="LATENCY", warmup=1):
        self.repo, self.config_path, self.model_path = repo.expanduser().resolve(), config.expanduser().resolve(), model.expanduser().resolve()
        self.device = device.upper()
        add_repo_to_path(self.repo)
        
        from groundingdino.util.utils import get_phrases_from_posmap
        self.get_phrases_from_posmap = get_phrases_from_posmap

        metadata = json.loads(self.model_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.height, self.width, self.text_len = int(metadata["height"]), int(metadata["width"]), int(metadata["text_len"])
        _, self.tokenizer = load_config_tokenizer(self.config_path)

        self.core = ov.Core()
        cache_dir = cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(cache_dir)})

        ov_model = self.core.read_model(str(self.model_path))
        self.compiled_model = self.core.compile_model(ov_model, self.device, {"PERFORMANCE_HINT": performance_hint.upper()})
        self.request = self.compiled_model.create_infer_request()

        if warmup > 0:
            inputs, _ = self._prepare_inputs(Image.new("RGB", (self.width, self.height)), "object .")
            for _ in range(warmup):
                self.request.infer(inputs, share_inputs=False)

    @staticmethod
    def _load_image(image_source):
        if isinstance(image_source, (bytes, bytearray, memoryview)):
            image = Image.open(io.BytesIO(bytes(image_source)))
            image.load()
            return image.convert("RGB")
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB").copy()
        if isinstance(image_source, np.ndarray):
            if image_source.ndim == 2: return Image.fromarray(image_source).convert("RGB")
            if image_source.shape[2] == 3: return Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
            if image_source.shape[2] == 4: return Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGRA2RGB))
            raise ValueError("NumPy image must have 3 or 4 channels.")
        
        image_path = Path(image_source).expanduser().resolve()
        if not image_path.is_file(): raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path)
        image.load()
        return image.convert("RGB")

    def _prepare_inputs(self, image, prompt):
        caption = normalize_caption(prompt)
        image_tensor = preprocess_image(image, self.height, self.width)
        text_inputs = prepare_text_inputs(self.tokenizer, caption, self.text_len)
        
        inputs = {
            "samples": image_tensor.numpy(), "input_ids": text_inputs[0].numpy(),
            "attention_mask": text_inputs[1].numpy(), "position_ids": text_inputs[2].numpy(),
            "token_type_ids": text_inputs[3].numpy(), "text_self_attention_masks": text_inputs[4].numpy(),
        }
        return inputs, (caption, image.size)

    @staticmethod
    def _make_tracker_box(box_xyxy, image_width, image_height):
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        x1_int, y1_int = max(0, min(image_width - 1, math.floor(x1))), max(0, min(image_height - 1, math.floor(y1)))
        x2_int, y2_int = max(x1_int + 1, min(image_width, math.ceil(x2))), max(y1_int + 1, min(image_height, math.ceil(y2)))
        return (x1_int, y1_int, x2_int - x1_int, y2_int - y1_int)

    @staticmethod
    def _store_result(image, result, run_directory):
        run_directory.mkdir(parents=True, exist_ok=True)
        image.save(run_directory / "input.jpg", format="JPEG", quality=95)
        
        annotated = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for d in result.detections:
            x, y, w, h = d.tracker_box_xywh
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, f"{d.label} {d.score:.3f}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
        cv2.imwrite(str(run_directory / "annotated.jpg"), annotated)
        (run_directory / "detections.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


# HELPER FUNCTIONS

def add_repo_to_path(repo: Path) -> None:
    repo = repo.expanduser().resolve()
    if not (repo / "groundingdino").is_dir():
        raise FileNotFoundError(
            f"{repo} does not contain a groundingdino directory."
        )
    sys.path.insert(0, str(repo))

def normalize_caption(prompt: str) -> str:
    prompt = prompt.strip().lower()
    if not prompt:
        raise ValueError("Prompt must not be empty.")
    if not prompt.endswith("."):
        prompt += "."
    return prompt

def load_config_tokenizer(config_path: Path) -> tuple[Any, Any]:
    from groundingdino.util import get_tokenlizer
    from groundingdino.util.slconfig import SLConfig

    cfg = SLConfig.fromfile(str(config_path))
    tokenizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    return cfg, tokenizer

def prepare_text_inputs(
    tokenizer: Any,
    caption: str,
    text_len: int,
) -> tuple[Any, Any, Any, Any, Any]:
    from groundingdino.models.GroundingDINO.bertwarper import (
        generate_masks_with_special_tokens_and_transfer_map,
    )

    tokenized = tokenizer(
        [caption],
        padding="max_length",
        truncation=True,
        max_length=text_len,
        return_tensors="pt",
    )
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    text_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens, tokenizer
    )

    # Keep every text-related input at exactly the exported sequence length.
    text_masks = text_masks[:, :text_len, :text_len]
    position_ids = position_ids[:, :text_len]

    return (
        tokenized["input_ids"][:, :text_len],
        tokenized["attention_mask"][:, :text_len],
        position_ids,
        tokenized["token_type_ids"][:, :text_len],
        text_masks,
    )

def preprocess_image(pil_image: Any, height: int, width: int) -> Any:
    import groundingdino.datasets.transforms as T

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(pil_image, None)
    image = resize(
        image,
        [height, width],
        interpolation=InterpolationMode.BICUBIC,
    )
    return image.unsqueeze(0).contiguous()

def cxcywh_to_xyxy(boxes: Any, width: int, height: int) -> Any:
    boxes = np.asarray(boxes, dtype=np.float32)
    cx = boxes[:, 0] * width
    cy = boxes[:, 1] * height
    bw = boxes[:, 2] * width
    bh = boxes[:, 3] * height
    result = np.stack(
        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
        axis=1,
    )
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, width - 1)
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, height - 1)
    return result
