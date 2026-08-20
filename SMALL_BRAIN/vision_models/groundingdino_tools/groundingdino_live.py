#!/usr/bin/env python3
"""
Interactive GroundingDINO OpenVINO runner.

The model is compiled once when this program starts. Enter as many image paths
as you want. Type q or quit to stop the process.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from PIL import Image
from torchvision.ops import nms

import services.groundingdino_openvino_onnx as gd


class LiveGroundingDINO:
    def __init__(
        self,
        repo: Path,
        config: Path,
        model: Path,
        device: str = "GPU",
        cache_dir: Path = Path(".ov_cache_onnx"),
        performance_hint: str = "LATENCY",
        warmup: int = 1,
    ) -> None:
        self.repo = repo.expanduser().resolve()
        self.config_path = config.expanduser().resolve()
        self.model_path = model.expanduser().resolve()
        self.device = device.upper()

        gd.add_repo_to_path(self.repo)

        from groundingdino.util.utils import get_phrases_from_posmap

        self.get_phrases_from_posmap = get_phrases_from_posmap

        metadata_path = self.model_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.height = int(metadata["height"])
        self.width = int(metadata["width"])
        self.text_len = int(metadata["text_len"])

        _, self.tokenizer = gd.load_config_tokenizer(self.config_path)

        self.core = ov.Core()
        print("Available OpenVINO devices:")
        for available_device in self.core.available_devices:
            try:
                full_name = self.core.get_property(
                    available_device, "FULL_DEVICE_NAME"
                )
            except Exception:
                full_name = ""
            print(f"  {available_device}: {full_name}")

        if self.device.startswith("GPU") and not any(
            name.startswith("GPU") for name in self.core.available_devices
        ):
            raise RuntimeError("No Intel GPU was detected by OpenVINO.")

        cache_dir = cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(cache_dir)})

        ov_model = self.core.read_model(str(self.model_path))

        print(f"Compiling {self.model_path.name} for {self.device} once...")
        started = time.perf_counter()
        self.compiled_model = self.core.compile_model(
            ov_model,
            self.device,
            {"PERFORMANCE_HINT": performance_hint.upper()},
        )
        self.request = self.compiled_model.create_infer_request()
        print(f"Ready in {time.perf_counter() - started:.2f}s")

        if warmup > 0:
            # Warm up using a blank image and short prompt.
            blank = Image.new("RGB", (self.width, self.height))
            inputs, _ = self._prepare_inputs(blank, "object .")
            for _ in range(warmup):
                self.request.infer(inputs, share_inputs=False)
            print("Warm-up complete.")

    def _prepare_inputs(self, image: Image.Image, prompt: str):
        caption = gd.normalize_caption(prompt)
        original_size = image.size

        image_tensor = gd.preprocess_image(image, self.height, self.width)
        text_inputs = gd.prepare_text_inputs(
            self.tokenizer,
            caption,
            self.text_len,
        )

        inputs = {
            "samples": image_tensor.numpy(),
            "input_ids": text_inputs[0].numpy(),
            "attention_mask": text_inputs[1].numpy(),
            "position_ids": text_inputs[2].numpy(),
            "token_type_ids": text_inputs[3].numpy(),
            "text_self_attention_masks": text_inputs[4].numpy(),
        }
        return inputs, (caption, original_size)

    def detect(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.80,
    ) -> None:
        image_path = image_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        inputs, context = self._prepare_inputs(image, prompt)
        caption, (original_width, original_height) = context

        started = time.perf_counter()
        self.request.infer(inputs, share_inputs=False)
        latency_ms = (time.perf_counter() - started) * 1000.0

        pred_logits = np.asarray(
            self.request.get_tensor("pred_logits").data
        )[0]
        pred_boxes = np.asarray(
            self.request.get_tensor("pred_boxes").data
        )[0]

        # Numerically stable sigmoid.
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(pred_logits, -50.0, 50.0))
        )
        scores = probabilities.max(axis=1)
        keep = scores > box_threshold

        probabilities = probabilities[keep]
        pred_boxes = pred_boxes[keep]
        scores = scores[keep]

        phrase_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.text_len,
        )

        labels: list[str] = []
        for row in probabilities:
            phrase = self.get_phrases_from_posmap(
                torch.from_numpy(row > text_threshold),
                phrase_tokens,
                self.tokenizer,
            )
            labels.append(phrase or "object")

        boxes = gd.cxcywh_to_xyxy(
            pred_boxes,
            original_width,
            original_height,
        )

        if len(boxes):
            keep_indices = nms(
                torch.from_numpy(boxes.astype(np.float32)),
                torch.from_numpy(scores.astype(np.float32)),
                nms_threshold,
            ).numpy()
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = [labels[int(index)] for index in keep_indices]

        gd.draw_detections(
            image_path,
            output_path,
            boxes,
            labels,
            scores,
        )

        print(f"\nInference: {latency_ms:.1f} ms on {self.device}")
        print(f"Detections: {len(boxes)}")
        for label, score, box in zip(labels, scores, boxes):
            print(
                f"  {label:24s} score={float(score):.3f} "
                f"box={box.round(1).tolist()}"
            )
        print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--prompt", default="object .")
    parser.add_argument("--cache-dir", type=Path, default=Path(".ov_cache_onnx"))
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.80)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    detector = LiveGroundingDINO(
        repo=args.repo,
        config=args.config,
        model=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        warmup=args.warmup,
    )

    current_prompt = args.prompt

    print("\nThe model is now staying alive on the GPU.")
    print("Enter an image path. Type q or quit to stop.")

    while True:
        try:
            raw_image = input("\nImage path: ").strip()
        except EOFError:
            break

        if raw_image.lower() in {"q", "quit", "exit"}:
            break
        if not raw_image:
            continue

        image_path = Path(raw_image).expanduser()
        if not image_path.is_file():
            print(f"Image not found: {image_path}")
            continue

        raw_prompt = input(f"Prompt [{current_prompt}]: ").strip()
        if raw_prompt:
            current_prompt = raw_prompt

        default_output = image_path.with_name(
            f"{image_path.stem}_detected.jpg"
        )
        raw_output = input(f"Output [{default_output}]: ").strip()
        output_path = (
            Path(raw_output).expanduser()
            if raw_output
            else default_output
        )

        try:
            detector.detect(
                image_path=image_path,
                prompt=current_prompt,
                output_path=output_path,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                nms_threshold=args.nms_threshold,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")

    print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
