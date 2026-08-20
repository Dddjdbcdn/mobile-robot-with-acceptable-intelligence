#!/usr/bin/env python3
"""
GroundingDINO -> static ONNX -> OpenVINO IR, plus OpenVINO inference.

This avoids the TorchScript SliceAssign/ScatterNDUpdate graph that can fail
during Intel GPU compilation with:
    PullReshapeThroughReduce ... Axis 1 out of tensor rank range

Target repository:
    https://github.com/wenyi5608/GroundingDINO
    branch: wenyi5608-openvino

Examples
--------
Export:
python groundingdino_openvino_onnx.py export \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --weights "$PWD/weights/groundingdino_swint_ogc.pth" \
  --output "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --height 512 --width 768 --text-len 32

Inference:
python groundingdino_openvino_onnx.py infer \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --model "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --image "$PWD/images/row_0_1.jpg" \
  --prompt "guitar ." \
  --device GPU \
  --output "$PWD/result.jpg"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


INPUT_NAMES = [
    "samples",
    "input_ids",
    "attention_mask",
    "position_ids",
    "token_type_ids",
    "text_self_attention_masks",
]
OUTPUT_NAMES = ["pred_logits", "pred_boxes"]


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


def export_model(args: argparse.Namespace) -> None:
    import numpy as np
    import onnx
    import openvino as ov
    import torch

    repo = Path(args.repo)
    config_path = Path(args.config).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    onnx_path = output_path.with_suffix(".onnx")

    add_repo_to_path(repo)

    from groundingdino.models import build_model
    from groundingdino.util.utils import clean_state_dict

    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    cfg, tokenizer = load_config_tokenizer(config_path)
    cfg.device = "cpu"
    cfg.use_checkpoint = False
    cfg.use_transformer_ckpt = False

    print("Building PyTorch GroundingDINO on CPU...")
    model = build_model(cfg)
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(clean_state_dict(state), strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing checkpoint keys")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected checkpoint keys")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    caption = normalize_caption(args.trace_prompt)
    text_inputs = prepare_text_inputs(tokenizer, caption, args.text_len)
    image = torch.randn(
        1, 3, args.height, args.width, dtype=torch.float32
    )
    dummy_inputs = (image, *text_inputs)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Exporting static ONNX: image=[1,3,{args.height},{args.width}], "
        f"text_len={args.text_len}, opset={args.opset}"
    )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=args.opset,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes=None,
            verbose=False,
        )

    print("Checking ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print("Converting ONNX to OpenVINO IR...")
    ov_model = ov.convert_model(str(onnx_path))

    # Ensure stable tensor names even if a converter modifies port aliases.
    if len(ov_model.inputs) != len(INPUT_NAMES):
        raise RuntimeError(
            f"Expected {len(INPUT_NAMES)} inputs, got {len(ov_model.inputs)}."
        )
    for port, name in zip(ov_model.inputs, INPUT_NAMES):
        port.get_tensor().set_names({name})

    if len(ov_model.outputs) != 2:
        raise RuntimeError(f"Expected 2 outputs, got {len(ov_model.outputs)}.")
    ov_model.outputs[0].get_tensor().set_names({"pred_logits"})
    ov_model.outputs[1].get_tensor().set_names({"pred_boxes"})

    ov.save_model(
        ov_model,
        str(output_path),
        compress_to_fp16=not args.fp32_weights,
    )

    metadata = {
        "conversion": "static_onnx_to_openvino",
        "height": args.height,
        "width": args.width,
        "text_len": args.text_len,
        "opset": args.opset,
        "config": str(config_path),
        "weights": str(weights_path),
        "fp16_weights": not args.fp32_weights,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if not args.keep_onnx:
        onnx_path.unlink(missing_ok=True)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.bin')}")
    print(f"Saved: {output_path.with_suffix('.json')}")
    if args.keep_onnx:
        print(f"Saved: {onnx_path}")


def preprocess_image(pil_image: Any, height: int, width: int) -> Any:
    import groundingdino.datasets.transforms as T
    from torchvision.transforms.functional import InterpolationMode, resize

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
    import numpy as np

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


def draw_detections(
    image_path: Path,
    output_path: Path,
    boxes: Any,
    labels: list[str],
    scores: Any,
) -> None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        text = f"{label} {float(score):.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            text,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write image: {output_path}")


def infer_model(args: argparse.Namespace) -> None:
    import numpy as np
    import openvino as ov
    import torch
    from PIL import Image
    from torchvision.ops import nms

    repo = Path(args.repo)
    config_path = Path(args.config).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    metadata_path = model_path.with_suffix(".json")

    add_repo_to_path(repo)

    from groundingdino.util.utils import get_phrases_from_posmap

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}. "
            "Use the export command from this script."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    height = int(metadata["height"])
    width = int(metadata["width"])
    text_len = int(metadata["text_len"])

    cfg, tokenizer = load_config_tokenizer(config_path)
    caption = normalize_caption(args.prompt)

    pil_image = Image.open(image_path).convert("RGB")
    original_width, original_height = pil_image.size
    image_tensor = preprocess_image(pil_image, height, width)
    text_inputs = prepare_text_inputs(tokenizer, caption, text_len)

    inputs = {
        "samples": image_tensor.numpy(),
        "input_ids": text_inputs[0].numpy(),
        "attention_mask": text_inputs[1].numpy(),
        "position_ids": text_inputs[2].numpy(),
        "token_type_ids": text_inputs[3].numpy(),
        "text_self_attention_masks": text_inputs[4].numpy(),
    }

    core = ov.Core()
    print("Available OpenVINO devices:")
    for device in core.available_devices:
        try:
            full_name = core.get_property(device, "FULL_DEVICE_NAME")
        except Exception:
            full_name = ""
        print(f"  {device}: {full_name}")

    requested = args.device.upper()
    if requested.startswith("GPU") and not any(
        device.startswith("GPU") for device in core.available_devices
    ):
        raise RuntimeError("No Intel GPU was detected by OpenVINO.")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    core.set_property({"CACHE_DIR": str(cache_dir)})

    model = core.read_model(str(model_path))
    print("Static model inputs:")
    for port in model.inputs:
        print(f"  {port.get_any_name()}: {port.get_partial_shape()}")

    print(f"Compiling {model_path.name} for {requested}...")
    start = time.perf_counter()
    compiled = core.compile_model(
        model,
        requested,
        {"PERFORMANCE_HINT": args.performance_hint.upper()},
    )
    print(f"Compile/load time: {time.perf_counter() - start:.2f}s")

    request = compiled.create_infer_request()
    for _ in range(max(0, args.warmup)):
        request.infer(inputs, share_inputs=False)

    start = time.perf_counter()
    request.infer(inputs, share_inputs=False)
    latency_ms = (time.perf_counter() - start) * 1000

    pred_logits = np.asarray(request.get_tensor("pred_logits").data)[0]
    pred_boxes = np.asarray(request.get_tensor("pred_boxes").data)[0]

    probabilities = 1.0 / (1.0 + np.exp(-np.clip(pred_logits, -50.0, 50.0)))
    scores = probabilities.max(axis=1)
    keep = scores > args.box_threshold

    probabilities = probabilities[keep]
    pred_boxes = pred_boxes[keep]
    scores = scores[keep]

    phrase_tokens = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=text_len,
    )
    labels: list[str] = []
    for row in probabilities:
        phrase = get_phrases_from_posmap(
            torch.from_numpy(row > args.text_threshold),
            phrase_tokens,
            tokenizer,
        )
        labels.append(phrase or "object")

    boxes = cxcywh_to_xyxy(
        pred_boxes,
        original_width,
        original_height,
    )

    if len(boxes):
        keep_indices = nms(
            torch.from_numpy(boxes.astype(np.float32)),
            torch.from_numpy(scores.astype(np.float32)),
            args.nms_threshold,
        ).numpy()
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = [labels[int(i)] for i in keep_indices]

    draw_detections(image_path, output_path, boxes, labels, scores)

    print(f"Device: {requested}")
    print(f"Model image shape: 1x3x{height}x{width}")
    print(f"Model text length: {text_len}")
    print(f"Inference latency: {latency_ms:.1f} ms")
    print(f"Detections: {len(boxes)}")
    for label, score, box in zip(labels, scores, boxes):
        print(
            f"  {label:24s} score={float(score):.3f} "
            f"box={box.round(1).tolist()}"
        )
    print(f"Saved annotated image: {output_path}")


def list_devices(_: argparse.Namespace) -> None:
    import openvino as ov

    core = ov.Core()
    for device in core.available_devices:
        try:
            full_name = core.get_property(device, "FULL_DEVICE_NAME")
        except Exception:
            full_name = ""
        print(f"{device}: {full_name}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    devices = commands.add_parser("devices")
    devices.set_defaults(func=list_devices)

    export = commands.add_parser("export")
    export.add_argument("--repo", required=True)
    export.add_argument("--config", required=True)
    export.add_argument("--weights", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--height", type=int, default=512)
    export.add_argument("--width", type=int, default=768)
    export.add_argument(
        "--text-len",
        type=int,
        default=32,
        help="Static BERT token length. 32 is enough for short class prompts.",
    )
    export.add_argument(
        "--trace-prompt",
        default="person . car . dog .",
    )
    export.add_argument("--opset", type=int, default=16)
    export.add_argument("--fp32-weights", action="store_true")
    export.add_argument("--keep-onnx", action="store_true")
    export.set_defaults(func=export_model)

    infer = commands.add_parser("infer")
    infer.add_argument("--repo", required=True)
    infer.add_argument("--config", required=True)
    infer.add_argument("--model", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--device", default="GPU")
    infer.add_argument("--output", default="groundingdino_result.jpg")
    infer.add_argument("--box-threshold", type=float, default=0.30)
    infer.add_argument("--text-threshold", type=float, default=0.25)
    infer.add_argument("--nms-threshold", type=float, default=0.80)
    infer.add_argument("--warmup", type=int, default=1)
    infer.add_argument("--cache-dir", default=".ov_cache_onnx")
    infer.add_argument(
        "--performance-hint",
        choices=("latency", "throughput"),
        default="latency",
    )
    infer.set_defaults(func=infer_model)

    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
