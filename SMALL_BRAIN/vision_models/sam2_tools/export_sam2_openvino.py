#!/usr/bin/env python3
"""Export SAM2/SAM2.1 image encoder and box decoder to OpenVINO.

Version 3 deliberately follows the current native SAM2ImagePredictor prompt path:
box corners are passed as point labels 2/3 to ``sam_prompt_encoder`` with
``boxes=None``. The prompt encoder then appends SAM2's padding/not-a-point token.
The decoder exports low-resolution logits; resizing to the source image is done
outside OpenVINO for easier parity testing.

Download checkpoint:

mkdir -p checkpoints

wget -O checkpoints/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

Export:

python export_sam2_openvino.py \
  --sam2-repo ./sam2 \
  --config 'configs/sam2.1/sam2.1_hiera_b+.yaml' \
  --checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
  --output-dir ./models/sam2.1_base_plus_openvino_v3_fp16 \
  --fp16 \
  --no-multimask
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import openvino as ov
import torch


class SAM2ImageEncoderExport(torch.nn.Module):
    def __init__(self, predictor) -> None:
        super().__init__()
        self.model = predictor.model
        self.bb_feature_sizes = predictor._bb_feat_sizes

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        backbone_out = self.model.forward_image(image)
        _, vision_features, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_features[-1] = vision_features[-1] + self.model.no_mem_embed

        features = [
            feature.permute(1, 2, 0).view(1, -1, *feature_size)
            for feature, feature_size in zip(
                vision_features[::-1], self.bb_feature_sizes[::-1]
            )
        ][::-1]
        return features[-1], features[0], features[1]


class SAM2BoxMaskDecoderExport(torch.nn.Module):
    """Prompt encoder + mask decoder for one box prompt.

    ``point_coords`` is [1, 2, 2] and ``point_labels`` is [1, 2] with labels
    [2, 3]. We intentionally call the native prompt encoder rather than
    reimplementing positional embeddings. With boxes=None, the native prompt
    encoder appends the required not-a-point padding token.
    """

    def __init__(self, model, *, multimask_output: bool) -> None:
        super().__init__()
        self.model = model
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder
        self.multimask_output = multimask_output

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        high_res_feats_256: torch.Tensor,
        high_res_feats_128: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=[high_res_feats_256, high_res_feats_128],
        )
        return torch.clamp(low_res_masks, -32.0, 32.0), iou_predictions


def add_repo_to_path(repo: Path) -> Path:
    repo = repo.expanduser().resolve()
    if not (repo / "sam2").is_dir():
        raise FileNotFoundError(f"{repo} does not contain the sam2 Python package")
    sys.path.insert(0, str(repo))
    return repo


def set_tensor_names(
    model: ov.Model, input_names: Sequence[str], output_names: Sequence[str]
) -> None:
    if len(model.inputs) != len(input_names):
        raise RuntimeError(
            f"Expected {len(input_names)} inputs, converted model has {len(model.inputs)}"
        )
    if len(model.outputs) != len(output_names):
        raise RuntimeError(
            f"Expected {len(output_names)} outputs, converted model has {len(model.outputs)}"
        )
    for port, name in zip(model.inputs, input_names):
        port.get_tensor().set_names({name})
    for port, name in zip(model.outputs, output_names):
        port.get_tensor().set_names({name})


def export_models(args: argparse.Namespace) -> None:
    os.environ.setdefault("SAM2_BUILD_CUDA", "0")
    repo = add_repo_to_path(args.sam2_repo)
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("Loading SAM2 on CPU...")
    sam2_model = build_sam2(
        args.config,
        str(checkpoint),
        device="cpu",
        mode="eval",
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(sam2_model)
    image_size = int(predictor.model.image_size)

    encoder_path = output_dir / "sam2_image_encoder.xml"
    decoder_path = output_dir / "sam2_box_mask_decoder.xml"

    print(f"Converting static encoder [1,3,{image_size},{image_size}]...")
    encoder_wrapper = SAM2ImageEncoderExport(predictor).eval()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        encoder_model = ov.convert_model(
            encoder_wrapper,
            example_input=torch.zeros(1, 3, image_size, image_size),
            input=([1, 3, image_size, image_size],),
        )
    set_tensor_names(
        encoder_model,
        ["image"],
        ["image_embeddings", "high_res_feats_256", "high_res_feats_128"],
    )
    ov.save_model(encoder_model, encoder_path, compress_to_fp16=args.fp16)

    print("Converting native box prompt encoder + mask decoder...")
    decoder_wrapper = SAM2BoxMaskDecoderExport(
        predictor.model, multimask_output=args.multimask
    ).eval()
    embed_dim = int(predictor.model.sam_prompt_encoder.embed_dim)
    embed_h, embed_w = predictor.model.sam_prompt_encoder.image_embedding_size
    high_res_sizes = predictor._bb_feat_sizes
    dummy_inputs = (
        torch.randn(1, embed_dim, embed_h, embed_w),
        torch.randn(1, 32, *high_res_sizes[0]),
        torch.randn(1, 64, *high_res_sizes[1]),
        torch.tensor([[[128.0, 128.0], [896.0, 896.0]]], dtype=torch.float32),
        torch.tensor([[2, 3]], dtype=torch.int64),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        decoder_model = ov.convert_model(decoder_wrapper, example_input=dummy_inputs)
    set_tensor_names(
        decoder_model,
        [
            "image_embeddings",
            "high_res_feats_256",
            "high_res_feats_128",
            "point_coords",
            "point_labels",
        ],
        ["low_res_masks", "iou_predictions"],
    )
    ov.save_model(decoder_model, decoder_path, compress_to_fp16=args.fp16)

    metadata = {
        "format": "sam2-openvino-box-v2",
        "sam2_repo": str(repo),
        "config": args.config,
        "checkpoint": str(checkpoint),
        "image_size": image_size,
        "prompt_count": 2,
        "prompt_labels": [2, 3],
        "native_prompt_padding": True,
        "multimask_output": bool(args.multimask),
        "candidate_count": 3 if args.multimask else 1,
        "mask_threshold": 0.0,
        "encoder": encoder_path.name,
        "decoder": decoder_path.name,
        "decoder_outputs_low_res": True,
        "compressed_to_fp16": bool(args.fp16),
    }
    (output_dir / "sam2_openvino.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved encoder: {encoder_path}")
    print(f"Saved decoder: {decoder_path}")
    print(f"Saved metadata: {output_dir / 'sam2_openvino.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="Hydra config name relative to the sam2 package",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--multimask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export 3 alternative masks. The service selects the highest IoU score.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compress constants to FP16. Keep disabled for the first parity test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export_models(parse_args())