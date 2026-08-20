from pathlib import Path

import openvino as ov
import torch

from depth_anything_v2.dpt import DepthAnythingV2


# Both values must be divisible by 14.
NET_H = 630
NET_W = 1120

SCENE = "indoor"  # Change to "outdoor" when using VKITTI.

CONFIG = {
    "indoor": {
        "checkpoint": "checkpoints/depth_anything_v2_metric_hypersim_vitb.pth",
        "max_depth": 20.0,
    },
    "outdoor": {
        "checkpoint": "checkpoints/depth_anything_v2_metric_vkitti_vits.pth",
        "max_depth": 80.0,
    },
}

# MODEL_CONFIG = {
#     "encoder": "vits",
#     "features": 64,
#     "out_channels": [48, 96, 192, 384],
# }

MODEL_CONFIG = {
    "encoder": "vitb",
    "features": 128,
    "out_channels": [96, 192, 384, 768],
}


def main() -> None:
    if NET_H % 14 != 0 or NET_W % 14 != 0:
        raise ValueError("NET_H and NET_W must both be divisible by 14.")

    scene_config = CONFIG[SCENE]
    checkpoint = Path(scene_config["checkpoint"])

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = DepthAnythingV2(
        **MODEL_CONFIG,
        max_depth=scene_config["max_depth"],
    )

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    example_input = torch.zeros(
        (1, 3, NET_H, NET_W),
        dtype=torch.float32,
    )

    print("Converting model...")
    ov_model = ov.convert_model(
        model,
        example_input=example_input,
        input=[1, 3, NET_H, NET_W],
    )

    output_dir = Path("openvino_models")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / (
        f"dav2_metric_{SCENE}_vitb_{NET_W}x{NET_H}_fp16.xml"
    )

    ov.save_model(
        ov_model,
        output_path,
        compress_to_fp16=True,
    )

    print(f"Saved: {output_path}")
    print(f"Maximum model depth: {scene_config['max_depth']} metres")


if __name__ == "__main__":
    main()