#!/usr/bin/env python3
"""Print the appearance similarity between two whole-person images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

SMALL_BRAIN_ROOT = Path(__file__).resolve().parents[2]
if str(SMALL_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(SMALL_BRAIN_ROOT))

from services.vision.person_reid_service import PersonReIDService


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two tightly cropped whole-person images.", )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--model", type=Path, default=None)
    return parser.parse_args()


def _read(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def main():
    args = parse_args()
    service = PersonReIDService(model_path=args.model, device=args.device)
    reference = service.extract_embedding(_read(args.reference))
    candidate = service.extract_embedding(_read(args.candidate))
    score = float(np.clip(np.dot(reference, candidate), -1.0, 1.0))
    print(f"cosine_similarity: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
