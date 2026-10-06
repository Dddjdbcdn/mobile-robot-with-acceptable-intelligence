"""Download and verify the Open Model Zoo person ReID model."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
from urllib.request import Request, urlopen


MODEL_NAME = "person-reidentification-retail-0288"
MODEL_ROOT = Path(__file__).resolve().parent / MODEL_NAME / "FP16"
BASE_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    f"2023.0/models_bin/1/{MODEL_NAME}/FP16"
)
FILES = {
    f"{MODEL_NAME}.xml": {
        "size": 614_533,
        "sha384": (
            "18814c6445b35224987bc44bcbdb47da6c3826e5da0b6aa00"
            "a88c06bf71491a8c4f42e1f62424c077cce46e76a41f7a2"
        ),
    },
    f"{MODEL_NAME}.bin": {
        "size": 364_098,
        "sha384": (
            "0a5ca9d000ce63b078149a3aa3993cd6e92d52fdecf1ff58f"
            "be717481c2043db348938c32fed964846817dc321825a4c"
        ),
    },
}


def sha384(path: Path) -> str:
    digest = hashlib.sha384()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_file(path: Path, metadata: dict[str, object]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == metadata["size"]
        and sha384(path) == metadata["sha384"]
    )


def download_file(name: str, force: bool = False) -> Path:
    metadata = FILES[name]
    destination = MODEL_ROOT / name
    if not force and valid_file(destination, metadata):
        print(f"Already verified: {destination}")
        return destination

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    request = Request(
        f"{BASE_URL}/{name}",
        headers={"User-Agent": "DJ-robot-model-downloader/1.0"},
    )
    print(f"Downloading {name} ...")
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not valid_file(temporary, metadata):
            raise RuntimeError(f"Downloaded file failed verification: {name}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Verified: {destination}")
    return destination


def download_model(force: bool = False) -> Path:
    for name in FILES:
        download_file(name, force=force)
    return MODEL_ROOT / f"{MODEL_NAME}.xml"


def verify_openvino_model(model_path: Path) -> None:
    try:
        import openvino as ov
    except ImportError:
        print("OpenVINO is unavailable; checksum verification succeeded.")
        return
    model = ov.Core().read_model(str(model_path))
    input_shape = tuple(model.input(0).shape)
    output_shape = tuple(model.output(0).shape)
    if input_shape != (1, 3, 256, 128) or output_shape != (1, 256):
        raise RuntimeError(
            f"Unexpected model shapes: input={input_shape}, output={output_shape}"
        )
    print(f"OpenVINO model verified: input={input_shape}, output={output_shape}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Download the FP16 OpenVINO {MODEL_NAME} model."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even when both local files pass verification",
    )
    args = parser.parse_args()
    try:
        model_path = download_model(force=args.force)
        verify_openvino_model(model_path)
    except Exception as error:
        print(f"Model download failed: {error}", file=sys.stderr)
        return 1
    print(f"Ready for PersonReIDService: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

