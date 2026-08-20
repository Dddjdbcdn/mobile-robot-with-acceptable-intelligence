#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-venv}"
CHECKPOINT="weights/groundingdino_swint_ogc.pth"
CHECKPOINT_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

echo "== GroundingDINO OpenVINO setup =="
echo "Project directory: $ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN was not found."
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(
        f"Use Python 3.10, 3.11, or 3.12; found {sys.version.split()[0]}"
    )
print("Python:", sys.version.split()[0])
PY

if command -v apt-get >/dev/null 2>&1; then
    echo
    echo "Installing Ubuntu system packages..."
    sudo apt-get update
    sudo apt-get install -y \
        git curl ca-certificates \
        python3-venv \
        libgl1 \
        ocl-icd-libopencl1 \
        intel-opencl-icd \
        clinfo \
        intel-gpu-tools

    # Optional on distributions where these package names exist.
    sudo apt-get install -y intel-level-zero-gpu level-zero 2>/dev/null || true

    if ! id -nG "$USER" | tr ' ' '\n' | grep -qx render; then
        echo
        echo "Adding $USER to render/video groups..."
        sudo usermod -aG render,video "$USER"
        echo "IMPORTANT: log out and back in before GPU inference."
    fi
fi

echo
echo "Creating virtual environment: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing CPU-only PyTorch..."
python -m pip install \
    "torch==2.4.1" \
    "torchvision==0.19.1" \
    --index-url https://download.pytorch.org/whl/cpu

echo
echo "Installing pinned OpenVINO/GroundingDINO dependencies..."
python -m pip install -r requirements_groundingdino_openvino.txt

echo
echo "Cloning the OpenVINO-compatible GroundingDINO branch..."
if [[ ! -d GroundingDINO/.git ]]; then
    git clone \
        --branch wenyi5608-openvino \
        --single-branch \
        https://github.com/wenyi5608/GroundingDINO.git
else
    echo "GroundingDINO already exists; leaving the checkout unchanged."
fi

mkdir -p weights models images .ov_cache_onnx

if [[ ! -s "$CHECKPOINT" ]]; then
    echo
    echo "Downloading GroundingDINO Swin-T checkpoint..."
    rm -f "${CHECKPOINT}.part"
    curl -fL --retry 3 \
        "$CHECKPOINT_URL" \
        -o "${CHECKPOINT}.part"
    mv "${CHECKPOINT}.part" "$CHECKPOINT"
else
    echo "Checkpoint already exists: $CHECKPOINT"
fi

chmod +x groundingdino_openvino_onnx.py groundingdino_live.py

# Avoid repeated Hugging Face tokenizers/fork warnings.
ACTIVATE_FILE="$VENV_DIR/bin/activate"
if ! grep -q 'TOKENIZERS_PARALLELISM=false' "$ACTIVATE_FILE"; then
    printf '\nexport TOKENIZERS_PARALLELISM=false\n' >> "$ACTIVATE_FILE"
fi
export TOKENIZERS_PARALLELISM=false

echo
echo "Installed versions:"
python - <<'PY'
import numpy
import onnx
import openvino
import torch
import torchvision
import transformers

print("numpy:", numpy.__version__)
print("onnx:", onnx.__version__)
print("openvino:", openvino.__version__)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
PY

echo
echo "OpenVINO devices:"
python groundingdino_openvino_onnx.py devices

echo
echo "Setup completed."
echo "Activate later with: source $VENV_DIR/bin/activate"
echo "Next: follow README.md section 'Export the model'."
