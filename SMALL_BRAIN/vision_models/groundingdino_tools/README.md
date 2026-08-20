# GroundingDINO with OpenVINO on Intel N97 iGPU

This is the complete, repeatable setup that produced working GroundingDINO inference on an Intel N97 UHD Graphics iGPU.

## Known-working software stack

- Ubuntu Linux
- Python 3.12; Python 3.10 and 3.11 should also work
- CPU-only PyTorch 2.4.1
- TorchVision 0.19.1
- OpenVINO 2024.6.0
- NumPy 1.26.4
- Transformers 4.37.2
- Tokenizers 0.15.2
- GroundingDINO branch `wenyi5608-openvino`
- Static ONNX export followed by ONNX-to-OpenVINO conversion
- Static image shape `1 × 3 × 512 × 768`
- Static text length `32`

Do not replace these versions unless you are prepared to troubleshoot compatibility again.

## Why this particular process is necessary

Three tempting approaches caused problems:

1. Installing GroundingDINO with `pip install -e` entered pip/setuptools build isolation and failed because the isolated build environment could not import Torch.
2. New Transformers releases removed the `BertModel.get_head_mask` interface expected by this GroundingDINO branch.
3. Direct PyTorch/TorchScript-to-OpenVINO conversion generated a `SliceAssign → ScatterNDUpdate → ReduceMax` graph that failed during Intel GPU compilation.

The working path is:

```text
GroundingDINO PyTorch checkpoint
        ↓
static ONNX model
        ↓
OpenVINO XML + BIN
        ↓
Intel N97 GPU
```

GroundingDINO does not need to be installed as a Python package. The supplied scripts add the cloned repository directly to `sys.path`.

---

# A. Starting completely from scratch

Place this bundle in an empty project directory and enter it:

```bash
cd /path/to/your/project
```

The directory should initially contain:

```text
README.md
setup_groundingdino_openvino.sh
requirements_groundingdino_openvino.txt
groundingdino_openvino_onnx.py
groundingdino_live.py
```

Run the setup:

```bash
chmod +x setup_groundingdino_openvino.sh
./setup_groundingdino_openvino.sh
```

The setup script:

- installs the Intel OpenCL runtime and basic Ubuntu packages;
- creates `venv`;
- installs CPU-only PyTorch instead of useless NVIDIA CUDA packages;
- installs the tested dependency versions;
- clones the required GroundingDINO branch;
- downloads the Swin-T checkpoint;
- creates the model, image, and OpenVINO cache directories;
- lists the OpenVINO devices.

If the setup script adds your account to `render` or `video`, log out and back in before continuing.

Activate the environment:

```bash
source venv/bin/activate
```

Confirm that Torch is CPU-only:

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

Expected:

```text
CUDA build: None
CUDA available: False
```

That is correct. PyTorch is used only to load and export the model. OpenVINO performs inference on the Intel GPU.

Confirm that OpenVINO sees the GPU:

```bash
python groundingdino_openvino_onnx.py devices
```

Expected device names include:

```text
CPU: Intel(R) N97
GPU: Intel(R) UHD Graphics (iGPU)
```

---

# B. Export the model

Export is needed once for a chosen image resolution and text length.

```bash
source venv/bin/activate
export TOKENIZERS_PARALLELISM=false

python groundingdino_openvino_onnx.py export \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --weights "$PWD/weights/groundingdino_swint_ogc.pth" \
  --output "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --height 512 \
  --width 768 \
  --text-len 32
```

The first export also downloads `bert-base-uncased` from Hugging Face.

Successful export creates:

```text
models/groundingdino_swint_512x768_onnx.xml
models/groundingdino_swint_512x768_onnx.bin
models/groundingdino_swint_512x768_onnx.json
```

Keep those three files together.

The temporary ONNX file is removed after conversion. Add `--keep-onnx` to the export command when you need to preserve it.

## Choosing a model shape

The N97 configuration that worked was:

```text
height: 512
width: 768
text length: 32
```

Every source image is automatically resized before inference, so its original dimensions can vary.

A new prompt does not require a new export as long as it fits within 32 BERT tokens. Export with `--text-len 64` for long descriptive prompts.

Changing `height`, `width`, or `text-len` requires a new export under a different model filename.

---

# C. Test one image

Put an image in `images`, for example:

```text
images/row_0_1.jpg
```

Run one-shot inference:

```bash
python groundingdino_openvino_onnx.py infer \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --model "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --image "$PWD/images/row_0_1.jpg" \
  --prompt "guitar ." \
  --device GPU \
  --output "$PWD/result.jpg"
```

A successful run prints:

```text
Available OpenVINO devices:
  CPU: Intel(R) N97
  GPU: Intel(R) UHD Graphics (iGPU)

Compiling ... for GPU...
Device: GPU
Inference latency: ...
Detections: ...
Saved annotated image: .../result.jpg
```

The line `Device: GPU` confirms that the OpenVINO network ran on the Intel iGPU.

Use periods between multiple object categories:

```text
person . guitar . microphone . chair .
```

Useful threshold options:

```bash
--box-threshold 0.30
--text-threshold 0.25
--nms-threshold 0.80
```

Reduce `--box-threshold` when legitimate objects are being missed. Raising it removes weak detections.

---

# D. Keep the model alive for many images

The one-shot command starts Python, loads the model, processes one image, and exits. Running that command again creates a new process and calls `compile_model()` again.

To load the model once and reuse it, start the interactive runner:

```bash
python groundingdino_live.py \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --model "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --device GPU \
  --prompt "guitar ."
```

It compiles once and then waits:

```text
The model is now staying alive on the GPU.
Image path:
```

Example interaction:

```text
Image path: images/row_0_1.jpg
Prompt [guitar .]:
Output [images/row_0_1_detected.jpg]:
```

Press Enter to keep the current prompt and default output name.

For the next image:

```text
Image path: images/row_0_2.jpg
Prompt [guitar .]: microphone .
Output [images/row_0_2_detected.jpg]:
```

The model does not compile again. The same OpenVINO `CompiledModel` and `InferRequest` remain in memory.

Stop it with:

```text
Image path: quit
```

or press `Ctrl+C`.

## What does and does not trigger model loading

| Action | Full export needed? | New `compile_model()` call? |
|---|---:|---:|
| New image in the live process | No | No |
| New short prompt in the live process | No | No |
| New original image dimensions | No | No |
| Restart the live Python program | No | Yes, usually loaded quickly from cache |
| Run the one-shot CLI again | No | Yes, usually loaded quickly from cache |
| Change OpenVINO version or device | No | Yes |
| Change model height/width/text length | Yes | Yes |

---

# E. Normal startup after reboot

You do not reinstall or re-export after every reboot.

Enter the project and activate the environment:

```bash
cd /path/to/your/project
source venv/bin/activate
export TOKENIZERS_PARALLELISM=false
```

Then start the long-running process:

```bash
python groundingdino_live.py \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --model "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --device GPU \
  --prompt "object ."
```

The `venv/bin/activate` file created by the setup script already exports `TOKENIZERS_PARALLELISM=false`, so the explicit export is optional after activation.

---

# F. OpenVINO cache behavior

The scripts use:

```text
.ov_cache_onnx
```

The first GPU compile may be much slower than later launches. A later process still calls `compile_model()`, but OpenVINO reuses cached compiled data.

Check the cache:

```bash
du -sh .ov_cache_onnx
find .ov_cache_onnx -type f | head
```

Do not delete it during normal use.

Clear it only after changing OpenVINO versions, changing drivers, replacing the model, or diagnosing a corrupt cache:

```bash
rm -rf .ov_cache_onnx
```

---

# G. Confirm actual GPU activity

Run this in a second terminal while inference is executing:

```bash
sudo intel_gpu_top
```

The setup script installs `intel-gpu-tools`.

You should see GPU engine activity during the approximately multi-second inference. The exact latency varies with power limits, memory pressure, drivers, and model shape.

---

# H. Harmless warnings

## Custom C++ operations warning

```text
Failed to load custom C++ ops. Running on CPU mode Only!
```

This refers to GroundingDINO's PyTorch CUDA extension. The extension is not required for the ONNX/OpenVINO inference path. The heavy network still runs on the OpenVINO GPU device.

## timm import warning

```text
Importing from timm.models.layers is deprecated
```

This is an old import path inside GroundingDINO. It does not affect inference.

## Hugging Face download warning

```text
resume_download is deprecated
```

It is harmless.

## BERT checkpoint unexpected keys

Keys related to BERT's pretraining heads may be reported as unexpected. GroundingDINO does not use those heads.

---

# I. Errors and their fixes

## `BertModel` has no attribute `get_head_mask`

The Transformers version is too new.

Repair:

```bash
source venv/bin/activate
python -m pip install --force-reinstall \
  "transformers==4.37.2" \
  "tokenizers==0.15.2"
```

## `No module named torch` while installing GroundingDINO

Do not run:

```bash
pip install -e ./GroundingDINO
```

The scripts directly import the checkout through `--repo`. No package installation is needed.

## `to_shape was called on a dynamic shape`

That came from the discarded direct-conversion script. Use only:

```text
groundingdino_openvino_onnx.py
```

and the `_onnx.xml` model.

## `PullReshapeThroughReduce` / `ScatterNDUpdate` / invalid ReduceMax axis

That IR was produced through direct TorchScript-to-OpenVINO conversion. Delete it and re-export with the ONNX-first script in this bundle.

Do not use:

```text
groundingdino_swint_512x768.xml
```

Use:

```text
groundingdino_swint_512x768_onnx.xml
```

## OpenVINO lists CPU but not GPU

Check OpenCL:

```bash
clinfo | grep -E "Platform Name|Device Name"
```

Check permissions:

```bash
groups
ls -l /dev/dri
```

Your account should be in the `render` group. Add it when necessary:

```bash
sudo usermod -aG render,video "$USER"
```

Then log out and back in.

## NumPy/OpenVINO dependency conflict

Restore the tested versions:

```bash
python -m pip uninstall -y optimum-intel openvino-tokenizers openvino-dev

python -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "openvino==2024.6.0"
```

This project does not need `optimum-intel`, `openvino-tokenizers`, or `openvino-dev`.

## NVIDIA CUDA packages appear during installation

Use the CPU PyTorch index:

```bash
python -m pip install --force-reinstall \
  "torch==2.4.1" \
  "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

The CUDA value should be `None`.

---

# J. Clean rebuild procedure

Use this only when rebuilding everything:

```bash
deactivate 2>/dev/null || true

rm -rf \
  venv \
  GroundingDINO \
  weights \
  models \
  .ov_cache_onnx

./setup_groundingdino_openvino.sh
source venv/bin/activate
```

Then repeat the export command in section B.

Images and result files are not removed by the command above.

---

# K. Final project layout

After setup and export:

```text
project/
├── README.md
├── setup_groundingdino_openvino.sh
├── requirements_groundingdino_openvino.txt
├── groundingdino_openvino_onnx.py
├── groundingdino_live.py
├── venv/
├── GroundingDINO/
├── weights/
│   └── groundingdino_swint_ogc.pth
├── models/
│   ├── groundingdino_swint_512x768_onnx.xml
│   ├── groundingdino_swint_512x768_onnx.bin
│   └── groundingdino_swint_512x768_onnx.json
├── images/
└── .ov_cache_onnx/
```

## Minimal daily command

After everything has been installed and exported, this is all you normally run:

```bash
cd /path/to/your/project
source venv/bin/activate

python groundingdino_live.py \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --model "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --device GPU \
  --prompt "object ."
```
