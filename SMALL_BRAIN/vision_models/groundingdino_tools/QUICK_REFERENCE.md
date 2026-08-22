# Quick reference

## First setup
```bash
./setup_groundingdino_openvino.sh
source venv/bin/activate
```

## Download Swin-T checkpoint

mkdir -p weights

wget -O weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

## Export once
```bash
python groundingdino_openvino_onnx.py export \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --weights "$PWD/weights/groundingdino_swint_ogc.pth" \
  --output "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --height 512 --width 768 --text-len 32
```

## One image
```bash
python groundingdino_openvino_onnx.py infer \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --model "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --image "$PWD/row_0_1.jpg" \
  --prompt "microphone ." \
  --device GPU \
  --output "$PWD/result.jpg"
```

## Keep alive
```bash
python groundingdino_live.py \
  --repo "$PWD/GroundingDINO" \
  --config "$PWD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --model "$PWD/models/groundingdino_swint_512x768_onnx.xml" \
  --device GPU \
  --prompt "object ."
```
