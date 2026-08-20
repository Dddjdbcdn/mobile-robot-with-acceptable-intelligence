from ultralytics import YOLOE

classes = [
    "guitar",
    "chair",
    "table",
    "person",
    "laptop",
    "door",
    "television",
    "fan",
    "bottle",
    "mirror",
    "toolbox",
    "dumbbell",
    "camera",
    "houseplant",
    "curtain",
    "power socket",
    "book",
    "microphone",
    "smartphone",
    "air conditioner",
    "remote control"
]

# 1. Get text prompt embeddings from the pretrained seg model
prompt_model = YOLOE("yoloe-11m-seg.pt")
prompt_model.set_classes(classes)
prompt_model.save_prompt_embeddings("household.npz")

# 2. Build detection-only YOLOE-11M
model = YOLOE("yoloe-11m.yaml").load("yoloe-11m-seg.pt")

# 3. Apply your household classes
model.load_prompt_embeddings("household.npz")

# 4. Export bbox-only OpenVINO
model.export(
    format="openvino",
    quantize=16,
    imgsz=(320, 640),
)