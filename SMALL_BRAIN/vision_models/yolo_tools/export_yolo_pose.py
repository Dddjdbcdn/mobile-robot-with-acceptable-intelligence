from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

model.export(
    format="openvino",
    imgsz=(320, 640),
    half=True,
)