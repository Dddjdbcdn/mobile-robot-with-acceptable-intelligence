from ultralytics import YOLO

model = YOLO("yolo11m-pose.pt")

model.export(
    format="openvino",
    imgsz=(320, 640),
    half=True,
)