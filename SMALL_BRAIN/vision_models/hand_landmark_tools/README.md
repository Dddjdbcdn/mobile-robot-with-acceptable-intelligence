# Hand landmark model

Automatic hand guidance uses MediaPipe Hand Landmarker on a crop around YOLO's
anatomical right wrist. Running `source SMALL_BRAIN/setup.sh` downloads the
official float16 `hand_landmarker.task` model from Google when it is missing.

The model runs while a person is being tracked. Gesture interpretation and
temporal navigation state live in `cognition/hand_guided_navigation.py`.
