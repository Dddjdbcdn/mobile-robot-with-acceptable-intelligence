# Hand landmark model

Automatic hand guidance uses MediaPipe Hand Landmarker on a crop around YOLO's
anatomical right wrist. Running `source SMALL_BRAIN/setup.sh` downloads the
official float16 `hand_landmarker.task` model from Google when it is missing.

The model runs while a person is being tracked. Gesture interpretation and
gesture classification and temporal confirmation live in
`cognition/hand/`, while robot action state lives in
`cognition/manager/action_state.py`.
