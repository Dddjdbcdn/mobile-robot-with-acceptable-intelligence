# Person ReID model

This directory provides the downloader for Open Model Zoo's
`person-reidentification-retail-0288` model. The model accepts a whole-person
BGR crop with shape `1x3x256x128` and returns a 256-value appearance embedding.

Download and validate the FP16 model from the repository root:

```bash
SMALL_BRAIN/venv/bin/python \
  SMALL_BRAIN/vision_models/reid_tools/download_model.py
```

The model is used by `PersonReIDService` for inference and by
`PersonTrackingService` as one signal in scene-local multi-person assignment.
The tracking service runs continuously from startup, maintains bounded
appearance memory automatically, and does not persist strangers across robot
restarts.

The camera display draws `ID <number>` around every visible scene track. The
ID, state, appearance sample count, and association confidence make crossings,
occlusions, and reacquisition directly testable before those IDs control robot
motion. `TrackAction` continues to use its existing person association for now.

Compare any two tightly cropped person images directly:

```bash
SMALL_BRAIN/venv/bin/python \
  SMALL_BRAIN/vision_models/reid_tools/check_reid.py reference.jpg candidate.jpg
```

Use `--device CPU` if GPU inference is unavailable.

Model documentation and license:

- <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-reidentification-retail-0288>
- <https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE>
