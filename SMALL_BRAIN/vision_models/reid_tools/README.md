# Person ReID model

This directory provides the downloader for Open Model Zoo's
`person-reidentification-retail-0288` model. The model accepts a whole-person
BGR crop with shape `1x3x256x128` and returns a 256-value appearance embedding.

From the repository root, download and verify the FP16 OpenVINO model with:

```bash
SMALL_BRAIN/venv/bin/python \
  SMALL_BRAIN/vision_models/reid_tools/download_model.py
```

The downloader verifies the official file sizes and SHA-384 checksums and then
asks OpenVINO to validate the input and output shapes. Re-running it is safe;
already verified files are retained. Use `--force` to replace them.

The resulting files are stored at:

```text
person-reidentification-retail-0288/FP16/
├── person-reidentification-retail-0288.xml
└── person-reidentification-retail-0288.bin
```

That is the default path used by `services.vision.person_reid_service`, so no
model path is required after downloading:

```python
from services.vision.person_reid_service import PersonReIDService

reid = PersonReIDService(device="GPU")
```

Model documentation and license:

- <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-reidentification-retail-0288>
- <https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE>
