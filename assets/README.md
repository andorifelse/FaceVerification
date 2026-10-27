# Detection model provenance

`face_detection_yunet_2023mar.onnx` is the **face detector**, not an identity recognition or verification network.

- Upstream: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Download: https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- Downloaded during this implementation; distributed with its upstream MIT license (see `LICENSE-YuNet`).
- Input: BGR image. Output: bounding box, five facial landmarks, confidence.

The recognition backbone comes from the supplied project, `checkpoints/18.pth`, and is actually fine-tuned by `train.py`; see `report.md` and `artifacts/training/training.json`.
