# Detection model provenance

`face_detection_yunet_2023mar.onnx` is the **face detector**, not an identity recognition or verification network.

- Upstream: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Download: https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- Downloaded during this implementation; distributed with its upstream MIT license (see `LICENSE-YuNet`).
- Input: BGR image. Output: bounding box, five facial landmarks, confidence.

The current recognition backbone is a locally fine-tuned CVLFace ViT; initialization provenance and code license are recorded in `third_party/`, and local refinement records are in `artifacts/refinement/`. The supplied project's `checkpoints/18.pth` is retained only as a comparison baseline. See `report.md` for details.
