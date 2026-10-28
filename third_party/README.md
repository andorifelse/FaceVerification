# CVLFace ViT provenance

The new recognition backbone is **ViT-Base** from [CVLface](https://github.com/mk-minchul/CVLface), not the supplied FaceMobileNet. `model/face_vit.py` preserves its layer names and architecture: RGB 112×112, 8×8 patches, 196 tokens, 512 internal dimensions, 24 Transformer blocks, 16 attention heads, ReLU6 MLP ratio 3, flattened token projection and 512-dimensional output. The attention computation uses PyTorch SDPA. No Hugging Face remote Python code is executed.

- [Upstream architecture](https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m/blob/c6fed2ff09ac0a29cccc72c09127ad7b4bd3fa9e/models/vit/vit.py)
- [Upstream configuration](https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m/blob/c6fed2ff09ac0a29cccc72c09127ad7b4bd3fa9e/models/vit/__init__.py)
- [Model card](https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m)
- Code license: MIT, reproduced in `CVLface-LICENSE`.
- Weights: the upstream model card requests compliance with the training dataset license; the MIT code license is not presented as a grant of dataset rights.
- Pretraining: AdaFace on WebFace4M, as declared by the author. Public benchmark numbers are not used as our experimental measurements.
- Fixed model revision and exact initialization checksum: `vit_initialization.json`.

Actual local CASIA fine-tuning updates the final two Transformer blocks, final LayerNorm and both feature projection matrices. BatchNorm parameters/statistics and earlier layers remain frozen. Stochastic depth is disabled for this fine-tuning implementation. The resulting `checkpoints/transformer.pth` is distinct from the downloaded initialization; changed tensors, losses, identity splits, selected epoch and checksums are recorded in `artifacts/transformer_training/`.

The subsequent refinement round is recorded in `artifacts/refinement/`: it trains the final **four** blocks with quality-weighted classification and clean-target consistency for downsampled inputs, then fits a regularized within-identity covariance projection. Candidate families are selected using CASIA validation identities only. The new default `transformer.pth` follows that selection; the first-round checkpoint remains `transformer_round1.pth`. The projection is a supervised closed-form fit, and is not represented as additional gradient updates. See the current `report.md` for all ablations, including unchanged or regressed metrics.

Download the initialization with `python scripts/download_transformer.py`, then train with `python train.py`. The assignment explicitly permits fine-tuning, and this project does not claim to have trained the initialization from scratch.
