# Face Verification：GPU 微调 Transformer

按 `说明.docx` 完成双图验证接口、人脸检测与对齐、照片/摄像头演示及实验报告。默认识别器现已替换为 **ViT-Base Transformer**，原 FaceMobileNet 仅保留用于对照。

## 实际结果

| 方法 | LFW DevTest（1,000 对） | 标准 LFW 十折（6,000 对） |
|---|---:|---:|
| 原 FaceMobileNet，单视图 | 96.80% | 96.72% ± 0.77% |
| 新 ViT，实际 CASIA 微调后，翻转融合 | **99.10%** | **99.17% ± 0.39%** |

在 RTX PRO 6000 GPU 上实际完成 4 轮微调，使用 4,096 个 CASIA 身份、65,536 张训练图像；另 256 个身份用于模型选择。最终选中第 2 轮，共 56,895,389 个浮点值相对初始化发生变化。

初始化来自作者发布的 CVLFace AdaFace ViT-Base WebFace4M 权重。预训练 ViT 本身已有较强效果（DevTest 99.1%），本次微调没有进一步提高该项准确率；性能提升不能全部归因于本次微调。来源、完整消融、局限和实际训练记录见 [report.md](report.md) 与 [third_party/](third_party/)。

## 安装与演示

建议 Python 3.11。旧 `venv/` 是 Windows 环境，请在当前系统配置环境。

```bash
cd /home/wzc/datasets/casia/faceRecognitionCASIA
python -m pip install -r requirements.txt
python app.py
```

浏览器打开 **http://127.0.0.1:7860**。上传两张单人照片点击“开始验证”，也可开启摄像头拍照或连续验证。已裁剪对齐的 CASIA/LFW 图片勾选“输入为已对齐的人脸图片”。普通照片自动检测与五点对齐；无脸、多脸会提示错误。图片不保存到服务器文件。

默认权重 `checkpoints/transformer.pth` 和配套阈值 `artifacts/calibration.json` 已在本地生成，使用演示不必重新训练。正常环境自动使用 CUDA，没有 CUDA 时可在 CPU 推理。多卡机器可选择：

```bash
FACE_DEVICE=cuda:7 python app.py
```

当前自动化工具沙箱可能无法访问 GPU，正常终端或授权的 GPU 执行环境可用。单卡机器请把 `cuda:7` 换成 `cuda:0` 或 `cuda`。

## 作业规定接口

```python
from FaceVerification import FaceVerification

root = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped"
Y = FaceVerification(
    root + "/Abel_Pacheco/Abel_Pacheco_0001.jpg",
    root + "/Abel_Pacheco/Abel_Pacheco_0004.jpg",
)
print(Y)  # Python int：1=同人，0=异人
```

规定接口接收已对齐的人脸。普通照片可用：

```python
from verification import FaceVerifier
result = FaceVerifier().compare("photo1.jpg", "photo2.jpg", detect=True)
print(result)
```

命令行示例（使用本地照片）：

```bash
python verification.py samples/cl.png samples/lxl.png --detect
# 或用随项目已有的样例检查同图一致性
python verification.py samples/009.jpg samples/009.jpg --detect
```

最终决策为“余弦相似度 >= 0.269504666 返回 1”，相似度不代表概率。模型、翻转配置与阈值通过 SHA-256 校验。环境变量 `FACE_CHECKPOINT`、`FACE_CALIBRATION` 可切换配套文件，`FACE_DEVICE` 可选择推理设备。

## 重新训练与评估

```bash
python -m pip install -r requirements-dev.txt
python scripts/download_transformer.py
python train.py --device cuda:7
python evaluate.py --device cuda:7
python scripts/validate_system.py --device cuda:7
python scripts/build_report.py
python -m pytest -q
```

`train.py` 现在默认训练新 Transformer。默认冻结前 22 个块和全部 BN，微调最后两个块、LayerNorm 与特征投影；使用 CosFace 分类和初始特征保持项、BF16、AdamW。`python train.py --help` 可查看数据路径、身份数、轮数、块数和学习率等配置。旧网络训练入口另存为 `train_mobile.py`。

`evaluate.py` 默认比较原权重、此前卷积微调权重（存在时）、ViT 初始化（存在时）及最终微调 ViT，分别测试单视图和翻转融合。输出的 `finetuned_single/flip` **现在指新 Transformer**；旧卷积微调结果标为 `mobile_finetuned_single/flip`。

DevTrain 2,200 对仅用于校准阈值，DevTest 1,000 对用于独立评估，二者身份和图片无重叠。标准 6,000 对十折在其余九折选择阈值后测试留出一折。默认模型由 CASIA 验证 AUC 选择，不用 LFW 测试标签挑轮次。运行评估会更新阈值、预测与指标；运行报告脚本会据此更新报告。

## 交付与记录

| 文件 | 作用 |
|---|---|
| `report.md` | 方法、运行方法、来源披露、实测结果及局限 |
| `FaceVerification.py` / `verification.py` | 规定接口、模型加载、按网络选择预处理、命令行 |
| `model/face_vit.py` | 本地 PyTorch ViT 实现，无远程 Python 执行 |
| `train.py` / `train_transformer.py` | 默认训练入口与实际 Transformer 微调 |
| `scripts/download_transformer.py` / `third_party/` | 固定版本初始化、SHA-256 和上游代码许可 |
| `checkpoints/transformer.pth` | 本次实际微调选出的最终权重 |
| `artifacts/transformer_training/` | 身份划分、逐轮损失、梯度更新参数数量与模型校验值 |
| `evaluate.py` / `metrics.py` | 统一评估、ROC、十折和配对比较 |
| `artifacts/calibration.json` | 与新 ViT 绑定的阈值 |
| `artifacts/evaluation.json` / `predictions.csv` | 所有方案指标与最终方案的逐对预测 |
| `artifacts/test_validation.json` / `browser_validation.json` | 自动测试和浏览器验证记录 |
| `app.py` / `templates/index.html` | 照片及摄像头演示 |
| `detection.py` / `assets/` | YuNet 检测、五点对齐及许可证 |
| `experiments/mobile_before_transformer/` | 本轮修改前的代码、用户已有报告与结果备份 |
| `train_mobile.py` / `artifacts/training/` | 此前卷积微调入口及记录 |
| `legacy/` / `blog/` | 最初的主要脚本及教程 |

权重、数据集和大型特征缓存被 `.gitignore` 排除。**Git 克隆不含最终权重**；迁移时单独复制 `checkpoints/transformer.pth`（约 438 MiB）及配套 `artifacts/calibration.json`，或按上面的下载/训练流程重新生成权重与阈值。`evaluate.py` 的原网络对照还需要本地原有的 `checkpoints/18.pth` 和 LFW 数据。

14 项自动测试以及真实浏览器上传、手机布局、模拟摄像头流程通过。未声称已测试真实摄像头硬件。公开测试成绩不保证未公开数据上的同等准确率。

## 致谢

新 ViT 基于 [CVLface](https://github.com/mk-minchul/CVLface)，其代码许可、模型卡及固定版本记录见 `third_party/`。原项目沿用 [insightface](https://github.com/deepinsight/insightface)、[InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)、[arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch) 的代码及思路，保留原始致谢与教程。
