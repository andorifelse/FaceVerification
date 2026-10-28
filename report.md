# 人脸验证实验报告

## 1. 任务与结果

本项目判断两张人脸照片是否属于同一个人，提供 `Y = FaceVerification(X1, X2)` 接口：输入两张图片的路径，同人返回 1，异人返回 0。系统还支持照片上传、人脸检测与对齐、摄像头拍照和连续验证。

我们完成了三组主要实验：微调 FaceMobileNet、直接评估预训练 ViT、微调 ViT。统一使用原图与水平翻转特征融合时，微调 FaceMobileNet 的 LFW DevTest 准确率为 **97.10%**；预训练 ViT 和微调 ViT 均为 **99.10%**。ViT 微调后的标准十折平均准确率从 **99.15% 提高到 99.20%**，6,000 对图片中净少错 3 对。提升较小，单视图和部分指标未改善，具体对照见第 5 节。

## 2. 安装与运行

建议使用 Python 3.11。克隆仓库后，在项目根目录安装依赖：

```bash
git clone git@github.com:andorifelse/FaceVerification.git
cd FaceVerification
python -m pip install -r requirements.txt
```

Git 仓库不包含数据集和模型权重。使用本次微调 ViT 时，需另行放置 `checkpoints/my_experiments/vit_robust.pth`，并保留配套的 `artifacts/my_experiments/vit_robust_eval/calibration.json`。预训练初始化可用 `python scripts/download_transformer.py` 下载，微调权重可按第 7 节生成。

### 2.1 网页演示

下面显式加载本次实验的 ViT，确保网页与报告使用同一份权重：

```bash
FACE_DEVICE=cuda:7 \
FACE_CHECKPOINT=checkpoints/my_experiments/vit_robust.pth \
FACE_CALIBRATION=artifacts/my_experiments/vit_robust_eval/calibration.json \
python -c 'from app import create_app; from verification import get_verifier; create_app(verifier=get_verifier()).run(host="127.0.0.1", port=7860)'
```

启动后访问 `http://127.0.0.1:7860`。普通照片会先检测和对齐人脸；已对齐的 CASIA、LFW 图片应勾选“输入为已对齐的人脸图片”。摄像头功能需要浏览器授权。`cuda:7` 表示第 8 张 GPU，单卡机器改为 `cuda:0`，无 GPU 时改为 `cpu`。

直接运行 `python app.py` 会使用项目默认的 `checkpoints/transformer.pth`，它是此前的“增强微调 + 类内投影”组合版，与本次三组实验中的 ViT 权重不同。

### 2.2 Python 接口与命令行

规定接口返回 Python `int`，以下示例使用本次微调权重，输入为已对齐的人脸：

```python
import os
os.environ["FACE_CHECKPOINT"] = "checkpoints/my_experiments/vit_robust.pth"
os.environ["FACE_CALIBRATION"] = "artifacts/my_experiments/vit_robust_eval/calibration.json"

from FaceVerification import FaceVerification
root = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped"
Y = FaceVerification(root + "/Abel_Pacheco/Abel_Pacheco_0001.jpg",
                     root + "/Abel_Pacheco/Abel_Pacheco_0004.jpg")
print(Y)  # 1
```

普通照片可通过命令行先检测再验证：

```bash
python verification.py samples/009.jpg samples/009.jpg \
  --detect \
  --checkpoint checkpoints/my_experiments/vit_robust.pth \
  --calibration artifacts/my_experiments/vit_robust_eval/calibration.json
```

输出包含 `same_person`、`similarity`、`threshold` 和人脸框 `boxes`。本次 ViT 的阈值为 **0.277990639**，相似度大于等于阈值时判为同人。两次输入同一张图片只能检查流程是否正常，不能用于衡量准确率。

## 3. 模型与微调方法

### 3.1 检测与特征提取

普通照片使用 YuNet 检测人脸，再根据双眼、鼻尖和嘴角五点做相似变换对齐。无脸、多脸或损坏图片会提示错误。评估数据已有裁剪和对齐，不重复执行检测。

FaceMobileNet 沿用项目提供的网络，输入为灰度 128×128。ViT 使用 CVLFace 发布的 `minchul/cvlface_adaface_vit_base_webface4m`，输入为 RGB 112×112，将图像划分为 196 个 8×8 patch，经过 24 个 Transformer 块，输出 512 维特征。

ViT 初始化由作者使用 WebFace4M 和 AdaFace 预训练；本项目在此基础上做本地微调，没有从零完成这部分预训练。固定模型版本和许可证记录在 `third_party/`，原 FaceMobileNet 初始化为项目已有的 `18.pth`，其历史训练信息无法从文件确认。

推理时，先分别归一化原图和水平翻转图的特征，再相加并归一化，最后计算两张人脸的余弦相似度。本文将这一设置记为“翻转融合”，只使用原图则记为“单视图”。

### 3.2 质量加权与清晰目标一致性

直接微调强预训练模型，可能只降低训练损失而没有改善验证效果。我们尝试降低离群样本的影响，并让降质输入保留清晰图像的身份信息。

首先，用初始化 ViT 提取训练图片的原图和翻转特征，融合后按身份求平均，得到身份原型。样本与原型的余弦相似度记为 `q`，训练权重为：

```text
w = clip((q - 0.25) / 0.5, 0, 1)
```

与同身份中心差异较大的样本权重更低。这是特征一致性估计，可能减弱错标或异常图片的影响，但也可能降低真正困难样本的权重。

训练时，以 0.5 概率水平翻转图片，另以 0.5 概率将图片缩至 40、56 或 72 像素后恢复到 112×112。模型处理增强后的输入，保持目标则始终使用对应清晰图片的初始化特征。总损失为：

```text
L = weighted_mean(CrossEntropy(CosFace(f(x_aug), y)), w)
  + 2 × weighted_mean(1 - cosine(f(x_aug), e_initial_clean), w)
```

CosFace 尺度为 32、间隔为 0.15，分类器由身份原型初始化。只更新最后 4 个 Transformer 块、最终 LayerNorm 和特征投影，冻结前 20 块及 BatchNorm 参数和统计量。

本项目的工作重点是将样本加权、分辨率增强和清晰目标一致性用于本地人脸迁移，并通过对照检查收益。这些技术均有已有研究基础，我们不将其称为全新的网络结构。

## 4. 实验设置

两组微调使用相同的 CASIA 清单：4,096 个训练身份、65,536 张图像；256 个验证身份、4,096 张图像，每人 16 张，随机种子为 42。训练与验证身份不重叠，LFW 不参与梯度更新。

| 设置 | FaceMobileNet 微调 | ViT 微调 |
|---|---|---|
| 更新范围 | 全网络 | 最后 4 块、LayerNorm、特征投影；固定 BN |
| 轮数 / batch size | 8 / 128 | 8 / 128 |
| 学习率 | 网络 5×10⁻⁵，分类器 10⁻³ | 块 2×10⁻⁵，特征头 4×10⁻⁵，分类器 10⁻³ |
| 特征保持权重 | 10 | 2 |
| 模型选择 | 清晰 CASIA 验证 balanced accuracy 最大 | 降质 CASIA 验证 AUC 最大，清晰 AUC 回退不超过 0.001 |
| 选中轮次 | 第 2 轮 | 第 6 轮 |
| 耗时（含初始化缓存等） | 402.77 秒 | 263.14 秒 |

实验使用 NVIDIA RTX PRO 6000 Blackwell、PyTorch 2.9.1+cu128；ViT 训练采用 BF16。两组脚本的加载方式、更新范围和选模型规则不同，耗时不能直接解释为同等条件下的速度比较。

LFW 采用两种评估协议：DevTrain 的 2,200 对仅用于选择阈值，固定阈值后在 DevTest 的 1,000 对上测试，同人、异人各 500 对；另按官方 6,000 对划分做十折，每折用其余 5,400 对选择阈值，在留出的 600 对上测试。十折期间只重新标定阈值，不重新训练网络，表中的“±”为十折准确率的标准差，不是置信区间。

Accuracy 表示全部配对中判断正确的比例；FAR 为异人误判同人的比例，FRR 为同人误判异人的比例。AUC 衡量不同阈值下的整体区分能力，EER 为 FAR 与 FRR 相等附近的错误率。Accuracy、AUC 越高越好，FAR、FRR、EER 越低越好。

## 5. 主要实验结果

本节全部来自本次 `artifacts/my_experiments/`，与下一节的早期探索分开记录。权重、训练清单和评估文件已交叉核对。

### 5.1 翻转融合

| 模型 | DevTest 准确率 | 标准十折均值 ± 标准差 | DevTest AUC | EER | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|
| 原始 FaceMobileNet | 96.70% | 96.55% ± 0.93% | 0.989524 | 3.60% | 1.60% | 5.00% |
| 微调 FaceMobileNet | 97.10% | 96.73% ± 0.85% | 0.991092 | 3.20% | 2.20% | 3.60% |
| 预训练 ViT | 99.10% | 99.15% ± 0.40% | 0.990520 | 1.60% | 0.00% | 1.80% |
| 微调 ViT | 99.10% | 99.20% ± 0.41% | 0.990468 | 1.60% | 0.00% | 1.80% |

FaceMobileNet 微调后，DevTest 从 96.7% 提高到 97.1%，十折错误从 207 对降到 196 对。但 DevTest 误接受从 8 对增至 11 对，误拒绝从 25 对降至 18 对，说明准确率改善也伴随了取舍。

ViT 微调前后的 DevTest 均为 99.1%。十折准确率提高 0.05 个百分点，错误从 51 对降至 48 对；DevTest AUC 略降，EER 不变。因此本次微调有小幅十折收益，但没有全面超过初始化。只有一个随机种子、净少错 3 对，还不足以说明稳定或显著的提升。

### 5.2 单视图对照

| 模型 | DevTest 准确率 | 标准十折均值 ± 标准差 | 十折错误数 |
|---|---:|---:|---:|
| 原始 FaceMobileNet | 96.80% | 96.72% ± 0.77% | 197 |
| 微调 FaceMobileNet | 97.00% | 96.92% ± 0.85% | 185 |
| 预训练 ViT | 99.00% | 99.17% ± 0.39% | 50 |
| 微调 ViT | 99.00% | 99.15% ± 0.36% | 51 |

翻转融合对不同模型的影响不一致：它提高了微调 ViT 的十折成绩，却降低了微调 FaceMobileNet 的十折成绩，所以两种设置均保留。ViT 整体准确率高于本次 FaceMobileNet，但骨干规模、输入通道和预训练数据也不同，不能把差异全部归因于 Transformer 结构。

![三组实验对比](artifacts/my_experiments/summary/comparison.png)

### 5.3 训练过程

FaceMobileNet 总损失从 2.8602 降至 1.2956，验证 balanced accuracy 在第 2 轮达到 93.33%，后续未再超过。ViT 总损失从 0.11285 降至 0.02596，第 6 轮降质验证 AUC 为 0.964114，高于初始化的 0.962823，因此按规则保留第 6 轮。训练损失持续下降不代表测试准确率会持续提高，两组损失数值也不能直接比较。

![训练与模型选择](artifacts/my_experiments/summary/training_curves.png)

## 6. 补充探索：类内变化抑制投影

在此前一次独立运行中，我们还尝试了类内协方差投影：过滤与身份中心余弦相似度小于 0.3 的训练样本，用各样本相对身份均值的残差估计类内协方差，再做收缩正则和部分白化，以减弱同人变化较大的特征方向。投影采用闭式拟合，不是额外的 Transformer 梯度训练。

正则系数在 `{0.05, 0.2, 0.5}`、白化强度在 `{0.25, 0.5, 0.75, 1.0}` 中选择，只使用 CASIA 验证数据。组合方法按清晰与降质验证 AUC 的均值选出。下面保留该次运行的模块对照，不将其与本次 ViT 的第 6 轮权重混用。

| 方法 | 原始 LFW 十折 | 降质 DevTest | 降质 LFW 十折 |
|---|---:|---:|---:|
| 预训练 ViT | 99.15% | 98.10% | 98.37% |
| 普通微调 | 99.17% | 97.90% | 98.28% |
| 仅类内投影 | 99.17% | 97.90% | 98.23% |
| 质量加权增强微调 | 99.22% | 98.00% | 98.27% |
| 增强微调 + 类内投影 | 99.10% | 98.30% | 98.33% |

降质诊断统一将图片缩为 40×40、以 JPEG 质量 45 编码，再恢复为 112×112，并在降质校准集上重新标定阈值。这是额外诊断，不能当作原始 LFW 成绩。组合方法改善了降质 DevTest，但原始十折略退，说明抑制类内变化不一定带来更高准确率。此前增强微调的 99.22% 与本次复现的 99.20% 属于不同运行，不应择高合并。

全部候选、验证选择和额外 CASIA 身份审计保存在 `artifacts/refinement/`。主实验只重复了三组模型，没有对本次权重重新执行这些附加诊断。

## 7. 复现与记录

以下命令需在项目根目录运行，GPU 编号按实际机器修改。CASIA 和 LFW 数据路径见 `config.py`。

```bash
# 1. 微调 FaceMobileNet，再评估
python train_mobile.py \
  --init checkpoints/18.pth --output checkpoints/my_experiments/mobile_full.pth \
  --work-dir artifacts/my_experiments/mobile_train \
  --train-ids 4096 --validation-ids 256 --per-id 16 \
  --epochs 8 --batch-size 128 --lr 0.00005 --anchor-weight 10 \
  --seed 42 --device cuda:7 --full
python evaluate.py \
  --checkpoint checkpoints/my_experiments/mobile_full.pth \
  --mobile-checkpoint checkpoints/my_experiments/mobile_full.pth \
  --output artifacts/my_experiments/mobile_eval --device cuda:7

# 2. 直接评估预训练 ViT
python scripts/download_transformer.py
python evaluate.py \
  --checkpoint checkpoints/pretrained_vit/model.safetensors \
  --mobile-checkpoint checkpoints/my_experiments/mobile_full.pth \
  --output artifacts/my_experiments/vit_pretrained_eval --device cuda:7

# 3. 微调 ViT，再评估
python train_transformer.py \
  --init checkpoints/pretrained_vit/model.safetensors \
  --output checkpoints/my_experiments/vit_robust.pth \
  --work-dir artifacts/my_experiments/vit_robust_train \
  --train-ids 4096 --validation-ids 256 --per-id 16 \
  --epochs 8 --batch-size 128 --workers 8 --train-blocks 4 \
  --lr 0.00002 --head-lr 0.00004 --anchor-weight 2 \
  --recipe robust --augmentation-probability 0.5 --seed 42 --device cuda:7
python evaluate.py \
  --checkpoint checkpoints/my_experiments/vit_robust.pth \
  --mobile-checkpoint checkpoints/my_experiments/mobile_full.pth \
  --pretrained checkpoints/pretrained_vit/model.safetensors \
  --output artifacts/my_experiments/vit_robust_eval --device cuda:7
```

训练参数、数据划分和选中轮次位于各训练目录的 `training.json`、`split.json`；逐对预测和十折指标位于各评估目录的 `predictions.csv`、`evaluation.json`。完整指标表见 [实验汇总](artifacts/my_experiments/summary/summary.md)，原始数字均可追溯。自动测试运行 `python -m pytest -q`；已有 16 项检查通过，覆盖接口、阈值、检测、网页请求和网络计算。摄像头流程使用模拟输入验证，真实硬件尚未测试。

本次只验证了一组训练种子，公开 LFW 在多轮实验中已被查看，微小收益仍需更多种子和外部数据确认。此外，上游预训练数据的身份重叠未完整核验，教师未公开测试集也尚未评估。

## 8. 参考资料

- [CVLFace 代码](https://github.com/mk-minchul/CVLface)与[固定版本 ViT 权重](https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m/tree/c6fed2ff09ac0a29cccc72c09127ad7b4bd3fa9e)。
- [Vision Transformer](https://arxiv.org/abs/2010.11929)、[AdaFace](https://arxiv.org/abs/2204.00964) 与 [CosFace](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html)。
- [YuNet 检测器](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。
- 原项目的网络、训练脚本与教程，来源和致谢见 `legacy/README.md`、`blog/`；第三方许可证见 `third_party/`、`assets/`。
