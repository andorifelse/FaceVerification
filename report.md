# 人脸验证实验报告

## 1. 任务与结果

本实验判断两张人脸照片是否属于同一个人，提供 `Y = FaceVerification(X1, X2)` 接口：输入两张图片的路径，同人返回 1，异人返回 0。
并设计了人脸识别系统，支持照片上传、人脸检测与对齐、摄像头拍照和连续验证。

我们完成了三组主要实验：
- 微调 FaceMobileNet
- 直接评估预训练 ViT
- 微调 ViT。

统一使用原图与水平翻转特征融合时，微调 FaceMobileNet 的 LFW DevTest 准确率为 **97.10%**；预训练 ViT 和微调 ViT 均为 **99.10%**。ViT 微调后的标准十折平均准确率从 **99.15% 提高到 99.20%**，6,000 对图片中净少错 3 对。具体对照见第 5 节。

## 2. 安装与运行

首先 git clone 我们的仓库，并下载权重：

```bash
git clone git@github.com:andorifelse/FaceVerification.git
huggingface-cli download wawawa111/FaceVerificationCheckpoint --local-dir ./checkpoints
```

在项目根目录安装依赖：

```bash
cd /home/wzc/datasets/casia/faceRecognitionCASIA # replace your path
python -m pip install -r requirements.txt
```

使用我们微调 ViT 的权重时，需将下载好的权重文件放到根目录的指定路径 `checkpoints/my_experiments/vit_robust.pth`，配套的校准配置文件也需要放到根目录的指定路径 `artifacts/my_experiments/vit_robust_eval/calibration.json`。
预训练初始化可用 `python scripts/download_transformer.py` 下载，微调权重可按第 7 节生成。

### 2.1 网页演示

下面显式加载我们微调后的 ViT 权重：

```bash
FACE_DEVICE=cuda:7 FACE_CHECKPOINT=checkpoints/my_experiments/vit_robust.pth FACE_CALIBRATION=artifacts/my_experiments/vit_robust_eval/calibration.json python app.py
```
其中：

- `FACE_DEVICE`：推理设备，`cuda:7` 表示第 8 张 GPU，单卡机器改为 `cuda:0`，无 GPU 时改为 `cpu`。
- `FACE_CHECKPOINT`：模型权重文件路径。
- `FACE_CALIBRATION`：模型校准配置文件路径。

启动后访问 `http://127.0.0.1:7860`，即可看到我们的人脸识别系统。关于app的使用，请看`app_demo.mp4`。

### 2.2 Python 接口与命令行

我们按照要求也提供了 `Y = FaceVerification(X1, X2)` 这样的统一接口，输入两张已对齐人脸图像路径，返回 Python `int`：同人 1、异人 0。

```python
from FaceVerification import FaceVerification
root = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped"
Y = FaceVerification(root + "/Abel_Pacheco/Abel_Pacheco_0001.jpg",
                     root + "/Abel_Pacheco/Abel_Pacheco_0004.jpg")
print(Y)  # int: 1 or 0
```

例如，打开终端，输入命令，我们会得到如下结果：

```bash
(cifar1) wzc@user-Rack-Server:~/datasets/casia/faceRecognitionCASIA$ python
Python 3.11.16 (main, Aug 27 2026, 14:44:21) [GCC 14.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
Ctrl click to launch VS Code Native REPL (https://aka.ms/python-native-repl)
>>> from FaceVerification import FaceVerification

>>> root = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped"
>>> Y = FaceVerification(root + "/Abel_Pacheco/Abel_Pacheco_0001.jpg",
...                      root + "/Abel_Pacheco/Abel_Pacheco_0004.jpg")
>>> print(Y)
1
```

普通照片可通过命令行先检测再验证：

```bash
python verification.py samples/009.jpg samples/009.jpg \
  --detect \
  --checkpoint checkpoints/my_experiments/vit_robust.pth \
  --calibration artifacts/my_experiments/vit_robust_eval/calibration.json
```

其中：

- `--checkpoint`：模型权重。
- `--calibration`：权重对应的判定阈值。
- `--detect`：先检测并对齐人脸，再进行验证。

打开终端，输入以上命令，我们会得到如下结果：

```bash
(cifar1) wzc@user-Rack-Server:~/datasets/casia/faceRecognitionCASIA$ python verification.py samples/009.jpg samples/009.jpg \
  --detect \
  --checkpoint checkpoints/my_experiments/vit_robust.pth \
  --calibration artifacts/my_experiments/vit_robust_eval/calibration.json
[ WARN:0@0.034] global net_impl_backend.cpp:345 setPreferableTarget Targets are not supported by the new graph engine for now
{
  "same_person": 1,
  "similarity": 1.0,
  "threshold": 0.2779906392097473,
  "boxes": [
    [
      63,
      42,
      188,
      212
    ],
    [
      63,
      42,
      188,
      212
    ]
  ]
}
```

输出包含 `same_person`、`similarity`、`threshold` 和人脸框 `boxes`。本次 ViT 的阈值为 **0.277990639**，相似度大于等于阈值时判为同人。

## 3. 模型与微调方法

### 3.1 从照片到验证结果

系统先用 YuNet 找到照片中的人脸，再根据双眼、鼻尖和两个嘴角的位置调整人脸的角度和大小。照片中没有人脸、出现多张人脸或文件损坏时，系统会提示错误。

识别部分使用 CVLFace 发布的预训练 ViT：`minchul/cvlface_adaface_vit_base_webface4m`。它接收 112×112 的彩色人脸图片，将图片分成 196 个小块，经过 24 层 Transformer，得到一个由 512 个数字组成的特征向量，用来表示这张脸。该模型已由作者在 WebFace4M 上使用 AdaFace 方法预训练，我们在它的基础上继续训练。

比较两张照片时，每张照片都处理两次：一次使用原图，一次使用左右镜像后的图片。模型分别提取特征，将两个向量各自调整为长度 1，再相加，并把结果也调整为长度 1。这叫作“翻转融合”；只提取原图特征则叫“单视图”。

两张照片各自得到最终特征后，计算它们的余弦相似度。分数越高，表示两张脸越相似；达到校准得到的阈值，就判断为同一个人。

### 3.2 如何微调 ViT

前期实验中，直接微调虽然让训练损失下降，但测试准确率变化很小。因此，我们从训练图片的权重和清晰度入手做了两项调整。

**第一，降低异常图片对训练的影响。** 用预训练 ViT 提取每张训练图片的特征，再将同一个人的特征取平均，作为这个人的参考特征。图片与参考特征越接近，训练时权重越大；差异特别大的图片，权重较低。设二者的余弦相似度为 `q`，权重为：

```text
w = clip((q - 0.25) / 0.5, 0, 1)
```

这里 `clip` 表示把结果限制在 0 到 1 之间。这种处理可以减弱部分错标或异常图片的影响，但也可能降低侧脸等困难样本的权重。

**第二，让模糊图片也尽量保留身份信息。** 训练时，每张图片有 50% 的概率左右翻转，另有 50% 的概率先缩小到 40、56 或 72 像素，再恢复为 112×112。模型需要从这些细节减少的图片中提取特征，并尽量接近预训练模型对对应清晰图片提取的特征。

训练目标由两部分组成：

```text
总损失 = 按样本权重计算的身份分类损失
       + 2 × 按样本权重计算的特征差异损失
```

身份分类损失使用 CosFace，让模型区分不同的人；特征差异损失使用 `1 - 余弦相似度`，让增强后图片的特征接近对应的清晰特征。CosFace 的尺度设为 32、间隔设为 0.15，分类器用各身份的参考特征初始化。

微调时只更新最后 4 个 Transformer 块、最后的 LayerNorm 和特征投影层，前 20 个块及 BatchNorm 的参数和统计量保持固定。这样既保留预训练模型已有的表示能力，也让部分网络适应本地训练数据。


## 4. 实验设置

对于原本的 FaceMobileNet 和 ViT 微调，我们使用相同的 CASIA 配置：4,096 个训练身份、65,536 张图像；256 个验证身份、4,096 张图像，每人 16 张，随机种子为 42。

| 设置 | FaceMobileNet 微调 | ViT 微调 |
|---|---|---|
| 更新范围 | 全网络 | 最后 4 块、LayerNorm、特征投影；固定 BN |
| 轮数 / batch size | 8 / 128 | 8 / 128 |
| 学习率 | 网络 5×10⁻⁵，分类器 10⁻³ | 块 2×10⁻⁵，特征头 4×10⁻⁵，分类器 10⁻³ |
| 特征保持权重 | 10 | 2 |
| 模型选择 | 清晰 CASIA 验证 balanced accuracy 最大 | 降质 CASIA 验证 AUC 最大，清晰 AUC 回退不超过 0.001 |
| 选中轮次 | 第 2 轮 | 第 6 轮 |

实验使用单张 NVIDIA RTX PRO 6000 Blackwell、PyTorch 2.9.1+cu128；ViT 训练采用 BF16。

LFW 采用两种评估协议：DevTrain 的 2,200 对仅用于选择阈值，固定阈值后在 DevTest 的 1,000 对上测试，同人、异人各 500 对；另按官方 6,000 对划分做十折，每折用其余 5,400 对选择阈值，在留出的 600 对上测试。十折期间只重新标定阈值，不重新训练网络，表中的“±”为十折准确率的标准差，不是置信区间。

Accuracy 表示全部配对中判断正确的比例；FAR 为异人误判同人的比例，FRR 为同人误判异人的比例。AUC 衡量不同阈值下的整体区分能力，EER 为 FAR 与 FRR 相等附近的错误率。Accuracy、AUC 越高越好，FAR、FRR、EER 越低越好。

## 5. 实验结果

### 5.1 翻转融合

| 模型 | DevTest 准确率 | 标准十折均值 ± 标准差 | DevTest AUC | EER | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|
| 原始 FaceMobileNet | 96.70% | 96.55% ± 0.93% | 0.989524 | 3.60% | 1.60% | 5.00% |
| 微调 FaceMobileNet | 97.10% | 96.73% ± 0.85% | 0.991092 | 3.20% | 2.20% | 3.60% |
| 预训练 ViT | 99.10% | 99.15% ± 0.40% | 0.990520 | 1.60% | 0.00% | 1.80% |
| 微调 ViT | 99.10% | 99.20% ± 0.41% | 0.990468 | 1.60% | 0.00% | 1.80% |

FaceMobileNet 微调后，DevTest 从 96.7% 提高到 97.1%，十折错误从 207 对降到 196 对。但 DevTest 误接受从 8 对增至 11 对，误拒绝从 25 对降至 18 对，说明准确率改善也伴随了取舍。

ViT 微调前后的 DevTest 均为 99.1%。十折准确率提高 0.05 个百分点，错误从 51 对降至 48 对；DevTest AUC 略降，EER 不变。因此本次微调有小幅十折收益，但没有全面超过预训练的模型。

### 5.2 单视图对照

| 模型 | DevTest 准确率 | 标准十折均值 ± 标准差 | 十折错误数 |
|---|---:|---:|---:|
| 原始 FaceMobileNet | 96.80% | 96.72% ± 0.77% | 197 |
| 微调 FaceMobileNet | 97.00% | 96.92% ± 0.85% | 185 |
| 预训练 ViT | 99.00% | 99.17% ± 0.39% | 50 |
| 微调 ViT | 99.00% | 99.15% ± 0.36% | 51 |

翻转融合对不同模型的影响不一致：它提高了微调 ViT 的十折成绩，却降低了微调 FaceMobileNet 的十折成绩，所以两种设置均保留。由上表可知，ViT 整体准确率高于本次 FaceMobileNet。

![三组实验对比](artifacts/my_experiments/summary/comparison.png)

### 5.3 训练过程

FaceMobileNet 总损失从 2.8602 降至 1.2956，验证 balanced accuracy 在第 2 轮达到 93.33%，后续未再超过。ViT 总损失从 0.11285 降至 0.02596，第 6 轮降质验证 AUC 为 0.964114，高于初始化的 0.962823，因此按规则保留第 6 轮。训练损失持续下降不代表测试准确率会持续提高，两组损失数值也不能直接比较。

![训练与模型选择](artifacts/my_experiments/summary/training_curves.png)


## 6. 复现与记录

以下命令需在项目根目录运行。CASIA 和 LFW 数据路径见 `config.py`。

```bash
# 1. 全量微调 FaceMobileNet，再评估
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

训练参数、数据划分和选中轮次位于各训练目录的 `training.json`、`split.json`；逐对预测和十折指标位于各评估目录的 `predictions.csv`、`evaluation.json`。

## 7. 参考资料

- [CVLFace 代码](https://github.com/mk-minchul/CVLface)与[固定版本 ViT 权重](https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m/tree/c6fed2ff09ac0a29cccc72c09127ad7b4bd3fa9e)。
- [Vision Transformer](https://arxiv.org/abs/2010.11929)、[AdaFace](https://arxiv.org/abs/2204.00964) 与 [CosFace](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html)。
- [YuNet 检测器](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。