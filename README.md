# Face Verification：有对照的 Transformer 微调

按 `说明.docx` 实现双图验证接口、人脸检测/对齐、照片及摄像头演示、真实 GPU 微调与实验报告。

## 用户独立复现实验

你在 `artifacts/my_experiments/` 完成的三组实验已单独整理，完整配置、单视图/翻转融合对照、训练曲线和指标解释见 [实验整理](artifacts/my_experiments/summary/summary.md)。以下为本次运行、统一翻转融合的结果，与下方此前自动实验分开记录。

| 本次模型 | DevTest 准确率 | 标准 LFW 十折均值 ± 折间标准差 |
|---|---:|---:|
| 微调 FaceMobileNet | 97.10% | 96.73% ± 0.85% |
| 预训练 ViT | 99.10% | 99.15% ± 0.40% |
| 微调 ViT | 99.10% | 99.20% ± 0.41% |

本次 ViT 微调十折净少错 3 对，提升 0.05 个百分点；DevTest 准确率不变，单视图十折略退，不能称为所有指标都提高。FaceMobileNet 微调也有准确率收益，但 FAR 从 1.6% 升到 2.2%。完整对照含原始 FaceMobileNet。汇总可用 `python scripts/summarize_experiments.py` 重新生成，并已加入主报告末尾。

## 本轮实际结果

所有比较均使用相同 ViT 初始化和相同原图/翻转融合，原始协议与降质诊断分开。

| 设置 | 原始 LFW DevTest | 原始 LFW 十折 | 40px/JPEG 诊断 DevTest |
|---|---:|---:|---:|
| 预训练 ViT | 99.10% | 99.15% | 98.10% |
| 质量加权增强微调 | 99.10% | 99.22% | 98.00% |
| 增强微调 + 类内变化抑制投影（默认） | 99.10% | 99.10% | 98.30% |

增强微调在原始十折中少错 4 对，提升约 0.067 个百分点，幅度小，不宣称统计显著。默认组合方法按 CASIA 清晰/降质验证 AUC 的既定规则选出：它改善了原始 DevTest AUC/EER 和降质 DevTest 准确率，但原始十折准确率略有下降，没有全面胜出。完整消融与新增身份审计见 [report.md](report.md) 和 [refinement/evaluation.json](artifacts/refinement/evaluation.json)。

## 安装与启动

当前 `cifar1` 环境已安装依赖。新环境建议 Python 3.11：

```bash
cd /home/wzc/datasets/casia/faceRecognitionCASIA
python -m pip install -r requirements.txt
FACE_DEVICE=cuda:7 python app.py
```

打开 **http://127.0.0.1:7860**。普通单人照片自动检测与对齐；已对齐图像勾选相应选项。支持摄像头拍照和连续验证。图片不保存到服务器文件。单卡机器改用 `cuda:0` 或 `cuda`；没有 CUDA 时可以在 CPU 推理，也可直接 `python app.py` 自动选设备。

默认是 `checkpoints/transformer.pth` 与 `artifacts/calibration.json`。本地已生成，使用演示不必重新训练。权重约 438 MiB，被 `.gitignore` 排除，Git 克隆不含权重；迁移时应单独复制模型及配套校准文件，或按下方步骤复现。

如果想体验**没有附加投影的增强微调模型**：

```bash
FACE_DEVICE=cuda:7 FACE_CHECKPOINT=checkpoints/transformer_robust.pth FACE_CALIBRATION=artifacts/refinement/calibration_robust.json python app.py
```

该模型为事先列出的消融候选；没有因为它的 LFW 十折分数更高就改写默认选择规则。

## 作业规定接口

```python
from FaceVerification import FaceVerification
root = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped"
Y = FaceVerification(root + "/Abel_Pacheco/Abel_Pacheco_0001.jpg",
                     root + "/Abel_Pacheco/Abel_Pacheco_0004.jpg")
print(Y)  # Python int，1=同人，0=异人
```

接口接收已对齐的人脸路径。普通照片可用 `FaceVerifier().compare(X1, X2, detect=True)`，或：

```bash
python verification.py samples/009.jpg samples/009.jpg --detect
python verification.py samples/cl.png samples/lxl.png --detect
```

默认决策为余弦相似度 `>= 0.252063662` 返回 1。相似度不是概率；阈值与模型 SHA-256 绑定，不允许混用。

## 训练和复现

一条命令复现本轮训练、选择、评估和报告：

```bash
python -m pip install -r requirements-dev.txt
python scripts/download_transformer.py
python scripts/run_refinement.py --device cuda:7
python -m pytest -q
```

分步执行：

```bash
python train.py --device cuda:7
python evaluate.py --device cuda:7
python evaluate_refinement.py --device cuda:7
python scripts/validate_system.py --device cuda:7
python scripts/build_report.py
python -m pytest -q
```

`train.py` 现在运行本轮完整训练与 CASIA 选择流程，不直接访问 LFW 测试标签。底层训练参数入口为 `train_transformer.py --help`：本轮使用质量加权 CosFace + 清晰目标一致性、50% 分辨率增强、最后 4 块微调、8 轮。`train_metric.py` 用训练身份闭式拟合正则化类内投影，并在 CASIA 验证集选择强度；这不是额外梯度训练。`scripts/select_refinement.py` 比较模块组合并启用默认模型。

训练身份 4,096 个/65,536 张，验证身份 256 个/4,096 张，二者不重叠。新增审计 256 个身份与这两部分均不重叠，不用于选模型；其中前 128 身份校准、后 128 身份评估。清单和选择规则都保存。LFW 采用 DevTrain 2,200 对校准、DevTest 1,000 对测试，以及标准 6,000 对十折协议。

低分辨率/JPEG 诊断固定为 40×40、JPEG quality=45、恢复 112×112，对各模型完全相同。它是额外实验，不能当作原始 LFW 成绩。LFW 已在前轮被观察，本轮是重复公开评估；微小变化仍需要新外部数据和多个随机种子验证。

## 文件与来源

| 文件/目录 | 内容 |
|---|---|
| `report.md` | 当前报告、完整模块消融、正负结果、运行方法和局限 |
| `verification.py` / `FaceVerification.py` | 模型加载与规定接口 |
| `model/face_vit.py` | ViT 与可选类内投影 |
| `train_transformer.py` / `train_metric.py` | 增强微调与监督度量拟合 |
| `refinement.py` | 确定性的降质诊断 |
| `scripts/run_refinement.py` | 训练到报告的完整流程 |
| `artifacts/refinement/experiment_plan.json` | 本轮候选与选择规则 |
| `artifacts/refinement/selection.json` | 只使用 CASIA 验证身份的选择结果 |
| `artifacts/refinement/evaluation.json` | 所有候选的清晰/降质/审计最终指标 |
| `artifacts/refinement/robust_training/` | 逐轮损失、实际梯度更新数、数据清单 |
| `artifacts/refinement/metric_training/` / `combined_training/` | 投影拟合与验证网格 |
| `artifacts/refinement/calibration_*.json` | 各模型可选部署配置 |
| `artifacts/evaluation.json` / `predictions.csv` | 默认模型与原网络的统一评估、逐对预测 |
| `third_party/` | CVLFace 固定初始化版本、指纹与 MIT 代码许可 |
| `assets/` | YuNet 检测器、来源和许可 |
| `experiments/vit_round1/` / `transformer_round1.pth` | 上一轮普通微调报告与模型备份（权重在 checkpoints） |
| `legacy/` / `blog/` | 原项目主要脚本和教程 |

初始化为作者公开的 CVLFace AdaFace ViT-Base WebFace4M，绝不声称从零完成其预训练。本轮微调与投影拟合均有实际记录；类内投影也不包装为新的注意力架构。16 项自动测试通过，网页上传、手机布局和模拟摄像头流程通过；真实摄像头硬件未测试。

原项目参考 insightface、InsightFace_Pytorch、arcface-pytorch；新骨干参考 [CVLFace](https://github.com/mk-minchul/CVLface)，完整致谢和参考资料见报告及 `third_party/`。
