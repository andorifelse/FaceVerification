# Face Verification 作业交付

已按 `说明.docx` 完成双图验证接口、检测与关键点对齐、上传/摄像头演示、真实 CASIA 微调、独立阈值校准及实验报告。

## 安装和运行

建议 Python 3.11。现有 `venv/` 是旧的 Windows 环境，请重新创建环境。

```bash
cd /home/wzc/datasets/casia/faceRecognitionCASIA
python -m pip install -r requirements.txt
python app.py
```

浏览器打开 **http://127.0.0.1:7860**，上传两张单人人脸照片并点击“开始验证”。可以开启摄像头、拍摄参考照和待验证照，也可以连续验证。数据集图片已经裁剪对齐时，勾选“输入为已对齐的人脸图片”。图片不写入服务器文件。

## 作业规定接口

```python
from FaceVerification import FaceVerification

X1 = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped/Abel_Pacheco/Abel_Pacheco_0001.jpg"
X2 = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped/Abel_Pacheco/Abel_Pacheco_0004.jpg"
Y = FaceVerification(X1, X2)  # Python int，1=同一个人，0=不同的人
print(Y)
```

规定接口接收已对齐的人脸图片。普通照片可调用 `FaceVerifier().compare(X1, X2, detect=True)`，或使用命令行：

```bash
python verification.py samples/cl.png samples/lxl.png --detect
```

默认使用已交付的 `checkpoints/finetuned.pth` 和 `artifacts/calibration.json`，首次运行无需再次训练或联网。模型与阈值文件通过 SHA-256 绑定；更换模型后应重新校准。可用 `FACE_CHECKPOINT`、`FACE_CALIBRATION` 环境变量指定文件。

## 复现实验

```bash
python train.py
python evaluate.py
python scripts/validate_system.py
python scripts/build_report.py
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`train.py` 默认冻结卷积层，在 512 个 CASIA 身份上进行 16 轮投影层微调；128 个不参与微调的身份用于选择模型。`--full` 微调全部层，`--scratch --full` 从随机初始化训练全部层。两者所需时间更长，未作为报告中的已完成实验。

数据路径可通过 `train.py --data-root`、`evaluate.py --root` 指定；所有默认路径相对项目目录解析。首次卷积特征缓存耗时较多，后续相同数据划分/权重复用缓存。

评估采用 LFW DevTrain 的 2,200 对校准阈值，DevTest 的 1,000 对用于独立评估；另提供官方 6,000 对的十折结果。**不会在同一份测试标签上选择阈值再报告最佳准确率。**

## 交付文件

| 文件 | 作用 |
|---|---|
| `report.md` | 实验报告：运行方法、方法设计、来源说明、实测结果、局限 |
| `FaceVerification.py` / `verification.py` | 规定接口、推理类和命令行 |
| `app.py` / `templates/index.html` | 上传照片及摄像头演示 |
| `detection.py` / `assets/` | YuNet 人脸检测与五点相似变换对齐 |
| `train.py` / `dataset.py` | CASIA 身份抽样、可复现微调 |
| `evaluate.py` / `metrics.py` | 校准、独立评估、ROC、混淆矩阵、十折协议 |
| `checkpoints/finetuned.pth` | 本次实际微调后选出的权重 |
| `artifacts/calibration.json` | 与模型绑定的阈值 |
| `artifacts/evaluation.json` / `predictions.csv` | 各组指标与逐对预测 |
| `artifacts/training/split.json` / `training.json` | 数据划分及逐轮训练记录 |
| `tests/` | 协议、阈值、接口、检测及网页 API 测试 |
| `legacy/` / `blog/` | 修改前的关键脚本备份和原项目教程 |

`18.pth` 为项目原有权重，训练来源未知；本次没有下载现成验证网络，实际更新了投影层参数，详见报告。下载的 YuNet 仅用于检测，来源和许可证位于 `assets/`。原有 `feature_dict*.pkl`、`datasets/data/` 和无标签 `test_lst.csv` 不参与训练、校准或准确率统计。

---

# 原项目与致谢

原项目为 Build Your Own Face Recognition Model，模型与度量函数参考了以下开源实现：

- [insightface](https://github.com/deepinsight/insightface)
- [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)

原始介绍和运行方法保存在 `legacy/README.md`；旧版依赖与路径仅供历史参考。原教程在 [blog/](blog/)，本次实验与交付请以本 README 和 [report.md](report.md) 为准。
