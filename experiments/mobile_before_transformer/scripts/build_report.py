"""Build report.md and figures from the saved experiment records."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    artifacts = ROOT/'artifacts'
    training = json.loads((artifacts/'training/training.json').read_text())
    evaluation = json.loads((artifacts/'evaluation.json').read_text())
    calibration = json.loads((artifacts/'calibration.json').read_text())
    final = evaluation['variants']['finetuned_flip']
    variants = evaluation['variants']
    names = {'baseline_single':'原权重，单视图','baseline_flip':'原权重，翻转融合',
             'finetuned_single':'微调，单视图','finetuned_flip':'微调，翻转融合（交付）'}
    table = '\n'.join(f'| {names[key]} | {row["threshold"]:.6f} | {row["accuracy"]:.2%} | {row["auc"]:.4f} | {row["far"]:.2%} | {row["frr"]:.2%} | {row["lfw_standard_cv"]["mean_accuracy"]:.2%} ± {row["lfw_standard_cv"]["std_accuracy"]:.2%} |' for key,row in variants.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = training['history']
    fig,axes = plt.subplots(1,2,figsize=(10,3.6))
    axes[0].plot([h['epoch'] for h in history],[h['loss'] for h in history],marker='.')
    axes[0].set(xlabel='Epoch',ylabel='Training loss',title='CosFace + feature anchoring')
    axes[1].plot([h['epoch'] for h in history],[h['validation_balanced_accuracy'] for h in history],marker='.')
    axes[1].axhline(training['initial_validation_balanced_accuracy'],color='gray',linestyle='--',label='Initial model')
    axes[1].axvline(training['best_epoch'],color='orange',linestyle=':',label='Selected epoch')
    axes[1].set(xlabel='Epoch',ylabel='Balanced accuracy',title='Held-out CASIA identities')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(artifacts/'training.png',dpi=160);plt.close(fig)
    extra = ''
    if (artifacts/'system_validation.json').exists():
        validation = json.loads((artifacts/'system_validation.json').read_text())
        extra += f"\n系统实测：CPU 上双图接口平均 {validation['mean_pair_seconds']*1000:.1f} ms（已加载模型，包含读取和翻转融合，10 次），单图检测与对齐平均 {validation['mean_detection_seconds']*1000:.1f} ms（10 次）。这是当前机器上的小规模计时，随硬件和输入尺寸变化。\n"
    if (artifacts/'demo.png').exists():
        extra += '\n![真实浏览器上传验证界面](artifacts/demo.png)\n'
    if (artifacts/'browser_validation.json').exists():
        browser = json.loads((artifacts/'browser_validation.json').read_text())
        extra += f"\n浏览器实测（Chrome {browser['browser']}）：通过 {len(browser['checks'])} 项上传、输入提示、手机布局、清空与控制台检查；另通过 {len(browser.get('simulated_camera_checks', []))} 项模拟摄像头检查，涵盖开启、拍摄、连续验证与关闭。没有真实摄像头硬件测试。记录位于 `artifacts/browser_validation.json`。\n"
    delta_head = (variants['finetuned_flip']['accuracy']-variants['baseline_flip']['accuracy'])*100
    delta_flip = (variants['finetuned_flip']['accuracy']-variants['finetuned_single']['accuracy'])*100
    cv = final['lfw_standard_cv']
    report = f'''# Face Verification 实验报告

## 1. 任务与完成情况

输入两张人脸图像的路径，输出是否属于同一个人，严格提供 `Y = FaceVerification(X1, X2)`：`Y` 为 Python `int`，同人返回 1，异人返回 0。同时交付支持照片上传、摄像头拍照、连续验证、人脸框和结果反馈的网页系统。

本报告数值均来自本地实际运行，原始记录保存在 `artifacts/`。独立 LFW DevTest 1,000 对上的交付模型准确率为 **{final['accuracy']:.2%}**，AUC 为 **{final['auc']:.4f}**；标准 LFW 6,000 对十折准确率为 **{cv['mean_accuracy']:.2%} ± {cv['std_accuracy']:.2%}**。两种协议分别报告，不能互相混用。教师未公开测试集未提供，未对其性能作推测。

## 2. 代码与权重来源声明

基于随项目提供的 FaceMobileNet、CosFace、ResIRSE 等代码改造，原始 README 已注明参考 insightface、InsightFace_Pytorch、arcface-pytorch，本次保留这些致谢及博客。修改前的关键脚本另存于 `legacy/`。

项目原有 `checkpoints/18.pth` 被用作初始化。其 SHA-256 为 `{training['initial_checkpoint_sha256']}`。仅凭已有文件无法证明其最初训练者、使用数据、训练轮数，也不能确认其原始训练是否与 LFW 存在身份重叠，因此本实验不声称完全从零训练或严格排除历史数据重叠。

**本次实际微调**了识别网络的 2048→512 投影层；最终 checkpoint 相对初始化有 **{training['changed_floating_values']:,} 个浮点值发生变化**。未下载或直接提交未经微调的现成身份验证模型。最终文件为 `checkpoints/finetuned.pth`，SHA-256 `{training['checkpoint_sha256']}`。训练代码和逐轮日志均随附。

YuNet 是额外使用的开源**检测器**，只提供人脸框和五个关键点，不输出身份特征；来源与 MIT 许可证保存在 `assets/`。作业说明允许使用开源人脸检测器，MTCNN 为可选方案，本实现使用较轻量的 YuNet。

## 3. 运行方法

建议 Python 3.11。首次运行无需训练或下载模型，随附微调权重、检测器和已校准阈值。

```bash
cd /home/wzc/datasets/casia/faceRecognitionCASIA
python -m pip install -r requirements.txt
python app.py
```

浏览器打开 `http://127.0.0.1:7860`。上传两张普通照片后点击“开始验证”；系统依次检测、对齐、提取特征、比较并显示分数和阈值。无脸或多脸会明确提示。摄像头功能由浏览器采集，可分别拍照或每轮完成后间隔 1 秒连续验证。访问摄像头需要浏览器允许权限；远程访问一般需要 HTTPS。服务器不会把上传图片保存到文件。

若输入是已经对齐的 CASIA/LFW 裁剪图，勾选“输入为已对齐的人脸图片”，避免再次检测改变裁剪分布。

```python
from FaceVerification import FaceVerification
root = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped"
Y = FaceVerification(
    root + "/Abel_Pacheco/Abel_Pacheco_0001.jpg",
    root + "/Abel_Pacheco/Abel_Pacheco_0004.jpg",
)
print(Y)  # int，1 或 0
```

规定接口采用已对齐输入。普通照片调用 `FaceVerifier().compare(X1, X2, detect=True)`。命令行示例：

```bash
python verification.py samples/009.jpg samples/009.jpg --detect
```

缺失文件、损坏图片、无脸或多脸属于输入错误，会抛出异常或在界面提示，不伪造为异人结果。余弦相似度不代表概率。

## 4. 方法

### 4.1 检测、关键点与预处理

普通照片首先缩放到最长边不超过 960 像素，使用 YuNet 检测；置信度阈值为 0.8，NMS 阈值为 0.3。用双眼、鼻尖、嘴角五点最小二乘求解无剪切的相似变换，得到 128×128 人脸。评估数据已对齐，直接缩放而不重复检测。

识别输入统一灰度化，缩放到 128×128，以 `(pixel/255 - 0.5)/0.5` 归一化。灰度可减轻色彩与部分照明差异，但也会损失信息；沿用原网络的单通道设计以保持权重兼容。

### 4.2 轻量表示与可负担微调

沿用项目 FaceMobileNet：深度可分离卷积、残差块、512 通道末端卷积、2048 维卷积表示、512 维线性投影与 BatchNorm。不同于把训练身份分类器当作验证器，推理时丢弃分类头，仅使用归一化的特征比较未见身份。

当前环境 `torch.cuda.is_available()` 为 False，未使用 GPU。默认冻结所有卷积层及 BN 参数和运行统计，只微调线性投影。缓存每张训练图像及水平翻转图像的卷积表示，避免每轮重复计算。这种缓存与冻结特征的直接前向传播数值等价；测试检查了分解前向传播的一致性。

分类器以各训练身份的初始平均归一化特征初始化，使用 CosFace 加性余弦间隔损失，尺度 `s=30`，间隔 `m=0.2`。额外加入初始表示保持项：

```text
L = CrossEntropy(CosFace(f(x), y)) + 10 * (1 - <normalize(f(x)), normalize(f_initial(x))>)
```

保持项约束小样本微调偏移，试图降低遗忘。它是本项目的工程探索，未与无保持项设置做独立消融，不能据此声称该项提高了准确率，也不声称该组合是新的论文级算法。

### 4.3 翻转融合与决策

分别计算原图和水平翻转图的单位特征，二者相加后再次 L2 归一化。两张人脸的余弦相似度 `s = <e1, e2>`，当且仅当 `s >= threshold` 时输出 1。

阈值只在独立校准数据上选择，目标为 balanced accuracy，等价于正负样本等权。排序后分组处理相同分数，同时覆盖全部接受/全部拒绝的边界情况，时间复杂度 O(n log n)，避免原脚本 O(n²) 搜索及在测试集本身选阈值的问题。

阈值文件绑定模型 SHA-256 和是否翻转融合；更换模型或融合设置时必须重新校准，防止错配。

## 5. 数据与实验设置

本地 `CASIA-WebFace-Align-96` 实际包含 10,575 个身份目录和 454,191 张 JPG 图片（本地版本计数，不套用标准数据集规模）。目录名作为身份标签。随机种子为 42，从每个至少有 16 张图像的身份抽取 16 张；512 个身份、8,192 张用于微调，另 128 个身份、2,048 张用于模型选择，二者身份完全分离。划分保存在 `artifacts/training/split.json`。

LFW 本地裁剪版本有 5,749 个身份和 13,233 张图片。DevTrain 2,200 对仅用于阈值标定；DevTest 1,000 对仅用于冻结阈值后的评估。本地检查校准/测试图像交集为 {evaluation['image_overlap']}，身份交集为 {evaluation['calibration_test_identity_overlap']}。另依照 `pairs.txt` 的顺序将 6,000 对分成官方十折，每次在其余九折标定阈值后评估留出的一折。

默认配置在最终 DevTest 结果计算前固定为“微调权重 + 翻转融合”。模型选择只看 CASIA 验证集，不看 LFW 测试标签。四组消融均报告，未根据 DevTest 重选默认方案。十折结果是补充协议，不是另一个完全独立的数据集；它与开发协议可能共享图片。初始权重的数据历史未知，因此不能据此宣称对完全未见人物的严格独立泛化。

微调参数：16 轮，batch size=128，AdamW，投影层初始学习率 0.00005，分类头学习率 0.001，weight decay=0.0005，CosineAnnealingLR，梯度范数上限 5；每批图像独立随机翻转。冻结模式保持 BN 为 eval。训练版本 PyTorch {training['torch_version']}、NumPy {training['numpy_version']}，CPU 线程数 8。

训练耗时 {training['elapsed_seconds']:.1f} 秒（含首次卷积缓存）；平均训练损失从 {history[0]['loss']:.4f} 降至 {history[-1]['loss']:.4f}。CASIA 验证 balanced accuracy 从 {training['initial_validation_balanced_accuracy']:.2%} 变为最佳 {training['best_validation_balanced_accuracy']:.2%}，第 {training['best_epoch']} 轮入选。该值用于选择模型，不能当作独立测试成绩。

![训练及模型选择曲线](artifacts/training.png)

原有 `datasets/data/` 中无身份标签的图像、`test_lst.csv` 无标签配对和 `feature_dict*.pkl` 未参与训练、阈值标定或准确率报告，避免把未知数据当训练数据。

## 6. 实测结果与消融

| 设置 | DevTrain 阈值 | DevTest 准确率 | AUC | FAR | FRR | 标准十折准确率（均值 ± 折间标准差） |
|---|---:|---:|---:|---:|---:|---:|
{table}

交付模型 DevTest 混淆矩阵：TP={final['tp']}，FN={final['fn']}，FP={final['fp']}，TN={final['tn']}。同人接受率 TAR={final['tar']:.2%}，异人误接受率 FAR={final['far']:.2%}，同人误拒绝率 FRR={final['frr']:.2%}。EER={final['eer']:.2%} 为 ROC 线性插值的描述性指标，未用它替换交付阈值。十折标准差反映折间波动，不是置信区间。

同样采用翻转融合时，本次微调相对原权重的 DevTest 准确率变化为 {delta_head:+.2f} 个百分点；微调模型使用翻转融合相对单视图变化为 {delta_flip:+.2f} 个百分点。该小规模实验不证明普遍优势，正负变化均如实保留。一个百分点在 1,000 对中只有 10 对，需更大数据和多个随机种子才能判断稳定性。

![ROC 与同人/异人相似度分布](artifacts/evaluation.png)

最终阈值为 `{calibration['threshold']:.9f}`，逐对预测在 `artifacts/predictions.csv`，各折指标在 `artifacts/evaluation.json`。相似度分布重叠处会产生误接受和误拒绝，默认目标为等权准确率，并未针对低误接受应用优化。

下图按协议顺序各展示最多三例误拒绝与误接受，标签来自开发测试协议，不做人工重标。误拒绝会把同人的姿态、表情或裁剪差异当成身份差异；误接受说明外观相似时单一余弦分数仍可能混淆。这里只是错误案例观察，不是对误差原因的受控实验。

![DevTest 错误样例](artifacts/error_examples.jpg)

## 7. 系统验证与复现

```bash
python train.py
python evaluate.py
python scripts/validate_system.py
python scripts/build_report.py
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

使用 `python train.py --help` 查看身份数量、图像数、学习率及数据路径等配置；`--full` 可微调全部网络，`--scratch --full` 可随机初始化训练。它们是可用扩展入口，本报告没有把未执行的全量训练当成已完成实验。

自动测试覆盖：阈值并列和边界、混淆矩阵/ROC、官方及路径配对格式、头部缓存前向等价性、同图一致性与交换对称性、模型/阈值校验、无脸/多脸拒绝、真实单脸检测、图片上传及损坏图片处理，以及规定接口的返回类型。摄像头实际采集取决于运行电脑的硬件与浏览器授权；当前无可交互摄像头，未声称实测真人实时采集。
{extra}
## 8. 局限与后续探索

1. 基础权重来源与历史训练划分不可追溯；报告明确区分项目已有工作和本次微调，若需要完全可审计的从零训练，应运行随机初始化的全网络训练并保存数据清单。
2. 只微调 512 个身份的投影层，不能替代在完整 CASIA 上训练卷积网络。CASIA 验证表现不保证跨数据集性能；保持项的独立价值仍需消融。
3. LFW 是公开数据且图像经过裁剪，本地评估不是教师未公开测试的替代品。不同检测裁剪、姿态、模糊、遮挡、光照可能导致分布偏移；普通照片检测后的最终误差没有单独的标注评估集。
4. 灰度轻量网络与两视图融合存在速度/准确率取舍；未来可在固定身份划分下比较 RGB、更强骨干、ArcFace，以及更多随机种子。
5. 系统只验证照片相似性，没有活体检测，也不把它当作防照片攻击的身份认证系统。涉及高安全阈值时，应在对应人群与设备上按目标 FAR 重新校准。

## 9. 参考资料与致谢

- 原项目 `README.md` 与 `blog/` 中的网络、训练、度量教程及其上游致谢。
- Wang et al., [CosFace: Large Margin Cosine Loss for Deep Face Recognition, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html)。本项目使用的加性余弦间隔目标来自该工作，并非本次提出。
- Wu, Peng, Yu, [YuNet 官方实现和模型说明](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。采用其中 2023mar 检测权重及 MIT 许可证。
- [scikit-learn LFW 数据加载与协议源码](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py)，用于核对开发训练、开发测试及十折协议；计算实际使用本地 `pairs*.txt`。
'''
    (ROOT/'report.md').write_text(report,encoding='utf-8')
    print('Saved report.md and artifacts/training.png')


if __name__ == '__main__':
    main()
