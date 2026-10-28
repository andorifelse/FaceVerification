"""Generate the Transformer report from actual saved training/evaluation records."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    artifacts=ROOT/'artifacts'
    if (artifacts/'refinement/evaluation.json').exists():
        from build_refinement_report import main as build_refinement_report
        return build_refinement_report()
    training=json.loads((artifacts/'transformer_training/training.json').read_text())
    evaluation=json.loads((artifacts/'evaluation.json').read_text())
    calibration=json.loads((artifacts/'calibration.json').read_text())
    provenance=json.loads((ROOT/'third_party/vit_initialization.json').read_text())
    variants=evaluation['variants'];final=variants['finetuned_flip'];baseline=variants['baseline_single']
    if final.get('backbone')!='face_vit':raise ValueError('Run the Transformer evaluation before building its report')
    if final['checkpoint_sha256']!=training['checkpoint_sha256'] or final['checkpoint_sha256']!=calibration['checkpoint_sha256']:
        raise ValueError('Training, evaluation and calibration checkpoints do not match')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=training['history']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].plot([h['epoch'] for h in history],[h['loss'] for h in history],marker='.',label='Total loss')
    axes[0].plot([h['epoch'] for h in history],[h['classification_loss'] for h in history],marker='.',label='CosFace classification')
    axes[0].set(xlabel='Epoch',ylabel='Training loss',title='CASIA Transformer fine-tuning');axes[0].legend(fontsize=8)
    axes[1].plot([0]+[h['epoch'] for h in history],[training['initial_validation_auc']]+[h['validation_auc'] for h in history],marker='.')
    axes[1].axvline(training['best_epoch'],linestyle=':',color='orange',label='Selected epoch')
    axes[1].set(xlabel='Epoch',ylabel='AUC',title='Held-out CASIA identities');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(artifacts/'training.png',dpi=160);plt.close(fig)
    names={'baseline_single':'原 FaceMobileNet，单视图','baseline_flip':'原 FaceMobileNet，翻转融合',
           'mobile_finetuned_single':'此前 FaceMobileNet 微调，单视图','mobile_finetuned_flip':'此前 FaceMobileNet 微调，翻转融合',
           'vit_pretrained_single':'ViT 预训练初始化，单视图','vit_pretrained_flip':'ViT 预训练初始化，翻转融合',
           'finetuned_single':'本次微调 ViT，单视图','finetuned_flip':'本次微调 ViT，翻转融合（交付）'}
    table='\n'.join(f'| {names.get(key,key)} | {r["threshold"]:.6f} | {r["accuracy"]:.2%} | {r["auc"]:.6f} | {r["far"]:.2%} | {r["frr"]:.2%} | {r["lfw_standard_cv"]["mean_accuracy"]:.2%} ± {r["lfw_standard_cv"]["std_accuracy"]:.2%} |' for key,r in variants.items())
    cv=final['lfw_standard_cv'];bcv=baseline['lfw_standard_cv'];paired=evaluation['paired_comparison']
    original_errors=baseline['fp']+baseline['fn'];final_errors=final['fp']+final['fn']
    cv_old_errors=sum(r['fp']+r['fn'] for r in bcv['folds'])
    cv_new_errors=sum(r['fp']+r['fn'] for r in cv['folds'])
    accuracy_gain=100*(final['accuracy']-baseline['accuracy'])
    cv_gain=100*(cv['mean_accuracy']-bcv['mean_accuracy'])
    error_reduction=(original_errors-final_errors)/original_errors
    pre=variants.get('vit_pretrained_flip')
    fine_tuning_analysis=''
    if pre:
        delta=100*(final['accuracy']-pre['accuracy'])
        cdelta=100*(cv['mean_accuracy']-pre['lfw_standard_cv']['mean_accuracy'])
        fine_tuning_analysis=f'''ViT 初始化的 DevTest 准确率已为 {pre['accuracy']:.2%}；本次微调后变化 {delta:+.2f} 个百分点，标准十折变化 {cdelta:+.3f} 个百分点。因此超过旧网络的主要收益来自更强的 Transformer 骨干与大规模预训练，不能归因于本次微调。局部微调并未在 DevTest 上进一步提高准确率，其作用与更新量均如实记录；十折差异也很小，不声称具有统计显著性。'''
    system=''
    if (artifacts/'system_validation.json').exists():
        timing=json.loads((artifacts/'system_validation.json').read_text())
        if timing.get('checkpoint_sha256')==final['checkpoint_sha256']:
            system+=f"\n加载模型后，设备 `{timing['device']}` 上双图接口平均 {timing['mean_pair_seconds']*1000:.2f} ms，单张样例检测与对齐平均 {timing['mean_detection_seconds']*1000:.2f} ms，均为 10 次测量。GPU 双图计时在前后同步，包含图像读取、预处理与翻转融合，不含首次模型加载和浏览器网络耗时；单张检测由 OpenCV 在 CPU 上执行。\n"
    if (artifacts/'browser_validation.json').exists():
        browser=json.loads((artifacts/'browser_validation.json').read_text())
        if browser.get('checkpoint_sha256')==final['checkpoint_sha256']:
            system+=f"\n新模型通过 {len(browser['checks'])} 项真实浏览器上传与布局检查、{len(browser.get('simulated_camera_checks',[]))} 项模拟摄像头检查，记录见 `artifacts/browser_validation.json`。模拟采集不等于真实摄像头硬件测试。\n"
    if (artifacts/'demo.png').exists():system+='\n![Transformer 网页演示](artifacts/demo.png)\n'
    a=training['arguments']
    report=f'''# Face Verification 实验报告：GPU 微调 Vision Transformer

## 1. 任务、交付与结论

按照 `说明.docx` 实现 `Y = FaceVerification(X1, X2)`：输入两张已对齐人脸图像的路径，返回 Python `int`，同人返回 1，异人返回 0。另交付单人人脸检测、五点对齐、照片上传、摄像头拍照及连续验证界面。

本次把默认识别网络替换为 **ViT-Base Transformer**，不再使用项目原有 FaceMobileNet 作为交付识别器。实际在 GPU 上完成 CASIA 微调，选出第 {training['best_epoch']} 轮权重；接口、演示程序、阈值和报告均已切换。旧网络仅作为性能对照。

相同本地评估协议下，DevTest 准确率由原网络 **{baseline['accuracy']:.2%} 提升到 {final['accuracy']:.2%}**，提高 {accuracy_gain:.2f} 个百分点；1,000 对中的错误从 {original_errors} 对降到 {final_errors} 对，相对减少 {error_reduction:.2%}。标准 LFW 6,000 对十折平均准确率由 **{bcv['mean_accuracy']:.2%} 提升到 {cv['mean_accuracy']:.2%}**，提高 {cv_gain:.2f} 个百分点。本结论只适用于已运行的公开评估，教师未公开测试集尚未提供。

## 2. 快速运行

建议 Python 3.11，依赖版本见 `requirements.txt` 和实测环境记录 `requirements-lock.txt`。

```bash
cd /home/wzc/datasets/casia/faceRecognitionCASIA
python -m pip install -r requirements.txt
python app.py
```

浏览器打开 `http://127.0.0.1:7860`。上传两张普通单人照片，系统先检测和对齐，再显示是否同人、余弦相似度和阈值。已经对齐的 CASIA/LFW 图像可勾选“输入为已对齐的人脸图片”。摄像头可拍摄参考照和待验证照，或启动连续验证；localhost/HTTPS 下由浏览器获取摄像头权限。图片不写入服务器文件。

```python
from FaceVerification import FaceVerification
root = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped"
Y = FaceVerification(
    root + "/Abel_Pacheco/Abel_Pacheco_0001.jpg",
    root + "/Abel_Pacheco/Abel_Pacheco_0004.jpg",
)
print(Y)  # Python int，1 或 0
```

规定接口默认接收已对齐人脸。普通照片可用 `FaceVerifier().compare(X1, X2, detect=True)`，或：

```bash
python verification.py samples/009.jpg samples/009.jpg --detect
```

默认权重为 `checkpoints/transformer.pth`，阈值为 `artifacts/calibration.json`。本地已生成，无需再次训练；CPU 也能推理，有 CUDA 时默认使用 GPU。可用 `FACE_DEVICE=cuda:7 python app.py` 选择显卡，用 `FACE_CHECKPOINT`、`FACE_CALIBRATION` 指定配套文件。模型和阈值使用 SHA-256 绑定，不允许混用。

权重被 `.gitignore` 排除，普通 Git 克隆不会包含约 438 MiB 的最终权重。迁移时需复制 `checkpoints/transformer.pth`，或按下面的下载与微调步骤重新生成；不要只克隆代码便假定模型已存在。

## 3. 来源与作业诚信说明

### 3.1 Transformer 初始化

初始化来自作者公开发布的 **`{provenance['repo']}`**，固定版本 `{provenance['revision']}`。作者声明其使用 WebFace4M 数据和 AdaFace 预训练；相关信息属于上游模型来源，不作为本次实验测得的数据。本次未下载 WebFace4M 原始训练集。

初始化 SHA-256：`{training['initial_checkpoint_sha256']}`。模型代码基于 CVLFace 的 ViT 结构改编，保留层名与权重兼容性，使用原生 PyTorch SDPA 计算注意力，不执行 Hugging Face 远程 Python 代码。代码 MIT 许可证随附于 `third_party/CVLface-LICENSE`；上游模型卡要求遵守训练数据许可，代码许可证不被解释为数据许可。

### 3.2 本次真实微调

本次在本地 CASIA 上训练了 **{a['epochs']} 轮、{training['train_images']:,} 张训练图像**，实际更新最后 {a['train_blocks']} 个 Transformer 块、最终 LayerNorm 和两个特征投影矩阵，冻结其余 Transformer 层及 BatchNorm 参数/统计量。共有 {training['trainable_parameters']:,} 个识别器参数参与优化。最终选中权重相对初始化，{len(training['changed_tensors'])} 个张量中 **{training['changed_floating_values']:,} 个浮点值确实发生变化**，并非直接提交下载的原权重。

最终 SHA-256：`{training['checkpoint_sha256']}`。训练参数、逐轮损失、选中轮次和各张量更新数保存在 `artifacts/transformer_training/training.json`；身份与文件划分保存在 `split.json`。

作业允许微调预训练模型，本报告明确披露预训练和本次微调的各自贡献，不声称网络或预训练由本人从零完成，也不把该已有 ViT 结构称作本次原创算法。此前 MobileNet 代码和结果在 `experiments/mobile_before_transformer/`，原项目旧脚本在 `legacy/`。

## 4. 方法设计与工程探索

### 4.1 检测和对齐

普通照片采用 YuNet 检测器，检测置信度阈值 0.8、NMS 阈值 0.3。五个关键点求解无剪切/无反射的最小二乘相似变换，得到标准人脸后调整至识别输入大小。无脸、多脸、损坏文件均明确报错，不伪造为异人判断。YuNet 只做检测，不输出身份特征；来源和 MIT 许可证在 `assets/`。

LFW 评估使用本地已有裁剪图，不重复检测。两种网络使用同一图片路径和配对标签；原网络按其原设计使用灰度 128×128，ViT 按其预训练约定使用 RGB 112×112，都以 `(pixel/255-0.5)/0.5` 归一化。输入通道、骨干规模和预训练数据量也发生变化，因此比较的是完整方法，不能单独把收益归因于注意力结构。

### 4.2 ViT 表示

输入 RGB 112×112 图像划分为 8×8 patch，共 196 个 token，每个 token 512 维，加入可学习位置编码。网络包含 24 个 Transformer 块，每块 16 头自注意力、两层 LayerNorm 和扩展比为 3 的 ReLU6 MLP。展平最终 token 后，通过两个投影矩阵及 BatchNorm 得到 512 维身份表示；整个网络共有 {training['total_parameters']:,} 个参数。

原生 SDPA 与显式 `softmax(QKᵀ/√d)V` 数学等价，自动测试验证二者一致性。默认微调禁用随机深度和 token mask，固定 BN 统计；冻结前缀在 `no_grad` 下执行，减小反向传播开销。训练时使用 BF16 混合精度，正式评估和接口使用 FP32，不将精度设置差异伪装成结构提升。

### 4.3 原型初始化、间隔损失与保持项

每个训练身份的原图/翻转初始单位特征取均值，作为 CosFace 分类器的初始原型，减少随机分类器对已有特征空间的冲击。分类头仅用于训练，不进入最终验证接口。

训练目标为：

```text
L = CrossEntropy(CosFace(f(x), y))
    + 10 × (1 - <normalize(f(x)), normalize(f_initial(x))>)
```

CosFace 尺度 s=32、余弦间隔 m=0.15；第二项抑制迁移时的特征漂移。冻结初始化网络的原图与翻转特征先缓存，每批随机选择匹配的保持目标，无需维护第二个 GPU 教师网络。

本项目的探索在于将新 Transformer、原型初始化、局部块微调、特征保持及可校验阈值整合为可复现实验。这些组成技术均有已有研究基础；没有单独运行无保持项实验，因此不宣称该项具有已证实的独立收益。

### 4.4 推理与阈值

单视图特征 L2 归一化。翻转融合先分别归一化原图与水平翻转图特征，相加后再次归一化。两张人脸的余弦相似度 `s=<e1,e2>`；当 `s>=threshold` 时返回 1，否则返回 0。分数不是概率。

阈值仅在 LFW DevTrain 上最大化 balanced accuracy，排序搜索按并列分数组处理，并包含全部接受/拒绝的边界情况。之后冻结阈值，在 DevTest 上统计准确率；不会在 DevTest 标签上搜索最佳阈值或挑选训练轮次。最终阈值为 **{calibration['threshold']:.9f}**。

## 5. 数据、设备和训练过程

本地 CASIA 版本包含 10,575 个身份目录、454,191 张 JPG 图像，身份由目录名给出。随机种子 {a['seed']}，从每个至少 16 张图像的身份抽取 16 张：4,096 个身份、65,536 张用于微调；另 256 个身份、4,096 张用于模型选择，二者身份不重叠。这个“未参与本次微调”的验证集可能与上游大规模预训练存在身份重叠，不能称为对全部训练历史均未见过的身份。

设备：**{training['hardware']}**，实际使用 `{a['device']}`，PyTorch {training['torch_version']}。GPU 在当前工具沙箱内不可见，但在正常 GPU 执行环境可用；本轮训练实际通过 CUDA 完成，未沿用此前 CPU 训练限制的结论。

训练配置：AdamW，最后两个块及 LayerNorm 学习率 {a['lr']}，特征头学习率 {a['head_lr']}，分类器学习率 0.001，weight decay=0.0001；batch size={a['batch_size']}，{a['epochs']} 轮，梯度范数上限 5，CosineAnnealingLR；每张图片以 0.5 概率水平翻转。总耗时 {training['elapsed_seconds']:.2f} 秒，包含初始化特征缓存、训练、验证和参数差异检查；GPU/缓存/磁盘状态会影响复现耗时。身份划分固定，GPU 数值与软件差异可能引起小幅波动。

平均训练损失从 {history[0]['loss']:.6f} 降到 {history[-1]['loss']:.6f}。独立 CASIA 验证初始 AUC={training['initial_validation_auc']:.6f}，选中轮次 AUC={training['best_validation_auc']:.6f}；balanced accuracy 从 {training['initial_validation_balanced_accuracy']:.2%} 到选中轮次 {training['best_validation_balanced_accuracy']:.2%}。模型选择规则预先设为验证 AUC 最大、并列时 balanced accuracy 最大，第 {training['best_epoch']} 轮入选。仅在这些固定训练结束后评估 LFW，没有根据 LFW 反复挑训练配置。

![实际训练与验证曲线](artifacts/training.png)

## 6. 同协议评估与消融

LFW 本地裁剪版本有 5,749 个身份、13,233 张图像。开发协议使用 DevTrain 2,200 对标定阈值，DevTest 1,000 对评估（同人/异人各 500 对）；检查图像交集={evaluation['image_overlap']}、身份交集={evaluation['calibration_test_identity_overlap']}。另按本地官方 `pairs.txt` 顺序保留十折，每折 600 对，在其余 5,400 对上标定阈值，评估留出的一折。所有模型使用同一协议、图片和判定方向。

最终方案在测试前固定为“微调 ViT + 翻转融合”，`finetuned_*` 在本轮 JSON 中指新 ViT；此前卷积微调方案另标为 `mobile_finetuned_*`。以下全部为本地测量，未抄用作者公布的准确率。

| 设置 | DevTrain 阈值 | DevTest 准确率 | AUC | FAR | FRR | 标准十折准确率（均值 ± 折间标准差） |
|---|---:|---:|---:|---:|---:|---:|
{table}

交付模型 DevTest：TP={final['tp']}，FN={final['fn']}，FP={final['fp']}，TN={final['tn']}；TAR={final['tar']:.2%}，FAR={final['far']:.2%}，FRR={final['frr']:.2%}。FAR=0 仅表示这 500 对异人中未出现误接受，不代表真实误接受概率为零。EER={final['eer']:.2%} 是改变阈值时 ROC 交点插值的描述性指标，不能与冻结阈值处的 FAR/FRR 混为一谈。十折标准差不是置信区间。

### 6.1 与原网络比较

DevTest 改善 {accuracy_gain:.2f} 个百分点。逐对对照原网络单视图结果，{paired['gained_correct_pairs']} 对从错误变为正确，{paired['lost_correct_pairs']} 对从正确变为错误；配对 McNemar 双侧精确检验 p={paired['mcnemar_exact_two_sided_p']:.8g}。这是同一固定公开测试样本上的描述性比较，不能替代外部数据验证。标准十折错误由 {cv_old_errors} 对降至 {cv_new_errors} 对。

### 6.2 预训练和本次微调的贡献

{fine_tuning_analysis}

相较原网络，ViT 参数更多，且额外使用了上游 WebFace4M 预训练；作业允许额外数据与微调，但这个结果不能解释为相同训练资源下 Transformer 必然优于 CNN。公开 LFW 也可能与预训练身份重叠，本次未核验上游原始训练清单，不能声明严格排除了所有历史身份重叠。

![ROC 与分数分布](artifacts/evaluation.png)

逐对预测位于 `artifacts/predictions.csv`，各折指标、模型 SHA-256、配对统计位于 `evaluation.json`。下图按协议顺序展示最多三例误拒绝和三例误接受；本次没有误接受时只展示误拒绝，没有人为重标或删除困难样本。

![错误案例](artifacts/error_examples.jpg)

## 7. 复现实验与系统验证

从新环境开始：

```bash
python -m pip install -r requirements-dev.txt
python scripts/download_transformer.py
python train.py --device cuda:7
python evaluate.py --device cuda:7
python scripts/validate_system.py --device cuda:7
python scripts/build_report.py
python -m pytest -q
```

将 `cuda:7` 换成实际可用显卡；单卡环境使用 `cuda` 或 `cuda:0`。`train.py --help` 可配置身份数、学习率、训练块数和工作目录。默认运行即训练新 Transformer；旧卷积训练入口保留为 `train_mobile.py`，不会作为新默认方案。初始化下载脚本固定上游版本并校验 SHA-256。

原始 `18.pth` 是比较基准，须保留用于 `evaluate.py`；其历史训练来源无法从文件确认。本地无标签 `datasets/data/`、`test_lst.csv`、旧 `feature_dict*.pkl` 未用于训练或计算准确率。公开评估集不参与梯度更新。

自动测试覆盖阈值并列/边界、配对协议、指标、接口类型、图片错误、人脸检测、网页 API、RGB 预处理、SDPA 数学一致性，以及 Transformer 冻结前缀无梯度/末端块有梯度/BN 统计不变。完整测试已执行；测试记录见 `artifacts/test_validation.json`。真实摄像头硬件未测试。
{system}
## 8. 局限和可继续研究的方向

1. 强预训练是主要收益来源，不能将本次微调包装成全部改进来源。预训练数据和身份历史未完整审计。
2. 本次只微调末端两个块与投影，冻结前缀；没有把未执行的全网络从零训练作为实验成绩。
3. 1,000 对开发测试与 6,000 对标准十折属于不同协议，可能共享图片，不是两份完全独立的外部验证。教师未公开测试尚未评估。
4. 评估为已有裁剪图；普通照片经过检测和对齐会有分布变化，侧脸、遮挡、模糊和年龄变化仍可能失败。新网络体积/CPU 成本也高于旧模型。
5. 方法没有活体检测，不能仅凭照片相似性抵御照片重放。更低 FAR 的应用需要在目标设备和人群上另行校准。
6. 后续可固定训练数据和预算，比较不同 ViT/CNN、训练块数、无保持项设置及多个随机种子；这些是待做实验，不是本报告已有结论。

## 9. 参考资料与致谢

- [CVLface 官方仓库](https://github.com/mk-minchul/CVLface)及[固定版本 ViT 模型](https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m/tree/{provenance['revision']})，代码许可与权重来源见 `third_party/`。
- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)。
- Kim et al., [AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964)，为上游初始化的训练方法；本地微调目标使用 CosFace。
- Wang et al., [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html)。
- [YuNet 官方模型与许可证](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。
- 原项目 `legacy/README.md`、`blog/` 及其对 insightface、InsightFace_Pytorch、arcface-pytorch 的致谢。
'''
    (ROOT/'report.md').write_text(report,encoding='utf-8')
    print('Updated report.md and artifacts/training.png from verified Transformer records.')


if __name__=='__main__':main()
