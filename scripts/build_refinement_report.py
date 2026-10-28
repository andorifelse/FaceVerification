"""Report the selected refinement and all measured controls without hiding regressions."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    work=ROOT/'artifacts/refinement'
    training=json.loads((work/'robust_training/training.json').read_text())
    result=json.loads((work/'evaluation.json').read_text())
    selection=json.loads((work/'selection.json').read_text())
    metrics=json.loads((work/'metric_training/training.json').read_text())
    combined=json.loads((work/'combined_training/training.json').read_text())
    current=json.loads((ROOT/'artifacts/evaluation.json').read_text())
    calibration=json.loads((ROOT/'artifacts/calibration.json').read_text())
    chosen=selection['selected'];models=result['models'];final=models[chosen];pre=models['pretrained']
    if final['checkpoint_sha256']!=calibration['checkpoint_sha256'] or final['checkpoint_sha256']!=current['variants']['finetuned_flip']['checkpoint_sha256']:
        raise ValueError('Default evaluation, selected refinement and calibration hashes disagree')
    names={'pretrained':'原预训练 ViT','round1':'上一轮普通微调','metric':'仅类内变化抑制投影','robust':'质量加权增强微调','combined':'增强微调 + 类内变化抑制投影'}
    clean=final['clean'];diag=final['diagnostic'];baseline=pre['clean'];bdiag=pre['diagnostic']
    def table(condition):
        return '\n'.join(f'| {names[name]}{"（选定）" if name==chosen else ""} | {r[condition]["devtest"]["accuracy"]:.2%} | {r[condition]["standard_cv"]["mean_accuracy"]:.2%} ± {r[condition]["standard_cv"]["std_accuracy"]:.2%} | {r[condition]["devtest"]["auc"]:.6f} | {r[condition]["casia_audit"]["accuracy"]:.2%} | {r[condition]["casia_audit"]["auc"]:.6f} |' for name,r in models.items())
    validation='\n'.join(f'| {names[name]} | {r["clean"]["auc"]:.6f} | {r["diagnostic"]["auc"]:.6f} | {r["selection_score"]:.6f} |' for name,r in selection['candidates'].items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=training['history']
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    epochs=[h['epoch'] for h in history]
    axes[0].plot(epochs,[h['loss'] for h in history],marker='.',label='Total')
    axes[0].plot(epochs,[h['classification_loss'] for h in history],marker='.',label='Weighted classification')
    axes[0].set(xlabel='Epoch',ylabel='Loss',title='Quality-weighted consistency fine-tuning');axes[0].legend(fontsize=8)
    axes[1].plot([0]+epochs,[training['initial_validation_auc']]+[h['validation_auc'] for h in history],marker='.',label='Clean validation')
    axes[1].plot([0]+epochs,[training['initial_robust_validation']['auc']]+[h['robust_validation']['auc'] for h in history],marker='.',label='40px JPEG diagnostic')
    axes[1].axvline(training['best_epoch'],color='gray',linestyle=':',label='Selected epoch')
    axes[1].set(xlabel='Epoch',ylabel='AUC',title='Held-out CASIA selection identities');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'artifacts/training.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    for ax,condition,title in zip(axes,['clean','diagnostic'],['Original LFW images','40px JPEG diagnostic']):
        positions=range(len(models));values=[r[condition]['standard_cv']['mean_accuracy']*100 for r in models.values()]
        ax.bar(positions,values,color=['#829a9c','#b7bfc3','#62a59e','#dca75c','#277a67'])
        labels={'pretrained':'Pretrained','round1':'Round1','metric':'Metric','robust':'Robust','combined':'Combined'}
        ax.set_xticks(list(positions),[labels[name] for name in models],rotation=25,ha='right')
        ax.set(ylim=(min(values)-1,100),ylabel='10-fold accuracy (%)',title=title)
        for i,value in enumerate(values):ax.text(i,value+.05,f'{value:.2f}',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(work/'ablation.png',dpi=160);plt.close(fig)
    differences={label:{'devtest_pp':100*(final[condition]['devtest']['accuracy']-pre[condition]['devtest']['accuracy']),
                         'cv_pp':100*(final[condition]['standard_cv']['mean_accuracy']-pre[condition]['standard_cv']['mean_accuracy']),
                         'audit_pp':100*(final[condition]['casia_audit']['accuracy']-pre[condition]['casia_audit']['accuracy'])}
                 for label,condition in [('clean','clean'),('diagnostic','diagnostic')]}
    gain_summary='\n'.join(f'| {"原始清晰图像" if label=="clean" else "低分辨率/JPEG 诊断"} | {d["devtest_pp"]:+.3f} | {d["cv_pp"]:+.3f} | {d["audit_pp"]:+.3f} |' for label,d in differences.items())
    robust=models['robust'];robust_gain=100*(robust['clean']['standard_cv']['mean_accuracy']-baseline['standard_cv']['mean_accuracy'])
    robust_errors=sum(fold['fp']+fold['fn'] for fold in robust['clean']['standard_cv']['folds'])
    initial_errors=sum(fold['fp']+fold['fn'] for fold in baseline['standard_cv']['folds'])
    system=''
    for file,title in [('system_validation.json','系统计时'),('browser_validation.json','浏览器验证'),('test_validation.json','自动测试')]:
        path=ROOT/'artifacts'/file
        if path.exists():
            d=json.loads(path.read_text())
            if d.get('checkpoint_sha256')!=final['checkpoint_sha256']:continue
            if title=='系统计时':system+=f"\n设备 `{d['device']}` 上加载模型后的双图接口平均 {d['mean_pair_seconds']*1000:.2f} ms，单图检测与对齐平均 {d['mean_detection_seconds']*1000:.2f} ms，各测量 10 次。GPU 计时前后同步，不含首次模型加载。\n"
            elif title=='浏览器验证':system+=f"\n新模型通过 {len(d['checks'])} 项浏览器检查与 {len(d.get('simulated_camera_checks',[]))} 项模拟摄像头检查，真实摄像头硬件未测试。\n"
            else:system+=f"\n自动测试 {d['passed']} 项通过，失败 {d['failed']} 项，记录位于 `artifacts/test_validation.json`。\n"
    if (ROOT/'artifacts/demo.png').exists():system+='\n![最新界面](artifacts/demo.png)\n'
    report=f'''# Face Verification 实验报告：质量加权一致性微调与类内变化抑制

## 1. 任务、运行与交付

按 `说明.docx` 提供 `Y = FaceVerification(X1, X2)`，输入两张已对齐人脸图像路径，返回 Python `int`：同人 1、异人 0。普通照片由 YuNet 检测、五点相似变换对齐，网页支持上传照片、摄像头拍照与连续验证。

本轮针对“普通微调与预训练 ViT 准确率基本持平”的问题，在同一预训练初始化上探索质量加权的清晰目标一致性微调、类内变化抑制投影及其组合。选择结果为 **{names[chosen]}**；方法选择只看 CASIA 验证身份，选择记录写入后才执行本轮最终 LFW 和新增 CASIA 审计评估。

本轮实测的明确收益包括：**单独质量加权增强微调**在原始 LFW 十折准确率上从预训练的 {baseline['standard_cv']['mean_accuracy']:.2%} 到 {robust['clean']['standard_cv']['mean_accuracy']:.2%}，提高 {robust_gain:.3f} 个百分点，6,000 对中从 {initial_errors} 个错误减至 {robust_errors} 个；**按 CASIA 规则选中的组合方法**在降质 DevTest 上从 {bdiag['devtest']['accuracy']:.2%} 到 {diag['devtest']['accuracy']:.2%}。这是不同设置下的有限改进，不代表同一模型在所有指标上都更好。

```bash
cd /home/wzc/datasets/casia/faceRecognitionCASIA
python -m pip install -r requirements.txt
FACE_DEVICE=cuda:7 python app.py
```

访问 `http://127.0.0.1:7860`，普通照片自动检测对齐；已对齐图像勾选相应选项。单卡环境把 `cuda:7` 换成 `cuda:0` 或 `cuda`。没有 CUDA 时可在 CPU 推理。图片不保存到服务器文件。

```python
from FaceVerification import FaceVerification
root = "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped"
Y = FaceVerification(root + "/Abel_Pacheco/Abel_Pacheco_0001.jpg",
                     root + "/Abel_Pacheco/Abel_Pacheco_0004.jpg")
print(Y)  # int: 1 or 0
```

普通照片命令行：`python verification.py samples/009.jpg samples/009.jpg --detect`。
默认权重 `checkpoints/transformer.pth`，配套阈值 `artifacts/calibration.json`，相似度大于等于 {calibration['threshold']:.9f} 判为同人。模型与阈值通过 SHA-256 校验，分数不是概率。权重约 438 MiB，被 Git 忽略；迁移需单独复制最终权重与配套阈值，或复现实验。

## 2. 贡献与结论的边界

我们不把“换用了更强的预训练骨干”全部称作自己的贡献，也不声称下述已有技术为论文级新算法。本轮贡献是一个可复现、有明确对照的迁移方案：用训练身份的初始特征一致性估计样本权重，以清晰图像特征指导降质输入，再学习一个抑制同人变化方向的正则化度量；附带各模块消融和独立身份审计。

下面直接给出最终选定方法相对**完全相同预训练 ViT 初始化、同样翻转融合**的准确率变化，单位为百分点。原始协议与额外降质诊断分开，不能用诊断成绩替换原始 LFW 成绩。

| 条件 | LFW DevTest 变化 | LFW 标准十折变化 | 新增 CASIA 审计变化 |
|---|---:|---:|---:|
{gain_summary}

零或负值均保留。微小变化不能自动称为统计显著提升。公开 LFW 已在前轮被观察，本轮结果属于公开基准的重复评估，不能宣称它是从未看过的外部保留集。教师未公开测试集尚未提供。

若只希望体验没有附加类内投影的梯度微调版本，可使用 `checkpoints/transformer_robust.pth` 搭配 `artifacts/refinement/calibration_robust.json`。它是完整消融中的预定候选，不因看到 LFW 分数而替换默认模型；其十折提升仅相当于少错 4 对，不宣称统计显著。默认组合版本保留 CASIA AUC 的选择规则。

## 3. 初始化、数据与许可

骨干来自 CVLFace 作者的 `minchul/cvlface_adaface_vit_base_webface4m`，固定版本 `c6fed2ff09ac0a29cccc72c09127ad7b4bd3fa9e`，初始化 SHA-256 为 `5fafd6b7d599a3ede5fac5bd1d01ad05e9e93e89b39b7687d4a3bc93ff2aebc0`。作者声明使用 WebFace4M、AdaFace 预训练，本次未下载其原始数据。代码采用 MIT 许可证，模型卡要求遵守训练数据许可；详见 `third_party/`。

ViT 输入 RGB 112×112，8×8 patch，196 token，内部维度 512，24 个 Transformer 块、16 头注意力、扩展比 3 的 ReLU6 MLP，最终投影为 512 维。原生 PyTorch SDPA 与显式注意力公式一致，测试覆盖；不执行模型仓库远程 Python。

本地 CASIA 版本有 10,575 个身份目录、454,191 张 JPG。沿用固定种子 42 的 4,096 个训练身份、65,536 张图像；256 个验证身份、4,096 张图像与训练身份不重叠。它们可能与上游预训练人物重叠，因此“未见身份”仅指未参与本地训练。

额外审计集从其余 CASIA 身份中以种子 314 选择 256 个身份，每人 16 张，不与训练/选模型身份重叠。前 128 个身份标定审计阈值，后 128 个身份评估，共 4,096 对测试配对；这些身份的结果不参与方法选择。清单保存于 `artifacts/refinement/audit_split.json`。

## 4. 方法

### 4.1 质量加权的一致性微调

先缓存初始化网络对每张训练图像及其翻转的单位特征。每个训练身份的融合特征求均值并归一化为原型。样本与其身份原型的余弦相似度为 q，权重 `w=clip((q-0.25)/0.5,0,1)`。这只是训练集内的特征一致性启发式，不是已标定的人脸质量概率；它可能降低离群/错误标注样本的影响，也可能抑制真正困难样本。

每张输入以 0.5 概率水平翻转；每批另以 0.5 概率选中部分图像，将其降至 40、56 或 72 像素再恢复为 112（尺寸随批次循环）。无论输入是否降质，保持目标都是相应清晰图像/翻转图的初始化特征。训练没有使用 LFW 图像或标签。

```text
L = weighted_mean(CrossEntropy(CosFace(f(x_aug), y)), w)
  + 2 × weighted_mean(1 - <normalize(f(x_aug)), e_initial_clean>, w)
```

CosFace 尺度 32、间隔 0.15；分类器由训练身份原型初始化。训练最后 4 个 Transformer 块、最终 LayerNorm 和两个特征投影矩阵，冻结前 20 块以及 BN 参数和统计。AdamW：块学习率 0.00002、特征头 0.00004、分类器 0.001，weight decay=0.0001，batch=128，BF16，8 轮，余弦学习率调度、梯度范数上限 5。当前 GPU 为 {training['hardware']}，实际运行 `{training['arguments']['device']}`。

训练损失从 {history[0]['loss']:.6f} 降至 {history[-1]['loss']:.6f}；第 {training['best_epoch']} 轮按 CASIA 降质验证 AUC 选中，并要求清晰验证 AUC 相对初始化下降不超过 0.001。共 {training['changed_floating_values']:,} 个浮点值实际改变，{len(training['changed_tensors'])} 个张量发生变化。初始清晰验证 AUC={training['initial_validation_auc']:.6f}，降质验证 AUC={training['initial_robust_validation']['auc']:.6f}；训练和选模型记录见 `artifacts/refinement/robust_training/training.json`。

![增强微调训练与验证](artifacts/training.png)

### 4.2 正则化类内变化抑制投影

先在 CASIA 训练身份中取原图/翻转融合的单位特征，过滤与该身份中心余弦相似度小于 0.3 的极端离群样本；重新计算各身份均值后，用残差估计类内协方差 `Cw`。采用收缩正则和部分白化：

```text
C_reg = (1 - lambda) × Cw + lambda × trace(Cw)/512 × I
W = C_reg^(-alpha/2)
e = normalize(W × f(x))
```

按特征值分解计算 W，并归一化其几何平均尺度。正则 `lambda∈{{0.05,0.2,0.5}}`、强度 `alpha∈{{0.25,0.5,0.75,1.0}}`，共 12 个配置，只用 CASIA 清晰验证 AUC 选择；恒等变换也作为候选，避免强行添加无收益投影。该投影是基于标签的闭式度量拟合，不是梯度更新，报告不将其称为新增的 Transformer 反向训练。

分别在初始化 ViT 和增强微调 ViT 上拟合投影，形成模块消融。最终推理只多一个 512×512 线性矩阵，不读取测试身份、不按人名设置规则，也不在推理时查询标签。

### 4.3 翻转融合、模型选择与阈值

所有主比较均分别提取原图与翻转图特征，归一化后相加并再次归一化。投影存在时在每个视图归一化前应用。余弦相似度越高越可能同人。

候选方法比较清晰/降质 CASIA 验证 AUC 的平均值，并要求清晰 AUC 不低于初始化减 0.001；该规则记录在本轮 LFW 最终评估前保存的 `experiment_plan.json`。选择表如下：

| 方法 | 清晰验证 AUC | 降质验证 AUC | 平均选择分数 |
|---|---:|---:|---:|
{validation}

选择结果：**{names[chosen]}**。完整配置尝试均保留，不根据以下 LFW 结果重新选择默认方法。

## 5. 评估协议

原始 LFW DevTrain 2,200 对仅标定阈值，DevTest 1,000 对（正负各 500 对）计算准确率；标准 6,000 对按官方十折顺序，每折在其余九折选择阈值，留出一折测试。十折标准差是折间波动，不是置信区间。清晰结果保持原始图像、标签和协议。

额外降质诊断对每张图像固定执行：双线性缩至 40×40 → JPEG quality=45 → 双线性恢复 112×112。每个模型处理完全相同的降质输入；在降质 DevTrain 上各自标定阈值，在降质 DevTest/十折评估。新增 CASIA 审计也分别以清晰/降质校准与测试。

这是一项有意贴近训练降质方向的诊断，能够说明该条件下的稳健性，不能外推为所有真实低质量照片或官方新的 LFW 成绩。训练增强不含 JPEG 编码，诊断包含 JPEG；仍需真实拍摄数据做外部验证。

## 6. 完整对照和消融结果

### 6.1 原始清晰输入

| 方法 | DevTest 准确率 | 标准十折准确率 | DevTest AUC | 新增 CASIA 审计准确率 | 审计 AUC |
|---|---:|---:|---:|---:|---:|
{table('clean')}

### 6.2 低分辨率/JPEG 诊断输入（额外协议）

| 方法 | DevTest 准确率 | 标准十折准确率 | DevTest AUC | 新增 CASIA 审计准确率 | 审计 AUC |
|---|---:|---:|---:|---:|---:|
{table('diagnostic')}

![方法消融与条件对照](artifacts/refinement/ablation.png)

原始清晰 DevTest 最终混淆矩阵：TP={clean['devtest']['tp']}、FN={clean['devtest']['fn']}、FP={clean['devtest']['fp']}、TN={clean['devtest']['tn']}；降质诊断为 TP={diag['devtest']['tp']}、FN={diag['devtest']['fn']}、FP={diag['devtest']['fp']}、TN={diag['devtest']['tn']}。这些都是有限样本计数，FAR=0 不代表真实误接受风险为零。

所有方法、条件、每折指标保存于 `artifacts/refinement/evaluation.json`。初次普通 ViT 微调记录与权重保留在 `experiments/vit_round1/` 和 `checkpoints/transformer_round1.pth`，失败或未选中的本轮候选也保留。新默认文件 SHA-256：`{final['checkpoint_sha256']}`。

## 7. 复现与系统测试

```bash
python -m pip install -r requirements-dev.txt
python scripts/download_transformer.py
python scripts/run_refinement.py --device cuda:7
```

上述流程会依次训练增强版本、拟合两个投影、在 CASIA 验证集选择并启用默认权重、执行统一原始评估与本轮附加诊断、更新计时和报告。GPU 编号应根据机器调整；需要现有 CASIA/LFW 数据与原网络对照权重。默认 8 轮使用完整固定子集，耗时依赖硬件、缓存和磁盘。分步命令见 README。

已有模型仅需 `python app.py`。复核代码：`python -m pytest -q`。测试覆盖接口/阈值/配对、单脸与多脸处理、RGB 预处理、注意力等价、冻结前缀梯度、BN 统计、诊断确定性以及恒等投影前后输出一致性。系统验证使用最终模型与其校准文件。
{system}
## 8. 局限与诚实解释

1. 结构、CosFace、特征保持和类内白化均有既有研究基础；本次是任务导向的组合与受控探索，不声称提出全新通用算法。
2. 质量权重、训练增强和额外投影可能影响不同人群与困难姿态；质量启发式不是标注真值，可能错误抑制困难但正确的样本。
3. 明确区分预训练收益、本轮微调收益、类内投影收益，清晰/降质条件的零收益或回退不能隐藏。很小的百分点变化需要更多样本和多个随机种子验证。
4. 只有一组训练随机种子；公开 LFW 已多轮观察，跨轮实验存在适应公开基准的风险。新增 CASIA 审计不参与选模型，但与 CASIA 同域且可能与上游预训练身份重叠。
5. 未公开测试、真实设备低质量采集及活体检测均未评估。照片相似性验证不能等同于安全身份认证。

## 9. 参考资料与致谢

- [CVLFace](https://github.com/mk-minchul/CVLface)及[固定初始化模型](https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m/tree/c6fed2ff09ac0a29cccc72c09127ad7b4bd3fa9e)，许可见 `third_party/`。
- [Vision Transformer 原论文](https://arxiv.org/abs/2010.11929)、[AdaFace 原论文](https://arxiv.org/abs/2204.00964)：分别为骨干与初始化来源。本轮质量启发式不是 AdaFace 自适应间隔的复现。
- [CosFace 原论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html)。
- [YuNet 官方实现与模型](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。
- 原始 `legacy/README.md`、`blog/` 及其对 insightface、InsightFace_Pytorch、arcface-pytorch 的致谢。
'''
    (ROOT/'report.md').write_text(report,encoding='utf-8')
    if (ROOT/'artifacts/my_experiments/vit_robust_eval/evaluation.json').exists():
        from summarize_experiments import main as summarize_user_experiments
        summarize_user_experiments([])
    print('Updated refinement report and figures with all controls.')


if __name__=='__main__':main()
