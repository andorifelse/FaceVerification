"""Organize the user's three completed experiments without retraining or deployment changes."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import cross_validate
from metrics import pair_scores, read_pairs, summarize
from verification import sha256

START = '<!-- USER_EXPERIMENTS_START -->'
END = '<!-- USER_EXPERIMENTS_END -->'
NAMES = {
    'baseline': '原始 FaceMobileNet',
    'mobile_finetuned': '本次微调 FaceMobileNet',
    'vit_pretrained': '预训练 ViT',
    'finetuned': '本次微调 ViT',
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def paired_test(before, after, labels):
    before, after = before == labels, after == labels
    gained = int(np.sum(after & ~before))
    lost = int(np.sum(before & ~after))
    n = gained + lost
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gained, lost)+1)) / 2**n) if n else 1.
    return {'pairs': len(labels), 'gained_correct': gained, 'lost_correct': lost,
            'net_fewer_errors': gained-lost, 'mcnemar_exact_two_sided_p': p,
            'accuracy_gain_percentage_points': 100*(gained-lost)/len(labels)}


def plot_results(destination, variants, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#879995', '#4d8674', '#849bc0', '#365e99']
    labels = ['Mobile original', 'Mobile fine-tuned', 'ViT pretrained', 'ViT fine-tuned']
    groups = list(NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, mode, title in zip(axes, ['single', 'flip'], ['Single view', 'Original + horizontal flip']):
        for offset, metric, alpha in [(-.19, 'accuracy', .55), (.19, 'cv', 1.)]:
            values = [100*(variants[f'{g}_{mode}']['accuracy'] if metric == 'accuracy'
                           else variants[f'{g}_{mode}']['lfw_standard_cv']['mean_accuracy']) for g in groups]
            bars = ax.bar(np.arange(4)+offset, values, .36, color=colors, alpha=alpha,
                          edgecolor='white', label='DevTest' if metric == 'accuracy' else '10-fold mean')
            for bar, value in zip(bars, values):
                ax.annotate(f'{value:.2f}', (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks(range(4), labels, rotation=20, ha='right')
        ax.set(ylim=(95.5, 100), ylabel='Accuracy (%)', title=title)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('User experiments: identical LFW protocols; truncated accuracy axis', fontsize=11)
    fig.tight_layout(); fig.savefig(destination/'comparison.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    mobile, vit = training['mobile'], training['vit']
    for row, record, name in [(0, mobile, 'FaceMobileNet'), (1, vit, 'ViT robust')]:
        history = record['history']; epochs = [r['epoch'] for r in history]
        axes[row, 0].plot(epochs, [r['loss'] for r in history], marker='.')
        axes[row, 0].set(title=name+' training loss', xlabel='Epoch', ylabel='Loss')
        if row == 0:
            axes[row, 1].plot([0]+epochs, [record['initial_validation_balanced_accuracy']]+
                             [r['validation_balanced_accuracy'] for r in history], marker='.')
            axes[row, 1].set(ylabel='Balanced accuracy', title='CASIA validation (selection metric)')
        else:
            axes[row, 1].plot([0]+epochs, [record['initial_robust_validation']['auc']]+
                             [r['robust_validation']['auc'] for r in history], marker='.')
            axes[row, 1].set(ylabel='AUC', title='CASIA degraded validation (selection metric)')
        axes[row, 1].axvline(record['best_epoch'], color='#d58b3c', linestyle=':', label='Selected epoch')
        axes[row, 1].set(xlabel='Epoch'); axes[row, 1].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(destination/'training_curves.png', dpi=180); plt.close(fig)


def append_report(root, content):
    report = root/'report.md'
    if not report.exists(): return
    text = report.read_text(encoding='utf-8')
    section = f'{START}\n## 10. 用户独立复现实验\n\n{content}\n{END}'
    if START in text:
        begin = text.index(START); end = text.index(END, begin)+len(END)
        text = text[:begin]+section+text[end:]
    else:
        text = text.rstrip()+'\n\n'+section+'\n'
    report.write_text(text, encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT/'artifacts/my_experiments')
    parser.add_argument('--no-update-report', action='store_true')
    args = parser.parse_args(argv)
    base = args.input_root.resolve(); destination = base/'summary'; destination.mkdir(exist_ok=True)
    evaluations = {name: load(base/name/'evaluation.json')
                   for name in ['mobile_eval', 'vit_pretrained_eval', 'vit_robust_eval']}
    source = evaluations['vit_robust_eval']; variants = source['variants']
    training = {'mobile': load(base/'mobile_train/training.json'),
                'vit': load(base/'vit_robust_train/training.json')}
    splits = {name: load(base/folder/'split.json') for name, folder in
              [('mobile', 'mobile_train'), ('vit', 'vit_robust_train')]}
    if splits['mobile'] != splits['vit']:
        raise ValueError('Training identity/image splits differ; do not present them as identical')
    split = splits['mobile']
    if set(split['train']) & set(split['validation']): raise ValueError('Train/validation identity overlap')
    protocol_keys = ['calibration_protocol_sha256', 'test_protocol_sha256', 'standard_protocol_sha256',
                     'test_root', 'calibration_pairs', 'test_pairs', 'standard_pairs']
    for name, evaluation in evaluations.items():
        for key in protocol_keys:
            if evaluation[key] != source[key]: raise ValueError(f'{name}: different protocol {key}')
        for key in ['baseline_single', 'baseline_flip', 'mobile_finetuned_single',
                    'mobile_finetuned_flip', 'vit_pretrained_single', 'vit_pretrained_flip']:
            if evaluation['variants'][key] != variants[key]:
                raise ValueError(f'{name}: inconsistent repeated evaluation of {key}')
    weight_checks = {}
    for model, record in training.items():
        checkpoint = resolve(record['arguments']['output'])
        if not checkpoint.exists(): weight_checks[model] = 'not available; using recorded hash'; continue
        fingerprint = sha256(checkpoint)
        if fingerprint != record['checkpoint_sha256']: raise ValueError(f'{model}: checkpoint changed since evaluation')
        prefix = 'mobile_finetuned' if model == 'mobile' else 'finetuned'
        if variants[prefix+'_flip']['checkpoint_sha256'] != fingerprint: raise ValueError('Evaluation/training mismatch')
        weight_checks[model] = 'verified against actual checkpoint and evaluation'
    for key in ['baseline', 'vit_pretrained']:
        checkpoint = ROOT/('checkpoints/18.pth' if key == 'baseline' else 'checkpoints/pretrained_vit/model.safetensors')
        if checkpoint.exists():
            if sha256(checkpoint) != variants[key+'_flip']['checkpoint_sha256']: raise ValueError(key+' hash mismatch')
    rows = []
    for prefix, name in NAMES.items():
        for mode in ['single', 'flip']:
            r = variants[f'{prefix}_{mode}']; cv = r['lfw_standard_cv']
            rows.append({'model': name, 'variant': f'{prefix}_{mode}', 'view': mode,
                         **{k: r[k] for k in ['backbone', 'accuracy', 'balanced_accuracy', 'threshold', 'auc',
                                             'eer', 'tar', 'far', 'frr', 'tp', 'fn', 'fp', 'tn', 'checkpoint_sha256']},
                         'devtest_errors': r['fp']+r['fn'], 'cv_mean_accuracy': cv['mean_accuracy'],
                         'cv_std_accuracy': cv['std_accuracy'],
                         'cv_errors': sum(f['fp']+f['fn'] for f in cv['folds'])})
    paired = {}; prediction_checks = {}
    root = Path(source['test_root'])
    protocol_files = {'calibration_protocol_sha256': ROOT/'datasets/scikit_learn_data/lfw_home/pairsDevTrain.txt',
                      'test_protocol_sha256': ROOT/'datasets/scikit_learn_data/lfw_home/pairsDevTest.txt',
                      'standard_protocol_sha256': ROOT/'datasets/pairs.txt'}
    cache_paths = [base/'vit_robust_eval'/f'embeddings_{variants[p+"_flip"]["checkpoint_sha256"][:16]}.npz' for p in NAMES]
    if all(p.exists() for p in cache_paths) and root.exists() and all(p.exists() for p in protocol_files.values()):
        for key, path in protocol_files.items():
            if sha256(path) != source[key]: raise ValueError('Current protocol file changed: '+str(path))
        test, _ = read_pairs(protocol_files['test_protocol_sha256'], root)
        standard, header = read_pairs(protocol_files['standard_protocol_sha256'], root)
        predictions = {}
        for prefix, cache in zip(NAMES, cache_paths):
            with np.load(cache, allow_pickle=False) as saved:
                paths = saved['paths'].tolist(); raw, flip = saved['raw'], saved['flip']
            for mode in ['single', 'flip']:
                feature = raw if mode == 'single' else raw+flip
                if mode == 'flip': feature = feature/np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
                key = f'{prefix}_{mode}'; row = variants[key]
                scores, labels = pair_scores(feature, paths, test)
                actual = summarize(scores, labels, row['threshold'])
                for field in ['accuracy', 'tp', 'fp', 'tn', 'fn']:
                    if actual[field] != row[field]: raise ValueError('Cached predictions disagree: '+key)
                standard_scores, standard_labels = pair_scores(feature, paths, standard)
                cv = cross_validate(standard_scores, standard_labels, header[0])
                if not np.isclose(cv['mean_accuracy'], row['lfw_standard_cv']['mean_accuracy'], atol=1e-12):
                    raise ValueError('Cached cross-validation disagrees: '+key)
                cv_prediction = np.zeros(len(standard), dtype=int)
                for fold, stats in zip(np.array_split(np.arange(len(standard)), header[0]), cv['folds']):
                    cv_prediction[fold] = standard_scores[fold] >= stats['threshold']
                predictions[key] = {'devtest': scores >= row['threshold'], 'cv': cv_prediction}
        for mode in ['single', 'flip']:
            for label, before, after in [('mobile', 'baseline', 'mobile_finetuned'), ('vit', 'vit_pretrained', 'finetuned')]:
                paired[f'{label}_{mode}'] = {
                    'devtest': paired_test(predictions[f'{before}_{mode}']['devtest'], predictions[f'{after}_{mode}']['devtest'], labels),
                    'standard_cv': paired_test(predictions[f'{before}_{mode}']['cv'], predictions[f'{after}_{mode}']['cv'], standard_labels)}
        prediction_checks = {'cached_metric_recalculation': 'passed for all 8 variants',
                             'interpretation': 'Paired p values are descriptive: LFW pairs may share identities/images and CV training folds overlap.'}
    summary = {'source_evaluations': {k: str((base/k/'evaluation.json').relative_to(ROOT)) for k in evaluations},
               'scope': 'User-run experiments only; earlier agent runs kept separate; no model/threshold activation',
               'protocol': {k: source[k] for k in protocol_keys},
               'verification': {'same_ordered_training_split': True, 'train_validation_identity_overlap': 0,
                                'weight_checks': weight_checks, **prediction_checks},
               'training': training, 'results': rows, 'paired_comparisons': paired}
    (destination/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (destination/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    plot_results(destination, variants, training)
    def result_table(mode):
        lines = []
        for prefix, name in NAMES.items():
            r = variants[f'{prefix}_{mode}']; cv = r['lfw_standard_cv']
            errors = sum(f['fp']+f['fn'] for f in cv['folds'])
            lines.append(f'| {name} | {r["accuracy"]:.2%} | {cv["mean_accuracy"]:.2%} ± {cv["std_accuracy"]:.2%} | '
                         f'{r["auc"]:.6f} | {r["eer"]:.2%} | {r["far"]:.2%} | {r["frr"]:.2%} | {errors} |')
        return '\n'.join(lines)
    m, v = training['mobile'], training['vit']; ma, va = m['arguments'], v['arguments']
    paired_text = ''
    if paired:
        for model, name in [('mobile', 'FaceMobileNet'), ('vit', 'ViT')]:
            d = paired[model+'_flip']['devtest']; c = paired[model+'_flip']['standard_cv']
            paired_text += (f'| {name} 微调前后（翻转融合） | {d["gained_correct"]}/{d["lost_correct"]} | '
                            f'{d["mcnemar_exact_two_sided_p"]:.4f} | {c["gained_correct"]}/{c["lost_correct"]} | '
                            f'{c["mcnemar_exact_two_sided_p"]:.4f} |\n')
        paired_text = ('\n对缓存特征重算逐对预测及十折结果，与记录一致。配对比较如下，“改正/新增错误”说明微调修正了多少旧错误、同时引入多少新错误：\n\n'
                       '| 比较 | DevTest 改正/新增错误 | 双侧精确 p | 十折改正/新增错误 | 双侧精确 p |\n'
                       '|---|---:|---:|---:|---:|\n'+paired_text+
                       '\np 值为描述性 McNemar 结果，图片对可能共享图片或人物，十折训练部分也相互重叠，不满足完全独立样本假设；不能据此替代多随机种子和新数据验证。\n')
    def gain(prefix, base_prefix, field):
        def value(key):
            r = variants[key+'_flip']
            return r['lfw_standard_cv']['mean_accuracy'] if field == 'cv' else r[field]
        return 100*(value(prefix)-value(base_prefix))
    confusion = '\n'.join(f'| {NAMES[p]} | {variants[p+"_flip"]["threshold"]:.6f} | '
                           f'{variants[p+"_flip"]["tp"]} | {variants[p+"_flip"]["fn"]} | '
                           f'{variants[p+"_flip"]["fp"]} | {variants[p+"_flip"]["tn"]} |' for p in NAMES)
    hash_table = '\n'.join(f'| {name} | `{variants[p+"_flip"]["checkpoint_sha256"]}` |' for p, name in NAMES.items())
    old_m, new_m = variants['baseline_flip'], variants['mobile_finetuned_flip']
    old_v, new_v = variants['vit_pretrained_flip'], variants['finetuned_flip']
    def errors(row, cv=False):
        return sum(f['fp']+f['fn'] for f in row['lfw_standard_cv']['folds']) if cv else row['fp']+row['fn']
    def cv_value(key): return variants[key]['lfw_standard_cv']['mean_accuracy']
    body = f'''本节只整理 `artifacts/my_experiments/` 中用户本次完成的三组实验，不将此前自动实验的 99.22% 等结果混入本次复现。这里只读取记录与缓存，没有重新训练，也没有切换当前演示权重。

### 实验配置与核验

两组本地训练使用完全相同、顺序也相同的 CASIA 身份/图像清单：4,096 个训练身份、65,536 张训练图像；256 个验证身份、4,096 张验证图像，每人 16 张，seed=42，训练/验证身份交集为 0。三个评估目录的配对文件指纹、测试路径和重复基准结果一致；模型文件 SHA-256 与训练/评估记录相符。

| 设置 | 全网络微调 FaceMobileNet | 直接使用预训练 ViT | 质量加权增强微调 ViT |
|---|---|---|---|
| 初始化 | 项目原有 `18.pth` | CVLFace ViT-Base WebFace4M | 相同预训练 ViT |
| 输入 | 灰度 128×128 | RGB 112×112 | RGB 112×112 |
| 本地更新层 | 全网络（`--full`） | 不更新 | 最后 4 块、LayerNorm、特征投影；固定 BN |
| 训练轮数 / batch | {ma['epochs']} / {ma['batch_size']} | 不适用 | {va['epochs']} / {va['batch_size']} |
| 学习率 | 网络 {ma['lr']}，分类器 {ma['lr']*20} | 不适用 | 块 {va['lr']}，特征头 {va['head_lr']}，分类器 0.001 |
| 初始特征保持权重 | {ma['anchor_weight']} | 不适用 | {va['anchor_weight']} |
| 训练目标与增强 | CosFace、特征保持、随机翻转 | 不适用 | 质量加权 CosFace、清晰目标一致性、翻转、50% 分辨率增强 |
| CASIA 模型选择规则 | 清晰验证 balanced accuracy 最大 | 不适用 | 降质验证 AUC 最大，且清晰 AUC 回退不超过 0.001 |
| 选中轮次 | {m['best_epoch']} | 不适用 | {v['best_epoch']} |
| 训练耗时（含初始化缓存等） | {m['elapsed_seconds']:.2f} 秒 | 无训练 | {v['elapsed_seconds']:.2f} 秒 |

两份训练清单相同，但脚本生成验证配对的顺序/随机过程和选模型目标不同，不能将两组 CASIA 验证数字当作完全同协议的模型排名。正式模型比较使用统一 LFW 配对。

### 评估协议

LFW DevTrain 2,200 对用于阈值校准；DevTest 1,000 对用于冻结阈值后的测试（同人、异人各 500 对）。校准与开发测试的身份、图像交集均为 0。另用标准 6,000 对做十折，每折用其余九折标定阈值、留出一折测试。十折标准差仅表示折间波动，不是置信区间。

`single` 仅原图，`flip` 融合原图和水平翻转图。以下同时保留两种设置，避免按不同模型挑选各自最好的设置造成混比。所有结果均为原始 LFW 裁剪图；本次没有额外运行降质 LFW 诊断或新增 CASIA 审计。

### 主对照：统一使用翻转融合

| 模型 | DevTest 准确率 | 标准十折均值 ± 标准差 | DevTest AUC | EER | FAR | FRR | 十折错误数 / 6000 |
|---|---:|---:|---:|---:|---:|---:|---:|
{result_table('flip')}

### 补充对照：统一使用单视图

| 模型 | DevTest 准确率 | 标准十折均值 ± 标准差 | DevTest AUC | EER | FAR | FRR | 十折错误数 / 6000 |
|---|---:|---:|---:|---:|---:|---:|---:|
{result_table('single')}

![本次用户实验对比](artifacts/my_experiments/summary/comparison.png)

### 结果分析

**FaceMobileNet 微调前后。** 翻转融合下 DevTest 从 {old_m['accuracy']:.2%} 到 {new_m['accuracy']:.2%}，变化 {gain('mobile_finetuned','baseline','accuracy'):+.2f} 个百分点，错误由 {errors(old_m)} 对变为 {errors(new_m)} 对；十折从 {cv_value('baseline_flip'):.2%} 到 {cv_value('mobile_finetuned_flip'):.2%}，变化 {gain('mobile_finetuned','baseline','cv'):+.3f} 个百分点，错误从 {errors(old_m,True)} 对变为 {errors(new_m,True)} 对。与此同时 FAR 从 {old_m['far']:.1%} 到 {new_m['far']:.1%}，即误接受从 {old_m['fp']} 对变为 {new_m['fp']} 对；误拒绝从 {old_m['fn']} 对变为 {new_m['fn']} 对，说明总体准确率改善也可能伴随误接受增多。单视图十折从 {cv_value('baseline_single'):.2%} 到 {cv_value('mobile_finetuned_single'):.2%}。

**ViT 微调前后。** 翻转融合下预训练和本次微调的 DevTest 分别为 {old_v['accuracy']:.2%} 和 {new_v['accuracy']:.2%}，错误分别为 {errors(old_v)} 对、{errors(new_v)} 对；十折从 {cv_value('vit_pretrained_flip'):.2%} 到 {cv_value('finetuned_flip'):.2%}，变化 {gain('finetuned','vit_pretrained','cv'):+.3f} 个百分点，错误从 {errors(old_v,True)} 对变为 {errors(new_v,True)} 对，净少错 {errors(old_v,True)-errors(new_v,True)} 对。DevTest AUC 从 {old_v['auc']:.6f} 到 {new_v['auc']:.6f}；EER 从 {old_v['eer']:.1%} 到 {new_v['eer']:.1%}。单视图十折从 {cv_value('vit_pretrained_single'):.2%} 到 {cv_value('finetuned_single'):.2%}。因此需要同时报告视图和各指标，不能把某一项的小幅收益写成全面优势。

**翻转融合不是必然提高所有成绩。** 本次 FaceMobileNet 微调模型的 DevTest 使用翻转由 {variants['mobile_finetuned_single']['accuracy']:.1%} 到 {new_m['accuracy']:.1%}，十折由 {cv_value('mobile_finetuned_single'):.2%} 到 {cv_value('mobile_finetuned_flip'):.2%}；本次 ViT 微调十折由单视图 {cv_value('finetuned_single'):.2%} 到翻转 {cv_value('finetuned_flip'):.2%}。应同时报告，不能只根据测试分数为每个模型挑一种视图，再称之为受控比较。

**不同网络之间的比较。** 预训练 ViT 与本次 FaceMobileNet 微调的主对照 DevTest 为 {old_v['accuracy']:.1%} 对 {new_m['accuracy']:.1%}。但网络规模、RGB/灰度预处理、上游预训练数据与本地训练方式都不同，这只能说明这些完整方案在本地协议中的表现，不能单独证明 Transformer 结构必然胜过 CNN。AUC 衡量整个分数排序，Accuracy 衡量某个阈值，不应把两者混用。
{paired_text}
### 训练过程

FaceMobileNet 平均总损失由 {m['history'][0]['loss']:.6f} 到 {m['history'][-1]['loss']:.6f}；CASIA 验证 balanced accuracy 初始为 {m['initial_validation_balanced_accuracy']:.2%}，第 {m['best_epoch']} 轮达到 {m['best_validation_balanced_accuracy']:.2%}。模型选择看验证表现，不能因为后续训练损失更低就默认使用最后一轮。

ViT 平均总损失由 {v['history'][0]['loss']:.6f} 到 {v['history'][-1]['loss']:.6f}；降质 CASIA 验证 AUC 初始为 {v['initial_robust_validation']['auc']:.6f}，按既定规则选中第 {v['best_epoch']} 轮，其降质 AUC 为 {v['best_selection_score']:.6f}，清晰 AUC 为 {v['best_validation_auc']:.6f}。这次选中轮次与此前自动实验不同；GPU 数值/训练波动可能影响微小差异，但本次没有对波动原因做受控验证。不能把此前少错 4 对的成绩替代本次实测净少错 {errors(old_v,True)-errors(new_v,True)} 对。

![本次训练与选模型过程](artifacts/my_experiments/summary/training_curves.png)

两组损失定义及权重不同，损失数值不能直接用来比较模型强弱。单一随机种子、小量错误差异和反复使用公开 LFW 都限制了结论强度；当前收益只能称为这次运行的实测小幅改善，不能声称统计显著或保证未公开测试也提高。

### 阈值与混淆矩阵（翻转融合）

| 模型 | DevTrain 阈值 | TP 正确同人 | FN 误拒绝 | FP 误接受 | TN 正确异人 |
|---|---:|---:|---:|---:|---:|
{confusion}

阈值按模型分别在校准集标定。数值较高/较低本身不代表模型较好；余弦相似度不是概率。FAR=0 只表示这 500 对异人中未出现误接受，不代表实际风险为零。

### 指标速查

| 指标 | 解释 / 公式 | 方向 |
|---|---|---|
| Accuracy | `(TP+TN)/全部配对数` | 越高越好 |
| Balanced accuracy | `0.5 × [TP/(TP+FN) + TN/(TN+FP)]`，两类等权 | 越高越好 |
| TAR | `TP/(TP+FN)`，同人被正确接受的比例 | 越高越好 |
| FAR | `FP/(FP+TN)`，异人被误判同人的比例 | 越低越好 |
| FRR | `FN/(TP+FN)`，同人被误判异人的比例 | 越低越好 |
| AUC | ROC 曲线下面积，描述不同阈值下的分数排序/区分能力，不是准确率 | 越接近 1 越好 |
| EER | 随阈值变化时 FAR 与 FRR 相等附近的插值错误率，不一定对应部署阈值 | 越低越好 |
| 十折均值 / 标准差 | 10 个留出折准确率的均值及折间波动，不是均值的置信区间 | 结合样本量解释 |
| Calibration accuracy | 选阈值所用校准集上的表现，不能当独立测试分数 | 不用于替代测试 |

### 原始文件、权重与复现

本次汇总以 `vit_robust_eval/evaluation.json` 为统一主表，并交叉核对另两个目录中的相同基准。`finetuned_*` 是评估脚本给 `--checkpoint` 模型的固定名称：在 `mobile_eval` 它指 FaceMobileNet，在 `vit_pretrained_eval` 指未微调 ViT，在 `vit_robust_eval` 才指本次微调 ViT；脚本里通用 `selection` 文案不作为实际网络/选择规则的依据。重复条目按真实权重指纹去重，没有将一份权重当成多次独立实验。

| 模型 | SHA-256 |
|---|---|
{hash_table}

原始结果目录：`artifacts/my_experiments/mobile_eval/`、`vit_pretrained_eval/`、`vit_robust_eval/`。训练记录：`mobile_train/training.json`、`vit_robust_train/training.json`；两组 `split.json` 完全相同。

整理脚本：`python scripts/summarize_experiments.py`。输出为 `artifacts/my_experiments/summary/summary.md`、`summary.json`、`comparison.csv` 和两张图。CSV 数值用 0~1 原始比例，百分比显示时乘以 100；本脚本不更换默认推理权重或校准文件。

使用本次 ViT 模型演示：

```bash
FACE_DEVICE=cuda:7 FACE_CHECKPOINT=checkpoints/my_experiments/vit_robust.pth FACE_CALIBRATION=artifacts/my_experiments/vit_robust_eval/calibration.json python app.py
```
'''
    standalone = '# 用户独立实验整理：FaceMobileNet 与 ViT\n\n'+body
    standalone = standalone.replace('(artifacts/my_experiments/summary/comparison.png)', '(comparison.png)')
    standalone = standalone.replace('(artifacts/my_experiments/summary/training_curves.png)', '(training_curves.png)')
    (destination/'summary.md').write_text(standalone, encoding='utf-8')
    if not args.no_update_report: append_report(ROOT, body)
    print('Saved', destination/'summary.md')
    print('Saved comparison.csv, summary.json, comparison.png and training_curves.png')
    for prefix in NAMES:
        row = variants[prefix+'_flip']
        print(NAMES[prefix], f'DevTest={row["accuracy"]:.2%}', f'10-fold={row["lfw_standard_cv"]["mean_accuracy"]:.2%}')


if __name__ == '__main__': main()
