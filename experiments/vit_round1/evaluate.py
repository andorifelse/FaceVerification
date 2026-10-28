"""Calibrate on LFW DevTrain; evaluate once at that threshold on DevTest."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from config import ROOT, config
from metrics import pair_scores, read_pairs, roc_curve, summarize, threshold_search
from verification import FaceVerifier, sha256


def normalized(array):
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def extract(verifier, paths, output, batch_size):
    cache = output/f'embeddings_{verifier.model_hash[:16]}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if saved['paths'].tolist() == paths:
                return saved['raw'], saved['flip']
    import hashlib
    import torch
    import torch.nn.functional as F
    from verification import open_image
    from model import FaceMobileNet
    # Head tuning leaves all convolution weights unchanged. Reuse their outputs,
    # keyed by their actual tensor bytes, across the two model variants.
    if isinstance(verifier.net,FaceMobileNet):
        digest = hashlib.sha256()
        for key,value in verifier.net.state_dict().items():
            if not key.startswith(('linear.','bn.')):
                digest.update(key.encode()); digest.update(value.cpu().numpy().tobytes())
        conv_cache = output/f'convolution_{digest.hexdigest()[:16]}.npz'
        convolution = None
        if conv_cache.exists():
            with np.load(conv_cache,allow_pickle=False) as saved:
                if saved['paths'].tolist() == paths:
                    convolution = saved['raw'],saved['flip']
        if convolution is None:
            normal,flipped = [],[]
            with torch.inference_mode():
                for start in range(0,len(paths),batch_size):
                    batch = torch.stack([verifier.transform(open_image(p)) for p in paths[start:start+batch_size]]).to(verifier.device)
                    normal.append(verifier.net.forward_features(batch).cpu().numpy())
                    flipped.append(verifier.net.forward_features(torch.flip(batch,[-1])).cpu().numpy())
                    if start % (batch_size*10) == 0:
                        print(f'Convolution: {min(start+batch_size,len(paths))}/{len(paths)} images',flush=True)
            convolution = np.concatenate(normal),np.concatenate(flipped)
            np.savez_compressed(conv_cache,paths=np.array(paths),raw=convolution[0],flip=convolution[1])
        embedded = []
        with torch.inference_mode():
            for array in convolution:
                batches = []
                for start in range(0,len(paths),batch_size):
                    tensor = torch.from_numpy(array[start:start+batch_size]).to(verifier.device)
                    batches.append(F.normalize(verifier.net.bn(verifier.net.linear(tensor)),dim=1).cpu().numpy())
                embedded.append(np.concatenate(batches))
        a,b = embedded
        np.savez_compressed(cache,paths=np.array(paths),raw=a,flip=b)
        return a,b
    from PIL import ImageOps
    a, b = [], []
    verifier.tta = False
    for start in range(0,len(paths),batch_size):
        images = [open_image(p) for p in paths[start:start+batch_size]]
        a.append(verifier.embed(images, batch_size))
        b.append(verifier.embed([ImageOps.mirror(p) for p in images], batch_size))
        if start % (batch_size*10) == 0:
            print(f'{verifier.checkpoint.name}: {min(start+batch_size,len(paths))}/{len(paths)} images', flush=True)
    a, b = np.concatenate(a), np.concatenate(b)
    np.savez_compressed(cache, paths=np.array(paths), raw=a, flip=b)
    return a, b


def cross_validate(scores, labels, folds):
    results = []
    for fold in np.array_split(np.arange(len(scores)),folds):
        train = np.ones(len(scores),dtype=bool); train[fold] = False
        _, threshold = threshold_search(scores[train],labels[train])
        results.append(summarize(scores[fold],labels[fold],threshold))
    acc = [r['accuracy'] for r in results]
    return {'mean_accuracy':float(np.mean(acc)), 'std_accuracy':float(np.std(acc)), 'folds':results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=config.checkpoints/config.test_model)
    parser.add_argument('--baseline',type=Path,default=config.checkpoints/'18.pth')
    parser.add_argument('--pretrained',type=Path,default=config.checkpoints/'pretrained_vit/model.safetensors')
    parser.add_argument('--mobile-checkpoint',type=Path,default=config.checkpoints/'finetuned.pth')
    parser.add_argument('--root',type=Path,default=config.test_root)
    parser.add_argument('--calibration-pairs',type=Path,default=config.calibration_list)
    parser.add_argument('--test-pairs',type=Path,default=config.test_list)
    parser.add_argument('--standard-pairs',type=Path,default=ROOT/'datasets/pairs.txt')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--device',default=config.device)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration, _ = read_pairs(args.calibration_pairs,args.root)
    test, _ = read_pairs(args.test_pairs,args.root)
    # Fail on even reversed-pair or image overlap instead of silently leaking data.
    cal_images = {p for pair in calibration for p in pair[:2]}
    test_images = {p for pair in test for p in pair[:2]}
    overlap = cal_images & test_images
    if overlap:
        raise ValueError(f'Calibration/test image overlap: {len(overlap)} images')
    standard, header = read_pairs(args.standard_pairs,args.root)
    paths = sorted({p for pair in calibration+test+standard for p in pair[:2]})
    rows, score_sets, selected_scores = {}, {}, None
    models = [('baseline',args.baseline)]
    if args.mobile_checkpoint.exists():models.append(('mobile_finetuned',args.mobile_checkpoint))
    if args.pretrained.exists():models.append(('vit_pretrained',args.pretrained))
    models.append(('finetuned',args.checkpoint))
    for name, checkpoint in models:
        verifier = FaceVerifier(checkpoint,threshold=0.,tta=False,device=args.device)
        raw, flipped = extract(verifier,paths,args.output,args.batch_size)
        for tta in [False,True]:
            variant = f'{name}_{"flip" if tta else "single"}'
            features = normalized(raw+flipped) if tta else raw
            cs, cy = pair_scores(features,paths,calibration)
            ts, ty = pair_scores(features,paths,test)
            ca, threshold = threshold_search(cs,cy)
            row = summarize(ts,ty,threshold)
            row['calibration_balanced_accuracy'] = ca
            row['checkpoint_sha256'] = verifier.model_hash
            row['backbone'] = verifier.metadata.get('backbone','fmobile')
            ss, sy = pair_scores(features,paths,standard)
            if not header or len(header) != 2:
                raise ValueError('Standard LFW evaluation requires its official folds header')
            row['lfw_standard_cv'] = cross_validate(ss,sy,header[0])
            rows[variant] = row
            score_sets[variant] = (ts,ty)
            print(variant, json.dumps({k:v for k,v in row.items() if k!='lfw_standard_cv'}),flush=True)
            if name == 'finetuned' and tta:
                settings = {'threshold':threshold,'tta':True,'checkpoint_sha256':verifier.model_hash,
                            'calibration_protocol':str(args.calibration_pairs),
                            'calibration_protocol_sha256':sha256(args.calibration_pairs),
                            'calibration_pairs':len(calibration),'decision':'cosine_similarity >= threshold',
                            'backbone':verifier.metadata.get('backbone','fmobile'),
                            'preprocessing':'RGB, resize 112x112, (pixel/255-0.5)/0.5' if verifier.metadata.get('backbone')=='face_vit' else 'grayscale, resize 128x128, (pixel/255-0.5)/0.5'}
                (args.output/'calibration.json').write_text(json.dumps(settings,indent=2))
                selected_scores = ts,ty,threshold
    record = {'selected_variant':'finetuned_flip', 'calibration_pairs':len(calibration),
              'test_pairs':len(test),'unique_images':len(paths),'image_overlap':len(overlap),
              'calibration_test_identity_overlap':len({Path(p).parent.name for p in cal_images}&{Path(p).parent.name for p in test_images}),
              'standard_pairs':len(standard),'standard_protocol_sha256':sha256(args.standard_pairs),
              'variants':rows,'elapsed_seconds':time.perf_counter()-started,
              'selection':'model selected using held-out CASIA identity validation; final variant predetermined as finetuned Transformer + flip',
              'calibration_protocol_sha256':sha256(args.calibration_pairs),
              'test_protocol_sha256':sha256(args.test_pairs),
              'test_root':str(args.root.resolve())}
    baseline_s,labels = score_sets['baseline_single']
    final_s,_,final_threshold = selected_scores
    old_correct = (baseline_s>=rows['baseline_single']['threshold'])==labels
    new_correct = (final_s>=final_threshold)==labels
    gained = int(np.sum(new_correct & ~old_correct))
    lost = int(np.sum(old_correct & ~new_correct))
    discordant = gained+lost
    exact_p = min(1.,2*sum(math.comb(discordant,i) for i in range(min(gained,lost)+1))/2**discordant) if discordant else 1.
    record['paired_comparison'] = {'baseline':'baseline_single','candidate':'finetuned_flip',
                                  'gained_correct_pairs':gained,'lost_correct_pairs':lost,
                                  'mcnemar_exact_two_sided_p':exact_p,
                                  'accuracy_gain_percentage_points':100*(rows['finetuned_flip']['accuracy']-rows['baseline_single']['accuracy'])}
    (args.output/'evaluation.json').write_text(json.dumps(record,indent=2))
    ts,ty,threshold = selected_scores
    with (args.output/'predictions.csv').open('w',newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['image1','image2','label','similarity','prediction'])
        for pair,score in zip(test,ts):
            writer.writerow([str(Path(pair[0]).relative_to(args.root)),str(Path(pair[1]).relative_to(args.root)),pair[2],float(score),int(score>=threshold)])
    plot_results(args.output,rows,score_sets)
    print(f'Results saved to {args.output}',flush=True)


def plot_results(output,rows,score_sets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.3))
    for name,(scores,labels) in score_sets.items():
        fpr,tpr = roc_curve(scores,labels)
        axes[0].plot(fpr,tpr,label=f'{name} (AUC {rows[name]["auc"]:.3f})')
    axes[0].set(xlabel='False acceptance rate',ylabel='True acceptance rate',title='LFW DevTest ROC')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2)
    scores,labels = score_sets['finetuned_flip']
    for label,color,name in [(0,'#d65d5d','Different people'),(1,'#278c90','Same person')]:
        axes[1].hist(scores[labels==label],bins=35,alpha=0.6,color=color,label=name)
    axes[1].axvline(rows['finetuned_flip']['threshold'],color='black',linestyle='--',label='DevTrain threshold')
    axes[1].set(xlabel='Cosine similarity',ylabel='Number of pairs',title='Frozen threshold evaluation')
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'evaluation.png',dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
