"""Select among preregistered refinement families using CASIA validation only."""
import argparse
import json
import sys
import shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verification import FaceVerifier,sha256,open_image
from refinement import diagnostic_image
from train_transformer import make_pairs
from metrics import roc_curve,threshold_search


def summarize(features,pairs,labels):
    scores=np.sum(features[pairs[:,0]]*features[pairs[:,1]],axis=1)
    far,tar=roc_curve(scores,labels)
    acc,threshold=threshold_search(scores,labels)
    return {'auc':float(np.trapezoid(tar,far)),'balanced_accuracy':acc,'threshold':threshold}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:7')
    parser.add_argument('--activate',action='store_true',help='Copy validation-selected checkpoint to the default inference path')
    args=parser.parse_args()
    work=ROOT/'artifacts/refinement';work.mkdir(exist_ok=True)
    split=json.loads((ROOT/'artifacts/transformer_training/split.json').read_text())
    paths=[str(ROOT/'datasets/CASIA-WebFace-Align-96'/p) for paths in split['validation'].values() for p in paths]
    per_id=len(next(iter(split['validation'].values())))
    pairs,labels=make_pairs(len(split['validation']),per_id,split['seed']+1)
    results={}
    for name,checkpoint in [('pretrained',ROOT/'checkpoints/pretrained_vit/model.safetensors'),
                            ('metric',ROOT/'checkpoints/transformer_metric.pth'),
                            ('robust',ROOT/'checkpoints/transformer_robust.pth'),
                            ('combined',ROOT/'checkpoints/transformer_combined.pth')]:
        verifier=FaceVerifier(checkpoint,threshold=0,tta=True,device=args.device)
        row={'checkpoint':str(checkpoint),'checkpoint_sha256':verifier.model_hash}
        for condition in ['clean','diagnostic']:
            cache=work/f'validation_{verifier.model_hash[:12]}_{condition}.npz'
            if cache.exists():
                with np.load(cache) as saved:features=saved['features']
            else:
                parts=[]
                for start in range(0,len(paths),64):
                    images=[open_image(p) for p in paths[start:start+64]]
                    if condition=='diagnostic':images=[diagnostic_image(im) for im in images]
                    parts.append(verifier.embed(images))
                features=np.concatenate(parts);np.savez_compressed(cache,features=features)
            row[condition]=summarize(features,pairs,labels)
        results[name]=row;print(name,json.dumps(row),flush=True)
        del verifier
    initial_clean=results['pretrained']['clean']['auc']
    eligible={k:r for k,r in results.items() if k!='pretrained' and r['clean']['auc']>=initial_clean-.001}
    for row in results.values():row['selection_score']=(row['clean']['auc']+row['diagnostic']['auc'])/2
    selected=max(eligible,key=lambda name:eligible[name]['selection_score']) if eligible else 'pretrained'
    record={'selection':'maximum mean of clean/diagnostic CASIA AUC, requiring clean AUC >= initialization -.001',
            'selected':selected,'candidates':results,'test_labels_used':False,
            'diagnostic':'Every image resized to 40x40, JPEG quality 45, then restored to 112x112',
            'validation_split_sha256':sha256(ROOT/'artifacts/transformer_training/split.json')}
    (work/'selection.json').write_text(json.dumps(record,indent=2))
    if args.activate:
        if selected=='pretrained':raise RuntimeError('No eligible refined model; retain the previous default')
        source=Path(results[selected]['checkpoint'])
        destination=ROOT/'checkpoints/transformer.pth'
        shutil.copy2(source,destination)
        record['activated_checkpoint']=str(destination)
        (work/'selection.json').write_text(json.dumps(record,indent=2))
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
