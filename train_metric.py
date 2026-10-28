"""Fit a quality-filtered, shrinkage within-identity metric; CASIA-only selection."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from config import ROOT
from metrics import roc_curve,threshold_search
from train_transformer import make_pairs
from verification import load_backbone,sha256


def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def measure(normal,flip,pairs,labels,matrix=None):
    if matrix is not None:normal=normal@matrix.T;flip=flip@matrix.T
    fused=normalize(normalize(normal)+normalize(flip))
    scores=np.sum(fused[pairs[:,0]]*fused[pairs[:,1]],axis=1)
    far,tar=roc_curve(scores,labels)
    accuracy,threshold=threshold_search(scores,labels)
    return {'auc':float(np.trapezoid(tar,far)),'balanced_accuracy':accuracy,'threshold':threshold}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init',type=Path,default=ROOT/'checkpoints/pretrained_vit/model.safetensors')
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--split',type=Path,default=ROOT/'artifacts/transformer_training/split.json')
    parser.add_argument('--output',type=Path,default=ROOT/'checkpoints/transformer_metric.pth')
    parser.add_argument('--work-dir',type=Path,default=ROOT/'artifacts/refinement/metric_training')
    args=parser.parse_args();args.work_dir.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();torch.set_num_threads(8)
    manifest=json.loads(args.split.read_text())
    n_train=len(manifest['train']);n_val=len(manifest['validation']);per_id=len(next(iter(manifest['train'].values())))
    cache=args.cache or next((ROOT/'artifacts/transformer_training').glob('teacher_*.npz'))
    with np.load(cache,allow_pickle=False) as saved:
        a,b=saved['normal'],saved['flipped']
        if 'checkpoint_sha256' in saved and str(saved['checkpoint_sha256'])!=sha256(args.init):
            raise ValueError('Metric fitting cache does not match initialization checkpoint')
    assert len(a)==(n_train+n_val)*per_id
    clean=normalize(a[:n_train*per_id]+b[:n_train*per_id]).reshape(n_train,per_id,-1)
    centroid=normalize(clean.mean(1))
    quality=np.sum(clean*centroid[:,None],axis=-1)
    mask=quality>=.3
    # Re-estimate identity means after excluding extreme low-consistency samples.
    mean=(clean*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1)
    residual=(clean-mean[:,None])[mask].astype(np.float64)
    within=residual.T@residual/max(1,len(residual)-n_train)
    eigenvalues,eigenvectors=np.linalg.eigh(within)
    pairs,labels=make_pairs(n_val,per_id,manifest['seed']+1)
    va,vb=a[n_train*per_id:],b[n_train*per_id:]
    baseline=measure(va,vb,pairs,labels)
    candidates=[];best_rank=(baseline['auc'],baseline['balanced_accuracy']);best_matrix=np.eye(512,dtype=np.float32);best=None
    for shrinkage in [.05,.2,.5]:
        values=(1-shrinkage)*eigenvalues+shrinkage*eigenvalues.mean()
        for strength in [.25,.5,.75,1.]:
            scales=np.maximum(values,1e-12)**(-strength/2)
            scales/=np.exp(np.log(scales).mean())
            matrix=((eigenvectors*scales)@eigenvectors.T).astype(np.float32)
            result=measure(va,vb,pairs,labels,matrix)
            row={'shrinkage':shrinkage,'strength':strength,**result};candidates.append(row)
            rank=(result['auc'],result['balanced_accuracy'])
            print(json.dumps(row),flush=True)
            if rank>best_rank:best_rank=rank;best_matrix=matrix;best=row
    net,metadata=load_backbone(args.init,'cpu')
    from torch import nn
    net.metric_projection=nn.Linear(512,512,bias=False)
    net.metric_projection.weight.data.copy_(torch.from_numpy(best_matrix))
    metadata={**metadata,'metric_projection':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'backbone':net.state_dict(),'model_config':metadata,
                'training':{'method':'quality-filtered within-class covariance normalization',
                            'init_sha256':sha256(args.init),'selected':best}},args.output)
    result={'method':'quality-filtered shrinkage within-class covariance projection',
            'selection':'maximum clean CASIA validation AUC, then balanced accuracy; identity transform included',
            'gradient_updates':False,'fitted_values':int(np.count_nonzero(best_matrix-np.eye(512))),
            'train_identities':n_train,'train_images':n_train*per_id,'kept_images':int(mask.sum()),
            'validation_identities':n_val,'baseline_validation':baseline,'selected':best,
            'candidates':candidates,'checkpoint_sha256':sha256(args.output),'initial_checkpoint_sha256':sha256(args.init),
            'cache':str(cache),'cache_sha256':sha256(cache),'split_sha256':sha256(args.split),
            'elapsed_seconds':time.perf_counter()-started}
    (args.work_dir/'training.json').write_text(json.dumps(result,indent=2))
    print('BASELINE',baseline,'SELECTED',best,'Saved',args.output,flush=True)


if __name__=='__main__':main()
