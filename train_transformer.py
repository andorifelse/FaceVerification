"""Actually fine-tune CVLFace ViT on CASIA, with disjoint identity validation."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

from config import ROOT,config
from dataset import select_identities
from metrics import threshold_search,roc_curve
from model.metric import CosFace
from verification import load_backbone,make_transform,open_image,sha256

MODEL_CONFIG = {'backbone':'face_vit','embedding_size':512,'input_size':112,
                'color_space':'RGB','depth':24,'patch_size':8,'embed_dim':512,'heads':16}


class IndexedFaces(Dataset):
    def __init__(self,paths,transform,per_id=16):
        self.paths,self.transform,self.per_id = paths,transform,per_id

    def __len__(self):return len(self.paths)

    def __getitem__(self,index):
        return self.transform(open_image(self.paths[index])),index//self.per_id,index


def make_pairs(identities,per_id,seed=43):
    rng = np.random.default_rng(seed)
    pairs,labels = [],[]
    for identity in range(identities):
        for i in range(per_id):
            j = (i+int(rng.integers(1,per_id)))%per_id
            pairs.append((identity*per_id+i,identity*per_id+j));labels.append(1)
            other = (identity+int(rng.integers(1,identities)))%identities
            pairs.append((identity*per_id+i,other*per_id+int(rng.integers(per_id))));labels.append(0)
    return np.asarray(pairs),np.asarray(labels)


def validate(net,loader,device,pairs,labels,amp=True):
    net.eval();features=[]
    with torch.inference_mode():
        for images,_,_ in loader:
            images = images.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp):
                a = F.normalize(net(images).float(),dim=1)
                b = F.normalize(net(images.flip(-1)).float(),dim=1)
            features.append(F.normalize(a+b,dim=1).cpu().numpy())
    features = np.concatenate(features)
    scores = np.sum(features[pairs[:,0]]*features[pairs[:,1]],axis=1)
    accuracy,threshold = threshold_search(scores,labels)
    far,tar = roc_curve(scores,labels)
    return {'balanced_accuracy':accuracy,'threshold':threshold,'auc':float(np.trapezoid(tar,far))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=config.train_root)
    p.add_argument('--init',type=Path,default=ROOT/'checkpoints/pretrained_vit/model.safetensors')
    p.add_argument('--output',type=Path,default=ROOT/'checkpoints/transformer.pth')
    p.add_argument('--work-dir',type=Path,default=ROOT/'artifacts/transformer_training')
    p.add_argument('--teacher-cache-dir',type=Path,help='Reuse initialization embeddings for the exact same split')
    p.add_argument('--train-ids',type=int,default=4096)
    p.add_argument('--validation-ids',type=int,default=256)
    p.add_argument('--per-id',type=int,default=16)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--train-blocks',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--head-lr',type=float,default=2e-5)
    p.add_argument('--anchor-weight',type=float,default=10.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--recipe',choices=['classification','robust'],default='classification')
    p.add_argument('--augmentation-probability',type=float,default=.5)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<2 or not 0<=args.train_blocks<=24:p.error('Invalid epochs/batch size/train blocks')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not accessible. Run outside a GPU-restricted sandbox or select --device cpu explicitly.')
    device=torch.device(args.device);amp=device.type=='cuda' and not args.no_amp
    torch.set_num_threads(8)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('high')
    args.work_dir.mkdir(parents=True,exist_ok=True);args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    train_groups,val_groups=select_identities(args.data_root,args.train_ids,args.validation_ids,args.per_id,args.seed)
    manifest={'seed':args.seed,'train':{},'validation':{}}
    for name,groups in [('train',train_groups),('validation',val_groups)]:
        manifest[name]={identity:[str(path.relative_to(args.data_root)) for path in paths] for identity,paths in groups}
    split_path=args.work_dir/'split.json';split_path.write_text(json.dumps(manifest,indent=2))
    net,metadata=load_backbone(args.init,device)
    if metadata.get('backbone')!='face_vit':raise ValueError('Transformer initialization is required')
    transform=make_transform(metadata)
    init_hash=sha256(args.init)
    train_paths=[path for _,paths in train_groups for path in paths]
    val_paths=[path for _,paths in val_groups for path in paths]
    paths=train_paths+val_paths
    def loader(paths,shuffle=False):
        return DataLoader(IndexedFaces(paths,transform,args.per_id),batch_size=args.batch_size,shuffle=shuffle,
                          num_workers=args.workers,pin_memory=device.type=='cuda',
                          persistent_workers=args.workers>0,drop_last=shuffle)
    all_loader=loader(paths);train_loader=loader(train_paths,True);val_loader=loader(val_paths)
    gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'
    print(json.dumps({'device':str(device),'hardware':gpu,'train_images':len(train_paths),'validation_images':len(val_paths),
                      'initialization_sha256':init_hash,'amp':amp}),flush=True)
    cache=(args.teacher_cache_dir or args.work_dir)/f'teacher_{init_hash[:12]}_{sha256(split_path)[:12]}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:normal,flipped=saved['normal'],saved['flipped']
    else:
        a,b=[],[];net.eval()
        with torch.inference_mode():
            for step,(images,_,_) in enumerate(all_loader):
                images=images.to(device,non_blocking=True)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp):
                    a.append(F.normalize(net(images).float(),dim=1).cpu().numpy())
                    b.append(F.normalize(net(images.flip(-1)).float(),dim=1).cpu().numpy())
                if step%20==0:print(f'Initial feature cache {min((step+1)*args.batch_size,len(paths))}/{len(paths)}',flush=True)
        normal,flipped=np.concatenate(a),np.concatenate(b)
        np.savez_compressed(cache,normal=normal,flipped=flipped)
    vpairs,vlabels=make_pairs(args.validation_ids,args.per_id,args.seed+1)
    initial_f=normal[len(train_paths):]+flipped[len(train_paths):]
    initial_f/=np.linalg.norm(initial_f,axis=1,keepdims=True)
    initial_s=np.sum(initial_f[vpairs[:,0]]*initial_f[vpairs[:,1]],axis=1)
    initial_accuracy,_=threshold_search(initial_s,vlabels)
    initial_far,initial_tar=roc_curve(initial_s,vlabels)
    initial_auc=float(np.trapezoid(initial_tar,initial_far))
    robust_val_loader=None
    initial_robust_validation=None
    if args.recipe=='robust':
        from refinement import DiagnosticTransform
        robust_val_loader=DataLoader(IndexedFaces(val_paths,DiagnosticTransform(transform),args.per_id),
                                    batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
        initial_robust_validation=validate(net,robust_val_loader,device,vpairs,vlabels,amp)
        print('Initial robust validation:',json.dumps(initial_robust_validation),flush=True)
    print(f'Initial CASIA held-out validation: balanced_accuracy={initial_accuracy:.6f}, auc={initial_auc:.6f}',flush=True)
    normal=torch.from_numpy(normal[:len(train_paths)]).to(device)
    flipped=torch.from_numpy(flipped[:len(train_paths)]).to(device)
    classifier=CosFace(512,args.train_ids,s=32.,m=.15).to(device)
    with torch.no_grad():
        prototypes=F.normalize(normal+flipped,dim=1).reshape(args.train_ids,args.per_id,512).mean(1)
        classifier.weight.copy_(F.normalize(prototypes,dim=1))
        quality=(F.normalize(normal+flipped,dim=1)*F.normalize(prototypes,dim=1).repeat_interleave(args.per_id,dim=0)).sum(1)
        quality=((quality-.25)/.5).clamp(0,1)
    net.set_trainable_blocks(args.train_blocks)
    block_parameters=[parameter for name,parameter in net.named_parameters() if parameter.requires_grad and not name.startswith('feature.')]
    head_parameters=[parameter for name,parameter in net.named_parameters() if parameter.requires_grad and name.startswith('feature.')]
    optimizer=torch.optim.AdamW([{'params':block_parameters,'lr':args.lr},
                                 {'params':head_parameters,'lr':args.head_lr},
                                 {'params':classifier.parameters(),'lr':1e-3}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    history=[];best=(-1.,-1.);best_epoch=0;best_validation=None
    trainable=sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad)
    total=sum(parameter.numel() for parameter in net.parameters())
    for epoch in range(args.epochs):
        epoch_start=time.perf_counter();net.train()
        n=0;loss_total=ce_total=anchor_total=0.
        for step,(images,labels,indexes) in enumerate(train_loader):
            images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True);indexes=indexes.to(device)
            mask=torch.rand(len(images),device=device)<.5
            images[mask]=images[mask].flip(-1)
            target=torch.where(mask[:,None],flipped[indexes],normal[indexes])
            if args.recipe=='robust':
                # Half of each batch is reduced to 40/56/72 pixels then restored;
                # the target always remains the clean initialization embedding.
                degraded=torch.rand(len(images),device=device)<args.augmentation_probability
                size=[40,56,72][step%3]
                small=F.interpolate(images[degraded],size=(size,size),mode='bilinear',align_corners=False)
                images[degraded]=F.interpolate(small,size=(112,112),mode='bilinear',align_corners=False)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp):embedding=net(images)
            embedding=embedding.float()
            ce_per=F.cross_entropy(classifier(embedding,labels),labels,reduction='none')
            anchor_per=1-(F.normalize(embedding,dim=1)*target).sum(1)
            if args.recipe=='robust':
                weights=quality[indexes]
                ce=(ce_per*weights).sum()/weights.sum().clamp_min(1)
                anchor=(anchor_per*weights).sum()/weights.sum().clamp_min(1)
            else:
                ce=ce_per.mean();anchor=anchor_per.mean()
            loss=ce+args.anchor_weight*anchor
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],5.)
            optimizer.step()
            n+=len(images);loss_total+=float(loss.detach())*len(images)
            ce_total+=float(ce.detach())*len(images);anchor_total+=float(anchor.detach())*len(images)
            if step%40==0:print(f'Epoch {epoch+1}/{args.epochs} step {step+1}/{len(train_loader)} loss={float(loss.detach()):.5f}',flush=True)
        scheduler.step()
        validation=validate(net,val_loader,device,vpairs,vlabels,amp)
        robust_validation=validate(net,robust_val_loader,device,vpairs,vlabels,amp) if robust_val_loader else None
        row={'epoch':epoch+1,'loss':loss_total/n,'classification_loss':ce_total/n,'anchor_loss':anchor_total/n,
             'validation_balanced_accuracy':validation['balanced_accuracy'],'validation_auc':validation['auc'],
             'seconds':time.perf_counter()-epoch_start}
        history.append(row)
        # AUC then balanced accuracy; selected only on held-out CASIA identities.
        if robust_validation:
            row['robust_validation']=robust_validation
            # Require no more than .001 absolute AUC regression on clean inputs,
            # then select by the diagnostic AUC; fixed before LFW evaluation.
            rank=(robust_validation['auc'] if validation['auc']>=initial_auc-.001 else -1.,validation['auc'])
        else:rank=(validation['auc'],validation['balanced_accuracy'])
        print(json.dumps(row),flush=True)
        if rank>best:
            best=rank;best_epoch=epoch+1;best_validation=validation
            torch.save({'backbone':{k:v.detach().cpu() for k,v in net.state_dict().items()},'model_config':MODEL_CONFIG,
                        'training':{'epoch':best_epoch,'seed':args.seed,'init_sha256':init_hash,
                                    'train_blocks':args.train_blocks,'train_images':len(train_paths)}},args.output)
        record={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'model_config':MODEL_CONFIG,'hardware':gpu,'total_parameters':total,'trainable_parameters':trainable,
                'initial_validation_balanced_accuracy':initial_accuracy,'initial_validation_auc':initial_auc,
                'best_epoch':best_epoch,'best_validation_auc':best_validation['auc'],
                'best_validation_balanced_accuracy':best_validation['balanced_accuracy'],
                'history':history,'initial_checkpoint_sha256':init_hash,'checkpoint_sha256':sha256(args.output),
                'train_images':len(train_paths),'validation_images':len(val_paths),
                'selection':'maximum held-out CASIA AUC, then balanced accuracy; no LFW labels used',
                'elapsed_seconds':time.perf_counter()-started,'torch_version':str(torch.__version__)}
        if args.recipe=='robust':
            record['initial_robust_validation']=initial_robust_validation
            record['best_selection_score']=best[0]
            record['selection']='Maximum CASIA diagnostic AUC, subject to clean CASIA AUC >= initial -.001; no LFW test labels'
        (args.work_dir/'training.json').write_text(json.dumps(record,indent=2))
    initial,_=load_backbone(args.init,'cpu');selected,_=load_backbone(args.output,'cpu')
    changed={}
    for name,value in selected.state_dict().items():
        original=initial.state_dict()[name]
        count=int(torch.count_nonzero(value-original)) if value.is_floating_point() else 0
        if count:changed[name]=count
    record['changed_floating_values']=sum(changed.values());record['changed_tensors']=changed
    record['elapsed_seconds']=time.perf_counter()-started
    (args.work_dir/'training.json').write_text(json.dumps(record,indent=2))
    print(f'Saved {args.output}; updated {record["changed_floating_values"]:,} values in {len(changed)} tensors.',flush=True)


if __name__=='__main__':main()
