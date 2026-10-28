"""Fine-tune the supplied backbone on CASIA; no downloaded verifier weights."""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ROOT, config
from dataset import FaceImages, select_identities
from metrics import threshold_search
from model.metric import CosFace
from verification import load_backbone, sha256


def validation_pairs(groups, seed):
    rng = np.random.default_rng(seed)
    per_id = len(groups[0][1])
    positive, negative = [], []
    for c in range(len(groups)):
        for j in range(per_id):
            other = (j + int(rng.integers(1, per_id))) % per_id
            positive.append((c*per_id+j, c*per_id+other))
            other_c = (c + int(rng.integers(1, len(groups)))) % len(groups)
            negative.append((c*per_id+j, other_c*per_id+int(rng.integers(per_id))))
    return np.array(positive + negative), np.r_[np.ones(len(positive)), np.zeros(len(negative))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=config.train_root)
    parser.add_argument('--init', type=Path, default=config.checkpoints/'18.pth')
    parser.add_argument('--output', type=Path, default=config.checkpoints/'finetuned.pth')
    parser.add_argument('--train-ids', type=int, default=512)
    parser.add_argument('--validation-ids', type=int, default=128)
    parser.add_argument('--per-id', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--anchor-weight', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default=config.device)
    parser.add_argument('--scratch', action='store_true', help='随机初始化并训练全部层；需要较长训练')
    parser.add_argument('--full', action='store_true', help='微调全部层；默认缓存卷积特征，仅微调投影层')
    parser.add_argument('--work-dir', type=Path, default=ROOT/'artifacts/training')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs >= 1 and batch-size >= 2 required')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_groups, val_groups = select_identities(args.data_root, args.train_ids, args.validation_ids, args.per_id, args.seed)
    manifest = {'seed': args.seed, 'train': {}, 'validation': {}}
    for name, groups in [('train', train_groups), ('validation', val_groups)]:
        manifest[name] = {identity: [str(p.relative_to(args.data_root)) for p in paths] for identity, paths in groups}
    (args.work_dir/'split.json').write_text(json.dumps(manifest, indent=2))
    if args.scratch:
        from model import FaceMobileNet
        net = FaceMobileNet(512).to(args.device).eval()
        init_hash = None
    else:
        net, _ = load_backbone(args.init, args.device)
        init_hash = sha256(args.init)
    reference = copy.deepcopy(net).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    paths = [p for _, images in train_groups+val_groups for p in images]
    count_train = args.train_ids * args.per_id
    labels = torch.arange(args.train_ids, device=args.device).repeat_interleave(args.per_id)
    loader = DataLoader(FaceImages(paths, config.test_transform), batch_size=args.batch_size, shuffle=False)
    cache_key = sha256(args.work_dir/'split.json')[:12] + '_' + (init_hash or 'scratch')[:12]
    cache_path = args.work_dir/f'features_{cache_key}.npz'
    if cache_path.exists() and not args.scratch:
        with np.load(cache_path, allow_pickle=False) as saved:
            raw, flip = torch.from_numpy(saved['raw']), torch.from_numpy(saved['flip'])
    else:
        normal, flipped = [], []
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                batch = batch.to(args.device)
                normal.append(reference.forward_features(batch).cpu())
                flipped.append(reference.forward_features(torch.flip(batch, [-1])).cpu())
                if i % 10 == 0:
                    print(f'Cache convolution features: {min((i+1)*args.batch_size,len(paths))}/{len(paths)}', flush=True)
        raw, flip = torch.cat(normal), torch.cat(flipped)
        np.savez_compressed(cache_path, raw=raw.numpy(), flip=flip.numpy())
    raw, flip = raw.to(args.device), flip.to(args.device)
    with torch.no_grad():
        original = F.normalize(reference.bn(reference.linear(raw)), dim=1)
        original_flip = F.normalize(reference.bn(reference.linear(flip)), dim=1)
    metric = CosFace(512, args.train_ids, s=30., m=0.2).to(args.device)
    with torch.no_grad():
        prototypes = original[:count_train].reshape(args.train_ids, args.per_id, -1).mean(1)
        metric.weight.copy_(F.normalize(prototypes, dim=1))
    full = args.full or args.scratch
    for parameter in net.parameters():
        parameter.requires_grad_(full)
    for parameter in net.linear.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr},
        {'params': metric.parameters(), 'lr': args.lr*20},
    ], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    vpairs, vlabels = validation_pairs(val_groups, args.seed+1)
    def validation():
        net.eval()
        with torch.no_grad():
            if full:
                embedded = []
                vloader = DataLoader(FaceImages(paths[count_train:], config.test_transform), batch_size=args.batch_size)
                for batch in vloader:
                    batch = batch.to(args.device)
                    embedded.append(F.normalize(F.normalize(net(batch), dim=1)+F.normalize(net(torch.flip(batch, [-1])), dim=1), dim=1))
                f = torch.cat(embedded).cpu().numpy()
            else:
                a = F.normalize(net.bn(net.linear(raw[count_train:])), dim=1)
                b = F.normalize(net.bn(net.linear(flip[count_train:])), dim=1)
                f = F.normalize(a+b, dim=1).cpu().numpy()
            scores = np.sum(f[vpairs[:,0]]*f[vpairs[:,1]], axis=1)
            return threshold_search(scores, vlabels)[0]
    baseline = validation()
    print(f'Initial CASIA validation balanced accuracy: {baseline:.6f}', flush=True)
    history, best_accuracy, best_epoch = [], -1., 0
    for epoch in range(args.epochs):
        net.train() if full else net.eval()  # preserve inherited BN statistics in head mode
        total_loss = total_n = 0
        order = torch.randperm(count_train, device=args.device)
        for offset in range(0, count_train, args.batch_size):
            indexes = order[offset:offset+args.batch_size]
            if len(indexes) < 2:
                continue
            flip_mask = torch.rand(len(indexes), device=args.device) < 0.5
            target = torch.where(flip_mask[:,None], original_flip[indexes], original[indexes])
            if full:
                images = [config.test_transform(__import__('verification').open_image(paths[i])) for i in indexes.tolist()]
                batch = torch.stack(images).to(args.device)
                batch[flip_mask] = torch.flip(batch[flip_mask], [-1])
                embedding = net(batch)
            else:
                features = torch.where(flip_mask[:,None], flip[indexes], raw[indexes])
                embedding = net.bn(net.linear(features))
            classification = F.cross_entropy(metric(embedding, labels[indexes]), labels[indexes])
            anchor = (1 - (F.normalize(embedding, dim=1)*target).sum(1)).mean()
            loss = classification + (0. if args.scratch else args.anchor_weight)*anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 5.)
            optimizer.step()
            total_loss += float(loss.detach())*len(indexes); total_n += len(indexes)
        scheduler.step()
        accuracy = validation()
        row = {'epoch':epoch+1, 'loss':total_loss/total_n, 'validation_balanced_accuracy':accuracy}
        history.append(row)
        print(json.dumps(row), flush=True)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch+1
            torch.save({'backbone':{k:v.detach().cpu() for k,v in net.state_dict().items()},
                        'model_config':{'backbone':'fmobile','embedding_size':512},
                        'training':{'mode':'scratch' if args.scratch else ('full' if full else 'head'),
                                    'epoch':best_epoch,'seed':args.seed,'init_sha256':init_hash}}, args.output)
    saved, _ = load_backbone(args.output)
    initial = reference.cpu().state_dict()
    changed = sum(int(torch.count_nonzero(v-initial[k])) for k,v in saved.state_dict().items() if v.is_floating_point())
    record = {'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'initial_validation_balanced_accuracy':baseline, 'best_epoch':best_epoch,
              'best_validation_balanced_accuracy':best_accuracy, 'history':history,
              'changed_floating_values':changed, 'checkpoint_sha256':sha256(args.output),
              'initial_checkpoint_sha256':init_hash, 'elapsed_seconds':time.perf_counter()-started,
              'train_images':count_train,'validation_images':len(paths)-count_train,
              'torch_version':str(torch.__version__),'numpy_version':str(np.__version__)}
    (args.work_dir/'training.json').write_text(json.dumps(record,indent=2))
    print(f'Saved {args.output}; changed parameters/buffers: {changed}',flush=True)


if __name__ == '__main__':
    main()
