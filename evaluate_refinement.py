"""Final comparisons after CASIA-only selection, with explicit clean/diagnostic labels."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np

from config import ROOT,config
from dataset import IMAGE_SUFFIXES
from evaluate import cross_validate
from metrics import read_pairs,pair_scores,summarize,threshold_search
from refinement import diagnostic_image
from train_transformer import make_pairs
from verification import FaceVerifier,open_image,sha256


def audit_split(output):
    file=output/'audit_split.json'
    if file.exists():return json.loads(file.read_text())
    split=json.loads((ROOT/'artifacts/transformer_training/split.json').read_text())
    excluded=set(split['train'])|set(split['validation'])
    folders=sorted(p for p in config.train_root.iterdir() if p.is_dir() and p.name not in excluded)
    rng=np.random.default_rng(314);rng.shuffle(folders)
    identities={}
    for folder in folders:
        files=sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if len(files)<16:continue
        indexes=rng.choice(len(files),16,replace=False)
        identities[folder.name]=[str(files[i].relative_to(config.train_root)) for i in indexes]
        if len(identities)==256:break
    result={'seed':314,'identities':identities,'excluded_train_validation_identities':len(excluded)}
    file.write_text(json.dumps(result,indent=2));return result


def extract(checkpoint,paths,condition,work,device):
    verifier=FaceVerifier(checkpoint,threshold=0,tta=True,device=device)
    fingerprint=hashlib.sha256('\n'.join(paths).encode()).hexdigest()[:12]
    cache=work/f'final_{verifier.model_hash[:12]}_{fingerprint}_{condition}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:return saved['features']
    if condition=='clean':
        embedding_cache=ROOT/'artifacts'/f'embeddings_{verifier.model_hash[:16]}.npz'
        if embedding_cache.exists():
            with np.load(embedding_cache,allow_pickle=False) as saved:
                if saved['paths'].tolist()==paths:
                    features=saved['raw']+saved['flip']
                    features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
                    np.savez_compressed(cache,features=features)
                    return features
    features=[]
    for start in range(0,len(paths),64):
        images=[open_image(p) for p in paths[start:start+64]]
        if condition=='diagnostic':images=[diagnostic_image(image) for image in images]
        features.append(verifier.embed(images))
        if start%2048==0:print(checkpoint.name,condition,min(start+64,len(paths)),'/',len(paths),flush=True)
    features=np.concatenate(features);np.savez_compressed(cache,features=features)
    return features


def final_audit(features,per_id=16):
    # Fixed first 128 identities for threshold calibration; other 128 for audit.
    middle=128*per_id
    cp,cy=make_pairs(128,per_id,315);tp,ty=make_pairs(128,per_id,316)
    cs=np.sum(features[cp[:,0]]*features[cp[:,1]],axis=1)
    _,threshold=threshold_search(cs,cy)
    ts=np.sum(features[middle+tp[:,0]]*features[middle+tp[:,1]],axis=1)
    return summarize(ts,ty,threshold)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:7')
    args=p.parse_args();work=ROOT/'artifacts/refinement'
    selection=json.loads((work/'selection.json').read_text())
    chosen=selection['candidates'][selection['selected']]['checkpoint']
    calibration,_=read_pairs(config.calibration_list,config.test_root)
    test,_=read_pairs(config.test_list,config.test_root)
    standard,header=read_pairs(ROOT/'datasets/pairs.txt',config.test_root)
    paths=sorted({p for pair in calibration+test+standard for p in pair[:2]})
    audit=audit_split(work)
    audit_paths=[str(config.train_root/p) for paths in audit['identities'].values() for p in paths]
    models={'pretrained':ROOT/'checkpoints/pretrained_vit/model.safetensors',
            'round1':ROOT/'checkpoints/transformer_round1.pth',
            'metric':ROOT/'checkpoints/transformer_metric.pth',
            'robust':ROOT/'checkpoints/transformer_robust.pth',
            'combined':ROOT/'checkpoints/transformer_combined.pth'}
    records={}
    for name,checkpoint in models.items():
        if name=='round1' and not checkpoint.exists():continue
        row={'checkpoint_sha256':sha256(checkpoint),'checkpoint':str(checkpoint)}
        for condition in ['clean','diagnostic']:
            features=extract(checkpoint,paths,condition,work,args.device)
            cs,cy=pair_scores(features,paths,calibration);ts,ty=pair_scores(features,paths,test)
            _,threshold=threshold_search(cs,cy)
            ss,sy=pair_scores(features,paths,standard)
            row[condition]={'devtest':summarize(ts,ty,threshold),'standard_cv':cross_validate(ss,sy,header[0])}
            af=extract(checkpoint,audit_paths,condition,work,args.device)
            row[condition]['casia_audit']=final_audit(af)
        records[name]=row
        print(name,json.dumps({c:{'devtest':row[c]['devtest']['accuracy'],
                                  'standard_cv':row[c]['standard_cv']['mean_accuracy'],
                                  'audit':row[c]['casia_audit']['accuracy']} for c in ['clean','diagnostic']}),flush=True)
    result={'selection':selection['selected'],'selected_checkpoint':chosen,
            'selection_locked_before_final_evaluation':True,'models':records,
            'diagnostic_protocol':'40px bilinear downsample -> JPEG quality45 ->112px restore, same images and settings for all models',
            'casia_audit':'256 additional identities exclude all training/selection identities; first128 calibrate threshold, last128 test; seed314',
            'limitations':'LFW results from previous rounds already observed; this is a repeated public benchmark evaluation, not an untouched external holdout.'}
    (work/'evaluation.json').write_text(json.dumps(result,indent=2))
    export_profiles(result,work)
    print('Final refinement comparisons saved.',flush=True)


def export_profiles(result,work):
    for name,row in result['models'].items():
        settings={'threshold':row['clean']['devtest']['threshold'],'tta':True,
                  'checkpoint_sha256':row['checkpoint_sha256'],'backbone':'face_vit',
                  'calibration_protocol':str(config.calibration_list),
                  'calibration_protocol_sha256':sha256(config.calibration_list),
                  'decision':'cosine_similarity >= threshold',
                  'preprocessing':'RGB, resize 112x112, (pixel/255-0.5)/0.5',
                  'profile':name,'deployment_note':'Optional profile; CASIA-AUC-selected default is '+result['selection']}
        (work/f'calibration_{name}.json').write_text(json.dumps(settings,indent=2))


if __name__=='__main__':main()
