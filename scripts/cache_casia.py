"""Cache original/flipped CASIA embeddings for a fixed identity manifest."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verification import FaceVerifier


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:7')
    args=p.parse_args()
    split=json.loads((ROOT/'artifacts/transformer_training/split.json').read_text())
    paths=[str(ROOT/'datasets/CASIA-WebFace-Align-96'/path) for part in ['train','validation'] for group in split[part].values() for path in group]
    verifier=FaceVerifier(args.checkpoint,threshold=0,tta=False,device=args.device)
    from PIL import ImageOps
    from verification import open_image
    a,b=[],[]
    for start in range(0,len(paths),128):
        images=[open_image(p) for p in paths[start:start+128]]
        a.append(verifier.embed(images,128));b.append(verifier.embed([ImageOps.mirror(im) for im in images],128))
        if start%4096==0:print('Cache',min(start+128,len(paths)),'/',len(paths),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,normal=np.concatenate(a),flipped=np.concatenate(b),checkpoint_sha256=verifier.model_hash)
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
