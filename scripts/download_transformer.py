"""Download the pinned, checksummed ViT initialization for reproducible fine-tuning."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    metadata=json.loads((ROOT/'third_party/vit_initialization.json').read_text())
    destination=ROOT/'checkpoints/pretrained_vit'/metadata['filename']
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()==metadata['sha256']:
        print(f'Already verified: {destination}');return
    url=f'https://huggingface.co/{metadata["repo"]}/resolve/{metadata["revision"]}/{metadata["filename"]}?download=true'
    temporary=destination.with_suffix('.part')
    digest=hashlib.sha256();downloaded=0
    with urllib.request.urlopen(url,timeout=90) as response,temporary.open('wb') as stream:
        while block:=response.read(4*1024*1024):
            stream.write(block);digest.update(block);downloaded+=len(block)
            if downloaded%(64*1024*1024)==0:print(f'{downloaded//1024**2} MiB downloaded',flush=True)
    if digest.hexdigest()!=metadata['sha256']:
        raise RuntimeError(f'Checksum mismatch; incomplete file retained at {temporary}')
    temporary.replace(destination)
    print(f'Verified initialization: {destination}')
    print('Next: python train.py --device cuda')


if __name__=='__main__':main()
