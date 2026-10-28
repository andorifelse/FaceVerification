"""Reproduce the full CASIA-only refinement and final evaluation pipeline."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--train-only',action='store_true',help='Stop after CASIA-only model selection and activation')
    args=parser.parse_args()
    def run(*command):
        print('Running:',sys.executable,*map(str,command),flush=True)
        subprocess.run([sys.executable,*map(str,command)],cwd=ROOT,check=True)
    robust_work=ROOT/'artifacts/refinement/robust_training'
    training=['train_transformer.py','--device',args.device,'--recipe','robust','--epochs','8',
              '--train-blocks','4','--lr','0.00002','--head-lr','0.00004','--anchor-weight','2',
              '--work-dir',robust_work,'--output','checkpoints/transformer_robust.pth']
    previous=ROOT/'artifacts/transformer_training'
    reusable=list(previous.glob('teacher_*.npz'))
    if reusable:training+=['--teacher-cache-dir',previous]
    run(*training)
    cache_folder=previous if reusable else robust_work
    cache=next(cache_folder.glob('teacher_5fafd6b7d599_*.npz'))
    run('train_metric.py','--cache',cache)
    robust_cache='artifacts/refinement/robust_casia.npz'
    run('scripts/cache_casia.py','--checkpoint','checkpoints/transformer_robust.pth','--output',robust_cache,'--device',args.device)
    run('train_metric.py','--init','checkpoints/transformer_robust.pth','--cache',robust_cache,
        '--output','checkpoints/transformer_combined.pth','--work-dir','artifacts/refinement/combined_training')
    run('scripts/select_refinement.py','--device',args.device,'--activate')
    if args.train_only:
        print('Selected model activated. Run python evaluate.py to recalibrate its threshold.',flush=True)
        return
    run('evaluate.py','--device',args.device)
    run('evaluate_refinement.py','--device',args.device)
    run('scripts/validate_system.py','--device',args.device)
    run('scripts/build_report.py')
    print('Completed refinement; run python -m pytest -q to verify code.',flush=True)


if __name__=='__main__':main()
