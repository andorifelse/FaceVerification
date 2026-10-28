"""Record inference timing and representative errors without changing the model."""
import csv
import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image, ImageDraw
from config import config
from detection import FaceDetector
from verification import FaceVerifier, open_image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default=config.device)
    args=parser.parse_args()
    verifier = FaceVerifier(device=args.device)
    os.environ['FACE_DEVICE']=args.device
    def synchronize():
        if str(args.device).startswith('cuda'):torch.cuda.synchronize(args.device)
    detector = FaceDetector()
    image = ROOT/'samples/009.jpg'
    verifier.compare(image,image)
    pair_times = []
    for _ in range(10):
        synchronize();started = time.perf_counter();verifier.compare(image,image);synchronize()
        pair_times.append(time.perf_counter()-started)
    detector.single(image)
    detection_times = []
    for _ in range(10):
        started = time.perf_counter();detector.single(image)
        detection_times.append(time.perf_counter()-started)
    from FaceVerification import FaceVerification
    value = FaceVerification(str(image),str(image))
    assert type(value) is int and value == 1
    results = {'device':args.device,'threads':torch.get_num_threads(),'timing_iterations':10,
               'backbone':verifier.metadata.get('backbone','fmobile'),'checkpoint_sha256':verifier.model_hash,
               'mean_pair_seconds':sum(pair_times)/len(pair_times),
               'mean_detection_seconds':sum(detection_times)/len(detection_times),
               'pair_seconds':pair_times,'detection_seconds':detection_times,
               'required_interface_same_image_result':value,
               'camera_hardware_tested':False}
    (ROOT/'artifacts/system_validation.json').write_text(json.dumps(results,indent=2))
    with (ROOT/'artifacts/predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    false_negatives = [r for r in rows if r['label']=='1' and r['prediction']=='0'][:3]
    false_positives = [r for r in rows if r['label']=='0' and r['prediction']=='1'][:3]
    selected = false_negatives+false_positives
    if selected:
        sheet = Image.new('RGB',(660,180*len(selected)),(248,250,248))
        draw = ImageDraw.Draw(sheet)
        for i,row in enumerate(selected):
            for j,key in enumerate(['image1','image2']):
                face = open_image(config.test_root/row[key]);face.thumbnail((130,150))
                sheet.paste(face,(15+j*145,i*180+20))
            kind = 'False rejection' if row['label']=='1' else 'False acceptance'
            draw.text((315,i*180+35),kind,fill=(40,65,55))
            draw.text((315,i*180+60),f"similarity = {float(row['similarity']):.4f}",fill=(40,65,55))
            draw.text((315,i*180+80),f"threshold = {verifier.threshold:.4f}",fill=(40,65,55))
            for j,key in enumerate(['image1','image2']):
                name = Path(row[key]).stem
                draw.text((315,i*180+110+j*20),name[:48],fill=(85,100,95))
        sheet.save(ROOT/'artifacts/error_examples.jpg',quality=90)
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
