"""Face embeddings and the required Y = FaceVerification(X1, X2) interface."""
import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from config import config
from model import FaceMobileNet, ResIRSE


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_backbone(checkpoint, device='cpu'):
    if Path(checkpoint).suffix == '.safetensors':
        from safetensors.torch import load_file
        payload = {'backbone':load_file(str(checkpoint)),
                   'model_config':{'backbone':'face_vit','embedding_size':512,'input_size':112,'color_space':'RGB'}}
    else:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = payload.get('model_config', {})
    state = payload.get('backbone', payload)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    size = metadata.get('embedding_size', state.get('linear.weight', torch.empty(512, 0)).shape[0])
    name = metadata.get('backbone', 'fmobile')
    if name == 'face_vit':
        from model.face_vit import FaceViT
        net = FaceViT(size,metric_projection=metadata.get('metric_projection',False))
        for prefix in ['model.net.','net.']:
            if all(k.startswith(prefix) for k in state):
                state = {k[len(prefix):]:v for k,v in state.items()}
                break
    elif name == 'resnet':
        net = ResIRSE(size,0.5)
    elif name == 'fmobile':
        net = FaceMobileNet(size)
    else:
        raise ValueError(f'Unsupported backbone: {name}')
    net.load_state_dict(state, strict=True)
    return net.to(device).eval(), metadata


def make_transform(metadata):
    if metadata.get('backbone') == 'face_vit':
        return T.Compose([T.Resize((112,112)),T.ToTensor(),T.Normalize([.5]*3,[.5]*3)])
    return config.test_transform


def open_image(value):
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert('RGB')
    with Image.open(value) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


class FaceVerifier:
    def __init__(self, checkpoint=None, calibration=None, device=None, threshold=None, tta=None):
        self.device = device or config.device
        # Small 112px preprocessing tensors otherwise oversubscribe large CPU
        # hosts, even when the network itself runs on a GPU.
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.checkpoint = Path(checkpoint or config.checkpoints / config.test_model).resolve()
        self.model_hash = sha256(self.checkpoint)
        self.net, self.metadata = load_backbone(self.checkpoint, self.device)
        self.transform = make_transform(self.metadata)
        calibration_path = Path(calibration or config.calibration)
        settings = {}
        if calibration_path.exists():
            settings = json.loads(calibration_path.read_text())
        if threshold is None:
            if not settings:
                raise FileNotFoundError('请先运行 python evaluate.py 生成阈值，或显式传入 threshold。')
            if settings.get('checkpoint_sha256') != self.model_hash:
                raise ValueError('校准文件与模型不匹配；请重新运行 evaluate.py。')
            if tta is not None and bool(tta) != settings['tta']:
                raise ValueError('翻转融合设置与校准文件不一致。')
            threshold = settings['threshold']
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold) or not -1.00001 <= self.threshold <= 1.00001:
            raise ValueError('余弦阈值必须在 [-1, 1] 范围内。')
        self.tta = bool(settings.get('tta', True) if tta is None else tta)

    @torch.inference_mode()
    def embed(self, images, batch_size=64):
        if batch_size < 1:
            raise ValueError('batch_size 必须为正数。')
        output = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(open_image(p)) for p in images[start:start+batch_size]])
            batch = batch.to(self.device)
            feature = F.normalize(self.net(batch), dim=1)
            if self.tta:
                flipped = F.normalize(self.net(torch.flip(batch, [-1])), dim=1)
                feature = F.normalize(feature + flipped, dim=1)
            if not torch.isfinite(feature).all() or (feature.norm(dim=1) < 0.5).any():
                raise ValueError('模型生成了无效特征。')
            output.append(feature.cpu().numpy())
        return np.concatenate(output) if output else np.empty((0, 512), dtype=np.float32)

    def compare(self, image1, image2, detect=False, detector=None):
        images = [open_image(image1), open_image(image2)]
        boxes = [None, None]
        if detect:
            if detector is None:
                from detection import FaceDetector
                detector = FaceDetector()
            detections = [detector.single(image) for image in images]
            images = [entry['face'] for entry in detections]
            boxes = [entry['box'] for entry in detections]
        features = self.embed(images)
        score = float(np.clip(np.dot(features[0], features[1]), -1, 1))
        return {'same_person': int(score >= self.threshold), 'similarity': score,
                'threshold': self.threshold, 'boxes': boxes}


@lru_cache(maxsize=1)
def get_verifier():
    return FaceVerifier(checkpoint=os.environ.get('FACE_CHECKPOINT'),
                        calibration=os.environ.get('FACE_CALIBRATION'),device=os.environ.get('FACE_DEVICE'))


def FaceVerification(X1, X2):
    """Return Python int 1/0 for two aligned face image paths; errors are explicit."""
    return int(get_verifier().compare(X1, X2)['same_person'])


def main():
    parser = argparse.ArgumentParser(description='验证两张人脸图片，默认输入已经对齐的人脸。')
    parser.add_argument('image1')
    parser.add_argument('image2')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--detect', action='store_true', help='对普通照片先执行人脸检测')
    args = parser.parse_args()
    verifier = FaceVerifier(args.checkpoint, args.calibration)
    print(json.dumps(verifier.compare(args.image1, args.image2, detect=args.detect), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
