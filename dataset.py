"""Deterministic identity selection; folder names supply identity labels."""
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from verification import open_image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


class FaceImages(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(open_image(self.paths[index]))


def select_identities(root, train_ids=512, validation_ids=128, per_id=16, seed=42):
    root = Path(root)
    if min(train_ids, validation_ids) < 2 or per_id < 2:
        raise ValueError('训练/验证至少各 2 个身份，每个身份至少 2 张图片。')
    rng = np.random.default_rng(seed)
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    rng.shuffle(folders)
    groups = []
    for folder in folders:
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) >= per_id:
            indexes = rng.choice(len(images), per_id, replace=False)
            groups.append((folder.name, [images[i] for i in indexes]))
        if len(groups) == train_ids + validation_ids:
            break
    if len(groups) < train_ids + validation_ids:
        raise ValueError('满足每个身份图片数要求的身份不足，请减少身份数或每人图片数。')
    return groups[:train_ids], groups[train_ids:]


def load_data(conf, training=True):
    data = ImageFolder(str(conf.train_root if training else conf.test_root),
                       transform=conf.train_transform if training else conf.test_transform)
    loader = DataLoader(data, batch_size=conf.train_batch_size if training else conf.test_batch_size,
                        shuffle=training, pin_memory=conf.pin_memory, num_workers=conf.num_workers,
                        drop_last=training)
    return loader, len(data.classes)
