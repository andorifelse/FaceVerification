"""Portable defaults; all paths are relative to the project."""
from pathlib import Path
import torch
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent


class Config:
    backbone = 'face_vit'
    metric = 'cosface'
    embedding_size = 512
    drop_ratio = 0.5
    # Compatibility transforms for the convolutional baselines. Transformer
    # preprocessing is selected from checkpoint metadata in verification.py.
    input_shape = [1, 128, 128]
    transformer_input_shape = [3, 112, 112]
    train_transform = T.Compose([
        T.Grayscale(), T.Resize((128, 128)), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize([0.5], [0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(), T.Resize((128, 128)), T.ToTensor(), T.Normalize([0.5], [0.5]),
    ])
    train_root = ROOT / 'datasets/CASIA-WebFace-Align-96'
    test_root = ROOT / 'datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped'
    test_list = ROOT / 'datasets/scikit_learn_data/lfw_home/pairsDevTest.txt'
    calibration_list = ROOT / 'datasets/scikit_learn_data/lfw_home/pairsDevTrain.txt'
    checkpoints = ROOT / 'checkpoints'
    test_model = 'transformer.pth'
    calibration = ROOT / 'artifacts/calibration.json'
    train_batch_size = 64
    test_batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 0
    pin_memory = torch.cuda.is_available()


config = Config()
