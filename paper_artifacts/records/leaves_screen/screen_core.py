"""Shared models, data, and deterministic utilities for Phase 2B-S1."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / 'configs' / 'controlled_screen.json'
VARIANTS = ('#0', '#1', '#2')


def load_config() -> Dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding='utf-8'))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180, fill=(255, 255, 255)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, shear=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class LeafDataset(Dataset):
    def __init__(self, csv_file: Path, image_root: Path, transform: transforms.Compose) -> None:
        self.frame = pd.read_csv(csv_file)
        labels = self.frame.iloc[:, 1].astype(str)
        ordered = list(dict.fromkeys(labels.tolist()))
        self.label_to_idx = {name: idx for idx, name in enumerate(ordered)}
        self.labels = labels.map(self.label_to_idx).to_numpy(np.int64)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[Tensor, int, int, str]:
        name = str(self.frame.iloc[index, 0])
        path = self.image_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, int(self.labels[index]), int(index), name


def split_indices(labels: np.ndarray, fold: int, split_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    train_idx, val_idx = list(splitter.split(np.zeros(len(labels)), labels))[fold]
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def build_loaders(fold: int, stage: int, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, LeafDataset, np.ndarray, np.ndarray]:
    dataset_cfg = config['dataset']
    protocol = config['common_training_protocol']
    seed = int(config['seed']['training_seed'])
    split_seed = int(config['split']['split_random_state'])
    csv_file = Path(dataset_cfg['train_csv'])
    root = Path(dataset_cfg['root'])
    train_data = LeafDataset(csv_file, root, train_transform())
    val_data = LeafDataset(csv_file, root, eval_transform())
    if train_data.label_to_idx != val_data.label_to_idx:
        raise ValueError('Train/eval label mapping mismatch')
    train_idx, val_idx = split_indices(train_data.labels, fold, split_seed)
    generator = torch.Generator()
    generator.manual_seed(seed + 1000 * stage)
    common = {
        'batch_size': int(protocol['batch_size']),
        'num_workers': int(protocol['num_workers']),
        'pin_memory': True,
        'worker_init_fn': seed_worker,
        'persistent_workers': int(protocol['num_workers']) > 0,
    }
    train_loader = DataLoader(
        Subset(train_data, train_idx.tolist()), shuffle=True, generator=generator, drop_last=False, **common
    )
    val_loader = DataLoader(
        Subset(val_data, val_idx.tolist()), shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader, train_data, train_idx, val_idx


class BaselineHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor, return_trace: bool = False):
        trace: Dict[str, Tuple[int, ...]] = {'backbone_gap': tuple(x.shape)}
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        trace['hidden_1'] = tuple(x.shape)
        feature = self.drop2(self.act2(self.bn2(self.fc2(x))))
        trace['feature_pre_classifier'] = tuple(feature.shape)
        logits = self.classifier(feature)
        trace['logits'] = tuple(logits.shape)
        return (logits, feature, trace) if return_trace else (logits, feature)


class MappingBlock(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(), nn.Dropout(0.3)
        )

    def forward(self, x: Tensor, shortcut_lambda: float) -> Tensor:
        return self.mapping(x) + shortcut_lambda * x


class MatchedShortcutHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, shortcut_lambda: float) -> None:
        super().__init__()
        self.shortcut_lambda = float(shortcut_lambda)
        self.input_projection = nn.Linear(in_features, 256)
        self.mapping_blocks = nn.ModuleList([MappingBlock(256) for _ in range(5)])
        self.aggregation_dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: Tensor, return_trace: bool = False):
        trace: Dict[str, Tuple[int, ...]] = {'backbone_gap': tuple(x.shape)}
        current = self.input_projection(x)
        trace['input_projection'] = tuple(current.shape)
        stages = []
        for index, block in enumerate(self.mapping_blocks, start=1):
            current = block(current, self.shortcut_lambda)
            stages.append(current)
            trace[f'mapping_stage_{index}'] = tuple(current.shape)
        stacked = torch.stack(stages, dim=1)
        trace['stacked_stages'] = tuple(stacked.shape)
        feature = self.aggregation_dropout(stacked.mean(dim=1))
        trace['mean_aggregation_pre_classifier'] = tuple(feature.shape)
        logits = self.classifier(feature)
        trace['logits'] = tuple(logits.shape)
        return (logits, feature, trace) if return_trace else (logits, feature)


class ScreenModel(nn.Module):
    def __init__(self, variant: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Forbidden or unknown variant: {variant}')
        self.variant = variant
        self.backbone = timm.create_model('inception_resnet_v2', pretrained=pretrained, num_classes=0)
        if self.backbone.num_features != 1536:
            raise ValueError(f'Expected 1536-D backbone, got {self.backbone.num_features}')
        if variant == '#0':
            self.head = BaselineHead(self.backbone.num_features, num_classes)
        else:
            self.head = MatchedShortcutHead(
                self.backbone.num_features, num_classes, shortcut_lambda=1.0 if variant == '#1' else 0.1
            )

    def forward(self, images: Tensor, return_trace: bool = False):
        gap = self.backbone(images)
        return self.head(gap, return_trace=return_trace)


def initialize_head(model: ScreenModel) -> None:
    for module in model.head.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def model_counts(model: nn.Module) -> Dict[str, int]:
    return {
        'total_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'frozen_parameters': sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
        'backbone_parameters': sum(parameter.numel() for parameter in model.backbone.parameters()),
        'head_parameters': sum(parameter.numel() for parameter in model.head.parameters()),
    }


def fold_class_weights(labels: np.ndarray, train_idx: Sequence[int], num_classes: int) -> Tensor:
    counts = np.bincount(labels[np.asarray(train_idx)], minlength=num_classes).astype(np.float64)
    weights = len(train_idx) / (num_classes * counts)
    minority = counts < np.median(counts)
    weights[minority] *= 1.1
    weights = np.clip(weights, 0.2, 5.0)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def batch_mix(images: Tensor, labels: Tensor, epoch: int, batch_index: int, epochs: int, device: torch.device):
    if epoch > epochs * 0.7:
        scale = max(0.0, 1.0 - (epoch - epochs * 0.7) / (epochs * 0.3))
    else:
        scale = 1.0
    rng = np.random.default_rng(42_000_000 + epoch * 10_000 + batch_index)
    choice = float(rng.random())
    generator = torch.Generator(device=device)
    generator.manual_seed(84_000_000 + epoch * 10_000 + batch_index)
    permutation = torch.randperm(images.shape[0], generator=generator, device=device)
    if choice < 0.5 and scale > 0:
        lam = float(rng.beta(1.0 * scale, 1.0 * scale))
        height, width = images.shape[-2:]
        ratio = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(width * ratio), int(height * ratio)
        cx, cy = int(rng.integers(width)), int(rng.integers(height))
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (height * width))
        return mixed, labels, labels[permutation], lam, 'cutmix'
    if choice < 0.9 and scale > 0:
        alpha = 0.4 * scale
        lam = float(rng.beta(alpha, alpha))
        return lam * images + (1.0 - lam) * images[permutation], labels, labels[permutation], lam, 'mixup'
    return images, labels, labels, 1.0, 'none'


def mixed_loss(criterion: nn.Module, logits: Tensor, labels_a: Tensor, labels_b: Tensor, lam: float) -> Tensor:
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
