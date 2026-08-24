from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 299


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class L2SPColorPerturbation:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = image + float(torch.empty(1).uniform_(-63.0 / 255.0, 63.0 / 255.0))
        image = TF.adjust_saturation(image, float(torch.empty(1).uniform_(0.5, 1.5)))
        return TF.adjust_contrast(image, float(torch.empty(1).uniform_(0.2, 1.8)))


def train_transform(method: str, dataset: str = ""):
    if method == "l2_sp":
        return transforms.Compose([
            transforms.Resize((342, 342)), transforms.ToTensor(), L2SPColorPerturbation(),
            transforms.RandomHorizontalFlip(0.5), transforms.RandomCrop(INPUT_SIZE), transforms.Normalize(MEAN, STD),
        ])
    if method == "mc_loss":
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.RandomCrop(INPUT_SIZE, padding=4),
            transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
    if method == "cal":
        return transforms.Compose([
            transforms.Resize((341, 341)), transforms.RandomCrop(INPUT_SIZE), transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
    spatial = [transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.RandomHorizontalFlip(0.5)]
    if dataset == "classifyleaves":
        spatial.extend([transforms.RandomVerticalFlip(0.5), transforms.RandomRotation(180, fill=(255, 255, 255))])
    elif dataset in {"flowers", "flowers102"}:
        spatial.append(transforms.RandomRotation(180, fill=(255, 255, 255)))
    return transforms.Compose(spatial + [
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, shear=5), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def load_dataset_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    return json.loads(source.read_text(encoding="utf-8")), source.parent


def load_records(dataset_config: str | Path, dataset: str) -> list[dict[str, Any]]:
    config, base = load_dataset_config(dataset_config)
    entry = config["datasets"][dataset]
    manifest = Path(entry["development_manifest"])
    if not manifest.is_absolute():
        manifest = (base / manifest).resolve()
    frame = pd.read_csv(manifest)
    required = {"path", "label", "sample_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest {manifest} must contain {sorted(required)}")
    rows = []
    for row in frame.to_dict("records"):
        image_path = Path(str(row["path"]))
        if not image_path.is_absolute():
            image_path = (manifest.parent / image_path).resolve()
        rows.append({"path": image_path, "label": int(row["label"]), "sample_id": str(row["sample_id"])})
    if not rows:
        raise ValueError(f"Manifest {manifest} is empty")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("sample_id values must be unique")
    labels = {row["label"] for row in rows}
    if min(labels) != 0 or max(labels) >= int(entry["num_classes"]):
        raise ValueError("Labels must be zero-based and smaller than num_classes")
    return rows


def fold_indices(records: Sequence[dict[str, Any]], fold: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    pairs = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    if fold not in range(len(pairs)):
        raise ValueError("fold must be in 0..4")
    return pairs[fold]


class RecordDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]], transform: Any) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row["label"], index, row["sample_id"]


def build_loaders(
    dataset_config: str | Path,
    dataset: str,
    method: str,
    fold: int,
    batch_size: int,
    workers: int,
    seed: int = 42,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    records = load_records(dataset_config, dataset)
    train_idx, heldout_idx = fold_indices(records, fold, seed)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    generator = torch.Generator().manual_seed(seed)
    train_subset = Subset(RecordDataset(records, train_transform(method, dataset)), train_idx.tolist())
    sampler = DistributedSampler(
        train_subset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False,
    ) if distributed else None
    train = DataLoader(
        train_subset, sampler=sampler, shuffle=sampler is None,
        generator=generator if sampler is None else None, drop_last=bool(distributed and method == "cal"), **common,
    )
    heldout = DataLoader(
        Subset(RecordDataset(records, eval_transform()), heldout_idx.tolist()),
        shuffle=False, drop_last=False, **common,
    )
    return train, heldout, records, train_idx, heldout_idx
