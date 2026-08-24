"""Frozen primitives for the new unified cross-dataset validation protocol.

This module defines a NEW protocol.  It is not a reconstruction of the old
CUB/Cars/Flowers training code.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
RUN_IDS = ("ours_ft", "progressive_lambda_0_7")


def load_config() -> Dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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


def train_transform(dataset_key: str) -> transforms.Compose:
    if dataset_key not in ("cub", "cars", "flowers"):
        raise ValueError(dataset_key)
    # The new unified implementation explicitly disables vertical flipping for
    # every external dataset.  This is not a claim about historical Flowers.
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
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


def serialized_transforms() -> Dict[str, Any]:
    return {
        key: {
            "train_repr": repr(train_transform(key)),
            "validation_repr": repr(eval_transform()),
            "official_test_repr": repr(eval_transform()),
            "vertical_flip_present": "RandomVerticalFlip" in repr(train_transform(key)),
            "primary_tta": False,
        }
        for key in ("cub", "cars", "flowers")
    }


def _read_space_table(path: Path) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        result[int(key)] = value
    return result


def cub_records(config: Dict[str, Any]) -> Tuple[List[dict], List[dict]]:
    root = Path(config["datasets"]["cub"]["root"])
    names = _read_space_table(root / "images.txt")
    labels = {key: int(value) - 1 for key, value in _read_space_table(root / "image_class_labels.txt").items()}
    split = {key: int(value) for key, value in _read_space_table(root / "train_test_split.txt").items()}
    if set(names) != set(labels) or set(names) != set(split):
        raise ValueError("CUB metadata ID sets differ")
    development: List[dict] = []
    official_test: List[dict] = []
    for image_id in sorted(names):
        record = {
            "path": root / "images" / names[image_id],
            "label": labels[image_id],
            "sample_id": f"cub:{image_id}",
            "image_name": names[image_id],
            "source_id": image_id,
        }
        (development if split[image_id] == 1 else official_test).append(record)
    return development, official_test


def cars_records(config: Dict[str, Any]) -> Tuple[List[dict], List[dict]]:
    cfg = config["datasets"]["cars"]
    result = []
    for split_name, csv_key, root_key in (
        ("train", "train_csv", "train_image_root"),
        ("test", "test_csv", "test_image_root"),
    ):
        frame = pd.read_csv(cfg[csv_key], header=None)
        if frame.shape[1] != 6:
            raise ValueError(f"Cars {split_name} annotation must have 6 columns")
        image_root = Path(cfg[root_key])
        rows: List[dict] = []
        for row in frame.itertuples(index=False, name=None):
            file_name, _, _, _, _, class_id = row
            label = int(class_id) - 1
            # The reorganized directory contains class subdirectories; locate
            # by the unique numeric filename without using bounding boxes.
            matches = list(image_root.glob(f"*/{file_name}"))
            if len(matches) != 1:
                raise ValueError(f"Cars image lookup expected one match for {file_name}, got {len(matches)}")
            rows.append({
                "path": matches[0], "label": label,
                "sample_id": f"cars_{split_name}:{file_name}",
                "image_name": str(matches[0].relative_to(image_root)).replace("\\", "/"),
            })
        result.append(rows)
    return result[0], result[1]


def flowers_records(config: Dict[str, Any]) -> Tuple[List[dict], List[dict], dict]:
    cfg = config["datasets"]["flowers"]
    labels = scipy.io.loadmat(cfg["labels_mat"])["labels"].reshape(-1).astype(np.int64) - 1
    split = scipy.io.loadmat(cfg["split_mat"])
    train_ids = split["trnid"].reshape(-1).astype(np.int64)
    val_ids = split["valid"].reshape(-1).astype(np.int64)
    test_ids = split["tstid"].reshape(-1).astype(np.int64)
    development_ids = np.concatenate([train_ids, val_ids])
    image_root = Path(cfg["image_root"])

    def rows(ids: Iterable[int], split_name: str) -> List[dict]:
        return [{
            "path": image_root / f"image_{int(image_id):05d}.jpg",
            "label": int(labels[int(image_id) - 1]),
            "sample_id": f"flowers_{split_name}:{int(image_id)}",
            "image_name": f"image_{int(image_id):05d}.jpg",
            "source_id": int(image_id),
        } for image_id in ids]

    metadata = {
        "official_train_ids": train_ids,
        "official_validation_ids": val_ids,
        "official_test_ids": test_ids,
    }
    return rows(development_ids, "development"), rows(test_ids, "test"), metadata


class RecordDataset(Dataset):
    def __init__(self, records: Sequence[dict], transform: transforms.Compose) -> None:
        self.records = list(records)
        self.labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), int(index), str(row["sample_id"]), str(row["image_name"])


def split_indices(labels: np.ndarray, fold: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, val_idx = list(splitter.split(np.zeros(len(labels)), labels))[fold]
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def build_cub_loaders(fold: int, stage: int, config: Dict[str, Any]):
    development, official_test = cub_records(config)
    training_data = RecordDataset(development, train_transform("cub"))
    validation_data = RecordDataset(development, eval_transform())
    test_data = RecordDataset(official_test, eval_transform())
    random_state = int(config["split"]["random_state"])
    train_idx, val_idx = split_indices(training_data.labels, fold, random_state)
    protocol = config["training"]
    seed = int(config["smoke_scope"]["training_seed"])
    generator = torch.Generator()
    generator.manual_seed(seed + 1000 * stage)
    common = {
        "batch_size": int(protocol["batch_size"]),
        "num_workers": int(protocol["num_workers"]),
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": int(protocol["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        Subset(training_data, train_idx.tolist()), shuffle=True, generator=generator,
        drop_last=False, **common,
    )
    validation_loader = DataLoader(
        Subset(validation_data, val_idx.tolist()), shuffle=False, drop_last=False, **common,
    )
    test_loader = DataLoader(test_data, shuffle=False, drop_last=False, **common)
    return train_loader, validation_loader, test_loader, training_data, test_data, train_idx, val_idx


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

    def forward(self, x: Tensor, return_stages: bool = False):
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        feature = self.drop2(self.act2(self.bn2(self.fc2(x))))
        logits = self.classifier(feature)
        return logits, feature, None


class MappingBlock(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(), nn.Dropout(0.3)
        )

    def forward(self, x: Tensor, shortcut_lambda: float) -> Tensor:
        return self.mapping(x) + shortcut_lambda * x


class ProgressiveHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, shortcut_lambda: float = 0.7) -> None:
        super().__init__()
        self.shortcut_lambda = float(shortcut_lambda)
        self.input_projection = nn.Linear(in_features, 256)
        self.mapping_blocks = nn.ModuleList([MappingBlock(256) for _ in range(5)])
        self.aggregation_dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: Tensor, return_stages: bool = False):
        current = self.input_projection(x)
        stages = []
        for block in self.mapping_blocks:
            current = block(current, self.shortcut_lambda)
            stages.append(current)
        stacked = torch.stack(stages, dim=1)
        feature = self.aggregation_dropout(stacked.mean(dim=1))
        logits = self.classifier(feature)
        return logits, feature, stacked if return_stages else None


class ExternalModel(nn.Module):
    def __init__(self, run_id: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        if run_id not in RUN_IDS:
            raise ValueError(run_id)
        self.run_id = run_id
        self.backbone = timm.create_model("inception_resnet_v2", pretrained=pretrained, num_classes=0)
        if self.backbone.num_features != 1536:
            raise ValueError(f"expected a 1536-D backbone, got {self.backbone.num_features}")
        self.head = (
            BaselineHead(self.backbone.num_features, num_classes)
            if run_id == "ours_ft"
            else ProgressiveHead(self.backbone.num_features, num_classes, shortcut_lambda=0.7)
        )

    def forward(self, images: Tensor, return_stages: bool = False):
        gap = self.backbone(images)
        return self.head(gap, return_stages=return_stages)


def initialize_head(model: ExternalModel) -> None:
    for module in model.head.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def model_counts(model: ExternalModel) -> Dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "backbone_parameters": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "head_parameters": sum(parameter.numel() for parameter in model.head.parameters()),
    }


def batch_mix(images: Tensor, labels: Tensor, epoch: int, batch_index: int, epochs: int, device: torch.device):
    # Exact existing controlled Leaves semantics, now frozen for this protocol.
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
        return mixed, labels, labels[permutation], lam, "cutmix", scale
    if choice < 0.9 and scale > 0:
        alpha = 0.4 * scale
        lam = float(rng.beta(alpha, alpha))
        return lam * images + (1.0 - lam) * images[permutation], labels, labels[permutation], lam, "mixup", scale
    return images, labels, labels, 1.0, "none", scale


def mixed_loss(criterion: nn.Module, logits: Tensor, labels_a: Tensor, labels_b: Tensor, lam: float) -> Tensor:
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def evaluate(model: ExternalModel, loader: DataLoader, device: torch.device, capture_stages: bool = False) -> dict:
    model.eval()
    logits_parts: List[np.ndarray] = []
    feature_parts: List[np.ndarray] = []
    stage_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    sample_ids: List[str] = []
    image_names: List[str] = []
    with torch.inference_mode():
        for images, labels, indices, batch_ids, batch_names in loader:
            images = images.to(device, non_blocking=True)
            logits, features, stages = model(images, return_stages=capture_stages)
            logits_parts.append(logits.float().cpu().numpy())
            feature_parts.append(features.float().cpu().numpy())
            label_parts.append(labels.numpy())
            index_parts.append(indices.numpy())
            sample_ids.extend(str(value) for value in batch_ids)
            image_names.extend(str(value) for value in batch_names)
            if capture_stages:
                if stages is None:
                    raise ValueError("stage capture requested from a non-progressive model")
                stage_parts.append(stages.float().cpu().numpy())
    logits_np = np.concatenate(logits_parts).astype(np.float32)
    labels_np = np.concatenate(label_parts).astype(np.int64)
    predictions = logits_np.argmax(axis=1).astype(np.int64)
    result = {
        "accuracy": float(accuracy_score(labels_np, predictions)),
        "macro_f1": float(f1_score(labels_np, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, predictions)),
        "logits": logits_np,
        "features": np.concatenate(feature_parts).astype(np.float32),
        "labels": labels_np,
        "dataset_indices": np.concatenate(index_parts).astype(np.int64),
        "sample_ids": np.asarray(sample_ids),
        "image_names": np.asarray(image_names),
        "predictions": predictions,
    }
    if capture_stages:
        result["post_shortcut_stages"] = np.concatenate(stage_parts).astype(np.float32)
    return result


def save_evaluation(output: Path, prefix: str, result: dict) -> dict:
    array_keys = (
        "logits", "features", "labels", "dataset_indices", "sample_ids",
        "image_names", "predictions", "post_shortcut_stages",
    )
    for key in array_keys:
        if key in result:
            np.save(output / f"{prefix}_{key}.npy", result[key])
    return {key: float(result[key]) for key in ("accuracy", "macro_f1", "balanced_accuracy")}


def centered_linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    numerator = float(np.sum(cross * cross))
    denominator = float(np.sqrt(np.sum((x.T @ x) ** 2) * np.sum((y.T @ y) ** 2)))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("degenerate centered-linear CKA denominator")
    return numerator / denominator


def representation_diagnostics(stages: np.ndarray) -> dict:
    if stages.ndim != 3 or stages.shape[1] != 5:
        raise ValueError(f"expected [N,5,D] stages, got {stages.shape}")
    values = np.asarray(stages, dtype=np.float64)
    norms = np.linalg.norm(values, axis=2, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero stage vector in cosine diagnostic")
    normalized = values / norms
    cosine_matrices = normalized @ np.swapaxes(normalized, 1, 2)
    mask = ~np.eye(5, dtype=bool)
    sample_cosine = cosine_matrices[:, mask].mean(axis=1)
    pair_rows = []
    for left in range(5):
        for right in range(left + 1, 5):
            pair_rows.append({
                "left_stage": left + 1,
                "right_stage": right + 1,
                "mean_sample_cosine": float(cosine_matrices[:, left, right].mean()),
                "centered_linear_cka": centered_linear_cka(values[:, left, :], values[:, right, :]),
            })
    return {
        "n": int(values.shape[0]),
        "stage_count": 5,
        "feature_dim": int(values.shape[2]),
        "mean_off_diagonal_cosine": float(sample_cosine.mean()),
        "mean_centered_linear_cka": float(np.mean([row["centered_linear_cka"] for row in pair_rows])),
        "pairwise": pair_rows,
    }


def batch_norm_snapshot(model: nn.Module) -> Dict[str, np.ndarray]:
    snapshot = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            snapshot[f"{name}.running_mean"] = module.running_mean.detach().cpu().numpy().copy()
            snapshot[f"{name}.running_var"] = module.running_var.detach().cpu().numpy().copy()
            snapshot[f"{name}.num_batches_tracked"] = module.num_batches_tracked.detach().cpu().numpy().copy()
    return snapshot


def changed_snapshot_keys(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> List[str]:
    if before.keys() != after.keys():
        raise ValueError("BN snapshot key set changed")
    return [key for key in before if not np.array_equal(before[key], after[key])]
