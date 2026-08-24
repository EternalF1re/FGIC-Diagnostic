"""Frozen common primitives for the controlled ResNet-50 reimplementation.

This directory is intentionally independent from every formal-result directory.
It may create technical preflight/smoke artifacts, but it never starts a formal
fold job unless a separate, explicit authorization layer is added.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROOT.parent
REPO_ROOT = VALIDATION_ROOT.parent
EXTERNAL_CONFIG = VALIDATION_ROOT / "final_external_protocol" / "final_external_protocol_config.json"
PROTOCOL_CORE_DIR = VALIDATION_ROOT / "corrected_external_protocol_smoke" / "scripts"
if str(PROTOCOL_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_CORE_DIR))

import protocol_core as external_protocol  # noqa: E402


METHODS = ("l2_sp", "mc_loss", "cal", "ours_ft", "progressive", "dfag")
DATASETS = ("cub", "cars")
FOLDS = tuple(range(5))
SEED = 42
INPUT_SIZE = 299
BACKBONE_ID = "timm/resnet50.a1_in1k"
TIMM_MODEL_NAME = "resnet50"
EXPECTED_FEATURE_CHANNELS = 2048
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SOURCE_COMMITS = {
    "l2_sp": "1426356195d79613b12c94ad8d6d0db576b09261",
    "mc_loss": "befb3692cd0d5382eb32fa4e093226247f609fd9",
    "cal": "0ba9d5084f2532eeb21c9ef051c23f8b339595ff",
    "ours_ft": "workspace-audited-final-cross-backbone-semantics",
    "progressive": "workspace-audited-final-cross-backbone-semantics",
    "dfag": "workspace-audited-final-cross-backbone-semantics",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def state_digest(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_external_config() -> dict[str, Any]:
    return json.loads(EXTERNAL_CONFIG.read_text(encoding="utf-8"))


def development_records(dataset: str) -> list[dict[str, Any]]:
    config = load_external_config()
    if dataset == "cub":
        records, _ = external_protocol.cub_records(config)
    elif dataset == "cars":
        records, _ = external_protocol.cars_records(config)
    else:
        raise ValueError(dataset)
    return records


def split_indices(labels: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    return external_protocol.split_indices(labels, fold, random_state=SEED)


def class_map(dataset: str, records: Sequence[dict[str, Any]]) -> dict[str, str]:
    if dataset == "cub":
        root = Path(load_external_config()["datasets"]["cub"]["root"])
        result: dict[str, str] = {}
        for line in (root / "classes.txt").read_text(encoding="utf-8").splitlines():
            index, name = line.split(maxsplit=1)
            result[str(int(index) - 1)] = name
        return result
    if dataset == "cars":
        by_label: dict[int, set[str]] = {}
        for row in records:
            by_label.setdefault(int(row["label"]), set()).add(Path(row["path"]).parent.name)
        if any(len(names) != 1 for names in by_label.values()):
            raise ValueError("Cars label-to-directory mapping is not one-to-one")
        return {str(label): next(iter(by_label[label])) for label in sorted(by_label)}
    raise ValueError(dataset)


def fold_manifest(dataset: str) -> dict[str, Any]:
    records = development_records(dataset)
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    heldout_fold = np.full(len(records), -1, dtype=np.int64)
    folds: list[dict[str, Any]] = []
    for fold in FOLDS:
        train_idx, heldout_idx = split_indices(labels, fold)
        if np.intersect1d(train_idx, heldout_idx).size:
            raise AssertionError("train/held-out overlap")
        heldout_fold[heldout_idx] = fold
        folds.append({
            "fold": fold,
            "train_count": int(train_idx.size),
            "heldout_count": int(heldout_idx.size),
            "train_index_sha256": canonical_sha(train_idx.tolist()),
            "heldout_index_sha256": canonical_sha(heldout_idx.tolist()),
        })
    if np.any(heldout_fold < 0):
        raise AssertionError("development sample missing held-out fold")
    assignments = [
        {"sample_id": str(row["sample_id"]), "label": int(row["label"]), "heldout_fold": int(heldout_fold[i])}
        for i, row in enumerate(records)
    ]
    cmap = class_map(dataset, records)
    payload = {
        "dataset": dataset,
        "splitter": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "development_count": len(records),
        "folds": folds,
        "assignments": assignments,
        "class_map": cmap,
    }
    payload["fold_manifest_sha256"] = canonical_sha(assignments)
    payload["class_map_sha256"] = canonical_sha(cmap)
    return payload


class L2SPColorPerturbation:
    """Cross-framework transcription of the official TensorFlow random ops."""

    def __call__(self, image: Tensor) -> Tensor:
        delta = float(torch.empty(1).uniform_(-63.0 / 255.0, 63.0 / 255.0))
        saturation = float(torch.empty(1).uniform_(0.5, 1.5))
        contrast = float(torch.empty(1).uniform_(0.2, 1.8))
        image = image + delta
        image = TF.adjust_saturation(image, saturation)
        image = TF.adjust_contrast(image, contrast)
        return image


def train_transform(method: str) -> transforms.Compose:
    if method == "l2_sp":
        return transforms.Compose([
            transforms.Resize((342, 342), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            L2SPColorPerturbation(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if method == "mc_loss":
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(INPUT_SIZE, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if method == "cal":
        return transforms.Compose([
            transforms.Resize((341, 341), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if method in {"ours_ft", "progressive", "dfag"}:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, shear=5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    raise ValueError(method)


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


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
        return tensor, int(row["label"]), index, str(row["sample_id"])


def create_backbone(pretrained: bool = True) -> nn.Module:
    model = timm.create_model(TIMM_MODEL_NAME, pretrained=pretrained, num_classes=0)
    if int(model.num_features) != EXPECTED_FEATURE_CHANNELS:
        raise RuntimeError(f"unexpected ResNet-50 feature dimension: {model.num_features}")
    return model


def locate_pretrained_artifact() -> Path:
    candidates = [Path(torch.hub.get_dir()) / "checkpoints" / "resnet50_a1_0-14fe96d1.pth"]
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--timm--resnet50.a1_in1k"
    if cache.exists():
        candidates.extend(cache.glob("snapshots/*/model.safetensors"))
        candidates.extend(cache.glob("snapshots/*/pytorch_model.bin"))
    hits = [path.resolve() for path in candidates if path.is_file()]
    if not hits:
        raise FileNotFoundError("cached timm/resnet50.a1_in1k weight artifact not found")
    return hits[0]


class ConventionalHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.silu(self.bn1(self.fc1(x)))
        x = F.dropout(x, 0.2, self.training)
        feature = F.silu(self.bn2(self.fc2(x)))
        feature = F.dropout(feature, 0.2, self.training)
        return self.classifier(feature), feature


class MappingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(256, 256)
        self.bn = nn.BatchNorm1d(256)

    def forward(self, x: Tensor) -> Tensor:
        mapped = F.dropout(F.silu(self.bn(self.linear(x))), 0.3, self.training)
        return mapped + 0.7 * x


class ProgressiveHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_features, 256)
        self.mapping_blocks = nn.ModuleList([MappingBlock() for _ in range(5)])
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        current = self.input_projection(x)
        stages = []
        for block in self.mapping_blocks:
            current = block(current)
            stages.append(current)
        feature = F.dropout(torch.stack(stages, dim=1).mean(dim=1), 0.2, self.training)
        return self.classifier(feature), feature


class SingleBranchModel(nn.Module):
    def __init__(self, method: str, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        if method not in {"ours_ft", "progressive"}:
            raise ValueError(method)
        self.method = method
        self.backbone = create_backbone(pretrained)
        self.head = ConventionalHead(2048, num_classes) if method == "ours_ft" else ProgressiveHead(2048, num_classes)
        initialize_linear(self.head)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.head(self.backbone(images))


class FeatureWiseGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 128, bias=False), nn.ReLU(inplace=True),
            nn.Linear(128, 2048, bias=False), nn.Sigmoid(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.layers(feature)


class DFAGModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.anchor = SingleBranchModel("ours_ft", num_classes, pretrained=False)
        self.plastic = SingleBranchModel("ours_ft", num_classes, pretrained=False)
        self.gate = FeatureWiseGate()
        self.freeze_anchor()

    def freeze_anchor(self) -> None:
        self.anchor.eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def load_same_fold_stage1(self, payload: dict[str, Any], dataset: str, fold: int) -> None:
        if payload.get("method") != "ours_ft" or payload.get("dataset") != dataset or int(payload.get("fold", -1)) != fold:
            raise ValueError("DFAG Stage-1 provenance mismatch")
        state = payload["model_state_dict"]
        self.anchor.load_state_dict(state, strict=True)
        self.plastic.load_state_dict(state, strict=True)
        self.freeze_anchor()
        if state_digest(self.anchor.state_dict()) != state_digest(self.plastic.state_dict()):
            raise RuntimeError("anchor/plastic Stage-1 initialization mismatch")

    def forward(self, images: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        with torch.no_grad():
            f_anc = self.anchor.backbone(images).detach()
        f_spec = self.plastic.backbone(images)
        gate = self.gate(f_spec)
        fused = gate * f_anc + (1.0 - gate) * f_spec
        logits, head_feature = self.plastic.head(fused)
        return logits, {"f_anc": f_anc, "f_spec": f_spec, "gate": gate, "fused": fused, "head_feature": head_feature}


def initialize_linear(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def pooled_oof_metrics(rows: Iterable[dict[str, Any]], expected_ids: Sequence[str]) -> dict[str, float]:
    rows = list(rows)
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("OOF rows must contain each development sample exactly once")
    ordered = sorted(rows, key=lambda row: str(row["sample_id"]))
    labels = np.asarray([int(row["label"]) for row in ordered])
    predictions = np.asarray([int(row["prediction"]) for row in ordered])
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }
