"""Frozen common definitions for the final Classify-Leaves cross-backbone run."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import timm
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROOT.parent
PHASE2B = VALIDATION_ROOT / "phase2b_screen"
if str(PHASE2B) not in sys.path:
    sys.path.insert(0, str(PHASE2B))

from screen_core import (  # noqa: E402
    LeafDataset,
    batch_mix,
    eval_transform,
    fold_class_weights,
    seed_everything,
    seed_worker,
    split_indices,
    train_transform,
)


CONFIG_PATH = ROOT / "final_cross_backbone_protocol_config.json"
PREFLIGHT_JSON = ROOT / "preflight_audit.json"
BACKBONES = ("resnet50", "convnext_tiny")
METHODS = ("ours_ft", "progressive", "dfag")
FOLDS = tuple(range(5))
NUM_CLASSES = 176
HIDDEN = 256
BLOCKS = 5
SHORTCUT_LAMBDA = 0.7
REDUCTION = 16
EXPECTED_DIMENSIONS = {"resnet50": 2048, "convnext_tiny": 768}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class ConventionalHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = NUM_CLASSES) -> None:
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        feature = self.drop2(self.act2(self.bn2(self.fc2(x))))
        return self.classifier(feature), feature


class MappingBlock(nn.Module):
    def __init__(self, dimension: int = HIDDEN) -> None:
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.BatchNorm1d(dimension),
            nn.SiLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mapping(x) + SHORTCUT_LAMBDA * x


class ProgressiveHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_features, HIDDEN)
        self.mapping_blocks = nn.ModuleList([MappingBlock(HIDDEN) for _ in range(BLOCKS)])
        self.aggregation_dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(HIDDEN, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        current = self.input_projection(x)
        stages = []
        for block in self.mapping_blocks:
            current = block(current)
            stages.append(current)
        feature = self.aggregation_dropout(torch.stack(stages, dim=1).mean(dim=1))
        return self.classifier(feature), feature


class CrossBackboneModel(nn.Module):
    def __init__(self, backbone_name: str, method: str, pretrained: bool) -> None:
        super().__init__()
        if backbone_name not in BACKBONES:
            raise ValueError(backbone_name)
        if method not in {"ours_ft", "progressive"}:
            raise ValueError(method)
        self.backbone_name = backbone_name
        self.method = method
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.dimension = int(self.backbone.num_features)
        if self.dimension != EXPECTED_DIMENSIONS[backbone_name]:
            raise ValueError(f"{backbone_name} dimension {self.dimension} != expected {EXPECTED_DIMENSIONS[backbone_name]}")
        self.head = ConventionalHead(self.dimension) if method == "ours_ft" else ProgressiveHead(self.dimension)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.head(self.backbone(images))


class DynamicGate(nn.Module):
    def __init__(self, dimension: int, reduction: int = REDUCTION) -> None:
        super().__init__()
        if dimension % reduction:
            raise ValueError("Gate dimension must be divisible by reduction")
        hidden = dimension // reduction
        self.gate = nn.Sequential(
            nn.Linear(dimension, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dimension, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f_spec: Tensor) -> Tensor:
        return self.gate(f_spec)


class StandaloneDFAG(nn.Module):
    def __init__(self, backbone_name: str) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.anchor = CrossBackboneModel(backbone_name, "ours_ft", pretrained=False)
        self.plastic = CrossBackboneModel(backbone_name, "ours_ft", pretrained=False)
        self.dimension = self.plastic.dimension
        self.dfag_gate = DynamicGate(self.dimension)
        self.freeze_anchor()

    def freeze_anchor(self) -> None:
        self.anchor.eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def load_common_stage1(self, checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        self.anchor.load_state_dict(state, strict=True)
        self.plastic.load_state_dict(state, strict=True)
        self.freeze_anchor()
        if state_digest(self.anchor.state_dict()) != state_digest(self.plastic.state_dict()):
            raise RuntimeError("DFAG common Stage-1 initialization mismatch")
        return checkpoint

    def optimized_parameters(self) -> list[nn.Parameter]:
        return list(self.plastic.parameters()) + list(self.dfag_gate.parameters())

    def forward_components(self, images: Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            f_anc = self.anchor.backbone(images).detach()
        f_spec = self.plastic.backbone(images)
        gate = self.dfag_gate(f_spec)
        f_fuse = gate * f_anc + (1.0 - gate) * f_spec
        logits, classifier_feature = self.plastic.head(f_fuse)
        return {
            "logits": logits,
            "f_anc": f_anc,
            "f_spec": f_spec,
            "f_fuse": f_fuse,
            "gate": gate,
            "classifier_feature": classifier_feature,
        }

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        result = self.forward_components(images)
        return result["logits"], result["f_fuse"]

    def checkpoint_payload_state(self) -> dict[str, dict[str, Tensor]]:
        return {
            "plastic_state_dict": self.plastic.state_dict(),
            "gate_state_dict": self.dfag_gate.state_dict(),
        }

    def load_payload_state(self, payload: dict[str, Any]) -> None:
        self.plastic.load_state_dict(payload["plastic_state_dict"], strict=True)
        self.dfag_gate.load_state_dict(payload["gate_state_dict"], strict=True)
        self.freeze_anchor()


def initialize_head(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def parameter_counts(model: nn.Module, method: str) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    if method in {"ours_ft", "progressive"}:
        return {
            "resident_parameters": total,
            "executed_parameters": total,
            "backbone_parameters": sum(parameter.numel() for parameter in model.backbone.parameters()),
            "head_parameters": sum(parameter.numel() for parameter in model.head.parameters()),
            "gate_parameters": 0,
            "branch_count": 1,
        }
    gate = sum(parameter.numel() for parameter in model.dfag_gate.parameters())
    anchor = sum(parameter.numel() for parameter in model.anchor.parameters())
    plastic = sum(parameter.numel() for parameter in model.plastic.parameters())
    executed = sum(parameter.numel() for parameter in model.anchor.backbone.parameters()) + plastic + gate
    return {
        "resident_parameters": total,
        "executed_parameters": executed,
        "anchor_parameters": anchor,
        "plastic_parameters": plastic,
        "gate_parameters": gate,
        "head_parameters": sum(parameter.numel() for parameter in model.plastic.head.parameters()),
        "branch_count": 2,
    }


def build_loaders(fold: int, stage: int, config: dict[str, Any], backbone: str) -> tuple[DataLoader, DataLoader, LeafDataset, np.ndarray, np.ndarray]:
    data_cfg = config["dataset"]
    protocol = config["common_training_protocol"]
    batch_size = int(config["backbones"][backbone]["batch_size"])
    seed = int(config["seed"]["training_seed"])
    train_data = LeafDataset(Path(data_cfg["train_csv"]), Path(data_cfg["root"]), train_transform())
    val_data = LeafDataset(Path(data_cfg["train_csv"]), Path(data_cfg["root"]), eval_transform())
    if train_data.label_to_idx != val_data.label_to_idx:
        raise ValueError("label mapping mismatch")
    train_idx, val_idx = split_indices(train_data.labels, fold, int(config["split"]["split_random_state"]))
    generator = torch.Generator().manual_seed(seed + 1000 * stage)
    common = {
        "batch_size": batch_size,
        "num_workers": int(protocol["num_workers"]),
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": int(protocol["num_workers"]) > 0,
        "drop_last": False,
    }
    train_loader = DataLoader(Subset(train_data, train_idx.tolist()), shuffle=True, generator=generator, **common)
    val_loader = DataLoader(Subset(val_data, val_idx.tolist()), shuffle=False, **common)
    return train_loader, val_loader, train_data, train_idx, val_idx


def output_dir(run_type: str, backbone: str, method: str, fold: int) -> Path:
    return ROOT / run_type / backbone / method / f"fold_{fold}"


def selected_stage1_path(run_type: str, backbone: str, fold: int) -> Path:
    return output_dir(run_type, backbone, "ours_ft", fold) / "best_stage1.pth"


def interpolate_complete_state(stage1: dict[str, Tensor], stage2: dict[str, Tensor], alpha: float = 0.5) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if set(stage1) != set(stage2):
        raise ValueError("Stage1/Stage2 state keys differ")
    output: dict[str, Tensor] = {}
    floating = nonfloating = 0
    for key in stage1:
        x, y = stage1[key], stage2[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"State mismatch: {key}")
        if torch.is_floating_point(x):
            output[key] = x.mul(alpha).add(y, alpha=1.0 - alpha)
            floating += 1
        else:
            output[key] = y.clone()
            nonfloating += 1
    return output, {
        "formula": "alpha*Stage1+(1-alpha)*Stage2",
        "alpha": alpha,
        "floating_state_tensors_interpolated": floating,
        "nonfloating_state_tensors_copied_from_stage2": nonfloating,
    }


def ensure_preflight_pass() -> dict[str, Any]:
    payload = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError("Cross-backbone preflight missing or failed")
    return payload
