"""Frozen helpers for the seed45/46 Phase2F confirmation experiment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


EXP_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = EXP_ROOT.parent
REPO_ROOT = VALIDATION_ROOT.parent
PHASE2B = VALIDATION_ROOT / "phase2b_screen"
ROUND1 = VALIDATION_ROOT / "phase2d_round1"
ROUND2A = VALIDATION_ROOT / "phase2d_round2a"
TASK_A = VALIDATION_ROOT / "phase2d_checkpoint_state_interpolation"
OLD_PHASE2F = VALIDATION_ROOT / "phase2f_unified_standalone_dfag"
if str(PHASE2B) not in sys.path:
    sys.path.insert(0, str(PHASE2B))

from screen_core import ScreenModel  # noqa: E402


SEEDS = (45, 46)
FIVE_SEEDS = (42, 43, 44, 45, 46)
FOLDS = (0, 1, 2, 3, 4)
EXPECTED_N = 18_353
FEATURE_DIM = 1536
REDUCTION = 16
HIDDEN_DIM = 96
GATE_PARAMETERS = 294_912


def config_path(seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(seed)
    return EXP_ROOT / "configs" / f"seed{seed}.json"


def load_config(seed: int) -> dict[str, Any]:
    return json.loads(config_path(seed).read_text(encoding="utf-8"))


def source_dir(seed: int, fold: int) -> Path:
    if seed in SEEDS and fold in FOLDS:
        return ROUND2A / f"baseline_seed{seed}" / f"fold_{fold}"
    raise ValueError((seed, fold))


def source_checkpoint(seed: int, fold: int) -> Path:
    return source_dir(seed, fold) / "best_stage1.pth"


def output_dir(seed: int, fold: int) -> Path:
    return EXP_ROOT / f"seed{seed}" / f"fold_{fold}"


def comparator_oof(seed: int, alpha: str = "0_5") -> Path:
    if alpha != "0_5":
        raise ValueError(alpha)
    if seed in SEEDS:
        return EXP_ROOT / "comparators" / f"seed{seed}_alpha_{alpha}.npz"
    if seed in (42, 43, 44):
        return TASK_A / "oof" / f"seed{seed}_alpha_{alpha}.npz"
    raise ValueError(seed)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
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


def bn_state(module: nn.Module) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
            result[f"{name}.running_mean"] = submodule.running_mean.detach().clone()
            result[f"{name}.running_var"] = submodule.running_var.detach().clone()
            result[f"{name}.num_batches_tracked"] = submodule.num_batches_tracked.detach().clone()
    return result


class DynamicResidualGate(nn.Module):
    """g(f_spec), where g->1 favors anchor and g->0 favors plastic."""

    def __init__(self, channels: int = FEATURE_DIM, reduction: int = REDUCTION) -> None:
        super().__init__()
        if channels != FEATURE_DIM or reduction != REDUCTION:
            raise ValueError("Phase2F gate dimensions are frozen")
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f_spec: Tensor) -> Tensor:
        return self.gate(f_spec)


class UnifiedStandaloneDFAG(nn.Module):
    """Two Stage1-initialized baseline branches plus one 1536-D dynamic gate."""

    def __init__(self, num_classes: int = 176) -> None:
        super().__init__()
        self.anchor = ScreenModel("#0", num_classes, pretrained=False)
        self.plastic = ScreenModel("#0", num_classes, pretrained=False)
        self.dfag_gate = DynamicResidualGate(FEATURE_DIM, REDUCTION)
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
            raise RuntimeError("anchor/plastic common initialization mismatch")
        return checkpoint

    def optimized_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.plastic.parameters()) + list(self.dfag_gate.parameters())
        anchor_ids = {id(parameter) for parameter in self.anchor.parameters()}
        if any(id(parameter) in anchor_ids for parameter in parameters):
            raise RuntimeError("anchor leaked into optimized parameter set")
        return parameters

    def forward_components(self, images: Tensor, gate_override: Tensor | float | None = None) -> dict[str, Tensor]:
        with torch.no_grad():
            f_anc = self.anchor.backbone(images).detach()
        f_spec = self.plastic.backbone(images)
        learned_gate = self.dfag_gate(f_spec)
        gate = learned_gate if gate_override is None else gate_override
        if not isinstance(gate, Tensor):
            gate = torch.tensor(float(gate), dtype=f_spec.dtype, device=f_spec.device)
        f_fuse = gate * f_anc + (1.0 - gate) * f_spec
        logits, classifier_feature = self.plastic.head(f_fuse)
        return {
            "logits": logits,
            "f_anc": f_anc,
            "f_spec": f_spec,
            "f_fuse": f_fuse,
            "gate": learned_gate,
            "applied_gate": gate,
            "classifier_feature": classifier_feature,
        }

    def forward(self, images: Tensor):
        result = self.forward_components(images)
        return result["logits"], result["f_fuse"]

    def checkpoint_payload_state(self) -> dict[str, dict[str, Tensor]]:
        return {
            "plastic_state_dict": self.plastic.state_dict(),
            "gate_state_dict": self.dfag_gate.state_dict(),
        }

    def load_payload_state(self, checkpoint: dict[str, Any]) -> None:
        self.plastic.load_state_dict(checkpoint["plastic_state_dict"], strict=True)
        self.dfag_gate.load_state_dict(checkpoint["gate_state_dict"], strict=True)
        self.freeze_anchor()


def parameter_counts(model: UnifiedStandaloneDFAG) -> dict[str, int]:
    gate = sum(parameter.numel() for parameter in model.dfag_gate.parameters())
    anchor = sum(parameter.numel() for parameter in model.anchor.parameters())
    plastic = sum(parameter.numel() for parameter in model.plastic.parameters())
    executed = (
        sum(parameter.numel() for parameter in model.anchor.backbone.parameters())
        + sum(parameter.numel() for parameter in model.plastic.parameters())
        + gate
    )
    return {
        "resident_parameters": anchor + plastic + gate,
        "executed_parameters": executed,
        "trainable_parameters": sum(parameter.numel() for parameter in model.optimized_parameters()),
        "frozen_parameters": anchor,
        "gate_parameters": gate,
        "anchor_branch_parameters": anchor,
        "plastic_branch_parameters": plastic,
    }
