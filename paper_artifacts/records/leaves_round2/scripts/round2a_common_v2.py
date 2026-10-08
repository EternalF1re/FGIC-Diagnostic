"""Shared paths, run specifications and models for Phase2D Round2A."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import timm
import torch
from torch import Tensor, nn


ROUND_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROUND_ROOT.parent
REPO_ROOT = VALIDATION_ROOT.parent
PHASE2B_ROOT = VALIDATION_ROOT / "phase2b_screen"
ROUND1_ROOT = VALIDATION_ROOT / "phase2d_round1"
if str(PHASE2B_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2B_ROOT))

from screen_core import BaselineHead, MappingBlock  # noqa: E402


RUN_SPECS = {
    "mhsa_lambda_0_1_seed42": {"family": "deep_narrow_mhsa", "lambda": 0.1, "seed": 42},
    "mhsa_lambda_0_7_seed42": {"family": "deep_narrow_mhsa", "lambda": 0.7, "seed": 42},
    "mhsa_lambda_1_0_seed42": {"family": "deep_narrow_mhsa", "lambda": 1.0, "seed": 42},
    "baseline_seed45": {"family": "baseline", "lambda": None, "seed": 45},
    "baseline_seed46": {"family": "baseline", "lambda": None, "seed": 46},
}
ROUND_IDS = tuple(RUN_SPECS)
MHSA_IDS = tuple(run_id for run_id, spec in RUN_SPECS.items() if spec["family"] == "deep_narrow_mhsa")
BASELINE_IDS = tuple(run_id for run_id, spec in RUN_SPECS.items() if spec["family"] == "baseline")


class MHSAShortcutHead(nn.Module):
    """Five progressive mappings followed by the frozen Round2A MHSA design."""

    def __init__(self, in_features: int, num_classes: int, shortcut_lambda: float) -> None:
        super().__init__()
        self.shortcut_lambda = float(shortcut_lambda)
        self.input_projection = nn.Linear(in_features, 256)
        self.mapping_blocks = nn.ModuleList([MappingBlock(256) for _ in range(5)])
        self.pre_attention_norm = nn.LayerNorm(256)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(256)
        self.aggregation_dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, num_classes)

    def forward(
        self,
        x: Tensor,
        return_trace: bool = False,
        return_attention: bool = False,
        return_stages: bool = False,
    ):
        trace = {"backbone_gap": tuple(x.shape)}
        current = self.input_projection(x)
        trace["input_projection"] = tuple(current.shape)
        stages = []
        for index, block in enumerate(self.mapping_blocks, start=1):
            current = block(current, self.shortcut_lambda)
            stages.append(current)
            trace[f"mapping_stage_{index}"] = tuple(current.shape)
        stacked = torch.stack(stages, dim=1)
        trace["stacked_stages"] = tuple(stacked.shape)
        normalized = self.pre_attention_norm(stacked)
        attention_output, attention_weights = self.mhsa(
            normalized,
            normalized,
            normalized,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        fused = self.fusion_norm(stacked + attention_output)
        trace["attention_output"] = tuple(attention_output.shape)
        trace["fused_stages"] = tuple(fused.shape)
        feature = self.aggregation_dropout(fused.mean(dim=1))
        trace["mean_pooling_pre_classifier"] = tuple(feature.shape)
        logits = self.classifier(feature)
        trace["logits"] = tuple(logits.shape)
        if return_attention:
            if attention_weights is None:
                raise RuntimeError("attention weights were requested but not returned")
            trace["attention_weights"] = tuple(attention_weights.shape)
        if return_trace or return_attention or return_stages:
            extras = {
                "trace": trace,
                "attention_weights": attention_weights if return_attention else None,
                "post_shortcut_stages": stacked if return_stages else None,
                "fused_stages": fused if return_stages else None,
            }
            return logits, feature, extras
        return logits, feature


class Round2AModel(nn.Module):
    def __init__(self, run_id: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        if run_id not in RUN_SPECS:
            raise ValueError(f"forbidden or unknown Round2A run: {run_id}")
        self.variant = run_id
        self.run_id = run_id
        spec = RUN_SPECS[run_id]
        self.backbone = timm.create_model("inception_resnet_v2", pretrained=pretrained, num_classes=0)
        if self.backbone.num_features != 1536:
            raise ValueError(f"expected 1536-D backbone, got {self.backbone.num_features}")
        if spec["family"] == "baseline":
            self.head = BaselineHead(self.backbone.num_features, num_classes)
        else:
            self.head = MHSAShortcutHead(
                self.backbone.num_features,
                num_classes,
                shortcut_lambda=float(spec["lambda"]),
            )

    def forward(self, images: Tensor, return_trace: bool = False):
        gap = self.backbone(images)
        if return_trace:
            logits, features, trace_or_extras = self.head(gap, return_trace=True)
            trace = trace_or_extras.get("trace", trace_or_extras)
            return logits, features, trace
        return self.head(gap)


def config_path(run_id: str) -> Path:
    if run_id not in RUN_SPECS:
        raise ValueError(f"forbidden or unknown Round2A run: {run_id}")
    return ROUND_ROOT / "configs" / f"{run_id}.json"


def load_round_config(run_id: str) -> dict:
    return json.loads(config_path(run_id).read_text(encoding="utf-8"))


def output_dir(run_id: str, fold: int) -> Path:
    if fold not in range(5):
        raise ValueError(f"invalid fold: {fold}")
    return ROUND_ROOT / run_id / f"fold_{fold}"


def build_round_model(run_id: str, num_classes: int, pretrained: bool) -> Round2AModel:
    return Round2AModel(run_id, num_classes, pretrained)
