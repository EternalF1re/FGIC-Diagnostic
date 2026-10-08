"""Shared Phase2D Round1 configuration/model helpers.

The training/data primitives are imported directly from the frozen Phase2B
controlled implementation so augmentation, loaders, losses, and architecture
code are not reimplemented here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROUND_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROUND_ROOT.parent
REPO_ROOT = VALIDATION_ROOT.parent
PHASE2B_ROOT = VALIDATION_ROOT / "phase2b_screen"
if str(PHASE2B_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2B_ROOT))

from screen_core import ScreenModel  # noqa: E402


ROUND_IDS = (
    "baseline_seed43",
    "baseline_seed44",
    "lambda_0_7_seed42",
    "lambda_0_9_seed42",
)


def config_path(run_id: str) -> Path:
    if run_id not in ROUND_IDS:
        raise ValueError(f"Forbidden or unknown Round1 run: {run_id}")
    return ROUND_ROOT / "configs" / f"{run_id}.json"


def load_round_config(run_id: str) -> dict:
    return json.loads(config_path(run_id).read_text(encoding="utf-8"))


def build_round_model(run_id: str, num_classes: int, pretrained: bool) -> ScreenModel:
    config = load_round_config(run_id)
    family = config["round1_run"]["architecture_family"]
    base_variant = "#0" if family == "baseline" else "#1"
    model = ScreenModel(base_variant, num_classes, pretrained=pretrained)
    model.variant = run_id
    if family == "deep_narrow":
        model.head.shortcut_lambda = float(config["round1_run"]["shortcut_lambda"])
    return model


def output_dir(run_id: str, fold: int) -> Path:
    config = load_round_config(run_id)
    return ROUND_ROOT / config["round1_run"]["output_subdir"] / f"fold_{fold}"
