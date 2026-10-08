"""Immutable helpers for the final 40-job external-validation run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = ROOT.parent
CORRECTED_ROOT = VALIDATION_ROOT / "corrected_external_protocol_smoke"
CORRECTED_SCRIPTS = CORRECTED_ROOT / "scripts"
if str(CORRECTED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTED_SCRIPTS))

import protocol_core  # noqa: E402
import corrected_transform  # noqa: E402


corrected_transform.install()

CONFIG_PATH = ROOT / "final_external_protocol_config.json"
SOURCE_LEDGER_PATH = ROOT / "formal_job_ledger.csv"
RUNTIME_LEDGER_PATH = ROOT / "final_job_ledger.csv"
RUNS_ROOT = ROOT / "formal_runs"
LOGS_ROOT = ROOT / "formal_logs"
EXPECTED_CONFIG_SHA256 = "7f667c125983ca82ec36214b727ee8fbe3215f21562abd260f4018559f87e3e5"
INSTRUCTION_SHA256 = "BD71375E67A7B6F53AD1797EE0C9E338F601A08ADFCFB4A82D05031BCB3027B5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> Dict[str, Any]:
    observed = sha256_file(CONFIG_PATH)
    if observed != EXPECTED_CONFIG_SHA256:
        raise ValueError(f"formal config SHA mismatch: {observed}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def dataset_key(name: str) -> str:
    mapping = {"CUB-200-2011": "cub", "Stanford Cars": "cars", "Oxford Flowers-102": "flowers"}
    if name not in mapping:
        raise ValueError(name)
    return mapping[name]


def records_for_dataset(name: str, config: dict):
    key = dataset_key(name)
    if key == "cub":
        development, official_test = protocol_core.cub_records(config)
    elif key == "cars":
        development, official_test = protocol_core.cars_records(config)
    else:
        development, official_test, _ = protocol_core.flowers_records(config)
    expected = config["datasets"][key]
    if len(development) != int(expected["development_count"]) or len(official_test) != int(expected["official_test_count"]):
        raise ValueError(f"{name} dataset count mismatch")
    return development, official_test


def build_loaders(name: str, fold: int, stage: int, config: dict):
    key = dataset_key(name)
    development, official_test = records_for_dataset(name, config)
    train_data = protocol_core.RecordDataset(development, corrected_transform.train_transform(key))
    validation_data = protocol_core.RecordDataset(development, protocol_core.eval_transform())
    test_data = protocol_core.RecordDataset(official_test, protocol_core.eval_transform())
    train_idx, val_idx = protocol_core.split_indices(train_data.labels, fold, int(config["split"]["random_state"]))
    generator = torch.Generator()
    generator.manual_seed(42 + 1000 * stage)
    common = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": True,
        "worker_init_fn": protocol_core.seed_worker,
        "persistent_workers": int(config["training"]["num_workers"]) > 0,
    }
    train_loader = DataLoader(Subset(train_data, train_idx.tolist()), shuffle=True, generator=generator, drop_last=False, **common)
    validation_loader = DataLoader(Subset(validation_data, val_idx.tolist()), shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, drop_last=False, **common)
    return train_loader, validation_loader, test_loader, train_data, test_data, train_idx, val_idx


def build_model(method: str, shortcut_lambda: str, num_classes: int, pretrained: bool = True):
    if method == "Ours-FT":
        if shortcut_lambda not in ("", None):
            raise ValueError("Ours-FT lambda must be empty")
        model = protocol_core.ExternalModel("ours_ft", num_classes, pretrained=pretrained)
        model.formal_method = method
        model.formal_lambda = None
    elif method == "Progressive Head":
        value = float(shortcut_lambda)
        if value not in (0.1, 0.7, 1.0):
            raise ValueError(f"unauthorized lambda {value}")
        model = protocol_core.ExternalModel("progressive_lambda_0_7", num_classes, pretrained=pretrained)
        model.head.shortcut_lambda = value
        model.formal_method = method
        model.formal_lambda = value
    else:
        raise ValueError(method)
    return model


def job_id(dataset: str, method: str, shortcut_lambda: str, fold: int) -> str:
    ds = {"CUB-200-2011": "cub", "Stanford Cars": "cars", "Oxford Flowers-102": "flowers"}[dataset]
    method_id = "ours_ft" if method == "Ours-FT" else f"progressive_lambda_{str(shortcut_lambda).replace('.', '_')}"
    return f"{ds}__{method_id}__fold_{int(fold)}"


def output_dir(job: str) -> Path:
    return RUNS_ROOT / job

