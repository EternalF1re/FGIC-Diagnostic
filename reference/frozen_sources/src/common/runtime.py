"""WSL-safe runtime helpers for the frozen controlled reimplementation.

This module never rewrites a frozen config.  It only maps Windows absolute
paths to their /mnt/<drive>/ equivalents while executing under Linux/WSL.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

from common import core


_WINDOWS_ABSOLUTE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
CARS_LOOKUP_IMPLEMENTATION = "one_pass_per_split_filename_index_v1"


def runtime_path(value: str) -> str:
    """Translate an absolute Windows path only when running on Linux/WSL."""
    if os.name == "nt":
        return value
    match = _WINDOWS_ABSOLUTE.match(value)
    if not match:
        return value
    drive, rest = match.groups()
    return f"/mnt/{drive.lower()}/{rest.replace(chr(92), '/')}"


def _translate(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _translate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_translate(item) for item in value]
    if isinstance(value, str):
        return runtime_path(value)
    return value


def external_config() -> dict[str, Any]:
    return _translate(copy.deepcopy(core.load_external_config()))


def _indexed_cars_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build identical Cars records using one directory enumeration per split.

    The frozen lookup issued ``image_root.glob(f"*/{file_name}")`` for every
    annotation. On WSL DrvFS that creates millions of metadata RPCs. This
    storage-only optimization indexes every class directory once, then performs
    the same unique-filename lookup in memory. CSV order, labels, sample IDs,
    relative image names, and absolute paths remain unchanged.
    """
    cfg = config["datasets"]["cars"]
    result: list[list[dict[str, Any]]] = []
    for split_name, csv_key, root_key in (
        ("train", "train_csv", "train_image_root"),
        ("test", "test_csv", "test_image_root"),
    ):
        frame = pd.read_csv(cfg[csv_key], header=None)
        if frame.shape[1] != 6:
            raise ValueError(f"Cars {split_name} annotation must have 6 columns")

        image_root = Path(cfg[root_key])
        paths_by_name: dict[str, list[Path]] = {}
        for class_directory in image_root.iterdir():
            if not class_directory.is_dir():
                continue
            for candidate in class_directory.iterdir():
                paths_by_name.setdefault(candidate.name, []).append(candidate)

        rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False, name=None):
            file_name, _, _, _, _, class_id = row
            file_name = str(file_name)
            matches = paths_by_name.get(file_name, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Cars image lookup expected one match for {file_name}, got {len(matches)}"
                )
            path = matches[0]
            rows.append({
                "path": path,
                "label": int(class_id) - 1,
                "sample_id": f"cars_{split_name}:{file_name}",
                "image_name": str(path.relative_to(image_root)).replace("\\", "/"),
            })
        result.append(rows)
    return result[0], result[1]


def records(dataset: str) -> list[dict[str, Any]]:
    config = external_config()
    if dataset == "cub":
        development, _ = core.external_protocol.cub_records(config)
    elif dataset == "cars":
        development, _ = _indexed_cars_records(config)
    else:
        raise ValueError(dataset)
    for row in development:
        row["path"] = runtime_path(str(row["path"]))
    return development


def fold_records(dataset: str, fold: int) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    development = records(dataset)
    labels = np.asarray([int(row["label"]) for row in development], dtype=np.int64)
    train_indices, heldout_indices = core.split_indices(labels, int(fold))
    return development, train_indices, heldout_indices


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)


def loaders(
    dataset: str,
    method: str,
    fold: int,
    batch_size: int,
    *,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, list[dict[str, Any]], np.ndarray, np.ndarray]:
    development, train_indices, heldout_indices = fold_records(dataset, fold)
    train_dataset = core.RecordDataset(development, core.train_transform(method))
    heldout_dataset = core.RecordDataset(development, core.eval_transform())
    train_subset = Subset(train_dataset, train_indices.tolist())
    heldout_subset = Subset(heldout_dataset, heldout_indices.tolist())
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=core.SEED,
            drop_last=False,
        )
    generator = torch.Generator().manual_seed(core.SEED)
    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "persistent_workers": int(num_workers) > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_subset,
        sampler=sampler,
        shuffle=sampler is None,
        generator=generator if sampler is None else None,
        drop_last=False,
        **common,
    )
    heldout_loader = DataLoader(heldout_subset, shuffle=False, drop_last=False, **common)
    return train_loader, heldout_loader, development, train_indices, heldout_indices


def expected_heldout(
    development: Sequence[dict[str, Any]], heldout_indices: Sequence[int], fold: int
) -> dict[str, np.ndarray]:
    return {
        "sample_ids": np.asarray([str(development[int(i)]["sample_id"]) for i in heldout_indices]),
        "labels": np.asarray([int(development[int(i)]["label"]) for i in heldout_indices], dtype=np.int64),
        "dataset_indices": np.asarray(heldout_indices, dtype=np.int64),
        "fold_ids": np.full(len(heldout_indices), int(fold), dtype=np.int64),
    }


def verify_export(payload: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict[str, Any]:
    checks = {
        "count": len(payload["sample_ids"]) == len(expected["sample_ids"]),
        "sample_ids": np.array_equal(payload["sample_ids"].astype(str), expected["sample_ids"].astype(str)),
        "labels": np.array_equal(payload["labels"].astype(np.int64), expected["labels"]),
        "dataset_indices": np.array_equal(payload["dataset_indices"].astype(np.int64), expected["dataset_indices"]),
        "fold_ids": np.array_equal(payload["fold_ids"].astype(np.int64), expected["fold_ids"]),
        "unique_ids": len(set(payload["sample_ids"].astype(str).tolist())) == len(payload["sample_ids"]),
        "finite_logits": bool(np.isfinite(payload["logits"]).all()),
    }
    checks["all_pass"] = all(checks.values())
    if not checks["all_pass"]:
        raise ValueError(f"held-out export alignment failed: {checks}")
    return checks
