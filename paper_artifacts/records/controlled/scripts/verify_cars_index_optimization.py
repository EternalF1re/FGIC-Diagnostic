"""Verify the optimized Cars lookup against the frozen formal manifest."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from common import core, runtime  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    records = runtime.records("cars")
    elapsed = time.perf_counter() - started

    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    heldout_fold = np.full(len(records), -1, dtype=np.int64)
    folds = []
    for fold in core.FOLDS:
        train_indices, heldout_indices = core.split_indices(labels, fold)
        heldout_fold[heldout_indices] = fold
        folds.append({
            "fold": fold,
            "train_count": len(train_indices),
            "heldout_count": len(heldout_indices),
            "train_index_sha256": core.canonical_sha(train_indices.tolist()),
            "heldout_index_sha256": core.canonical_sha(heldout_indices.tolist()),
        })

    assignments = [
        {
            "sample_id": str(row["sample_id"]),
            "label": int(row["label"]),
            "heldout_fold": int(heldout_fold[index]),
        }
        for index, row in enumerate(records)
    ]
    class_map = core.class_map("cars", records)
    reference = json.loads(
        (ROOT / "manifests" / "cars_fold_and_class_manifest.json").read_text(encoding="utf-8")
    )
    train_root = Path(runtime.external_config()["datasets"]["cars"]["train_image_root"])
    paths_semantically_equal = all(
        Path(row["path"]).name == row["sample_id"].split(":", 1)[1]
        and Path(row["path"]).parent.name == class_map[str(row["label"])]
        and row["image_name"]
        == str(Path(row["path"]).relative_to(train_root)).replace("\\", "/")
        for row in records
    )

    result = {
        "status": "PASS",
        "implementation": runtime.CARS_LOOKUP_IMPLEMENTATION,
        "elapsed_seconds": elapsed,
        "development_count": len(records),
        "assignments_equal": assignments == reference["assignments"],
        "folds_equal": folds == reference["folds"],
        "class_map_equal": class_map == reference["class_map"],
        "fold_manifest_sha256": core.canonical_sha(assignments),
        "expected_fold_manifest_sha256": reference["fold_manifest_sha256"],
        "class_map_sha256": core.canonical_sha(class_map),
        "expected_class_map_sha256": reference["class_map_sha256"],
        "paths_semantically_equal": paths_semantically_equal,
    }
    checks = [
        result["development_count"] == reference["development_count"],
        result["assignments_equal"],
        result["folds_equal"],
        result["class_map_equal"],
        result["fold_manifest_sha256"] == result["expected_fold_manifest_sha256"],
        result["class_map_sha256"] == result["expected_class_map_sha256"],
        result["paths_semantically_equal"],
    ]
    if not all(checks):
        result["status"] = "FAIL"
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
