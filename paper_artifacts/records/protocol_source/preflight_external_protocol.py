"""Read-only dataset/architecture preflight for the new external protocol."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from protocol_core import (
    CONFIG_PATH,
    ROOT,
    ExternalModel,
    cars_records,
    cub_records,
    flowers_records,
    load_config,
    model_counts,
    seed_everything,
    serialized_transforms,
    sha256_file,
    split_indices,
)


def check(name: str, condition: bool, detail) -> dict:
    return {"name": name, "pass": bool(condition), "detail": detail}


def main() -> None:
    config = load_config()
    started = time.time()
    checks = []

    cub_dev, cub_test = cub_records(config)
    cars_dev, cars_test = cars_records(config)
    flowers_dev, flowers_test, flowers_meta = flowers_records(config)
    expected = config["datasets"]
    checks.extend([
        check("cub_development_count_5994", len(cub_dev) == 5994, len(cub_dev)),
        check("cub_official_test_count_5794", len(cub_test) == 5794, len(cub_test)),
        check("cub_classes_200", len(set(row["label"] for row in cub_dev + cub_test)) == 200,
              len(set(row["label"] for row in cub_dev + cub_test))),
        check("cars_development_count_8144", len(cars_dev) == 8144, len(cars_dev)),
        check("cars_official_test_count_8041", len(cars_test) == 8041, len(cars_test)),
        check("cars_classes_196", len(set(row["label"] for row in cars_dev + cars_test)) == 196,
              len(set(row["label"] for row in cars_dev + cars_test))),
        check("flowers_official_train_count_1020", len(flowers_meta["official_train_ids"]) == 1020,
              len(flowers_meta["official_train_ids"])),
        check("flowers_official_validation_count_1020", len(flowers_meta["official_validation_ids"]) == 1020,
              len(flowers_meta["official_validation_ids"])),
        check("flowers_development_count_2040", len(flowers_dev) == 2040, len(flowers_dev)),
        check("flowers_official_test_count_6149", len(flowers_test) == 6149, len(flowers_test)),
        check("flowers_classes_102", len(set(row["label"] for row in flowers_dev + flowers_test)) == 102,
              len(set(row["label"] for row in flowers_dev + flowers_test))),
    ])

    dataset_rows = {"cub": (cub_dev, cub_test), "cars": (cars_dev, cars_test), "flowers": (flowers_dev, flowers_test)}
    fold_summary = {}
    for dataset_key, (development, official_test) in dataset_rows.items():
        labels = np.asarray([row["label"] for row in development], dtype=np.int64)
        dev_ids = {row["sample_id"] for row in development}
        test_ids = {row["sample_id"] for row in official_test}
        checks.append(check(f"{dataset_key}_no_development_test_id_overlap", not (dev_ids & test_ids), len(dev_ids & test_ids)))
        folds = []
        validation_seen = np.zeros(len(development), dtype=np.int64)
        for fold in range(5):
            train_idx, val_idx = split_indices(labels, fold, 42)
            validation_seen[val_idx] += 1
            train_ids = set(train_idx.tolist())
            val_ids = set(val_idx.tolist())
            folds.append({
                "fold": fold,
                "train_count": len(train_idx),
                "validation_count": len(val_idx),
                "overlap": len(train_ids & val_ids),
                "validation_class_counts": dict(sorted(Counter(labels[val_idx].tolist()).items())),
            })
            checks.append(check(f"{dataset_key}_fold{fold}_train_validation_disjoint", not (train_ids & val_ids), 0))
        checks.append(check(f"{dataset_key}_each_development_sample_validation_once",
                            np.all(validation_seen == 1), Counter(validation_seen.tolist())))
        fold_summary[dataset_key] = folds

    for fold in fold_summary["flowers"]:
        counts = list(fold["validation_class_counts"].values())
        checks.append(check(f"flowers_fold{fold['fold']}_four_validation_per_class",
                            len(counts) == 102 and set(counts) == {4}, sorted(set(counts))))
        checks.append(check(f"flowers_fold{fold['fold']}_train_count_1632", fold["train_count"] == 1632, fold["train_count"]))
        checks.append(check(f"flowers_fold{fold['fold']}_validation_count_408", fold["validation_count"] == 408, fold["validation_count"]))

    transform_manifest = serialized_transforms()
    for dataset_key, record in transform_manifest.items():
        checks.append(check(f"{dataset_key}_vertical_flip_absent", not record["vertical_flip_present"], record["train_repr"]))
        checks.append(check(f"{dataset_key}_deterministic_validation_transform",
                            "Random" not in record["validation_repr"], record["validation_repr"]))
        checks.append(check(f"{dataset_key}_deterministic_test_transform",
                            "Random" not in record["official_test_repr"], record["official_test_repr"]))

    seed_everything(42)
    architecture = {}
    with torch.no_grad():
        sample = torch.randn(2, 3, 299, 299)
        for run_id in ("ours_ft", "progressive_lambda_0_7"):
            model = ExternalModel(run_id, 200, pretrained=False).eval()
            logits, features, stages = model(sample, return_stages=run_id.startswith("progressive"))
            names = [name.lower() for name, _ in model.named_modules()]
            module_types = [type(module).__name__ for module in model.modules()]
            record = {
                "counts": model_counts(model),
                "logits_shape": list(logits.shape),
                "feature_shape": list(features.shape),
                "stages_shape": list(stages.shape) if stages is not None else None,
                "module_types": module_types,
            }
            architecture[run_id] = record
            checks.append(check(f"{run_id}_finite_forward", torch.isfinite(logits).all().item(), record["logits_shape"]))
            checks.append(check(f"{run_id}_mhsa_absent", "MultiheadAttention" not in module_types, module_types))
            checks.append(check(f"{run_id}_terminal_residual_absent", not any("terminal" in name for name in names), names))
            if run_id == "progressive_lambda_0_7":
                checks.append(check("progressive_five_post_shortcut_stages", tuple(stages.shape) == (2, 5, 256), list(stages.shape)))
                checks.append(check("progressive_lambda_exact_0_7", model.head.shortcut_lambda == 0.7, model.head.shortcut_lambda))
                # Directly prove each saved stage obeys F_i(input)+lambda*input.
                projected = model.head.input_projection(model.backbone(sample))
                current = projected
                formula_ok = True
                for index, block in enumerate(model.head.mapping_blocks):
                    expected_stage = block.mapping(current) + 0.7 * current
                    formula_ok = formula_ok and torch.allclose(expected_stage, stages[:, index], atol=1e-6, rtol=1e-5)
                    current = expected_stage
                checks.append(check("progressive_shortcut_formula_exact", formula_ok, "f_i=F_i(f_{i-1})+0.7*f_{i-1}"))
                expected_feature = model.head.aggregation_dropout(stages.mean(dim=1))
                checks.append(check("progressive_mean_f1_to_f5_used",
                                    torch.allclose(expected_feature, features, atol=1e-6, rtol=1e-5), list(features.shape)))

    all_pass = all(row["pass"] for row in checks)
    manifest = {
        "status": "PASS" if all_pass else "FAIL",
        "generated_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "historical_reproduction": False,
        "instruction_sha256": config["instruction_sha256"],
        "config_path": str(CONFIG_PATH.resolve()),
        "config_sha256": sha256_file(CONFIG_PATH),
        "python_executable": sys.executable,
        "checks": checks,
        "dataset_counts": {
            "cub": {"development": len(cub_dev), "official_test": len(cub_test)},
            "cars": {"development": len(cars_dev), "official_test": len(cars_test)},
            "flowers": {"official_train": len(flowers_meta["official_train_ids"]),
                        "official_validation": len(flowers_meta["official_validation_ids"]),
                        "development": len(flowers_dev), "official_test": len(flowers_test)},
        },
        "fold_summary": fold_summary,
        "transforms": transform_manifest,
        "architecture": architecture,
        "training_jobs_launched": 0,
    }
    manifests = ROOT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "preflight.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "checks": len(checks), "failed": [row["name"] for row in checks if not row["pass"]]}, ensure_ascii=False), flush=True)
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
