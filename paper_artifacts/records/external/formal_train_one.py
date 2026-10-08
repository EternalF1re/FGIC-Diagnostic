"""Train one and only one row from the frozen 40-job formal ledger."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import formal_common as common


if str(common.CORRECTED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(common.CORRECTED_SCRIPTS))
import protocol_core  # noqa: E402
import train_cub_smoke as frozen_train  # noqa: E402


def checkpoint_payload(model, run_id: str, stage: int, epoch: int, val_accuracy: float, config: dict) -> dict:
    context = CHECKPOINT_CONTEXT
    return {
        "model_state_dict": model.state_dict(),
        "experiment_id": config["experiment_id"],
        "formal_job": True,
        "job_id": context["job_id"],
        "dataset": context["dataset"],
        "method": context["method"],
        "shortcut_lambda": context["shortcut_lambda"],
        "fold": context["fold"],
        "stage": stage,
        "epoch": epoch,
        "metric": val_accuracy,
        "metric_type": "deterministic_original_view_fold_validation_accuracy",
        "split_random_state": 42,
        "training_seed": 42,
        "stage1_label_smoothing": 0.05,
        "stage2_label_smoothing": 0.03,
        "batch_size": 32,
        "mhsa": False,
        "terminal_residual": False,
        "dfag": False,
        "class_weights": None,
        "config_sha256": common.EXPECTED_CONFIG_SHA256,
        "runner_sha256": common.sha256_file(Path(__file__)),
        "common_sha256": common.sha256_file(Path(common.__file__)),
    }


CHECKPOINT_CONTEXT = {}
frozen_train.checkpoint_payload = checkpoint_payload


def save_result(output: Path, prefix: str, result: dict, include_features: bool, include_stages: bool) -> dict:
    keys = ["logits", "labels", "dataset_indices", "sample_ids", "image_names", "predictions"]
    if include_features:
        keys.append("features")
    if include_stages:
        keys.append("post_shortcut_stages")
    for key in keys:
        np.save(output / f"{prefix}_{key}.npy", result[key])
    return {key: float(result[key]) for key in ("accuracy", "macro_f1", "balanced_accuracy")}


def assert_alignment(result: dict, expected_ids: np.ndarray, label: str) -> None:
    actual = result["sample_ids"].astype(str)
    if not np.array_equal(actual, expected_ids.astype(str)):
        raise ValueError(f"{label} sample-ID alignment failure")
    if len(set(actual.tolist())) != len(actual):
        raise ValueError(f"{label} duplicate sample IDs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--dataset", required=True, choices=("CUB-200-2011", "Stanford Cars", "Oxford Flowers-102"))
    parser.add_argument("--method", required=True, choices=("Ours-FT", "Progressive Head"))
    parser.add_argument("--lambda", dest="shortcut_lambda", default="")
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--device", required=True, choices=("cuda:0", "cuda:1"))
    args = parser.parse_args()
    config = common.load_config()
    expected_job_id = common.job_id(args.dataset, args.method, args.shortcut_lambda, args.fold)
    if args.job_id != expected_job_id:
        raise ValueError(f"job identity mismatch: {args.job_id} != {expected_job_id}")
    output = common.output_dir(args.job_id)
    if output.exists():
        manifest_path = output / "run_manifest.json"
        if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "COMPLETE":
            print(json.dumps({"status": "ALREADY_COMPLETE", "job_id": args.job_id}), flush=True)
            return
        raise RuntimeError(f"refusing to reuse incomplete output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "run_manifest.json"
    key = common.dataset_key(args.dataset)
    num_classes = int(config["datasets"][key]["num_classes"])
    context = {
        "job_id": args.job_id, "dataset": args.dataset, "method": args.method,
        "shortcut_lambda": None if args.method == "Ours-FT" else float(args.shortcut_lambda), "fold": args.fold,
    }
    CHECKPOINT_CONTEXT.clear()
    CHECKPOINT_CONTEXT.update(context)
    device = torch.device(args.device)
    protocol_core.seed_everything(42)
    started = time.time()
    manifest = {
        "status": "RUNNING", **context, "training_seed": 42, "split_random_state": 42,
        "device": str(device), "started_unix": started, "formal_training": True,
        "config_path": str(common.CONFIG_PATH.resolve()), "config_sha256": common.sha256_file(common.CONFIG_PATH),
        "runner_sha256": common.sha256_file(Path(__file__)), "common_sha256": common.sha256_file(Path(common.__file__)),
        "transforms": protocol_core.serialized_transforms()[key], "primary_tta": False,
        "fold_ensemble": False, "class_weights": None, "official_test_used_for_selection": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        train_loader, val_loader, test_loader, development, official_test, train_idx, val_idx = common.build_loaders(args.dataset, args.fold, 1, config)
        train_ids = np.asarray([development.records[index]["sample_id"] for index in train_idx])
        val_ids = np.asarray([development.records[index]["sample_id"] for index in val_idx])
        test_ids = np.asarray([row["sample_id"] for row in official_test.records])
        if set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) & set(test_ids):
            raise ValueError("split leakage")
        np.savez_compressed(output / "split_indices.npz", train_indices=train_idx, validation_indices=val_idx,
                            train_sample_ids=train_ids, validation_sample_ids=val_ids,
                            split_random_state=np.asarray([42]), training_seed=np.asarray([42]))
        model = common.build_model(args.method, args.shortcut_lambda, num_classes, pretrained=True)
        protocol_core.initialize_head(model)
        counts = protocol_core.model_counts(model)
        model.to(device)
        history = []
        stage1_path, stage1_epoch, stage1_best = frozen_train.stage1_train(
            model, train_loader, val_loader, device, config, args.job_id, output, history
        )
        stage1_checkpoint = torch.load(stage1_path, map_location=device, weights_only=False)
        model.load_state_dict(stage1_checkpoint["model_state_dict"], strict=True)
        capture = args.dataset == "CUB-200-2011" and args.method == "Progressive Head"
        stage1_val = protocol_core.evaluate(model, val_loader, device, capture_stages=capture)
        stage1_test = protocol_core.evaluate(model, test_loader, device, capture_stages=False)
        assert_alignment(stage1_val, val_ids, "Stage1 validation")
        assert_alignment(stage1_test, test_ids, "Stage1 official test")
        stage1_val_metrics = save_result(output, "stage1_validation", stage1_val, True, capture)
        stage1_test_metrics = save_result(output, "stage1_official_test", stage1_test, False, False)
        frozen_train.write_history(output / "stage1_history.csv", [row for row in history if row["stage"] == 1])
        diagnostics = {}
        if capture:
            diagnostics["stage1"] = protocol_core.representation_diagnostics(stage1_val["post_shortcut_stages"])
            pd.DataFrame(diagnostics["stage1"]["pairwise"]).to_csv(output / "stage1_representation_pairs.csv", index=False)

        model.load_state_dict(stage1_checkpoint["model_state_dict"], strict=True)
        del train_loader, val_loader, test_loader
        train_loader, val_loader, test_loader, _, _, train_idx2, val_idx2 = common.build_loaders(args.dataset, args.fold, 2, config)
        if not np.array_equal(train_idx, train_idx2) or not np.array_equal(val_idx, val_idx2):
            raise ValueError("Stage1/Stage2 split mismatch")
        bn_record = frozen_train.adapt_batch_norm(model, train_loader, device, 50)
        stage2_path, stage2_epoch, stage2_best = frozen_train.stage2_train(
            model, train_loader, val_loader, device, config, args.job_id, output, history
        )
        stage2_checkpoint = torch.load(stage2_path, map_location=device, weights_only=False)
        model.load_state_dict(stage2_checkpoint["model_state_dict"], strict=True)
        stage2_val = protocol_core.evaluate(model, val_loader, device, capture_stages=capture)
        stage2_test = protocol_core.evaluate(model, test_loader, device, capture_stages=False)
        assert_alignment(stage2_val, val_ids, "Stage2 validation")
        assert_alignment(stage2_test, test_ids, "Stage2 official test")
        stage2_val_metrics = save_result(output, "stage2_validation", stage2_val, True, capture)
        stage2_test_metrics = save_result(output, "stage2_official_test", stage2_test, False, False)
        frozen_train.write_history(output / "stage2_history.csv", [row for row in history if row["stage"] == 2])
        if capture:
            diagnostics["stage2"] = protocol_core.representation_diagnostics(stage2_val["post_shortcut_stages"])
            pd.DataFrame(diagnostics["stage2"]["pairwise"]).to_csv(output / "stage2_representation_pairs.csv", index=False)
            (output / "representation_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
        metrics = {
            **context,
            "stage1": {"validation": stage1_val_metrics, "secondary_official_test": stage1_test_metrics,
                       "best_epoch_zero_based": stage1_epoch, "best_validation_accuracy": stage1_best},
            "stage2": {"validation": stage2_val_metrics, "secondary_official_test": stage2_test_metrics,
                       "best_epoch_zero_based": stage2_epoch, "best_validation_accuracy": stage2_best},
            "representation_diagnostics": diagnostics,
        }
        metrics_path = output / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.update({
            "status": "COMPLETE", "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
            "environment": frozen_train.environment(device), "model_counts": counts,
            "train_count": len(train_idx), "validation_count": len(val_idx), "official_test_count": len(test_ids),
            "no_leakage": True, "stage2_started_from_selected_stage1": True,
            "bn_recalibration": bn_record, "history_counts": {"stage1": 150, "stage2": 60},
            "best_stage1_checkpoint": str(stage1_path.resolve()), "best_stage1_sha256": common.sha256_file(stage1_path),
            "best_stage2_checkpoint": str(stage2_path.resolve()), "best_stage2_sha256": common.sha256_file(stage2_path),
            "metrics_path": str(metrics_path.resolve()), "metrics_sha256": common.sha256_file(metrics_path),
            "prediction_artifacts": {
                stage: {split: str((output / f"{stage}_{split}_predictions.npy").resolve()) for split in ("validation", "official_test")}
                for stage in ("stage1", "stage2")
            },
            "metrics": metrics,
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "COMPLETE", "job_id": args.job_id}), flush=True)
    except Exception as exc:
        manifest.update({"status": "FAILED", "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
                         "error": repr(exc), "traceback": traceback.format_exc()})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
