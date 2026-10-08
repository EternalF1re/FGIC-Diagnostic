"""Train one authorized Phase2D Round1 configuration/fold.

Training functions are invoked directly from the frozen Phase2B train_one.py.
Only run identity, training seed, and the two declared shortcut lambdas differ.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from round1_common import PHASE2B_ROOT, ROUND_IDS, config_path, build_round_model, load_round_config, output_dir

from screen_core import (  # type: ignore
    build_loaders,
    fold_class_weights,
    initialize_head,
    model_counts,
    seed_everything,
    sha256_file,
)


SPEC = importlib.util.spec_from_file_location("phase2b_train_one_frozen", PHASE2B_ROOT / "train_one.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen Phase2B train_one.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def checkpoint_payload(model, run_id: str, fold: int, stage: int, epoch: int, val_accuracy: float, config: dict) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "run_id": run_id,
        "variant": run_id,
        "architecture_family": config["round1_run"]["architecture_family"],
        "shortcut_lambda": config["round1_run"]["shortcut_lambda"],
        "fold": fold,
        "stage": stage,
        "epoch": epoch,
        "metric": val_accuracy,
        "metric_type": "heldout_original_view_accuracy",
        "split_random_state": int(config["split"]["split_random_state"]),
        "training_seed": int(config["seed"]["training_seed"]),
        "config_sha256": sha256_file(config_path(run_id)),
        "phase2b_train_one_sha256": sha256_file(PHASE2B_ROOT / "train_one.py"),
        "phase2b_screen_core_sha256": sha256_file(PHASE2B_ROOT / "screen_core.py"),
    }


BASE.checkpoint_payload = checkpoint_payload


def save_evaluation(output: Path, stage: int, result: dict) -> dict:
    prefix = f"stage{stage}_validation"
    for key in ("logits", "features", "labels", "sample_ids", "image_names", "predictions"):
        np.save(output / f"{prefix}_{key}.npy", result[key])
    return {key: result[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, choices=ROUND_IDS)
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--device", required=True, choices=("cuda:0", "cuda:1"))
    args = parser.parse_args()
    run_id, fold = args.run_id, args.fold
    config = load_round_config(run_id)
    output = output_dir(run_id, fold)
    if output.exists():
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "COMPLETE":
            print(json.dumps({"status": "ALREADY_COMPLETE", "run_id": run_id, "fold": fold}), flush=True)
            return
        raise RuntimeError(f"refusing to reuse non-complete output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "run_manifest.json"
    device = torch.device(args.device)
    seed = int(config["seed"]["training_seed"])
    seed_everything(seed)
    started = time.time()
    manifest = {
        "status": "RUNNING", "phase": "Phase2D Round1", "run_id": run_id, "fold": fold,
        "split_id": f"skf42_fold{fold}", "split_random_state": 42, "training_seed": seed,
        "shortcut_lambda": config["round1_run"]["shortcut_lambda"],
        "architecture_family": config["round1_run"]["architecture_family"],
        "mhsa": False, "terminal_residual": False, "device": str(device),
        "started_unix": started, "formal_training": True,
        "config_sha256": sha256_file(config_path(run_id)),
        "round1_runner_sha256": sha256_file(Path(__file__).resolve()),
        "phase2b_train_one_sha256": sha256_file(PHASE2B_ROOT / "train_one.py"),
        "phase2b_screen_core_sha256": sha256_file(PHASE2B_ROOT / "screen_core.py"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(config_path(run_id), output / "config.json")
    try:
        train_loader, val_loader, dataset, train_idx, val_idx = build_loaders(fold, 1, config)
        phase2b_split = np.load(PHASE2B_ROOT / "#0" / f"fold_{fold}" / "split_indices.npz")
        if not np.array_equal(train_idx, phase2b_split["train_indices"]) or not np.array_equal(val_idx, phase2b_split["validation_indices"]):
            raise ValueError("Phase2D/Phase2B fold assignment mismatch")
        np.savez_compressed(output / "split_indices.npz", train_indices=train_idx, validation_indices=val_idx,
                            split_random_state=np.asarray([42]), training_seed=np.asarray([seed]))
        class_weights = fold_class_weights(dataset.labels, train_idx, int(config["dataset"]["num_classes"]))
        phase2b_weights = np.load(PHASE2B_ROOT / "#0" / f"fold_{fold}" / "class_weights.npy")
        if not np.array_equal(class_weights.numpy(), phase2b_weights):
            raise ValueError("class weights differ from Phase2B for identical fold partition")
        np.save(output / "class_weights.npy", class_weights.numpy())
        model = build_round_model(run_id, int(config["dataset"]["num_classes"]), pretrained=True)
        initialize_head(model)
        counts = model_counts(model)
        model.to(device)
        history = []
        stage1_path, stage1_epoch, stage1_best = BASE.stage1_train(
            model, train_loader, val_loader, class_weights, device, config, run_id, fold, output, history
        )
        stage1_checkpoint = torch.load(stage1_path, map_location=device, weights_only=False)
        model.load_state_dict(stage1_checkpoint["model_state_dict"], strict=True)
        stage1_result = BASE.evaluate(model, val_loader, device, collect=True)
        if not np.array_equal(stage1_result["sample_ids"], val_idx):
            raise ValueError("Stage1 held-out order mismatch")
        stage1_metrics = save_evaluation(output, 1, stage1_result)
        BASE.write_history(output / "stage1_history.csv", [row for row in history if row["stage"] == 1])

        del train_loader, val_loader
        train_loader, val_loader, dataset, train_idx2, val_idx2 = build_loaders(fold, 2, config)
        if not np.array_equal(train_idx, train_idx2) or not np.array_equal(val_idx, val_idx2):
            raise ValueError("Stage1/Stage2 split mismatch")
        bn_batches = BASE.adapt_batch_norm(model, train_loader, device, int(config["common_training_protocol"]["bn_adaptation_batches_before_stage2"]))
        stage2_path, stage2_epoch, stage2_best = BASE.stage2_train(
            model, train_loader, val_loader, class_weights, device, config, run_id, fold, output, history
        )
        stage2_checkpoint = torch.load(stage2_path, map_location=device, weights_only=False)
        model.load_state_dict(stage2_checkpoint["model_state_dict"], strict=True)
        stage2_result = BASE.evaluate(model, val_loader, device, collect=True)
        if not np.array_equal(stage2_result["sample_ids"], val_idx):
            raise ValueError("Stage2 held-out order mismatch")
        stage2_metrics = save_evaluation(output, 2, stage2_result)
        BASE.write_history(output / "stage2_history.csv", [row for row in history if row["stage"] == 2])
        metric_record = {
            "run_id": run_id, "fold": fold, "n": len(val_idx),
            "stage1": {**stage1_metrics, "best_epoch_zero_based": stage1_epoch, "best_accuracy": stage1_best},
            "stage2": {**stage2_metrics, "best_epoch_zero_based": stage2_epoch, "best_accuracy": stage2_best},
            "feature_extraction_location": config["architectures"]["#0" if config["round1_run"]["architecture_family"] == "baseline" else "#1"]["feature_extraction_location"],
        }
        (output / "metrics.json").write_text(json.dumps(metric_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage1_overflows = int(sum(int(row["amp_overflow_batches"]) for row in history if row["stage"] == 1))
        stage2_overflows = int(sum(int(row["amp_overflow_batches"]) for row in history if row["stage"] == 2))
        manifest.update({
            "status": "COMPLETE", "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
            "environment": BASE.environment(device), "model_counts": counts,
            "bn_adaptation_batches_completed": bn_batches,
            "best_stage1_checkpoint": str(stage1_path.resolve()), "best_stage1_sha256": sha256_file(stage1_path),
            "best_stage1_epoch_zero_based": stage1_epoch, "best_stage1_accuracy": stage1_best,
            "best_stage2_checkpoint": str(stage2_path.resolve()), "best_stage2_sha256": sha256_file(stage2_path),
            "best_stage2_epoch_zero_based": stage2_epoch, "best_stage2_accuracy": stage2_best,
            "stage1_amp_overflow_batches_total": stage1_overflows,
            "stage2_amp_overflow_batches_total": stage2_overflows,
            "history_counts": {"stage1": sum(r["stage"] == 1 for r in history), "stage2": sum(r["stage"] == 2 for r in history)},
            "validation_artifacts": {
                "stage1": {key: f"stage1_validation_{key}.npy" for key in ("logits","features","labels","sample_ids","image_names","predictions")},
                "stage2": {key: f"stage2_validation_{key}.npy" for key in ("logits","features","labels","sample_ids","image_names","predictions")},
            },
            "metrics": metric_record,
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "COMPLETE", "run_id": run_id, "fold": fold, "metrics": metric_record}, ensure_ascii=False), flush=True)
    except Exception as exc:
        manifest.update({"status": "FAILED", "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
                         "error": repr(exc), "traceback": traceback.format_exc()})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
