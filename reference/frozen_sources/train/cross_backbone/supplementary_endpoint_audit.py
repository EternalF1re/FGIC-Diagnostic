"""Read-only endpoint-fidelity audit for the completed cross-backbone run.

This script never trains, recalibrates BatchNorm, or writes into formal run
directories.  It only loads frozen checkpoints, performs eval-mode inference,
and writes one audit JSON per backbone.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cross_backbone_common import (
    FOLDS,
    ROOT,
    CrossBackboneModel,
    build_loaders,
    interpolate_complete_state,
    load_config,
    output_dir,
    seed_everything,
    sha256_file,
)
from train_one import evaluate


AUDIT_DIR = ROOT / "supplementary_audit"


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def all_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], keys: list[str]) -> bool:
    return all(torch.equal(left[key], right[key]) for key in keys)


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def endpoint_result(
    model: CrossBackboneModel,
    state: dict[str, torch.Tensor],
    loader,
    device: torch.device,
    reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    model.load_state_dict(state, strict=True)
    result = evaluate(model, loader, device)
    sample_id_exact = np.array_equal(result["sample_id"], reference["sample_id"])
    labels_exact = np.array_equal(result["y_true"], reference["y_true"])
    predictions_exact = np.array_equal(result["y_pred"], reference["y_pred"])
    logits_exact = np.array_equal(result["logits"], reference["logits"])
    return {
        "sample_id_exact": bool(sample_id_exact),
        "labels_exact": bool(labels_exact),
        "predictions_exact": bool(predictions_exact),
        "prediction_mismatch_count": int(np.count_nonzero(result["y_pred"] != reference["y_pred"]))
        if result["y_pred"].shape == reference["y_pred"].shape
        else -1,
        "logits_exact": bool(logits_exact),
        "logits_max_abs_difference": max_abs_difference(result["logits"], reference["logits"]),
        "pass": bool(sample_id_exact and labels_exact and predictions_exact and logits_exact),
    }


def audit_fold(backbone: str, fold: int, device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    run_dir = output_dir("formal", backbone, "ours_ft", fold)
    stage1_path = run_dir / "best_stage1.pth"
    stage2_path = run_dir / "best_stage2.pth"
    stage1_payload = torch.load(stage1_path, map_location="cpu", weights_only=False)
    stage2_payload = torch.load(stage2_path, map_location="cpu", weights_only=False)
    stage1 = stage1_payload["model_state_dict"]
    stage2 = stage2_payload["model_state_dict"]
    if set(stage1) != set(stage2):
        raise ValueError(f"state keys differ for {backbone}/fold_{fold}")

    keys = list(stage1)
    floating_keys = [key for key in keys if torch.is_floating_point(stage1[key])]
    nonfloating_keys = [key for key in keys if not torch.is_floating_point(stage1[key])]
    running_mean_keys = [key for key in keys if key.endswith("running_mean")]
    running_var_keys = [key for key in keys if key.endswith("running_var")]
    tracked_keys = [key for key in keys if key.endswith("num_batches_tracked")]
    nonfloating_diff = [key for key in nonfloating_keys if not torch.equal(stage1[key], stage2[key])]

    alpha0, alpha0_meta = interpolate_complete_state(stage1, stage2, alpha=0.0)
    alpha1, alpha1_meta = interpolate_complete_state(stage1, stage2, alpha=1.0)

    state_checks = {
        "total_state_tensors": len(keys),
        "floating_state_tensors": len(floating_keys),
        "nonfloating_state_tensors": len(nonfloating_keys),
        "running_mean_tensors": len(running_mean_keys),
        "running_var_tensors": len(running_var_keys),
        "num_batches_tracked_tensors": len(tracked_keys),
        "stage1_vs_stage2_nonfloating_difference_count": len(nonfloating_diff),
        "stage1_vs_stage2_nonfloating_difference_keys": nonfloating_diff,
        "alpha0_all_state_exact_stage2": all_equal(alpha0, stage2, keys),
        "alpha0_floating_state_exact_stage2": all_equal(alpha0, stage2, floating_keys),
        "alpha0_nonfloating_state_exact_stage2": all_equal(alpha0, stage2, nonfloating_keys),
        "alpha1_all_state_exact_stage1": all_equal(alpha1, stage1, keys),
        "alpha1_floating_state_exact_stage1": all_equal(alpha1, stage1, floating_keys),
        "alpha1_nonfloating_state_exact_stage1": all_equal(alpha1, stage1, nonfloating_keys),
        "alpha1_nonfloating_state_exact_stage2": all_equal(alpha1, stage2, nonfloating_keys),
        "alpha0_running_mean_exact_stage2": all_equal(alpha0, stage2, running_mean_keys),
        "alpha0_running_var_exact_stage2": all_equal(alpha0, stage2, running_var_keys),
        "alpha1_running_mean_exact_stage1": all_equal(alpha1, stage1, running_mean_keys),
        "alpha1_running_var_exact_stage1": all_equal(alpha1, stage1, running_var_keys),
        "alpha0_num_batches_tracked_exact_stage2": all_equal(alpha0, stage2, tracked_keys),
        "alpha1_num_batches_tracked_exact_stage1": all_equal(alpha1, stage1, tracked_keys),
        "alpha1_num_batches_tracked_exact_stage2": all_equal(alpha1, stage2, tracked_keys),
        "alpha0_interpolation_metadata": alpha0_meta,
        "alpha1_interpolation_metadata": alpha1_meta,
    }

    _, val_loader, _, _, val_idx = build_loaders(fold, 2, config, backbone)
    stage1_prediction = load_prediction(run_dir / "stage1_predictions.npz")
    stage2_prediction = load_prediction(run_dir / "stage2_predictions.npz")
    if not np.array_equal(val_idx, stage1_prediction["sample_id"]):
        raise ValueError(f"Stage1 validation order mismatch for {backbone}/fold_{fold}")
    if not np.array_equal(val_idx, stage2_prediction["sample_id"]):
        raise ValueError(f"Stage2 validation order mismatch for {backbone}/fold_{fold}")

    model = CrossBackboneModel(backbone, "ours_ft", pretrained=False).to(device)
    alpha0_eval = endpoint_result(model, alpha0, val_loader, device, stage2_prediction)
    alpha1_eval = endpoint_result(model, alpha1, val_loader, device, stage1_prediction)

    del model, val_loader, alpha0, alpha1, stage1, stage2, stage1_payload, stage2_payload
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "backbone": backbone,
        "fold": fold,
        "device": str(device),
        "stage1_checkpoint": str(stage1_path.resolve()),
        "stage2_checkpoint": str(stage2_path.resolve()),
        "stage1_checkpoint_sha256": sha256_file(stage1_path),
        "stage2_checkpoint_sha256": sha256_file(stage2_path),
        "state_checks": state_checks,
        "alpha0_vs_stage2": alpha0_eval,
        "alpha1_vs_stage1": alpha1_eval,
        "endpoint_prediction_logits_pass": bool(alpha0_eval["pass"] and alpha1_eval["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("resnet50", "convnext_tiny"), required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    device = torch.device(args.device)
    config = load_config()
    rows = []
    for fold in FOLDS:
        row = audit_fold(args.backbone, fold, device, config)
        rows.append(row)
        print(json.dumps({
            "backbone": args.backbone,
            "fold": fold,
            "alpha0_pass": row["alpha0_vs_stage2"]["pass"],
            "alpha1_pass": row["alpha1_vs_stage1"]["pass"],
        }), flush=True)

    payload = {
        "backbone": args.backbone,
        "device": str(device),
        "folds": rows,
        "all_folds_endpoint_prediction_logits_pass": all(row["endpoint_prediction_logits_pass"] for row in rows),
    }
    target = AUDIT_DIR / f"endpoint_{args.backbone}.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(target.resolve())}), flush=True)


if __name__ == "__main__":
    main()
