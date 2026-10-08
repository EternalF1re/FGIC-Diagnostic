"""Generate one endpoint-faithful seed45/46 alpha=0.5 comparator fold."""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from confirmation_common import comparator_task_dir, make_interpolated_state, metrics, reference_fold
from dfag_common import FOLDS, SEEDS, load_config, sha256_file, source_dir, state_digest
from screen_core import LeafDataset, ScreenModel, eval_transform, seed_everything, seed_worker, split_indices


def evaluate(model, loader, device) -> dict[str, np.ndarray]:
    logits, labels, sample_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels, indices, _ in loader:
            output, _ = model(images.to(device, non_blocking=True))
            logits.append(output.float().cpu().numpy())
            labels.append(batch_labels.numpy())
            sample_ids.append(indices.numpy())
    merged_logits = np.concatenate(logits).astype(np.float32)
    return {
        "sample_id": np.concatenate(sample_ids).astype(np.int64),
        "label": np.concatenate(labels).astype(np.int64),
        "logits": merged_logits,
        "prediction": merged_logits.argmax(axis=1).astype(np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--fold", type=int, required=True, choices=FOLDS)
    parser.add_argument("--device", required=True, choices=("cuda:0", "cuda:1"))
    args = parser.parse_args()
    seed, fold, device = args.seed, args.fold, torch.device(args.device)
    static = json.loads(
        (source_dir(seed, fold).parents[2] / "phase2f_seed45_46_confirmation" / "manifests" / "static_preflight_audit.json").read_text(encoding="utf-8")
    )
    if static.get("status") != "PASS":
        raise RuntimeError("static preflight missing or failed")
    output = comparator_task_dir(seed, fold)
    if output.exists():
        raise RuntimeError(f"refusing to reuse comparator task output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    started = time.time()
    manifest = {
        "status": "RUNNING",
        "seed": seed,
        "fold": fold,
        "device": str(device),
        "inference_only": True,
        "optimizer": False,
        "scheduler": False,
        "backward": False,
        "bn_refresh": False,
        "model_eval": True,
        "alphas": [0.0, 1.0, 0.5],
        "started_unix": started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        config = load_config(seed)
        dataset = LeafDataset(
            __import__("pathlib").Path(config["dataset"]["train_csv"]),
            __import__("pathlib").Path(config["dataset"]["root"]),
            eval_transform(),
        )
        _, validation_indices = split_indices(
            dataset.labels, fold, int(config["split"]["split_random_state"])
        )
        loader = DataLoader(
            Subset(dataset, validation_indices.tolist()),
            batch_size=int(config["common_training_protocol"]["batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=int(config["common_training_protocol"]["num_workers"]),
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=int(config["common_training_protocol"]["num_workers"]) > 0,
        )
        source = source_dir(seed, fold)
        stage1_path, stage2_path = source / "best_stage1.pth", source / "best_stage2.pth"
        source_manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
        if sha256_file(stage1_path) != source_manifest["best_stage1_sha256"]:
            raise RuntimeError("Stage1 source hash mismatch")
        if sha256_file(stage2_path) != source_manifest["best_stage2_sha256"]:
            raise RuntimeError("Stage2 source hash mismatch")
        stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)["model_state_dict"]
        stage2 = torch.load(stage2_path, map_location="cpu", weights_only=False)["model_state_dict"]
        endpoint_rows = []
        alpha05 = None
        for alpha in (0.0, 1.0, 0.5):
            seed_everything(seed)
            model = ScreenModel("#0", 176, pretrained=False)
            state, rule = make_interpolated_state(model.state_dict(), stage1, stage2, alpha)
            model.load_state_dict(state, strict=True)
            del state
            before = state_digest(model.state_dict())
            model.to(device).eval().requires_grad_(False)
            result = evaluate(model, loader, device)
            after = state_digest(model.state_dict())
            row = {
                "alpha": alpha,
                "state_rule": rule,
                "parameter_state_unchanged": before == after,
                "all_gradients_none": all(parameter.grad is None for parameter in model.parameters()),
                **metrics(result["label"], result["prediction"]),
            }
            if alpha in (0.0, 1.0):
                stage = 2 if alpha == 0.0 else 1
                reference = reference_fold(seed, fold, stage)
                row["endpoint_stage"] = stage
                row["sample_ids_exact"] = bool(np.array_equal(result["sample_id"], reference["sample_id"]))
                row["labels_exact"] = bool(np.array_equal(result["label"], reference["label"]))
                row["predictions_exact"] = bool(np.array_equal(result["prediction"], reference["prediction"]))
                row["logits_allclose_1e_6"] = bool(
                    np.allclose(result["logits"], reference["logits"], atol=1e-6, rtol=1e-6)
                )
                row["max_abs_logit_difference"] = float(
                    np.max(np.abs(result["logits"] - reference["logits"]))
                )
                if not all(row[key] for key in ("sample_ids_exact", "labels_exact", "predictions_exact")):
                    raise RuntimeError(f"ENDPOINT FIDELITY FAIL alpha={alpha}: {row}")
            else:
                alpha05 = result
            endpoint_rows.append(row)
            del model, result
            torch.cuda.empty_cache()
            gc.collect()
        if alpha05 is None:
            raise RuntimeError("alpha=0.5 output missing")
        artifact = output / "alpha_0_5.npz"
        np.savez_compressed(
            artifact,
            sample_id=alpha05["sample_id"],
            fold=np.full(len(alpha05["sample_id"]), fold, dtype=np.int8),
            label=alpha05["label"],
            prediction=alpha05["prediction"],
            logits=alpha05["logits"],
        )
        manifest.update(
            {
                "status": "COMPLETE",
                "completed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "stage1_checkpoint": str(stage1_path.resolve()),
                "stage1_sha256": sha256_file(stage1_path),
                "stage2_checkpoint": str(stage2_path.resolve()),
                "stage2_sha256": sha256_file(stage2_path),
                "endpoint_fidelity": endpoint_rows,
                "artifact": str(artifact.resolve()),
                "artifact_sha256": sha256_file(artifact),
                "counterfactual_training": False,
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "COMPLETE", "seed": seed, "fold": fold}), flush=True)
    except Exception as exc:
        manifest.update(
            {
                "status": "FAILED",
                "completed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
