"""Phase 2C P1 deterministic Stage1 checkpoint inference for one variant/fold.

This module is deliberately inference-only. It has no optimizer, loss, backward,
or parameter-update path and writes only inside the new Phase 2C directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


VALIDATION_ROOT = Path(__file__).resolve().parents[2]
SCREEN = VALIDATION_ROOT / "phase2b_screen"
OUT_ROOT = Path(__file__).resolve().parents[1] / "p1_stage1_only_oof" / "fold_outputs"
sys.path.insert(0, str(SCREEN))
from screen_core import LeafDataset, ScreenModel, eval_transform, load_config, seed_everything, sha256_file, split_indices  # noqa: E402


def expected_ledger_row(variant: str, fold: int) -> dict[str, str]:
    with (SCREEN / "run_ledger.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["variant"] == variant and int(row["fold"]) == fold]
    if len(rows) != 1 or rows[0]["status"] != "COMPLETE":
        raise RuntimeError(f"missing unique COMPLETE ledger row for {variant} fold {fold}")
    return rows[0]


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("#0", "#1", "#2"))
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    variant, fold = args.variant, args.fold
    tag = variant.replace("#", "variant_")
    out = OUT_ROOT / tag / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "stage1_validation.npz"
    manifest_path = out / "inference_manifest.json"
    if npz_path.exists() or manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite existing Phase2C artifact: {out}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("P1 requires CUDA inference; CPU fallback is intentionally disabled")

    started = time.time()
    config = load_config()
    seed = int(config["seed"]["training_seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    dataset_cfg = config["dataset"]
    dataset = LeafDataset(Path(dataset_cfg["train_csv"]), Path(dataset_cfg["root"]), eval_transform())
    recomputed_train, recomputed_val = split_indices(dataset.labels, fold, int(config["split"]["split_random_state"]))
    split_path = SCREEN / variant / f"fold_{fold}" / "split_indices.npz"
    saved_split = np.load(split_path)
    saved_train = saved_split["train_indices"].astype(np.int64)
    saved_val = saved_split["validation_indices"].astype(np.int64)
    if not np.array_equal(saved_train, recomputed_train) or not np.array_equal(saved_val, recomputed_val):
        raise RuntimeError("INTEGRITY_FAILURE: saved and recomputed fold assignments differ")

    loader = DataLoader(
        Subset(dataset, saved_val.tolist()), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    checkpoint_path = SCREEN / variant / f"fold_{fold}" / "best_stage1.pth"
    source_manifest_path = SCREEN / variant / f"fold_{fold}" / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(checkpoint_path)
    expected_hash = source_manifest["best_stage1_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(f"INTEGRITY_FAILURE: Stage1 checkpoint SHA256 mismatch {actual_hash} != {expected_hash}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ledger = expected_ledger_row(variant, fold)
    expected_meta = {
        "variant": variant, "fold": fold, "stage": 1,
        "epoch": int(ledger["best_stage1_epoch_zero_based"]),
        "metric": float(ledger["best_stage1_accuracy"]),
        "split_random_state": int(ledger["split_random_state"]),
        "training_seed": int(ledger["training_seed"]),
    }
    for key, expected in expected_meta.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"INTEGRITY_FAILURE: checkpoint {key}={checkpoint.get(key)!r} != {expected!r}")

    model = ScreenModel(variant, int(dataset_cfg["num_classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    ids_parts: list[np.ndarray] = []
    names: list[str] = []
    repeat_equal = False
    repeat_max_abs_diff = float("nan")
    with torch.inference_mode():
        for batch_index, (images, labels, sample_ids, image_names) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            output, _ = model(images)
            if batch_index == 0:
                repeated, _ = model(images)
                repeat_equal = bool(torch.equal(output, repeated))
                repeat_max_abs_diff = float((output - repeated).abs().max().item())
                if not repeat_equal:
                    raise RuntimeError(f"INTEGRITY_FAILURE: repeated inference differs, max abs {repeat_max_abs_diff}")
            logits_parts.append(output.float().cpu().numpy().astype(np.float32, copy=False))
            labels_parts.append(labels.numpy().astype(np.int64, copy=False))
            ids_parts.append(sample_ids.numpy().astype(np.int64, copy=False))
            names.extend(str(name) for name in image_names)

    logits = np.concatenate(logits_parts)
    labels = np.concatenate(labels_parts)
    ids = np.concatenate(ids_parts)
    predictions = logits.argmax(axis=1).astype(np.int64)
    correct_count = int(np.sum(predictions == labels))
    expected_accuracy = float(ledger["best_stage1_accuracy"])
    expected_correct_count = int(round(expected_accuracy * len(labels)))
    reproduced_accuracy = correct_count / len(labels)
    reproduction_pass = correct_count == expected_correct_count and reproduced_accuracy == expected_accuracy
    result = {
        "status": "PASS" if reproduction_pass else "STAGE1_CHECKPOINT_REPRODUCTION_FAILED",
        "variant": variant, "fold": fold, "n": len(labels),
        "expected_correct_count": expected_correct_count,
        "reproduced_correct_count": correct_count,
        "expected_accuracy": expected_accuracy,
        "reproduced_accuracy": reproduced_accuracy,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": actual_hash,
        "source_manifest": str(source_manifest_path), "split_indices": str(split_path),
        "sample_ids_exactly_saved_validation_order": bool(np.array_equal(ids, saved_val)),
        "labels_exactly_dataset_aligned": bool(np.array_equal(labels, dataset.labels[ids])),
        "logits_finite": bool(np.isfinite(logits).all()),
        "logits_argmax_equals_predictions": bool(np.array_equal(logits.argmax(axis=1), predictions)),
        "deterministic_repeat_exact": repeat_equal,
        "deterministic_repeat_max_abs_diff": repeat_max_abs_diff,
        "eval_mode": not model.training, "inference_mode": True,
        "transform": "Resize(299,299), ToTensor, ImageNet normalization; original view only",
        "fp32_saved_logits": str(logits.dtype) == "float32",
        "training_performed": False,
        "elapsed_seconds": time.time() - started,
    }
    if not all((result["sample_ids_exactly_saved_validation_order"], result["labels_exactly_dataset_aligned"], result["logits_finite"], result["logits_argmax_equals_predictions"])):
        raise RuntimeError("INTEGRITY_FAILURE: inferred arrays failed alignment/finiteness checks")
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not reproduction_pass:
        raise RuntimeError("STAGE1_CHECKPOINT_REPRODUCTION_FAILED")
    atomic_npz(npz_path, sample_id=ids, fold=np.full(len(ids), fold, dtype=np.int8), label=labels,
               prediction=predictions, logits=logits, image_name=np.asarray(names))
    print(f"P1 PASS {variant} fold {fold}: {correct_count}/{len(labels)}")


if __name__ == "__main__":
    main()
