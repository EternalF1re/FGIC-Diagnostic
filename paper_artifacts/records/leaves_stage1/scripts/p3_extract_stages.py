"""Phase 2C P3: extract true post-shortcut f1..f5 for one #1/#2 fold."""
from __future__ import annotations

import argparse
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
OUT_ROOT = Path(__file__).resolve().parents[1] / "p3_mslfa_representation" / "fold_features"
sys.path.insert(0, str(SCREEN))
from screen_core import LeafDataset, ScreenModel, eval_transform, load_config, seed_everything, sha256_file, split_indices  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("#1", "#2"))
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    variant, fold = args.variant, args.fold
    out = OUT_ROOT / variant.replace("#", "variant_") / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    feature_path = out / "post_shortcut_stages.npz"
    manifest_path = out / "feature_manifest.json"
    if feature_path.exists() or manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite existing Phase2C artifact: {out}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("P3 requires CUDA inference; CPU fallback is intentionally disabled")

    started = time.time()
    config = load_config(); seed_everything(int(config["seed"]["training_seed"]))
    dataset_cfg = config["dataset"]
    dataset = LeafDataset(Path(dataset_cfg["train_csv"]), Path(dataset_cfg["root"]), eval_transform())
    train_idx, val_idx = split_indices(dataset.labels, fold, int(config["split"]["split_random_state"]))
    split_path = SCREEN / variant / f"fold_{fold}" / "split_indices.npz"
    saved = np.load(split_path)
    if not np.array_equal(train_idx, saved["train_indices"]) or not np.array_equal(val_idx, saved["validation_indices"]):
        raise RuntimeError("INTEGRITY_FAILURE: saved/recomputed split mismatch")
    loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    source_dir = SCREEN / variant / f"fold_{fold}"
    checkpoint_path = source_dir / "best_stage2.pth"
    source_manifest_path = source_dir / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != source_manifest["best_stage2_sha256"]:
        raise RuntimeError("INTEGRITY_FAILURE: Stage2 checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != variant or checkpoint.get("fold") != fold or checkpoint.get("stage") != 2:
        raise RuntimeError("INTEGRITY_FAILURE: Stage2 checkpoint metadata mismatch")
    model = ScreenModel(variant, int(dataset_cfg["num_classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(args.device)); model.eval()
    if len(model.head.mapping_blocks) != 5 or model.head.input_projection.out_features != 256:
        raise RuntimeError("INTEGRITY_FAILURE: unexpected controlled-head architecture")

    stage_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    repeat_exact = False
    forward_equivalence_exact = False
    with torch.inference_mode():
        for batch_index, (images, labels, ids, _) in enumerate(loader):
            images = images.to(args.device, non_blocking=True)
            gap = model.backbone(images)
            current = model.head.input_projection(gap)
            stages = []
            for block in model.head.mapping_blocks:
                current = block(current, model.head.shortcut_lambda)
                stages.append(current)
            stacked = torch.stack(stages, dim=1)
            feature = model.head.aggregation_dropout(stacked.mean(dim=1))
            logits = model.head.classifier(feature)
            if batch_index == 0:
                direct_logits, _ = model(images)
                forward_equivalence_exact = bool(torch.equal(logits, direct_logits))
                if not forward_equivalence_exact:
                    raise RuntimeError("INTEGRITY_FAILURE: manual stage path != model.forward")
                gap2 = model.backbone(images)
                current2 = model.head.input_projection(gap2)
                stages2 = []
                for block in model.head.mapping_blocks:
                    current2 = block(current2, model.head.shortcut_lambda); stages2.append(current2)
                repeat_exact = bool(torch.equal(stacked, torch.stack(stages2, dim=1)))
                if not repeat_exact:
                    raise RuntimeError("INTEGRITY_FAILURE: repeated stage extraction differs")
            stage_parts.append(stacked.float().cpu().numpy().astype(np.float32, copy=False))
            logits_parts.append(logits.float().cpu().numpy().astype(np.float32, copy=False))
            label_parts.append(labels.numpy().astype(np.int64, copy=False))
            id_parts.append(ids.numpy().astype(np.int64, copy=False))

    stages = np.concatenate(stage_parts); logits = np.concatenate(logits_parts)
    labels = np.concatenate(label_parts); ids = np.concatenate(id_parts)
    predictions = logits.argmax(1).astype(np.int64)
    source_ids = np.load(source_dir / "validation_sample_ids.npy")
    source_labels = np.load(source_dir / "validation_labels.npy")
    source_predictions = np.load(source_dir / "validation_predictions.npy")
    source_logits = np.load(source_dir / "validation_logits.npy")
    if stages.shape != (len(ids), 5, 256):
        raise RuntimeError(f"INTEGRITY_FAILURE: actual stage shape {stages.shape}")
    if not np.isfinite(stages).all() or not np.isfinite(logits).all():
        raise RuntimeError("INTEGRITY_FAILURE: NaN/Inf")
    if not (np.array_equal(ids, val_idx) and np.array_equal(ids, source_ids) and np.array_equal(labels, source_labels)):
        raise RuntimeError("INTEGRITY_FAILURE: sample/label alignment")
    if not np.array_equal(predictions, source_predictions):
        raise RuntimeError("INTEGRITY_FAILURE: extracted predictions != saved Stage2 predictions")
    logits_max_abs_diff = float(np.max(np.abs(logits.astype(np.float64) - source_logits.astype(np.float64))))
    logits_exact = bool(np.array_equal(logits, source_logits))
    if logits_max_abs_diff > 1e-6:
        raise RuntimeError(f"INTEGRITY_FAILURE: Stage2 logits max abs diff {logits_max_abs_diff}")

    temp = feature_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, sample_id=ids, fold=np.full(len(ids), fold, dtype=np.int8), label=labels,
                        prediction=predictions, f1=stages[:, 0], f2=stages[:, 1], f3=stages[:, 2],
                        f4=stages[:, 3], f5=stages[:, 4])
    temp.replace(feature_path)
    manifest = {
        "status": "PASS", "phase": "Phase2C P3", "training_performed": False,
        "variant": variant, "fold": fold, "n": len(ids), "actual_stage_shape": list(stages.shape),
        "feature_semantics": "f1..f5 are outputs after mapping(x)+lambda*x, exactly as used by model.forward",
        "shortcut_lambda": model.head.shortcut_lambda, "mhsa": False,
        "aggregation": "unweighted mean of f1..f5, then Dropout(0.2), then classifier",
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "source_manifest": str(source_manifest_path), "split_indices": str(split_path),
        "source_logits_exact": logits_exact, "source_logits_max_abs_diff": logits_max_abs_diff,
        "source_predictions_exact": True, "forward_equivalence_exact": forward_equivalence_exact,
        "deterministic_repeat_exact": repeat_exact, "eval_mode": True, "inference_mode": True,
        "transform": "Resize(299,299), ToTensor, ImageNet normalization; original view only",
        "elapsed_seconds": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"P3 PASS {variant} fold {fold}: {stages.shape}")


if __name__ == "__main__":
    main()
