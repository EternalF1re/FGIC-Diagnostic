"""Extract true post-shortcut f1..f5 for one lambda/stage/fold checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from round1_common import PHASE2B_ROOT, ROUND_ROOT, VALIDATION_ROOT, output_dir
from screen_core import (
    LeafDataset,
    ScreenModel,
    eval_transform,
    load_config,
    seed_everything,
    sha256_file,
    split_indices,
)


MAP = {
    "lambda_0_1": ("#2", 0.1),
    "lambda_1_0": ("#1", 1.0),
    "lambda_0_7": ("lambda_0_7_seed42", 0.7),
    "lambda_0_9": ("lambda_0_9_seed42", 0.9),
}


def source(lambda_key: str, stage: int, fold: int):
    identity, _ = MAP[lambda_key]
    if identity.startswith("#"):
        directory = PHASE2B_ROOT / identity / f"fold_{fold}"
        checkpoint = directory / f"best_stage{stage}.pth"
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        expected_hash = manifest[f"best_stage{stage}_sha256"]
        if stage == 2:
            prediction_path = directory / "validation_predictions.npy"
        else:
            prediction_path = (
                VALIDATION_ROOT
                / "phase2c_inference_diagnostic"
                / "p1_stage1_only_oof"
                / "fold_outputs"
                / identity.replace("#", "variant_")
                / f"fold_{fold}"
                / "stage1_validation.npz"
            )
    else:
        directory = output_dir(identity, fold)
        checkpoint = directory / f"best_stage{stage}.pth"
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        expected_hash = manifest[f"best_stage{stage}_sha256"]
        prediction_path = directory / f"stage{stage}_validation_predictions.npy"
    return directory, checkpoint, expected_hash, prediction_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-key", choices=MAP, required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    args = parser.parse_args()

    destination = (
        ROUND_ROOT
        / "representation"
        / "fold_features"
        / args.lambda_key
        / f"stage{args.stage}"
        / f"fold_{args.fold}"
    )
    destination.mkdir(parents=True, exist_ok=False)

    config = load_config()
    seed_everything(42)
    dataset = LeafDataset(
        Path(config["dataset"]["train_csv"]),
        Path(config["dataset"]["root"]),
        eval_transform(),
    )
    _, validation_ids = split_indices(dataset.labels, args.fold, 42)
    loader = DataLoader(
        Subset(dataset, validation_ids.tolist()),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    _, checkpoint, expected_hash, prediction_path = source(args.lambda_key, args.stage, args.fold)
    actual_hash = sha256_file(checkpoint)
    if actual_hash != expected_hash:
        raise RuntimeError("checkpoint hash mismatch")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    variant = "#2" if args.lambda_key == "lambda_0_1" else "#1"
    model = ScreenModel(variant, 176, pretrained=False)
    model.head.shortcut_lambda = MAP[args.lambda_key][1]
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(args.device).eval()

    stages_by_layer: list[list[np.ndarray]] = [[] for _ in range(5)]
    sample_ids: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    manual_forward_exact = False
    with torch.inference_mode():
        for batch_index, (images, targets, ids, _) in enumerate(loader):
            images = images.to(args.device, non_blocking=True)
            gap = model.backbone(images)
            current = model.head.input_projection(gap)
            stages = []
            for block in model.head.mapping_blocks:
                current = block(current, model.head.shortcut_lambda)
                stages.append(current)
            stack = torch.stack(stages, dim=1)
            features = model.head.aggregation_dropout(stack.mean(dim=1))
            batch_logits = model.head.classifier(features)
            if batch_index == 0:
                direct_logits, _ = model(images)
                manual_forward_exact = torch.equal(batch_logits, direct_logits)
                if not manual_forward_exact:
                    raise RuntimeError("manual forward mismatch")
            for layer in range(5):
                stages_by_layer[layer].append(stack[:, layer].float().cpu().numpy())
            sample_ids.append(ids.numpy())
            labels.append(targets.numpy())
            logits.append(batch_logits.float().cpu().numpy())

    ids_array = np.concatenate(sample_ids)
    labels_array = np.concatenate(labels)
    logits_array = np.concatenate(logits)
    feature_arrays = [np.concatenate(values) for values in stages_by_layer]
    predictions = logits_array.argmax(axis=1)
    if prediction_path.suffix == ".npz":
        with np.load(prediction_path) as saved:
            source_predictions = saved["prediction"]
    else:
        source_predictions = np.load(prediction_path)
    if not np.array_equal(ids_array, validation_ids):
        raise RuntimeError("validation sample-id mismatch")
    if not np.array_equal(predictions, source_predictions):
        raise RuntimeError("source prediction mismatch")
    if not all(np.isfinite(values).all() for values in feature_arrays):
        raise RuntimeError("non-finite representation")

    np.savez_compressed(
        destination / "stages.npz",
        sample_id=ids_array,
        label=labels_array,
        prediction=predictions,
        **{f"f{layer + 1}": feature_arrays[layer] for layer in range(5)},
    )
    manifest = {
        "status": "PASS",
        "lambda_key": args.lambda_key,
        "lambda": MAP[args.lambda_key][1],
        "stage": args.stage,
        "fold": args.fold,
        "n": len(ids_array),
        "shape": [len(ids_array), 5, 256],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_hash,
        "source_predictions_exact": True,
        "manual_forward_exact": manual_forward_exact,
        "training_performed": False,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "lambda_key": args.lambda_key, "stage": args.stage, "fold": args.fold}))


if __name__ == "__main__":
    main()
