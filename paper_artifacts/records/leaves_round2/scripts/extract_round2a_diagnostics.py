"""Extract one Round2A MHSA-ON OOF representation/attention diagnostic.

This is a checkpoint-only post-hoc pass.  It never trains or mutates a model.
The atomic unit saved for attention is sample x head x query, before any
aggregation, as required by the frozen Round2A analysis plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from round2a_common import MHSA_IDS, RUN_SPECS, build_round_model, load_round_config, output_dir
from screen_core import LeafDataset, eval_transform, seed_everything, sha256_file, split_indices


ROUND_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_ROOT = ROUND_ROOT / "diagnostics" / "fold_outputs"


def lambda_key(run_id: str) -> str:
    value = float(RUN_SPECS[run_id]["lambda"])
    return f"lambda_{str(value).replace('.', '_')}"


def destination(run_id: str, stage: int, fold: int) -> Path:
    return DIAGNOSTIC_ROOT / lambda_key(run_id) / f"stage{stage}" / f"fold_{fold}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=MHSA_IDS, required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    args = parser.parse_args()

    target = destination(args.run_id, args.stage, args.fold)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_root = ROUND_ROOT / "diagnostics" / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f"{args.run_id}_s{args.stage}_f{args.fold}_", dir=temp_root))

    try:
        config = load_round_config(args.run_id)
        seed = int(config["seed"]["training_seed"])
        seed_everything(seed)
        dataset = LeafDataset(
            Path(config["dataset"]["train_csv"]),
            Path(config["dataset"]["root"]),
            eval_transform(),
        )
        _, validation_ids = split_indices(dataset.labels, args.fold, int(config["split"]["split_random_state"]))
        loader = DataLoader(
            Subset(dataset, validation_ids.tolist()),
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        source = output_dir(args.run_id, args.fold)
        source_manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
        if source_manifest.get("status") != "COMPLETE":
            raise RuntimeError(f"source training manifest is not COMPLETE: {source}")
        checkpoint = source / f"best_stage{args.stage}.pth"
        actual_hash = sha256_file(checkpoint)
        expected_hash = source_manifest[f"best_stage{args.stage}_sha256"]
        if actual_hash != expected_hash:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")

        device = torch.device(args.device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_round_model(args.run_id, int(config["dataset"]["num_classes"]), pretrained=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device).eval()

        ids_parts: list[np.ndarray] = []
        labels_parts: list[np.ndarray] = []
        logits_parts: list[np.ndarray] = []
        stage_parts: list[list[np.ndarray]] = [[] for _ in range(5)]
        fused_parts: list[list[np.ndarray]] = [[] for _ in range(5)]
        weight_parts: list[np.ndarray] = []
        entropy_parts: list[np.ndarray] = []
        maximum_parts: list[np.ndarray] = []
        argmax_parts: list[np.ndarray] = []
        attention_row_max_error = 0.0

        with torch.inference_mode():
            for images, labels, sample_ids, _ in loader:
                images = images.to(device, non_blocking=True)
                gap = model.backbone(images)
                logits, _, extras = model.head(
                    gap,
                    return_attention=True,
                    return_stages=True,
                )
                stacked = extras["post_shortcut_stages"]
                fused = extras["fused_stages"]
                weights = extras["attention_weights"]
                if stacked is None or fused is None or weights is None:
                    raise RuntimeError("Round2A model did not return requested diagnostics")
                if tuple(weights.shape[1:]) != (4, 5, 5):
                    raise RuntimeError(f"unexpected attention shape: {tuple(weights.shape)}")
                row_error = float((weights.sum(dim=-1) - 1.0).abs().max().item())
                attention_row_max_error = max(attention_row_max_error, row_error)

                w64 = weights.double()
                entropy = -(w64.clamp_min(1e-300).log() * w64).sum(dim=-1) / math.log(5.0)
                maximum, argmax = weights.max(dim=-1)
                for index in range(5):
                    stage_parts[index].append(stacked[:, index].float().cpu().numpy())
                    fused_parts[index].append(fused[:, index].float().cpu().numpy())
                ids_parts.append(sample_ids.numpy())
                labels_parts.append(labels.numpy())
                logits_parts.append(logits.float().cpu().numpy())
                weight_parts.append(weights.float().cpu().numpy())
                entropy_parts.append(entropy.float().cpu().numpy())
                maximum_parts.append(maximum.float().cpu().numpy())
                argmax_parts.append(argmax.to(torch.int8).cpu().numpy())

        ids = np.concatenate(ids_parts)
        labels = np.concatenate(labels_parts)
        logits = np.concatenate(logits_parts)
        predictions = logits.argmax(axis=1)
        weights = np.concatenate(weight_parts)
        normalized_entropy = np.concatenate(entropy_parts)
        max_attention_share = np.concatenate(maximum_parts)
        argmax_key = np.concatenate(argmax_parts)
        stages = [np.concatenate(parts) for parts in stage_parts]
        fused_stages = [np.concatenate(parts) for parts in fused_parts]

        saved_ids = np.load(source / f"stage{args.stage}_validation_sample_ids.npy")
        saved_labels = np.load(source / f"stage{args.stage}_validation_labels.npy")
        saved_predictions = np.load(source / f"stage{args.stage}_validation_predictions.npy")
        saved_logits = np.load(source / f"stage{args.stage}_validation_logits.npy")
        if not np.array_equal(ids, validation_ids) or not np.array_equal(ids, saved_ids):
            raise RuntimeError("held-out sample order differs from formal training output")
        if not np.array_equal(labels, saved_labels):
            raise RuntimeError("held-out labels differ from formal training output")
        if not np.array_equal(predictions, saved_predictions):
            raise RuntimeError("post-hoc predictions differ from formal training output")
        logits_max_abs_error = float(np.max(np.abs(logits.astype(np.float64) - saved_logits.astype(np.float64))))
        if logits_max_abs_error > 1e-4:
            raise RuntimeError(f"post-hoc logits mismatch: max_abs_error={logits_max_abs_error}")
        if attention_row_max_error > 1e-5:
            raise RuntimeError(f"attention rows do not sum to one: max_error={attention_row_max_error}")
        arrays = stages + fused_stages + [weights, normalized_entropy, max_attention_share]
        if not all(np.isfinite(array).all() for array in arrays):
            raise RuntimeError("non-finite diagnostic array")

        np.savez_compressed(
            temp / "stages.npz",
            sample_id=ids,
            label=labels,
            prediction=predictions,
            **{f"f{i + 1}": stages[i] for i in range(5)},
            **{f"fused_f{i + 1}": fused_stages[i] for i in range(5)},
        )
        np.savez_compressed(
            temp / "attention.npz",
            sample_id=ids,
            attention_weights=weights,
            normalized_entropy=normalized_entropy,
            max_attention_share=max_attention_share,
            argmax_key=argmax_key,
        )
        manifest = {
            "status": "PASS",
            "run_id": args.run_id,
            "lambda_key": lambda_key(args.run_id),
            "lambda": float(RUN_SPECS[args.run_id]["lambda"]),
            "mhsa": "ON",
            "stage": args.stage,
            "fold": args.fold,
            "n": int(len(ids)),
            "post_shortcut_shape": [int(len(ids)), 5, 256],
            "attention_shape": [int(len(ids)), 4, 5, 5],
            "attention_atomic_unit": "sample_x_head_x_query",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": actual_hash,
            "source_predictions_exact": True,
            "source_sample_order_exact": True,
            "source_labels_exact": True,
            "source_logits_max_abs_error": logits_max_abs_error,
            "attention_rows_max_abs_error": attention_row_max_error,
            "training_performed": False,
            "device": str(device),
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, target)
        print(json.dumps({"status": "PASS", "output": str(target), **manifest}, ensure_ascii=False), flush=True)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


if __name__ == "__main__":
    main()
