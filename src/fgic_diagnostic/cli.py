from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_records
from .engine import evaluate_checkpoint, interpolate_checkpoints, run_experiment
from .metrics import classification_metrics, paired_accuracy_statistics


def command_validate(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    records = load_records(args.dataset_config, config["dataset"])
    expected_classes = int(config["num_classes"])
    observed = len({row["label"] for row in records})
    if observed != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, observed {observed}")
    missing = [str(row["path"]) for row in records if not Path(row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing; first: {missing[0]}")
    print(json.dumps({"status": "PASS", "samples": len(records), "classes": observed}, indent=2))


def command_train(args: argparse.Namespace) -> None:
    result = run_experiment(
        args.experiment_config, args.dataset_config, args.fold, args.output,
        device_name=args.device, workers=args.workers, stage1_checkpoint=args.stage1_checkpoint, smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def command_aggregate(args: argparse.Namespace) -> None:
    files = sorted(Path(args.runs).glob("fold_*/heldout_predictions.npz"))
    if len(files) != 5:
        raise ValueError(f"Expected five fold exports, found {len(files)}")
    parts = [load_prediction(path) for path in files]
    sample_ids = np.concatenate([part["sample_ids"].astype(str) for part in parts])
    if len(sample_ids) != len(set(sample_ids.tolist())):
        raise ValueError("OOF sample IDs are not unique")
    labels = np.concatenate([part["labels"] for part in parts]).astype(np.int64)
    logits = np.concatenate([part["logits"] for part in parts]).astype(np.float32)
    predictions = logits.argmax(1).astype(np.int64)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(labels, predictions)
    np.savez_compressed(output / "pooled_oof_predictions.npz", sample_ids=sample_ids, labels=labels, logits=logits, predictions=predictions)
    (output / "pooled_oof_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def command_paired(args: argparse.Namespace) -> None:
    first = load_prediction(Path(args.first))
    second = load_prediction(Path(args.second))
    order_a = np.argsort(first["sample_ids"].astype(str))
    order_b = np.argsort(second["sample_ids"].astype(str))
    ids_a = first["sample_ids"].astype(str)[order_a]
    ids_b = second["sample_ids"].astype(str)[order_b]
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("Paired OOF sample IDs do not match")
    labels_a = first["labels"].astype(np.int64)[order_a]
    labels_b = second["labels"].astype(np.int64)[order_b]
    if not np.array_equal(labels_a, labels_b):
        raise ValueError("Paired OOF labels do not match")
    pred_a = first.get("predictions", first["logits"].argmax(1)).astype(np.int64)[order_a]
    pred_b = second.get("predictions", second["logits"].argmax(1)).astype(np.int64)[order_b]
    result = paired_accuracy_statistics(labels_a, pred_a, pred_b, args.bootstrap_samples, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def command_manifest(args: argparse.Namespace) -> None:
    root = Path(args.image_root).resolve()
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = sorted(path for path in root.iterdir() if path.is_dir())
    rows = []
    for label, directory in enumerate(classes):
        for path in sorted(item for item in directory.rglob("*") if item.suffix.lower() in extensions):
            relative = path.relative_to(root).as_posix()
            rows.append({"path": relative, "label": label, "sample_id": f"{args.dataset}:{relative}"})
    if not rows:
        raise ValueError(f"No images found below {root}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps({"output": str(output), "samples": len(rows), "classes": len(classes)}, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    result = evaluate_checkpoint(
        args.experiment_config, args.dataset_config, args.checkpoint, args.fold,
        args.output, device_name=args.device, workers=args.workers,
    )
    print(json.dumps(result, indent=2))


def command_interpolate(args: argparse.Namespace) -> None:
    result = interpolate_checkpoints(args.stage1, args.stage2, args.output, args.alpha)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="fgic")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data", help="Validate a dataset manifest against an experiment config")
    validate.add_argument("--experiment-config", required=True)
    validate.add_argument("--dataset-config", required=True)
    validate.set_defaults(function=command_validate)
    train = commands.add_parser("train", help="Run one controlled fold")
    train.add_argument("--experiment-config", required=True)
    train.add_argument("--dataset-config", required=True)
    train.add_argument("--fold", required=True, type=int, choices=range(5))
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--workers", default=4, type=int)
    train.add_argument("--stage1-checkpoint")
    train.add_argument("--smoke", action="store_true", help="Run shortened mechanics-only smoke training; not a paper result")
    train.set_defaults(function=command_train)
    aggregate = commands.add_parser("aggregate", help="Pool five held-out fold exports")
    aggregate.add_argument("--runs", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(function=command_aggregate)
    paired = commands.add_parser("paired", help="Paired bootstrap CI and exact two-sided McNemar test")
    paired.add_argument("--first", required=True)
    paired.add_argument("--second", required=True)
    paired.add_argument("--output", required=True)
    paired.add_argument("--bootstrap-samples", default=100_000, type=int)
    paired.add_argument("--seed", default=42, type=int)
    paired.set_defaults(function=command_paired)
    manifest = commands.add_parser("make-imagefolder-manifest", help="Create a manifest from class-named directories")
    manifest.add_argument("--image-root", required=True)
    manifest.add_argument("--dataset", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(function=command_manifest)
    evaluate = commands.add_parser("evaluate", help="Evaluate one checkpoint on its deterministic held-out fold")
    evaluate.add_argument("--experiment-config", required=True)
    evaluate.add_argument("--dataset-config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--fold", required=True, type=int, choices=range(5))
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--workers", default=4, type=int)
    evaluate.set_defaults(function=command_evaluate)
    interpolate = commands.add_parser("interpolate", help="Interpolate complete Stage-1/Stage-2 states")
    interpolate.add_argument("--stage1", required=True)
    interpolate.add_argument("--stage2", required=True)
    interpolate.add_argument("--alpha", default=0.5, type=float)
    interpolate.add_argument("--output", required=True)
    interpolate.set_defaults(function=command_interpolate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
