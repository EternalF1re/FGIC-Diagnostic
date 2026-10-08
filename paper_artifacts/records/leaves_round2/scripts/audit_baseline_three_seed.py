"""Zero-training-cost baseline seed42/43/44 prediction and provenance audit."""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np


ROUND_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROUND_ROOT.parent
PHASE2B = VALIDATION_ROOT / "phase2b_screen"
ROUND1 = VALIDATION_ROOT / "phase2d_round1"
P1 = VALIDATION_ROOT / "phase2c_inference_diagnostic" / "p1_stage1_only_oof"
SEEDS = (42, 43, 44)
N = 18_353


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_seed42(stage: int) -> dict[str, np.ndarray]:
    if stage == 1:
        path = P1 / "stage1_oof_predictions_variant_0.npz"
        with np.load(path) as data:
            return {
                "sample_id": data["sample_id"],
                "fold": data["fold"],
                "label": data["label"],
                "prediction": data["prediction"],
                "logits": data["logits"],
            }
    base = PHASE2B / "#0"
    folds = np.empty(N, dtype=np.int8)
    fold_logits = []
    fold_ids = []
    for fold in range(5):
        ids = np.load(base / f"fold_{fold}" / "validation_sample_ids.npy")
        folds[ids] = fold
        fold_ids.append(ids)
        fold_logits.append(np.load(base / f"fold_{fold}" / "validation_logits.npy"))
    ids_concat = np.concatenate(fold_ids)
    logits_concat = np.concatenate(fold_logits)
    order = np.argsort(ids_concat)
    return {
        "sample_id": np.load(base / "oof_sample_ids.npy"),
        "fold": folds,
        "label": np.load(base / "oof_labels.npy"),
        "prediction": np.load(base / "oof_predictions.npy"),
        "logits": logits_concat[order],
    }


def load_round1(seed: int, stage: int) -> dict[str, np.ndarray]:
    run_id = f"baseline_seed{seed}"
    path = ROUND1 / "oof" / f"{run_id}_stage{stage}_oof.npz"
    with np.load(path) as data:
        return {name: data[source] for name, source in {
            "sample_id": "sample_id",
            "fold": "fold",
            "label": "label",
            "prediction": "prediction",
            "logits": "logits",
        }.items()}


def source_predictions(seed: int, stage: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    if seed == 42 and stage == 1:
        path = P1 / "fold_outputs" / "variant_0" / f"fold_{fold}" / "stage1_validation.npz"
        with np.load(path) as data:
            return data["sample_id"], data["prediction"]
    if seed == 42:
        directory = PHASE2B / "#0" / f"fold_{fold}"
        return np.load(directory / "validation_sample_ids.npy"), np.load(directory / "validation_predictions.npy")
    directory = ROUND1 / f"baseline_seed{seed}" / f"fold_{fold}"
    return (
        np.load(directory / f"stage{stage}_validation_sample_ids.npy"),
        np.load(directory / f"stage{stage}_validation_predictions.npy"),
    )


def checkpoint(seed: int, stage: int, fold: int) -> tuple[Path, str]:
    if seed == 42:
        directory = PHASE2B / "#0" / f"fold_{fold}"
    else:
        directory = ROUND1 / f"baseline_seed{seed}" / f"fold_{fold}"
    manifest = read_json(directory / "run_manifest.json")
    return directory / f"best_stage{stage}.pth", manifest[f"best_stage{stage}_sha256"]


def verify_oof(data: dict[str, np.ndarray], label: str) -> None:
    if not (
        data["sample_id"].shape == data["fold"].shape == data["label"].shape == data["prediction"].shape == (N,)
        and data["logits"].shape == (N, 176)
        and np.array_equal(data["sample_id"], np.arange(N))
        and len(np.unique(data["sample_id"])) == N
        and set(np.unique(data["fold"]).tolist()) == {0, 1, 2, 3, 4}
        and np.isfinite(data["logits"]).all()
        and np.array_equal(data["logits"].argmax(axis=1), data["prediction"])
    ):
        raise RuntimeError(f"OOF integrity failure: {label}")


def main() -> None:
    output_csv = ROUND_ROOT / "statistics" / "baseline_seed42_44_pairwise_prediction_audit.csv"
    output_md = ROUND_ROOT / "BASELINE_THREE_SEED_PREDICTION_AUDIT.md"
    output_json = ROUND_ROOT / "manifests" / "baseline_three_seed_pretraining_audit.json"
    for path in (output_csv, output_md, output_json):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    data = {(seed, stage): load_seed42(stage) if seed == 42 else load_round1(seed, stage)
            for seed in SEEDS for stage in (1, 2)}
    checks = []
    for seed in SEEDS:
        for stage in (1, 2):
            item = data[(seed, stage)]
            verify_oof(item, f"seed{seed} stage{stage}")
            for fold in range(5):
                ids, predictions = source_predictions(seed, stage, fold)
                mask = item["fold"] == fold
                ok = np.array_equal(ids, item["sample_id"][mask]) and np.array_equal(predictions, item["prediction"][mask])
                checks.append({"check": f"seed{seed}_stage{stage}_fold{fold}_prediction_source", "status": "PASS" if ok else "FAIL"})

    reference_ids = data[(42, 1)]["sample_id"]
    reference_labels = data[(42, 1)]["label"]
    reference_folds = data[(42, 1)]["fold"]
    for key, item in data.items():
        aligned = (
            np.array_equal(item["sample_id"], reference_ids)
            and np.array_equal(item["label"], reference_labels)
            and np.array_equal(item["fold"], reference_folds)
        )
        checks.append({"check": f"seed{key[0]}_stage{key[1]}_alignment", "status": "PASS" if aligned else "FAIL"})

    checkpoint_hashes: dict[str, str] = {}
    for stage in (1, 2):
        for fold in range(5):
            hashes = []
            for seed in SEEDS:
                path, expected = checkpoint(seed, stage, fold)
                actual = sha256(path)
                hashes.append(actual)
                checkpoint_hashes[f"seed{seed}_stage{stage}_fold{fold}"] = actual
                checks.append({"check": f"seed{seed}_stage{stage}_fold{fold}_checkpoint_hash", "status": "PASS" if actual == expected else "FAIL"})
            distinct = len(set(hashes)) == len(SEEDS)
            checks.append({"check": f"stage{stage}_fold{fold}_checkpoint_hashes_distinct", "status": "PASS" if distinct else "FAIL"})

    rows = []
    stop_reasons = []
    for stage in (1, 2):
        for seed_a, seed_b in combinations(SEEDS, 2):
            a, b = data[(seed_a, stage)], data[(seed_b, stage)]
            y = reference_labels
            prediction_disagreement = a["prediction"] != b["prediction"]
            correct_a = a["prediction"] == y
            correct_b = b["prediction"] == y
            wrong_a = ~correct_a
            wrong_b = ~correct_b
            n10 = int(np.sum(correct_a & wrong_b))
            n01 = int(np.sum(wrong_a & correct_b))
            both_wrong = int(np.sum(wrong_a & wrong_b))
            union_wrong = int(np.sum(wrong_a | wrong_b))
            count = int(np.sum(prediction_disagreement))
            percentage = count * 100.0 / N
            if stage == 2 and (count < 10 or percentage < 0.1):
                stop_reasons.append(f"seed{seed_a}_vs_seed{seed_b}: disagreement={count} ({percentage:.6f}%)")
            rows.append({
                "stage": stage,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "n": N,
                "accuracy_a": float(np.mean(correct_a)),
                "accuracy_b": float(np.mean(correct_b)),
                "prediction_disagreement_count": count,
                "prediction_disagreement_percentage": percentage,
                "n10_a_correct_b_wrong": n10,
                "n01_a_wrong_b_correct": n01,
                "correctness_discordant_total": n10 + n01,
                "both_wrong_count": both_wrong,
                "a_only_wrong": n01,
                "b_only_wrong": n10,
                "error_set_jaccard": both_wrong / union_wrong if union_wrong else 1.0,
            })

    failed_checks = [check for check in checks if check["status"] != "PASS"]
    if failed_checks:
        stop_reasons.append(f"integrity/provenance checks failed: {len(failed_checks)}")
    write_csv(output_csv, rows)

    lines = [
        "# Baseline three-seed prediction audit",
        "",
        f"Status: {'STOP' if stop_reasons else 'PASS — Round2A training may proceed' }.",
        "",
        "Prediction disagreement and correctness discordants are reported as distinct quantities.",
        "",
        "| Stage | Seed pair | Prediction disagreement | Disagreement % | n10 | n01 | Both wrong | Error Jaccard |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['seed_a']} vs {row['seed_b']} | "
            f"{row['prediction_disagreement_count']} | {row['prediction_disagreement_percentage']:.4f}% | "
            f"{row['n10_a_correct_b_wrong']} | {row['n01_a_wrong_b_correct']} | "
            f"{row['both_wrong_count']} | {row['error_set_jaccard']:.6f} |"
        )
    lines += [
        "",
        "If Stage2 performance variance contracts while disagreement remains material, this supports only the phrase `performance-level convergence`, not solution convergence.",
    ]
    if stop_reasons:
        lines += ["", "## STOP reasons", ""] + [f"- {reason}" for reason in stop_reasons]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "status": "STOP" if stop_reasons else "PASS",
        "training_authorized_by_step1": not stop_reasons,
        "samples": N,
        "seeds": list(SEEDS),
        "stages": [1, 2],
        "checks_total": len(checks),
        "checks_failed": len(failed_checks),
        "stop_threshold": {"minimum_disagreement_count": 10, "minimum_disagreement_percentage": 0.1},
        "stop_reasons": stop_reasons,
        "checkpoint_sha256": checkpoint_hashes,
        "csv": str(output_csv),
        "markdown": str(output_md),
    }
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": len(failed_checks), "stop_reasons": stop_reasons}, ensure_ascii=False))
    if stop_reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
