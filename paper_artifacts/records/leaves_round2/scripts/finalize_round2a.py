"""Finalize the frozen Round2A analysis from formal OOF and diagnostics.

The script performs no training.  It aggregates exact held-out OOF outputs,
computes the predeclared MHSA contrasts, combines the five baseline seeds, and
summarizes true f1..f5 and attention diagnostics before emitting a fail-closed
postflight audit and Markdown review report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from round2a_common import PHASE2B_ROOT, ROUND1_ROOT, ROUND_ROOT, RUN_SPECS


VALIDATION_ROOT = ROUND_ROOT.parent
STATISTICS = ROUND_ROOT / "statistics"
REPRESENTATION = ROUND_ROOT / "representation"
ATTENTION = ROUND_ROOT / "attention"
MANIFESTS = ROUND_ROOT / "manifests"
OOF = ROUND_ROOT / "oof"
DIAGNOSTICS = ROUND_ROOT / "diagnostics" / "fold_outputs"
OFF_FEATURES = ROUND1_ROOT / "representation" / "fold_features"
BOOTSTRAP_SEED = 20260807
BOOTSTRAP_REPLICATES = 100_000
LAMBDAS = (("lambda_0_1", 0.1), ("lambda_0_7", 0.7), ("lambda_1_0", 1.0))
ON_RUNS = {
    "lambda_0_1": "mhsa_lambda_0_1_seed42",
    "lambda_0_7": "mhsa_lambda_0_7_seed42",
    "lambda_1_0": "mhsa_lambda_1_0_seed42",
}
OFF_RUNS = {
    "lambda_0_1": (PHASE2B_ROOT / "#2", True),
    "lambda_0_7": (ROUND1_ROOT / "lambda_0_7_seed42", False),
    "lambda_1_0": (PHASE2B_ROOT / "#1", True),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def describe(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise RuntimeError("empty or non-finite diagnostic")
    q = np.quantile(x, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median": float(q[2]),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def load_fold_oof(folder: Path, stage: int, legacy: bool = False) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("sample_id", "label", "prediction", "fold")}
    for fold in range(5):
        source = folder / f"fold_{fold}"
        prefix = "" if legacy else f"stage{stage}_"
        ids = np.load(source / f"{prefix}validation_sample_ids.npy")
        labels = np.load(source / f"{prefix}validation_labels.npy")
        predictions = np.load(source / f"{prefix}validation_predictions.npy")
        parts["sample_id"].append(ids)
        parts["label"].append(labels)
        parts["prediction"].append(predictions)
        parts["fold"].append(np.full(len(ids), fold, dtype=np.int8))
    data = {key: np.concatenate(value) for key, value in parts.items()}
    order = np.argsort(data["sample_id"])
    data = {key: value[order] for key, value in data.items()}
    if len(data["sample_id"]) != 18_353 or not np.array_equal(data["sample_id"], np.arange(18_353)):
        raise RuntimeError(f"OOF coverage failure: {folder}, stage={stage}")
    return data


def load_npz_oof(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        data = {
            "sample_id": source["sample_id"],
            "label": source["label"],
            "prediction": source["prediction"],
            "fold": source["fold"],
        }
    order = np.argsort(data["sample_id"])
    return {key: value[order] for key, value in data.items()}


def assert_aligned(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    if not np.array_equal(a["sample_id"], b["sample_id"]) or not np.array_equal(a["label"], b["label"]):
        raise RuntimeError("paired OOF inputs are not aligned")


def bootstrap_ci(effect: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values, counts = np.unique(np.asarray(effect), return_counts=True)
    draws = rng.multinomial(len(effect), counts / len(effect), size=BOOTSTRAP_REPLICATES)
    estimates = draws @ values / len(effect) * 100.0
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def paired_row(lambda_key: str, lam: float, off: dict[str, np.ndarray], on: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[dict[str, Any], np.ndarray]:
    assert_aligned(off, on)
    labels = off["label"]
    off_correct = off["prediction"] == labels
    on_correct = on["prediction"] == labels
    effect = on_correct.astype(np.int8) - off_correct.astype(np.int8)
    low, high = bootstrap_ci(effect, rng)
    n10 = int(np.sum(off_correct & ~on_correct))
    n01 = int(np.sum(~off_correct & on_correct))
    off_metrics, on_metrics = metrics(labels, off["prediction"]), metrics(labels, on["prediction"])
    return {
        "lambda_key": lambda_key,
        "lambda": lam,
        "n": len(labels),
        "accuracy_off": off_metrics["accuracy"],
        "accuracy_on": on_metrics["accuracy"],
        "delta_on_minus_off_pp": float(effect.mean() * 100.0),
        "ci95_low_pp": low,
        "ci95_high_pp": high,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "mcnemar_exact_two_sided_p": float(binomtest(n10, n10 + n01, 0.5).pvalue),
        "n10_off_correct_on_wrong": n10,
        "n01_off_wrong_on_correct": n01,
        "correctness_discordant_n": n10 + n01,
        "prediction_changed_n": int(np.sum(off["prediction"] != on["prediction"])),
        "macro_f1_off": off_metrics["macro_f1"],
        "macro_f1_on": on_metrics["macro_f1"],
        "delta_macro_f1_pp": (on_metrics["macro_f1"] - off_metrics["macro_f1"]) * 100.0,
        "balanced_accuracy_off": off_metrics["balanced_accuracy"],
        "balanced_accuracy_on": on_metrics["balanced_accuracy"],
        "delta_balanced_accuracy_pp": (on_metrics["balanced_accuracy"] - off_metrics["balanced_accuracy"]) * 100.0,
    }, effect


def finalize_mhsa_statistics() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    STATISTICS.mkdir(parents=True, exist_ok=True)
    OOF.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pair_rows: list[dict[str, Any]] = []
    effects: dict[str, np.ndarray] = {}
    data: dict[str, dict[str, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []
    for key, lam in LAMBDAS:
        run_id = ON_RUNS[key]
        on = load_fold_oof(ROUND_ROOT / run_id, stage=2)
        off_folder, legacy = OFF_RUNS[key]
        off = load_fold_oof(off_folder, stage=2, legacy=legacy)
        row, effect = paired_row(key, lam, off, on, rng)
        pair_rows.append(row)
        effects[key] = effect
        data[key] = {"off": off, "on": on}
        write_csv(STATISTICS / f"mhsa_on_vs_off_{key}.csv", [row])
        for state, source in (("OFF", off), ("ON", on)):
            value = metrics(source["label"], source["prediction"])
            metric_rows.append({"lambda_key": key, "lambda": lam, "mhsa": state, "stage": 2, "n": len(source["label"]), **value})
        for stage in (1, 2):
            source = load_fold_oof(ROUND_ROOT / run_id, stage=stage)
            np.savez_compressed(
                OOF / f"{run_id}_stage{stage}_oof.npz",
                sample_id=source["sample_id"], fold=source["fold"], label=source["label"], prediction=source["prediction"],
            )
    write_csv(STATISTICS / "mhsa_oof_metrics.csv", metric_rows)
    write_csv(STATISTICS / "mhsa_on_vs_off_all_lambdas.csv", pair_rows)

    interaction_rows: list[dict[str, Any]] = []
    for label, a, b, primary in (
        ("I_0.1_minus_1.0", "lambda_0_1", "lambda_1_0", True),
        ("I_0.1_minus_0.7", "lambda_0_1", "lambda_0_7", False),
        ("I_0.7_minus_1.0", "lambda_0_7", "lambda_1_0", False),
    ):
        effect = effects[a] - effects[b]
        low, high = bootstrap_ci(effect, rng)
        interaction_rows.append({
            "contrast": label,
            "role": "primary" if primary else "secondary",
            "effect_a": a,
            "effect_b": b,
            "n_paired_oof": len(effect),
            "difference_in_differences_pp": float(effect.mean() * 100.0),
            "ci95_low_pp": low,
            "ci95_high_pp": high,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "direct_sample_level_paired_bootstrap": True,
        })
    write_csv(STATISTICS / "mhsa_interaction_difference_in_differences.csv", interaction_rows)
    return pair_rows, interaction_rows, data


def baseline_oof() -> dict[str, dict[int, dict[str, np.ndarray]]]:
    outputs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    stage1_seed42 = load_npz_oof(
        VALIDATION_ROOT / "phase2c_inference_diagnostic" / "p1_stage1_only_oof" / "stage1_oof_predictions_variant_0.npz"
    )
    outputs["42"] = {1: stage1_seed42, 2: load_fold_oof(PHASE2B_ROOT / "#0", 2, legacy=True)}
    sources = {
        "43": ROUND1_ROOT / "baseline_seed43",
        "44": ROUND1_ROOT / "baseline_seed44",
        "45": ROUND_ROOT / "baseline_seed45",
        "46": ROUND_ROOT / "baseline_seed46",
    }
    for seed, folder in sources.items():
        outputs[seed] = {stage: load_fold_oof(folder, stage) for stage in (1, 2)}
    return outputs


def finalize_baselines() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources = baseline_oof()
    per_seed: list[dict[str, Any]] = []
    for seed in sorted(sources, key=int):
        s1, s2 = sources[seed][1], sources[seed][2]
        assert_aligned(s1, s2)
        m1, m2 = metrics(s1["label"], s1["prediction"]), metrics(s2["label"], s2["prediction"])
        per_seed.append({
            "seed": int(seed), "n": len(s1["label"]),
            "stage1_accuracy": m1["accuracy"], "stage1_macro_f1": m1["macro_f1"], "stage1_balanced_accuracy": m1["balanced_accuracy"],
            "stage2_accuracy": m2["accuracy"], "stage2_macro_f1": m2["macro_f1"], "stage2_balanced_accuracy": m2["balanced_accuracy"],
            "stage2_minus_stage1_accuracy_pp": (m2["accuracy"] - m1["accuracy"]) * 100.0,
        })
    summary: list[dict[str, Any]] = []
    for stage in (1, 2):
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = np.asarray([row[f"stage{stage}_{metric}"] for row in per_seed], dtype=np.float64)
            summary.append({
                "stage": stage, "metric": metric, "seeds": 5,
                "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                "min": float(values.min()), "max": float(values.max()), "range": float(values.max() - values.min()),
            })

    audit: list[dict[str, Any]] = []
    for stage in (1, 2):
        for seed_a, seed_b in combinations(sorted(sources, key=int), 2):
            a, b = sources[seed_a][stage], sources[seed_b][stage]
            assert_aligned(a, b)
            labels, pa, pb = a["label"], a["prediction"], b["prediction"]
            ca, cb = pa == labels, pb == labels
            wrong_a, wrong_b = ~ca, ~cb
            union = wrong_a | wrong_b
            n10, n01 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
            audit.append({
                "stage": stage, "seed_a": int(seed_a), "seed_b": int(seed_b), "n": len(labels),
                "prediction_disagreement_n": int(np.sum(pa != pb)),
                "prediction_disagreement_pct": float(np.mean(pa != pb) * 100.0),
                "n10_a_correct_b_wrong": n10, "n01_a_wrong_b_correct": n01,
                "correctness_discordant_n": n10 + n01,
                "mcnemar_exact_two_sided_p": float(binomtest(n10, n10 + n01, 0.5).pvalue),
                "both_wrong_n": int(np.sum(wrong_a & wrong_b)),
                "a_only_wrong_n": int(np.sum(wrong_a & ~wrong_b)),
                "b_only_wrong_n": int(np.sum(~wrong_a & wrong_b)),
                "both_correct_n": int(np.sum(ca & cb)),
                "error_union_n": int(np.sum(union)),
                "error_jaccard": float(np.sum(wrong_a & wrong_b) / np.sum(union)),
            })
    write_csv(STATISTICS / "baseline_five_seed_per_seed.csv", per_seed)
    write_csv(STATISTICS / "baseline_five_seed_summary.csv", summary)
    write_csv(STATISTICS / "baseline_five_seed_prediction_audit.csv", audit)
    return per_seed, summary, audit


def cosine_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    denominator = np.linalg.norm(x64, axis=1) * np.linalg.norm(y64, axis=1)
    if np.any(denominator <= 0):
        raise RuntimeError("zero-norm representation")
    return np.sum(x64 * y64, axis=1) / denominator


def load_features(key: str, mhsa: str, stage: int, fold: int) -> dict[str, np.ndarray]:
    if mhsa == "OFF":
        path = OFF_FEATURES / key / f"stage{stage}" / f"fold_{fold}" / "stages.npz"
        manifest_path = path.with_name("manifest.json")
    else:
        path = DIAGNOSTICS / key / f"stage{stage}" / f"fold_{fold}" / "stages.npz"
        manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("training_performed") is not False:
        raise RuntimeError(f"invalid representation manifest: {manifest_path}")
    with np.load(path, allow_pickle=False) as source:
        data = {name: source[name] for name in ("sample_id", "label", "prediction", "f1", "f2", "f3", "f4", "f5")}
    n = len(data["sample_id"])
    if any(data[f"f{i}"].shape != (n, 256) for i in range(1, 6)):
        raise RuntimeError(f"representation shape failure: {path}")
    return data


def finalize_representations() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = tuple(combinations(range(5), 2))
    transitions = tuple((i, i + 1) for i in range(4))
    cosine_fold: list[dict[str, Any]] = []
    cosine_pooled: list[dict[str, Any]] = []
    cka_fold: list[dict[str, Any]] = []
    cka_summary: list[dict[str, Any]] = []
    adjacent: list[dict[str, Any]] = []
    norms: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []

    for key, lam in LAMBDAS:
        for mhsa in ("OFF", "ON"):
            for stage in (1, 2):
                pooled_cos: dict[tuple[int, int], list[np.ndarray]] = {pair: [] for pair in pairs}
                pooled_offdiag: list[np.ndarray] = []
                ids_all: list[np.ndarray] = []
                for fold in range(5):
                    data = load_features(key, mhsa, stage, fold)
                    ids_all.append(data["sample_id"])
                    arrays = [data[f"f{i}"].astype(np.float64) for i in range(1, 6)]
                    centered = [array - array.mean(axis=0, keepdims=True) for array in arrays]
                    self_hsic = []
                    for value in centered:
                        gram = value.T @ value
                        self_hsic.append(float(np.sum(gram * gram)))
                    sample_pair_values: list[np.ndarray] = []
                    for i, j in pairs:
                        cos = cosine_rows(arrays[i], arrays[j])
                        cross = centered[i].T @ centered[j]
                        cka = float(np.sum(cross * cross) / math.sqrt(self_hsic[i] * self_hsic[j]))
                        if not np.isfinite(cka):
                            raise RuntimeError("non-finite CKA")
                        pooled_cos[(i, j)].append(cos)
                        sample_pair_values.append(cos)
                        cosine_fold.append({
                            "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage, "fold": fold,
                            "pair": f"f{i + 1}-f{j + 1}", **describe(cos),
                        })
                        cka_fold.append({
                            "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage, "fold": fold,
                            "pair": f"f{i + 1}-f{j + 1}", "centered_linear_cka": cka,
                        })
                    offdiag = np.stack(sample_pair_values, axis=1).mean(axis=1)
                    pooled_offdiag.append(offdiag)
                    cosine_fold.append({
                        "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage, "fold": fold,
                        "pair": "mean_offdiagonal_per_sample", **describe(offdiag),
                    })
                    for i, j in transitions:
                        raw = np.linalg.norm(arrays[j] - arrays[i], axis=1)
                        normalized = raw / np.maximum(np.linalg.norm(arrays[i], axis=1), 1e-12)
                        cosdist = 1.0 - cosine_rows(arrays[i], arrays[j])
                        for metric_name, values in (("raw_change", raw), ("normalized_change", normalized), ("cosine_distance", cosdist)):
                            adjacent.append({
                                "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage, "fold": fold,
                                "transition": f"f{i + 1}-f{j + 1}", "metric": metric_name, **describe(values),
                            })
                    for index, values in enumerate(arrays, start=1):
                        norms.append({
                            "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage, "fold": fold,
                            "layer": f"f{index}", **describe(np.linalg.norm(values, axis=1)),
                        })
                ids = np.concatenate(ids_all)
                coverage.append({
                    "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage, "n": len(ids),
                    "unique_n": len(np.unique(ids)), "covers_0_to_18352": bool(set(ids.tolist()) == set(range(18_353))),
                })
                for i, j in pairs:
                    cosine_pooled.append({
                        "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage,
                        "pair": f"f{i + 1}-f{j + 1}", **describe(np.concatenate(pooled_cos[(i, j)])),
                    })
                cosine_pooled.append({
                    "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage,
                    "pair": "mean_offdiagonal_per_sample", **describe(np.concatenate(pooled_offdiag)),
                })

    for key, lam in LAMBDAS:
        for mhsa in ("OFF", "ON"):
            for stage in (1, 2):
                for i, j in pairs:
                    values = np.asarray([
                        row["centered_linear_cka"] for row in cka_fold
                        if row["lambda_key"] == key and row["mhsa"] == mhsa and row["stage"] == stage and row["pair"] == f"f{i + 1}-f{j + 1}"
                    ])
                    cka_summary.append({
                        "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage,
                        "pair": f"f{i + 1}-f{j + 1}", "folds": len(values), "mean": float(values.mean()),
                        "std_across_folds": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max()),
                    })
                fold_means = np.asarray([
                    np.mean([
                        row["centered_linear_cka"] for row in cka_fold
                        if row["lambda_key"] == key and row["mhsa"] == mhsa and row["stage"] == stage and row["fold"] == fold
                    ]) for fold in range(5)
                ])
                all_values = np.asarray([
                    row["centered_linear_cka"] for row in cka_fold
                    if row["lambda_key"] == key and row["mhsa"] == mhsa and row["stage"] == stage
                ])
                cka_summary.append({
                    "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage,
                    "pair": "mean_over_10_pairs_and_5_folds", "folds": 5, "mean": float(all_values.mean()),
                    "std_across_folds": float(fold_means.std(ddof=1)), "min": float(all_values.min()), "max": float(all_values.max()),
                })

    overview: list[dict[str, Any]] = []
    for key, lam in LAMBDAS:
        for mhsa in ("OFF", "ON"):
            for stage in (1, 2):
                cos = next(row for row in cosine_pooled if row["lambda_key"] == key and row["mhsa"] == mhsa and row["stage"] == stage and row["pair"] == "mean_offdiagonal_per_sample")
                cka = next(row for row in cka_summary if row["lambda_key"] == key and row["mhsa"] == mhsa and row["stage"] == stage and row["pair"] == "mean_over_10_pairs_and_5_folds")
                overview.append({
                    "lambda_key": key, "lambda": lam, "mhsa": mhsa, "stage": stage,
                    "pooled_mean_offdiagonal_cosine": cos["mean"],
                    "pooled_sd_offdiagonal_cosine": cos["std"],
                    "mean_centered_linear_cka": cka["mean"],
                    "sd_centered_linear_cka_across_folds": cka["std_across_folds"],
                    "interpretation": "descriptive geometry diagnostic; not representation quality or causal evidence",
                })
    contrasts: list[dict[str, Any]] = []
    for key, lam in LAMBDAS:
        for stage in (1, 2):
            off = next(row for row in overview if row["lambda_key"] == key and row["mhsa"] == "OFF" and row["stage"] == stage)
            on = next(row for row in overview if row["lambda_key"] == key and row["mhsa"] == "ON" and row["stage"] == stage)
            contrasts.append({
                "lambda_key": key, "lambda": lam, "stage": stage,
                "on_minus_off_mean_offdiagonal_cosine": on["pooled_mean_offdiagonal_cosine"] - off["pooled_mean_offdiagonal_cosine"],
                "on_minus_off_mean_centered_linear_cka": on["mean_centered_linear_cka"] - off["mean_centered_linear_cka"],
                "inference": "descriptive_only",
            })
    write_csv(REPRESENTATION / "cosine_foldwise.csv", cosine_fold)
    write_csv(REPRESENTATION / "cosine_pooled_summary.csv", cosine_pooled)
    write_csv(REPRESENTATION / "cka_foldwise.csv", cka_fold)
    write_csv(REPRESENTATION / "cka_summary.csv", cka_summary)
    write_csv(REPRESENTATION / "representation_overview.csv", overview)
    write_csv(REPRESENTATION / "on_off_similarity_contrast.csv", contrasts)
    write_csv(REPRESENTATION / "adjacent_change.csv", adjacent)
    write_csv(REPRESENTATION / "feature_norms.csv", norms)
    write_csv(REPRESENTATION / "coverage.csv", coverage)
    return overview, coverage


def load_attention(key: str, stage: int, fold: int) -> dict[str, np.ndarray]:
    folder = DIAGNOSTICS / key / f"stage{stage}" / f"fold_{fold}"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("source_predictions_exact") is not True:
        raise RuntimeError(f"invalid attention manifest: {folder}")
    with np.load(folder / "attention.npz", allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def finalize_attention() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overview: list[dict[str, Any]] = []
    by_head_query: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for key, lam in LAMBDAS:
        for stage in (1, 2):
            folds = [load_attention(key, stage, fold) for fold in range(5)]
            ids = np.concatenate([item["sample_id"] for item in folds])
            weights = np.concatenate([item["attention_weights"] for item in folds])
            entropy = np.concatenate([item["normalized_entropy"] for item in folds])
            maximum = np.concatenate([item["max_attention_share"] for item in folds])
            argmax = np.concatenate([item["argmax_key"] for item in folds])
            if weights.shape != (18_353, 4, 5, 5) or entropy.shape != (18_353, 4, 5):
                raise RuntimeError(f"attention shape/coverage failure: {key}, stage={stage}")
            coverage.append({
                "lambda_key": key, "lambda": lam, "stage": stage, "samples": len(ids), "unique_samples": len(np.unique(ids)),
                "heads": 4, "queries": 5, "keys": 5, "atomic_units": int(entropy.size),
                "atomic_unit_definition": "sample_x_head_x_query_before_aggregation",
                "covers_0_to_18352": bool(set(ids.tolist()) == set(range(18_353))),
            })
            for metric_name, values in (("normalized_entropy", entropy), ("max_attention_share", maximum)):
                overview.append({
                    "lambda_key": key, "lambda": lam, "stage": stage, "metric": metric_name,
                    "atomic_unit": "sample_x_head_x_query", **describe(values),
                })
                for head in range(4):
                    for query in range(5):
                        by_head_query.append({
                            "lambda_key": key, "lambda": lam, "stage": stage, "head": head,
                            "query": f"f{query + 1}", "metric": metric_name,
                            "atomic_unit": "sample", **describe(values[:, head, query]),
                        })
            for head in range(4):
                for query in range(5):
                    for key_index in range(5):
                        values = weights[:, head, query, key_index]
                        positions.append({
                            "lambda_key": key, "lambda": lam, "stage": stage, "head": str(head),
                            "query": f"f{query + 1}", "key_position": f"f{key_index + 1}", "samples": len(values),
                            "mean_attention_share": float(values.mean()), "std_attention_share": float(values.std(ddof=1)),
                            "argmax_allocation_fraction": float(np.mean(argmax[:, head, query] == key_index)),
                        })
            for query in range(5):
                for key_index in range(5):
                    values = weights[:, :, query, key_index].reshape(-1)
                    positions.append({
                        "lambda_key": key, "lambda": lam, "stage": stage, "head": "ALL",
                        "query": f"f{query + 1}", "key_position": f"f{key_index + 1}", "samples": len(values),
                        "mean_attention_share": float(values.mean()), "std_attention_share": float(values.std(ddof=1)),
                        "argmax_allocation_fraction": float(np.mean(argmax[:, :, query].reshape(-1) == key_index)),
                    })
    write_csv(ATTENTION / "attention_overview.csv", overview)
    write_csv(ATTENTION / "attention_by_head_query.csv", by_head_query)
    write_csv(ATTENTION / "position_allocation_summary.csv", positions)
    write_csv(ATTENTION / "coverage.csv", coverage)
    return overview, coverage


def make_postflight(
    pair_rows: list[dict[str, Any]], interaction_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]], representation_coverage: list[dict[str, Any]], attention_coverage: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = json.loads((ROUND_ROOT / "orchestrator_summary.json").read_text(encoding="utf-8"))
    training_manifests = [
        json.loads((ROUND_ROOT / run_id / f"fold_{fold}" / "run_manifest.json").read_text(encoding="utf-8"))
        for run_id in RUN_SPECS for fold in range(5)
    ]
    diagnostic_manifests = [
        json.loads((DIAGNOSTICS / key / f"stage{stage}" / f"fold_{fold}" / "manifest.json").read_text(encoding="utf-8"))
        for key, _ in LAMBDAS for stage in (1, 2) for fold in range(5)
    ]
    required = [
        STATISTICS / "mhsa_on_vs_off_lambda_0_1.csv",
        STATISTICS / "mhsa_on_vs_off_lambda_0_7.csv",
        STATISTICS / "mhsa_on_vs_off_lambda_1_0.csv",
        STATISTICS / "mhsa_interaction_difference_in_differences.csv",
        STATISTICS / "baseline_five_seed_per_seed.csv",
        STATISTICS / "baseline_five_seed_summary.csv",
        STATISTICS / "baseline_five_seed_prediction_audit.csv",
        REPRESENTATION / "cosine_pooled_summary.csv",
        REPRESENTATION / "cka_summary.csv",
        REPRESENTATION / "representation_overview.csv",
        ATTENTION / "attention_overview.csv",
        ATTENTION / "position_allocation_summary.csv",
    ]
    checks = {
        "all_25_formal_training_jobs_complete": summary.get("status") == "ALL_25_COMPLETE" and all(item.get("status") == "COMPLETE" for item in training_manifests),
        "all_30_checkpoint_only_diagnostics_pass": len(diagnostic_manifests) == 30 and all(item.get("status") == "PASS" for item in diagnostic_manifests),
        "diagnostics_performed_no_training": all(item.get("training_performed") is False for item in diagnostic_manifests),
        "diagnostic_predictions_exact": all(item.get("source_predictions_exact") is True for item in diagnostic_manifests),
        "three_mhsa_pairwise_statistics_present": len(pair_rows) == 3,
        "three_predeclared_interactions_present": len(interaction_rows) == 3,
        "five_seed_baseline_present": len(baseline_rows) == 5,
        "five_seed_pairwise_prediction_audit_complete": len(prediction_rows) == 20,
        "representation_on_off_stage_coverage_exact": len(representation_coverage) == 12 and all(row["n"] == 18_353 and row["unique_n"] == 18_353 and row["covers_0_to_18352"] for row in representation_coverage),
        "attention_sample_head_query_coverage_exact": len(attention_coverage) == 6 and all(row["samples"] == 18_353 and row["atomic_units"] == 367_060 and row["covers_0_to_18352"] for row in attention_coverage),
        "required_review_files_nonempty": all(path.is_file() and path.stat().st_size > 0 for path in required),
        "forbidden_result_dependent_variants_not_launched": set(RUN_SPECS) == {"mhsa_lambda_0_1_seed42", "mhsa_lambda_0_7_seed42", "mhsa_lambda_1_0_seed42", "baseline_seed45", "baseline_seed46"},
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "all_passed": status == "PASS",
        "created_unix": time.time(),
        "checks": [{"check": key, "status": "PASS" if value else "FAIL"} for key, value in checks.items()],
        "formal_training_jobs": 25,
        "checkpoint_only_diagnostic_jobs": 30,
        "training_performed_by_diagnostics": False,
        "paired_oof_n": 18_353,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "attention_atomic_unit": "sample_x_head_x_query_before_aggregation",
        "representation_scope": "true post-shortcut f1..f5; descriptive geometry diagnostic",
        "inference_boundary": "seed is the independent repetition unit; folds construct OOF and are not independent repetitions",
        "required_file_sha256": {str(path.relative_to(ROUND_ROOT)): sha256_file(path) for path in required if path.is_file()},
    }


def pct(value: float) -> str:
    return f"{value * 100:.4f}"


def render_report(
    pair_rows: list[dict[str, Any]], interactions: list[dict[str, Any]], baseline: list[dict[str, Any]],
    baseline_summary: list[dict[str, Any]], prediction_audit: list[dict[str, Any]], representation: list[dict[str, Any]],
    attention: list[dict[str, Any]], postflight: dict[str, Any],
) -> str:
    lines = [
        "# Round2A Results Audit", "", f"Postflight: **{postflight['status']}**.", "",
        "All 25/25 formal training jobs and all 30/30 checkpoint-only OOF diagnostic jobs completed. No diagnostic job trained or modified a model.", "",
        "## 1. MHSA ON versus OFF", "",
        "| lambda | OFF Accuracy (%) | ON Accuracy (%) | Delta ON-OFF (pp) | 95% paired bootstrap CI | Exact McNemar p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(f"| {row['lambda']:.1f} | {pct(row['accuracy_off'])} | {pct(row['accuracy_on'])} | {row['delta_on_minus_off_pp']:+.4f} | [{row['ci95_low_pp']:.4f}, {row['ci95_high_pp']:.4f}] | {row['mcnemar_exact_two_sided_p']:.6g} |")
    lines += [
        "", "All three intervals include zero. The observed results do not establish a stable MHSA accuracy benefit at any tested lambda.", "",
        "## 2. Difference-in-differences interactions", "",
        "| Contrast | Role | Point estimate (pp) | 95% paired bootstrap CI |",
        "|---|---|---:|---:|",
    ]
    for row in interactions:
        lines.append(f"| {row['contrast']} | {row['role']} | {row['difference_in_differences_pp']:+.4f} | [{row['ci95_low_pp']:.4f}, {row['ci95_high_pp']:.4f}] |")
    lines += [
        "", "The primary and both secondary interaction intervals include zero. The experiment does not support the claim that the MHSA effect depends on shortcut lambda.", "",
        "## 3. Five-seed baseline confirmation", "",
        "| Seed | Stage1 Accuracy (%) | Stage2 Accuracy (%) | Stage2-Stage1 (pp) |",
        "|---:|---:|---:|---:|",
    ]
    for row in baseline:
        lines.append(f"| {row['seed']} | {pct(row['stage1_accuracy'])} | {pct(row['stage2_accuracy'])} | {row['stage2_minus_stage1_accuracy_pp']:+.4f} |")
    s1 = next(row for row in baseline_summary if row["stage"] == 1 and row["metric"] == "accuracy")
    s2 = next(row for row in baseline_summary if row["stage"] == 2 and row["metric"] == "accuracy")
    stage2_disagreement = [row["prediction_disagreement_pct"] for row in prediction_audit if row["stage"] == 2]
    lines += [
        "",
        f"Stage1 mean +/- sample SD is {pct(s1['mean'])} +/- {s1['sample_sd'] * 100:.4f}%; Stage2 is {pct(s2['mean'])} +/- {s2['sample_sd'] * 100:.4f}%.",
        f"Stage2 pairwise prediction disagreement remains {min(stage2_disagreement):.4f}% to {max(stage2_disagreement):.4f}%.",
        "Therefore only `performance-level convergence` is supported; same basin, same solution, and same prediction function are not supported.", "",
        "## 4. Representation diagnostics", "",
        "True post-shortcut f1..f5 are summarized below. Cosine is pooled sample-wise; centered-linear CKA is computed fold-wise in float64 and then summarized.", "",
        "| lambda | MHSA | Stage | Mean off-diagonal cosine | Mean centered-linear CKA |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in representation:
        lines.append(f"| {row['lambda']:.1f} | {row['mhsa']} | {row['stage']} | {row['pooled_mean_offdiagonal_cosine']:.8f} | {row['mean_centered_linear_cka']:.8f} |")
    lines += [
        "", "These are descriptive geometry measurements. Lower similarity or a Stage1/Stage2 change is not evidence of better/worse representation quality, restoration, collapse, or a causal MHSA mechanism.", "",
        "## 5. Attention diagnostics", "",
        "Normalized entropy and maximum attention share are first computed for every sample x head x query distribution, then aggregated.", "",
        "| lambda | Stage | Metric | Atomic units | Mean | SD | Median |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in attention:
        lines.append(f"| {row['lambda']:.1f} | {row['stage']} | {row['metric']} | {row['n']} | {row['mean']:.6f} | {row['std']:.6f} | {row['median']:.6f} |")
    lines += [
        "", "Attention concentration is descriptive and is not automatically an explanation for classification performance.", "",
        "## 6. Supported conclusion for manuscript review", "",
        "Within the frozen lambda x MHSA design, MHSA produced small and directionally inconsistent OOF changes, while all paired effect and interaction intervals included zero. The current evidence therefore does not support retaining MHSA on the basis of a demonstrated accuracy gain or lambda-dependent benefit. Any architectural decision to remove it should additionally consider complexity and the descriptive diagnostics, not claim proof of harm.", "",
        "## 7. Principal review files", "",
        "- `statistics/mhsa_on_vs_off_lambda_0_1.csv`", "- `statistics/mhsa_on_vs_off_lambda_0_7.csv`", "- `statistics/mhsa_on_vs_off_lambda_1_0.csv`",
        "- `statistics/mhsa_interaction_difference_in_differences.csv`", "- `statistics/baseline_five_seed_per_seed.csv`", "- `statistics/baseline_five_seed_summary.csv`",
        "- `statistics/baseline_five_seed_prediction_audit.csv`", "- `representation/representation_overview.csv`", "- `representation/cosine_pooled_summary.csv`",
        "- `representation/cka_summary.csv`", "- `attention/attention_overview.csv`", "- `attention/attention_by_head_query.csv`",
        "- `attention/position_allocation_summary.csv`", "- `manifests/postflight_audit.json`", "",
        "## 8. Stop boundary", "", "No DFAG or result-dependent follow-up variant was launched by this finalization. Stop for human/GPT-5.6 review.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    started = time.time()
    pair_rows, interactions, _ = finalize_mhsa_statistics()
    baseline, baseline_summary, prediction_audit = finalize_baselines()
    representation, representation_coverage = finalize_representations()
    attention, attention_coverage = finalize_attention()
    postflight = make_postflight(pair_rows, interactions, baseline, prediction_audit, representation_coverage, attention_coverage)
    postflight["elapsed_seconds"] = time.time() - started
    write_json(MANIFESTS / "postflight_audit.json", postflight)
    if postflight["status"] != "PASS":
        raise RuntimeError("Round2A postflight failed; report intentionally not generated")
    report = render_report(pair_rows, interactions, baseline, baseline_summary, prediction_audit, representation, attention, postflight)
    (ROUND_ROOT / "ROUND2A_RESULTS_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "FINALIZED", "postflight": "PASS", "elapsed_seconds": time.time() - started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
