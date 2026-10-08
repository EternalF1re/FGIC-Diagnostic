"""Compute final sample-level paired OOF statistics for the frozen protocol."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
EXPECTED_CONFIG_SHA256 = "7f667c125983ca82ec36214b727ee8fbe3215f21562abd260f4018559f87e3e5"
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_RESAMPLES = 100_000

CONFIG_PATH = ROOT / "final_external_protocol_config.json"
RESULTS_PATH = ROOT / "final_external_validation_results.json"
OOF_SUMMARY_PATH = ROOT / "final_external_oof_summary.csv"
FOLD_METRICS_PATH = ROOT / "final_external_fold_metrics.csv"
REPRESENTATION_PATH = ROOT / "final_cub_lambda_representation_diagnostics.csv"
OOF_ROOT = ROOT / "final_oof_predictions"

CSV_PATH = ROOT / "FINAL_EXTERNAL_PAIRED_STATISTICS.csv"
JSON_PATH = ROOT / "FINAL_EXTERNAL_PAIRED_STATISTICS.json"
CUB_SUMMARY_PATH = ROOT / "FINAL_CUB_LAMBDA_DIAGNOSTIC_SUMMARY.csv"
AUDIT_PATH = ROOT / "FINAL_EXTERNAL_STATISTICAL_AUDIT.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lambda_key(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return ""
    return f"{float(text):.1f}"


def fail_audit(checks: dict, message: str) -> None:
    lines = [
        "# Final External Statistical Audit — FAIL",
        "",
        f"- Failure: `{message}`",
        "- Statistics were not computed.",
        "",
        "## Integrity checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in checks.items())
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    checks: dict[str, object] = {}
    config_hash = sha256_file(CONFIG_PATH)
    checks["config_sha256"] = config_hash
    checks["config_sha256_pass"] = config_hash == EXPECTED_CONFIG_SHA256

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    summary = pd.read_csv(OOF_SUMMARY_PATH, keep_default_na=False)
    folds = pd.read_csv(FOLD_METRICS_PATH, keep_default_na=False)
    representation = pd.read_csv(REPRESENTATION_PATH, keep_default_na=False)

    checks["formal_jobs_complete"] = int(results.get("formal_jobs_complete", -1))
    checks["formal_jobs_failed"] = int(results.get("formal_jobs_failed", -1))
    checks["formal_jobs_pass"] = (
        results.get("formal_jobs_expected") == 40
        and results.get("formal_jobs_complete") == 40
        and results.get("formal_jobs_failed") == 0
    )
    checks["oof_coverage_pass"] = results.get("oof_coverage_pass") is True

    artifact_records = results.get("primary_oof_results", [])
    artifact_files = sorted(OOF_ROOT.glob("*.npz"))
    checks["recorded_artifact_count"] = len(artifact_records)
    checks["filesystem_artifact_count"] = len(artifact_files)
    checks["artifact_count_pass"] = len(artifact_records) == len(artifact_files) == 16

    artifact_audit = []
    artifact_map = {}
    for record in artifact_records:
        path = Path(record["prediction_artifact"])
        actual_hash = sha256_file(path) if path.is_file() else None
        expected_hash = str(record["artifact_sha256"]).lower()
        matched = actual_hash == expected_hash
        key = (
            str(record["dataset"]),
            str(record["method"]),
            lambda_key(record.get("lambda")),
            int(record["stage"]),
        )
        artifact_map[key] = {
            "path": path,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "hash_match": matched,
            "recorded_n": int(record["n"]),
            "recorded_accuracy": float(record["accuracy"]),
        }
        artifact_audit.append(
            {
                "dataset": key[0],
                "method": key[1],
                "lambda": key[2],
                "stage": key[3],
                "path": str(path),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "hash_match": matched,
            }
        )
    checks["artifact_hashes_pass"] = (
        len(artifact_audit) == 16 and all(row["hash_match"] for row in artifact_audit)
    )
    checks["summary_rows"] = len(summary)
    checks["summary_oof_coverage_pass"] = (
        len(summary) == 16
        and summary["oof_coverage_pass"].astype(str).str.lower().eq("true").all()
    )
    checks["fold_metric_rows"] = len(folds)

    required = [
        "config_sha256_pass",
        "formal_jobs_pass",
        "oof_coverage_pass",
        "artifact_count_pass",
        "artifact_hashes_pass",
        "summary_oof_coverage_pass",
    ]
    checks["provenance_pass"] = all(bool(checks[key]) for key in required)
    if not checks["provenance_pass"]:
        fail_audit(checks, "preflight provenance or integrity check failed")
        raise RuntimeError("preflight provenance or integrity check failed")
    return config, results, summary, folds, representation, {
        "checks": checks,
        "artifacts": artifact_audit,
        "map": artifact_map,
    }


def load_aligned(record: dict) -> dict:
    with np.load(record["path"], allow_pickle=False) as artifact:
        required = {"sample_ids", "labels", "predictions", "folds"}
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"missing NPZ arrays in {record['path']}: {sorted(missing)}")
        arrays = {key: np.asarray(artifact[key]).copy() for key in required}
    ids = arrays["sample_ids"].astype(str)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"duplicate sample_id in {record['path']}")
    order = np.argsort(ids, kind="stable")
    aligned = {key: value[order] for key, value in arrays.items()}
    aligned["sample_ids"] = ids[order]
    aligned["original_sample_ids"] = ids
    aligned["original_order_was_sorted"] = bool(np.array_equal(order, np.arange(len(ids))))
    return aligned


def bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[float, float]:
    difference = correct_b.astype(np.int8) - correct_a.astype(np.int8)
    n = len(difference)
    negative = int(np.count_nonzero(difference == -1))
    zero = int(np.count_nonzero(difference == 0))
    positive = int(np.count_nonzero(difference == 1))
    probabilities = np.asarray([negative, zero, positive], dtype=np.float64) / n
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # For a mean of {-1, 0, +1} paired correctness differences, multinomial
    # category counts are exactly equivalent to resampling paired sample indices.
    counts = rng.multinomial(n, probabilities, size=BOOTSTRAP_RESAMPLES)
    deltas_pp = (counts[:, 2] - counts[:, 0]) * (100.0 / n)
    low, high = np.percentile(deltas_pp, [2.5, 97.5])
    return float(low), float(high)


def compare(
    scope: str,
    dataset: str,
    stage: int,
    method_a: str,
    lambda_a: str,
    method_b: str,
    lambda_b: str,
    artifact_map: dict,
    bootstrap_cache: dict,
) -> dict:
    key_a = (dataset, method_a, lambda_a, stage)
    key_b = (dataset, method_b, lambda_b, stage)
    record_a = artifact_map[key_a]
    record_b = artifact_map[key_b]
    a = load_aligned(record_a)
    b = load_aligned(record_b)

    checks = {
        "n_match": len(a["sample_ids"]) == len(b["sample_ids"]),
        "sample_id_match": bool(np.array_equal(a["sample_ids"], b["sample_ids"])),
        "label_match": bool(np.array_equal(a["labels"], b["labels"])),
        "fold_match": bool(np.array_equal(a["folds"], b["folds"])),
        "sample_ids_unique_a": len(np.unique(a["sample_ids"])) == len(a["sample_ids"]),
        "sample_ids_unique_b": len(np.unique(b["sample_ids"])) == len(b["sample_ids"]),
    }
    checks["pass"] = all(checks.values())
    if not checks["pass"]:
        raise ValueError(f"paired alignment failed for {key_a} vs {key_b}: {checks}")

    labels = a["labels"]
    correct_a = a["predictions"] == labels
    correct_b = b["predictions"] == labels
    n = len(labels)
    accuracy_a = float(correct_a.mean())
    accuracy_b = float(correct_b.mean())
    delta_pp = (accuracy_b - accuracy_a) * 100.0

    if not math.isclose(accuracy_a, record_a["recorded_accuracy"], abs_tol=1e-14):
        raise ValueError(f"recomputed accuracy mismatch for {key_a}")
    if not math.isclose(accuracy_b, record_b["recorded_accuracy"], abs_tol=1e-14):
        raise ValueError(f"recomputed accuracy mismatch for {key_b}")

    # n10: A correct and B wrong. n01: A wrong and B correct.
    n10 = int(np.count_nonzero(correct_a & ~correct_b))
    n01 = int(np.count_nonzero(~correct_a & correct_b))
    discordant_correctness = n10 + n01
    mcnemar_p = (
        float(binomtest(n10, discordant_correctness, p=0.5, alternative="two-sided").pvalue)
        if discordant_correctness
        else 1.0
    )
    cache_key = (record_a["actual_sha256"], record_b["actual_sha256"])
    if cache_key not in bootstrap_cache:
        bootstrap_cache[cache_key] = bootstrap_ci(correct_a, correct_b)
    ci_low, ci_high = bootstrap_cache[cache_key]

    negative = positive = tie = 0
    fold_deltas = []
    for fold in sorted(np.unique(a["folds"]).tolist()):
        mask = a["folds"] == fold
        fold_delta_pp = float((correct_b[mask].mean() - correct_a[mask].mean()) * 100.0)
        fold_deltas.append({"fold": int(fold), "delta_accuracy_pp": fold_delta_pp})
        correct_difference = int(correct_b[mask].sum()) - int(correct_a[mask].sum())
        if correct_difference < 0:
            negative += 1
        elif correct_difference > 0:
            positive += 1
        else:
            tie += 1

    prediction_disagreement = int(np.count_nonzero(a["predictions"] != b["predictions"]))
    return {
        "comparison_scope": scope,
        "dataset": dataset,
        "stage": stage,
        "method_a": method_a,
        "method_b": method_b,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "n": n,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "delta_accuracy_pp": delta_pp,
        "ci95_low_pp": ci_low,
        "ci95_high_pp": ci_high,
        "n10": n10,
        "n01": n01,
        "mcnemar_p": mcnemar_p,
        "prediction_disagreement_count": prediction_disagreement,
        "prediction_disagreement_rate": prediction_disagreement / n,
        "negative_fold_count": negative,
        "positive_fold_count": positive,
        "tie_fold_count": tie,
        "fold_deltas": fold_deltas,
        "alignment_checks": checks,
        "artifact_a": {
            "path": str(record_a["path"]),
            "sha256": record_a["actual_sha256"],
            "original_order_was_sorted": a["original_order_was_sorted"],
        },
        "artifact_b": {
            "path": str(record_b["path"]),
            "sha256": record_b["actual_sha256"],
            "original_order_was_sorted": b["original_order_was_sorted"],
        },
    }


def p_text(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def direction_text(row: dict) -> str:
    return (
        f"negative {row['negative_fold_count']}/5, "
        f"positive {row['positive_fold_count']}/5, tie {row['tie_fold_count']}/5"
    )


def comparison_table(rows: list[dict]) -> str:
    lines = [
        "| Dataset | Stage | Comparison (B - A) | A Acc. | B Acc. | Delta (pp) | 95% CI (pp) | n10 | n01 | Exact McNemar | Fold direction |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        a = row["method_a"] + (f" lambda={row['lambda_a']}" if row["lambda_a"] else "")
        b = row["method_b"] + (f" lambda={row['lambda_b']}" if row["lambda_b"] else "")
        lines.append(
            f"| {row['dataset']} | {row['stage']} | {b} - {a} | "
            f"{100 * row['accuracy_a']:.2f}% | {100 * row['accuracy_b']:.2f}% | "
            f"{row['delta_accuracy_pp']:.2f} | "
            f"[{row['ci95_low_pp']:.2f}, {row['ci95_high_pp']:.2f}] | "
            f"{row['n10']} | {row['n01']} | {p_text(row['mcnemar_p'])} | "
            f"{direction_text(row)} |"
        )
    return "\n".join(lines)


def build_cub_summary(artifact_map: dict, representation: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    means = representation[representation["aggregation_level"] == "unweighted_five_fold_mean"].copy()
    rep_checks = []
    values = {}
    for stage in (1, 2):
        for lam in ("0.1", "0.7", "1.0"):
            record = artifact_map[("CUB-200-2011", "Progressive Head", lam, stage)]
            artifact = load_aligned(record)
            accuracy = float((artifact["predictions"] == artifact["labels"]).mean())
            mean_row = means[(means["stage"] == stage) & (means["lambda"].astype(str) == lam)]
            if len(mean_row) != 1:
                raise ValueError(f"missing representation mean for stage={stage}, lambda={lam}")
            mean_row = mean_row.iloc[0]
            fold_rows = representation[
                (representation["stage"] == stage)
                & (representation["lambda"].astype(str) == lam)
                & (representation["aggregation_level"] == "fold")
            ]
            cosine = float(mean_row["mean_off_diagonal_cosine"])
            cka = float(mean_row["centered_linear_cka"])
            cosine_recomputed = float(fold_rows["mean_off_diagonal_cosine"].mean())
            cka_recomputed = float(fold_rows["centered_linear_cka"].mean())
            check = {
                "stage": stage,
                "lambda": lam,
                "five_fold_rows": len(fold_rows),
                "cosine_mean_match": math.isclose(cosine, cosine_recomputed, abs_tol=1e-15),
                "cka_mean_match": math.isclose(cka, cka_recomputed, abs_tol=1e-15),
            }
            check["pass"] = (
                check["five_fold_rows"] == 5
                and check["cosine_mean_match"]
                and check["cka_mean_match"]
            )
            if not check["pass"]:
                raise ValueError(f"representation aggregation mismatch: {check}")
            rep_checks.append(check)
            values[(stage, lam)] = {"accuracy": accuracy, "cosine": cosine, "cka": cka}
            rows.append(
                {
                    "record_type": "lambda_summary",
                    "stage": stage,
                    "lambda": lam,
                    "oof_accuracy": accuracy,
                    "mean_off_diagonal_cosine": cosine,
                    "centered_linear_cka": cka,
                    "contrast": "",
                    "accuracy_delta_pp": "",
                    "cosine_delta": "",
                    "cka_delta": "",
                }
            )
        low = values[(stage, "0.1")]
        high = values[(stage, "1.0")]
        rows.append(
            {
                "record_type": "descriptive_contrast",
                "stage": stage,
                "lambda": "",
                "oof_accuracy": "",
                "mean_off_diagonal_cosine": "",
                "centered_linear_cka": "",
                "contrast": "lambda_1.0_minus_lambda_0.1",
                "accuracy_delta_pp": (high["accuracy"] - low["accuracy"]) * 100.0,
                "cosine_delta": high["cosine"] - low["cosine"],
                "cka_delta": high["cka"] - low["cka"],
            }
        )
    return pd.DataFrame(rows), {"checks": rep_checks, "values": values}


def main() -> None:
    started = time.time()
    config, results, summary, folds, representation, provenance = preflight()
    artifact_map = provenance.pop("map")
    bootstrap_cache: dict = {}
    comparisons = []

    for dataset in ("CUB-200-2011", "Stanford Cars", "Oxford Flowers-102"):
        for stage in (1, 2):
            comparisons.append(
                compare(
                    "primary_external", dataset, stage,
                    "Ours-FT", "", "Progressive Head", "0.7",
                    artifact_map, bootstrap_cache,
                )
            )

    for stage in (1, 2):
        for lam in ("0.1", "0.7", "1.0"):
            comparisons.append(
                compare(
                    "cub_lambda_vs_ours", "CUB-200-2011", stage,
                    "Ours-FT", "", "Progressive Head", lam,
                    artifact_map, bootstrap_cache,
                )
            )
        for lambda_a, lambda_b in (("0.1", "0.7"), ("0.1", "1.0"), ("0.7", "1.0")):
            comparisons.append(
                compare(
                    "cub_lambda_to_lambda", "CUB-200-2011", stage,
                    "Progressive Head", lambda_a, "Progressive Head", lambda_b,
                    artifact_map, bootstrap_cache,
                )
            )

    all_alignment_pass = all(row["alignment_checks"]["pass"] for row in comparisons)
    if not all_alignment_pass:
        fail_audit(provenance["checks"], "paired sample alignment failed")
        raise RuntimeError("paired sample alignment failed")

    cub_summary, cub_details = build_cub_summary(artifact_map, representation)

    flat_columns = [
        "comparison_scope", "dataset", "stage", "method_a", "method_b",
        "lambda_a", "lambda_b", "n", "accuracy_a", "accuracy_b",
        "delta_accuracy_pp", "ci95_low_pp", "ci95_high_pp", "n10", "n01",
        "mcnemar_p", "prediction_disagreement_count", "prediction_disagreement_rate",
        "negative_fold_count", "positive_fold_count", "tie_fold_count",
    ]
    pd.DataFrame([{key: row[key] for key in flat_columns} for row in comparisons]).to_csv(
        CSV_PATH, index=False
    )
    cub_summary.to_csv(CUB_SUMMARY_PATH, index=False)

    input_hashes = {
        path.name: sha256_file(path)
        for path in (
            CONFIG_PATH, RESULTS_PATH, OOF_SUMMARY_PATH,
            FOLD_METRICS_PATH, REPRESENTATION_PATH,
        )
    }
    json_payload = {
        "schema_version": 1,
        "generated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - started,
        "protocol": {
            "experiment_id": config["experiment_id"],
            "historical_reproduction": config["historical_reproduction"],
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "primary_unit": "complete pooled five-fold OOF samples",
            "official_test_used_for_significance": False,
            "tta_used": False,
            "cross_fold_ensemble_used": False,
        },
        "provenance": {
            **provenance,
            "input_file_sha256": input_hashes,
        },
        "alignment": {
            "explicit_alignment_key": "sample_id",
            "all_comparisons_pass": all_alignment_pass,
            "required_equal_fields": ["N", "sample_id", "ground_truth_label", "fold_assignment"],
        },
        "definitions": {
            "difference": "accuracy(method_b) - accuracy(method_a), in percentage points",
            "n10": "method_a correct and method_b wrong",
            "n01": "method_a wrong and method_b correct",
            "prediction_disagreement": "predicted class differs, including cases where both predictions are wrong",
            "mcnemar": "exact two-sided binomial McNemar test on n10 and n01",
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "paired sample",
            "ci_method": "percentile 95% CI",
            "implementation": (
                "multinomial counts over paired correctness differences {-1,0,+1}; "
                "exactly distribution-equivalent to paired sample-index resampling for delta accuracy"
            ),
        },
        "comparisons": comparisons,
        "cub_representation_diagnostics": {
            "aggregation": "unweighted arithmetic mean of five fold-level values",
            "cross_fold_feature_concatenation": False,
            "checks": cub_details["checks"],
            "summary_rows": cub_summary.to_dict(orient="records"),
        },
    }
    JSON_PATH.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    primary = [row for row in comparisons if row["comparison_scope"] == "primary_external"]
    cub_vs_ours = [row for row in comparisons if row["comparison_scope"] == "cub_lambda_vs_ours"]
    cub_pairs = [row for row in comparisons if row["comparison_scope"] == "cub_lambda_to_lambda"]

    rep_table_lines = [
        "| Stage | Lambda | OOF Accuracy | Mean off-diagonal cosine | Centered-linear CKA |",
        "|---:|---:|---:|---:|---:|",
    ]
    summary_only = cub_summary[cub_summary["record_type"] == "lambda_summary"]
    for _, row in summary_only.iterrows():
        rep_table_lines.append(
            f"| {int(row['stage'])} | {row['lambda']} | {100 * float(row['oof_accuracy']):.2f}% | "
            f"{float(row['mean_off_diagonal_cosine']):.3f} | {float(row['centered_linear_cka']):.3f} |"
        )
    contrast_lines = []
    for _, row in cub_summary[cub_summary["record_type"] == "descriptive_contrast"].iterrows():
        contrast_lines.append(
            f"- Stage {int(row['stage'])}, lambda 1.0 - lambda 0.1: "
            f"accuracy {float(row['accuracy_delta_pp']):.2f} pp; "
            f"cosine {float(row['cosine_delta']):.3f}; CKA {float(row['cka_delta']):.3f}."
        )

    audit = f"""# Final External Statistical Audit

## A. Integrity / provenance

- `PROVENANCE_PASS = YES`
- `CONFIG_SHA256 = {EXPECTED_CONFIG_SHA256}`
- `FORMAL_JOBS_COMPLETE = 40/40`
- `FORMAL_JOBS_FAILED = 0`
- `OOF_COVERAGE_PASS = YES`
- `OOF_ARTIFACT_COUNT = 16/16`
- `OOF_ARTIFACT_HASH_MATCH = 16/16`
- `ALL_PAIRED_SAMPLE_ALIGNMENT_PASS = YES`
- Every paired comparison was explicitly aligned by `sample_id`; N, ground-truth labels, and fold assignments matched exactly.

## B. Protocol summary

This analysis belongs to `FINAL_NEWLY_FROZEN_UNIFIED_CROSS_DATASET_VALIDATION_PROTOCOL`. The statistical unit is the complete pooled five-fold OOF sample. No fold-mean significance test, official-test primary analysis, TTA, cross-fold ensemble, checkpoint reselection, model training, or OOF modification was performed.

Paired bootstrap used {BOOTSTRAP_RESAMPLES:,} resamples, seed {BOOTSTRAP_SEED}, and a percentile 95% CI. The difference is always method B minus method A in percentage points. `n10` means method A is correct and method B is wrong; `n01` means method A is wrong and method B is correct. McNemar p-values are exact and two-sided.

## C. Primary six paired comparisons

{comparison_table(primary)}

## D. CUB lambda versus Ours-FT

{comparison_table(cub_vs_ours)}

## E. CUB lambda-to-lambda diagnostics

{comparison_table(cub_pairs)}

These lambda-to-lambda results are diagnostic sensitivity checks only; they do not establish universal lambda superiority.

## F. Representation-similarity summary

The values below are unweighted arithmetic means of five independently computed fold-level diagnostics. Feature matrices from different fold models were not concatenated for global CKA.

{chr(10).join(rep_table_lines)}

{chr(10).join(contrast_lines)}

These results describe systematic representation similarity changes associated with the shortcut coefficient. Cosine or CKA magnitude is not interpreted as representation quality and no causal accuracy claim is made.

## G. Fold-direction consistency

Primary directions are reported in the final column of Section C. Counts use the sign of each fold's unrounded paired accuracy difference; folds are descriptive strata, not independent samples for significance testing.

## H. Interpretation boundaries

1. This is a new unified cross-dataset controlled validation, not an exact reproduction of the historical main experiments.
2. These numbers must not directly replace the historical CUB, Cars, or Flowers main-table values unless the entire table is explicitly switched to the new protocol.
3. The primary comparison supports only the paired OOF performance difference between Progressive Head lambda=0.7 and Ours-FT under this frozen protocol. It does not show that Progressive Head intrinsically harms performance or that shortcut scaling universally reduces accuracy.
4. CUB representation diagnostics describe similarity structure changes associated with the shortcut coefficient; they do not measure representation quality.

## I. Manuscript-ready rounded preview

{comparison_table(primary)}

Machine-readable CSV and JSON retain full precision. Accuracy is shown to two decimals, delta and CI to two decimals, p-values to three decimals or `p<0.001`, and cosine/CKA to three decimals only in this Markdown preview.
"""
    AUDIT_PATH.write_text(audit, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "provenance_pass": True,
                "artifact_hashes": "16/16",
                "paired_alignment_pass": all_alignment_pass,
                "comparisons": len(comparisons),
                "outputs": [str(CSV_PATH), str(JSON_PATH), str(CUB_SUMMARY_PATH), str(AUDIT_PATH)],
                "elapsed_seconds": time.time() - started,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
