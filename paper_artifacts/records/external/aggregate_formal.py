"""Aggregate complete fold predictions only after all 40 formal jobs succeed."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import formal_common as common


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def configuration_id(dataset: str, method: str, shortcut_lambda: str) -> str:
    ds = common.dataset_key(dataset)
    return f"{ds}__ours_ft" if method == "Ours-FT" else f"{ds}__progressive_lambda_{shortcut_lambda.replace('.', '_')}"


def main() -> None:
    ledger = pd.read_csv(common.RUNTIME_LEDGER_PATH, keep_default_na=False)
    if len(ledger) != 40 or set(ledger["formal_status"]) != {"COMPLETE"}:
        raise RuntimeError("aggregation requires 40 COMPLETE jobs")
    config = common.load_config()
    oof_root = common.ROOT / "final_oof_predictions"
    oof_root.mkdir(parents=True, exist_ok=False)
    oof_rows, fold_rows, official_rows = [], [], []
    cache = {}
    grouped = ledger.groupby(["dataset", "method", "lambda"], dropna=False, sort=False)
    for (dataset, method, shortcut_lambda), group in grouped:
        shortcut_lambda = str(shortcut_lambda)
        config_id = configuration_id(dataset, method, shortcut_lambda)
        expected_dev, _ = common.records_for_dataset(dataset, config)
        expected_ids = {row["sample_id"] for row in expected_dev}
        for stage in (1, 2):
            parts = {key: [] for key in ("sample_ids", "labels", "predictions", "logits", "folds")}
            for row in group.sort_values("fold").itertuples(index=False):
                run = Path(row.output_dir)
                metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                stage_key = f"stage{stage}"
                val_metrics = metrics[stage_key]["validation"]
                test_metrics = metrics[stage_key]["secondary_official_test"]
                fold_rows.append({
                    "dataset": dataset, "method": method, "lambda": shortcut_lambda,
                    "stage": stage, "fold": int(row.fold), "n": int(json.loads((run / "run_manifest.json").read_text())["validation_count"]),
                    **val_metrics,
                })
                official_rows.append({
                    "dataset": dataset, "method": method, "lambda": shortcut_lambda,
                    "stage": stage, "fold": int(row.fold), "secondary_only": True,
                    "n": int(json.loads((run / "run_manifest.json").read_text())["official_test_count"]),
                    **test_metrics,
                })
                ids = np.load(run / f"stage{stage}_validation_sample_ids.npy", allow_pickle=False).astype(str)
                labels = np.load(run / f"stage{stage}_validation_labels.npy", allow_pickle=False).astype(np.int64)
                predictions = np.load(run / f"stage{stage}_validation_predictions.npy", allow_pickle=False).astype(np.int64)
                logits = np.load(run / f"stage{stage}_validation_logits.npy", mmap_mode="r").astype(np.float32)
                if len(set(ids.tolist())) != len(ids):
                    raise ValueError(f"duplicate fold IDs in {row.job_id} stage{stage}")
                parts["sample_ids"].append(ids)
                parts["labels"].append(labels)
                parts["predictions"].append(predictions)
                parts["logits"].append(np.asarray(logits))
                parts["folds"].append(np.full(len(ids), int(row.fold), dtype=np.int8))
            joined = {key: np.concatenate(value) for key, value in parts.items()}
            ids = joined["sample_ids"]
            if len(ids) != len(expected_ids) or len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != expected_ids:
                raise ValueError(f"OOF coverage failure: {config_id} stage{stage}")
            order = np.argsort(ids)
            joined = {key: value[order] for key, value in joined.items()}
            metrics = metric(joined["labels"], joined["predictions"])
            out = oof_root / f"{config_id}__stage{stage}.npz"
            np.savez_compressed(out, **joined)
            oof_rows.append({
                "dataset": dataset, "method": method, "lambda": shortcut_lambda,
                "stage": stage, "n": len(ids), "oof_coverage_pass": True,
                **metrics, "prediction_artifact": str(out.resolve()), "artifact_sha256": common.sha256_file(out),
            })
            cache[(dataset, method, shortcut_lambda, stage)] = joined

    paired = []
    for dataset in ("CUB-200-2011", "Stanford Cars", "Oxford Flowers-102"):
        for stage in (1, 2):
            ours = cache[(dataset, "Ours-FT", "", stage)]
            progressive = cache[(dataset, "Progressive Head", "0.7", stage)]
            if not np.array_equal(ours["sample_ids"], progressive["sample_ids"]) or not np.array_equal(ours["labels"], progressive["labels"]):
                raise ValueError(f"paired OOF alignment failure: {dataset} stage{stage}")
            ours_acc = float(np.mean(ours["predictions"] == ours["labels"]))
            prog_acc = float(np.mean(progressive["predictions"] == progressive["labels"]))
            paired.append({"dataset": dataset, "stage": stage, "n": len(ours["labels"]),
                           "ours_ft_accuracy": ours_acc, "progressive_lambda_0_7_accuracy": prog_acc,
                           "progressive_minus_ours_accuracy": prog_acc - ours_acc})

    diagnostic_rows = []
    for shortcut_lambda in ("0.1", "0.7", "1.0"):
        group = ledger[(ledger["dataset"] == "CUB-200-2011") & (ledger["method"] == "Progressive Head") & (ledger["lambda"].astype(str) == shortcut_lambda)]
        for stage in (1, 2):
            fold_values = []
            for row in group.sort_values("fold").itertuples(index=False):
                diagnostics = json.loads((Path(row.output_dir) / "representation_diagnostics.json").read_text(encoding="utf-8"))[f"stage{stage}"]
                value = {
                    "lambda": shortcut_lambda, "stage": stage, "fold": int(row.fold), "aggregation_level": "fold",
                    "n": diagnostics["n"], "mean_off_diagonal_cosine": diagnostics["mean_off_diagonal_cosine"],
                    "centered_linear_cka": diagnostics["mean_centered_linear_cka"],
                }
                diagnostic_rows.append(value)
                fold_values.append(value)
            diagnostic_rows.append({
                "lambda": shortcut_lambda, "stage": stage, "fold": "", "aggregation_level": "unweighted_five_fold_mean",
                "n": sum(row["n"] for row in fold_values),
                "mean_off_diagonal_cosine": float(np.mean([row["mean_off_diagonal_cosine"] for row in fold_values])),
                "centered_linear_cka": float(np.mean([row["centered_linear_cka"] for row in fold_values])),
            })

    pd.DataFrame(oof_rows).to_csv(common.ROOT / "final_external_oof_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(common.ROOT / "final_external_fold_metrics.csv", index=False)
    pd.DataFrame(official_rows).to_csv(common.ROOT / "final_external_official_test_secondary.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(common.ROOT / "final_cub_lambda_representation_diagnostics.csv", index=False)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config_sha256": common.EXPECTED_CONFIG_SHA256,
        "formal_jobs_expected": 40, "formal_jobs_complete": 40, "formal_jobs_failed": 0,
        "oof_coverage_pass": all(row["oof_coverage_pass"] for row in oof_rows),
        "primary_oof_results": oof_rows, "fold_metrics": fold_rows,
        "official_test_secondary": official_rows, "paired_oof_accuracy_differences": paired,
        "cub_representation_diagnostics": diagnostic_rows,
        "statistical_significance_interpreted": False,
    }
    (common.ROOT / "final_external_validation_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = [
        f"| {row['dataset']} | {row['method']} | {row['lambda'] or '-'} | {row['stage']} | {row['n']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} |"
        for row in oof_rows
    ]
    audit = f"""# Final External Validation Audit

- `FORMAL_JOBS_EXPECTED = 40`
- `FORMAL_JOBS_COMPLETE = 40`
- `FORMAL_JOBS_FAILED = 0`
- `CONFIG_SHA_MATCH = YES`
- `OOF_COVERAGE_PASS = YES`
- `CUB_DIAGNOSTICS_COMPLETE = YES`
- `FINAL_EXTERNAL_VALIDATION_READY_FOR_ANALYSIS = YES`

Primary metrics below are recomputed on each complete pooled five-fold OOF prediction set, not arithmetic means of fold metrics. Official-test metrics remain secondary.

| Dataset | Method | Lambda | Stage | N | OOF Accuracy | Macro-F1 | Balanced Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Prediction-level OOF artifacts are preserved under `final_oof_predictions/`. Paired OOF accuracy differences are stored in `final_external_validation_results.json`; no bootstrap, McNemar, or significance interpretation was invented automatically.
"""
    (common.ROOT / "FINAL_EXTERNAL_VALIDATION_AUDIT.md").write_text(audit, encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "jobs": 40, "oof_configurations": len(oof_rows), "diagnostics": len(diagnostic_rows)}), flush=True)


if __name__ == "__main__":
    main()
