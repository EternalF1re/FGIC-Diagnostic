"""Postflight integrity audit for completed Phase2D Round1 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from round1_common import PHASE2B_ROOT, ROUND_IDS, ROUND_ROOT, config_path, output_dir


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    target = ROUND_ROOT / "manifests" / "postflight_audit.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")

    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    preflight_path = ROUND_ROOT / "manifests" / "preflight_audit.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    check("preflight_passed", preflight.get("all_passed") is True, str(preflight_path))

    summary_path = ROUND_ROOT / "orchestrator_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    check("orchestrator_complete", summary.get("status") == "ALL_20_COMPLETE", str(summary))

    ledger = csv_rows(ROUND_ROOT / "run_ledger.csv")
    check("ledger_20_rows", len(ledger) == 20, f"rows={len(ledger)}")
    check(
        "ledger_all_complete",
        all(row["status"] == "COMPLETE" for row in ledger),
        json.dumps({status: sum(row["status"] == status for row in ledger) for status in sorted({row["status"] for row in ledger})}),
    )

    events = [
        json.loads(line)
        for line in (ROUND_ROOT / "logs" / "orchestrator_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    starts = [event for event in events if event.get("event") == "START"]
    exits = [event for event in events if event.get("event") == "EXIT"]
    check("events_20_starts", len(starts) == 20, f"starts={len(starts)}")
    check("events_20_clean_exits", len(exits) == 20 and all(event.get("exit_code") == 0 for event in exits), f"exits={len(exits)}")

    runner_hash = sha256(ROUND_ROOT / "scripts" / "train_one_round1.py")
    source_hashes = {
        "train_one.py": sha256(PHASE2B_ROOT / "train_one.py"),
        "screen_core.py": sha256(PHASE2B_ROOT / "screen_core.py"),
    }
    run_count = 0
    validation_array_count = 0
    for run_id in ROUND_IDS:
        expected_dim = 1024 if run_id.startswith("baseline") else 256
        config_hash = sha256(config_path(run_id))
        for fold in range(5):
            run_count += 1
            directory = output_dir(run_id, fold)
            manifest_path = directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prefix = f"{run_id}_fold{fold}"
            check(f"{prefix}_manifest_complete", manifest.get("status") == "COMPLETE", str(manifest_path))
            check(
                f"{prefix}_identity",
                manifest.get("run_id") == run_id and manifest.get("fold") == fold,
                f"run_id={manifest.get('run_id')}, fold={manifest.get('fold')}",
            )
            check(f"{prefix}_config_hash", manifest.get("config_sha256") == config_hash, config_hash)
            check(f"{prefix}_runner_hash", manifest.get("round1_runner_sha256") == runner_hash, runner_hash)
            check(
                f"{prefix}_frozen_source_hashes",
                manifest.get("phase2b_train_one_sha256") == source_hashes["train_one.py"]
                and manifest.get("phase2b_screen_core_sha256") == source_hashes["screen_core.py"],
                json.dumps(source_hashes),
            )

            stage1_history = csv_rows(directory / "stage1_history.csv")
            training_history = csv_rows(directory / "training_history.csv")
            check(f"{prefix}_history_counts", len(stage1_history) == 150 and len(training_history) == 210, f"stage1={len(stage1_history)}, total={len(training_history)}")
            check(
                f"{prefix}_manifest_history_counts",
                manifest.get("history_counts") == {"stage1": 150, "stage2": 60},
                str(manifest.get("history_counts")),
            )

            for stage in (1, 2):
                checkpoint = directory / f"best_stage{stage}.pth"
                expected_hash = manifest[f"best_stage{stage}_sha256"]
                check(f"{prefix}_stage{stage}_checkpoint_hash", checkpoint.exists() and sha256(checkpoint) == expected_hash, expected_hash)

                artifacts = manifest["validation_artifacts"][f"stage{stage}"]
                required = {"logits", "features", "labels", "sample_ids", "image_names", "predictions"}
                paths = {key: directory / artifacts[key] for key in required}
                check(f"{prefix}_stage{stage}_artifact_set", set(artifacts) == required and all(path.exists() for path in paths.values()), str(paths))

                labels = np.load(paths["labels"], mmap_mode="r")
                predictions = np.load(paths["predictions"], mmap_mode="r")
                sample_ids = np.load(paths["sample_ids"], mmap_mode="r")
                logits = np.load(paths["logits"], mmap_mode="r")
                features = np.load(paths["features"], mmap_mode="r")
                n = int(manifest["metrics"][f"stage{stage}"] and manifest["metrics"]["n"])
                shapes_ok = (
                    labels.shape == predictions.shape == sample_ids.shape == (n,)
                    and logits.shape == (n, 176)
                    and features.shape == (n, expected_dim)
                )
                check(f"{prefix}_stage{stage}_array_shapes", shapes_ok, f"n={n}, logits={logits.shape}, features={features.shape}")
                prediction_ok = np.array_equal(np.asarray(logits).argmax(axis=1), np.asarray(predictions))
                metric_accuracy = float(np.mean(np.asarray(predictions) == np.asarray(labels)))
                saved_accuracy = float(manifest["metrics"][f"stage{stage}"]["accuracy"])
                check(f"{prefix}_stage{stage}_prediction_integrity", prediction_ok and np.isfinite(logits).all(), f"accuracy={metric_accuracy}")
                check(f"{prefix}_stage{stage}_metric_accuracy", metric_accuracy == saved_accuracy, f"computed={metric_accuracy}, saved={saved_accuracy}")
                validation_array_count += 5

    check("formal_run_count", run_count == 20, f"runs={run_count}")
    check("validated_numeric_arrays", validation_array_count == 200, f"arrays={validation_array_count}")

    oof_dir = ROUND_ROOT / "oof"
    oof_files = sorted(oof_dir.glob("*_oof.npz"))
    check("eight_new_oof_files", len(oof_files) == 8, f"files={len(oof_files)}")
    for path in oof_files:
        with np.load(path) as data:
            ids = data["sample_id"]
            labels = data["label"]
            predictions = data["prediction"]
            logits = data["logits"]
            folds = data["fold"]
            ok = (
                len(ids) == 18353
                and np.array_equal(ids, np.arange(18353))
                and np.array_equal(logits.argmax(axis=1), predictions)
                and labels.shape == predictions.shape == folds.shape == (18353,)
                and set(np.unique(folds).tolist()) == {0, 1, 2, 3, 4}
            )
        check(f"oof_{path.stem}", ok, str(path))

    expected_csv_rows = {
        "oof/round1_oof_metrics.csv": 48,
        "statistics/stage1_vs_stage2_paired.csv": 4,
        "statistics/stage1_vs_stage2_per_fold.csv": 20,
        "statistics/baseline_multiseed_per_seed.csv": 3,
        "statistics/baseline_multiseed_summary.csv": 1,
        "statistics/lambda_direct_paired.csv": 12,
        "statistics/lambda_direct_paired_per_fold.csv": 60,
        "statistics/lambda_0_9_vs_0_7_paired.csv": 2,
        "statistics/lambda_0_9_vs_0_7_paired_per_fold.csv": 10,
    }
    result_hashes: dict[str, str] = {}
    for relative, expected in expected_csv_rows.items():
        path = ROUND_ROOT / relative
        rows = csv_rows(path)
        check(f"csv_{path.stem}_rows", len(rows) == expected, f"rows={len(rows)}, expected={expected}")
        result_hashes[relative] = sha256(path)
    for path in oof_files:
        result_hashes[str(path.relative_to(ROUND_ROOT)).replace("\\", "/")] = sha256(path)

    failures = [item for item in checks if item["status"] != "PASS"]
    result = {
        "all_passed": not failures,
        "formal_runs": run_count,
        "checks_total": len(checks),
        "checks_failed": len(failures),
        "checks": checks,
        "result_sha256": result_hashes,
    }
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "POSTFLIGHT_PASS" if not failures else "POSTFLIGHT_FAIL", "checks": len(checks), "failed": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
