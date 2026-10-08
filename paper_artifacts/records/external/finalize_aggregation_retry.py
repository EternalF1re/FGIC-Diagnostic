"""Finalize orchestrator metadata after a verified aggregation retry."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    with (ROOT / "final_job_ledger.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = json.loads(
        (ROOT / "final_external_validation_results.json").read_text(encoding="utf-8")
    )
    oof_files = list((ROOT / "final_oof_predictions").glob("*.npz"))

    checks = {
        "ledger_40_complete": len(rows) == 40
        and all(row["formal_status"] == "COMPLETE" for row in rows),
        "results_40_complete": results.get("formal_jobs_complete") == 40
        and results.get("formal_jobs_failed") == 0,
        "oof_coverage_pass": results.get("oof_coverage_pass") is True,
        "oof_artifact_count_16": len(oof_files) == 16,
        "audit_exists": (ROOT / "FINAL_EXTERNAL_VALIDATION_AUDIT.md").is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"aggregation retry finalization checks failed: {checks}")

    status_path = ROOT / "formal_orchestrator_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    prior_status = status.get("status")
    prior_returncode = status.get("aggregation_returncode")
    status.update(
        {
            "status": "COMPLETE",
            "updated_unix": time.time(),
            "counts": {"PENDING": 0, "RUNNING": 0, "COMPLETE": 40, "FAILED": 0},
            "aggregation_returncode": 0,
            "aggregation_retry": True,
            "aggregation_retry_entrypoint": "aggregate_formal_retry.py",
            "initial_aggregation_status": prior_status,
            "initial_aggregation_returncode": prior_returncode,
            "post_retry_checks": checks,
        }
    )
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status_path)
    print(json.dumps({"status": "COMPLETE", "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
