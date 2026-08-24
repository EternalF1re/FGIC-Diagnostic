"""Technical-only gate for the six fold-0 smoke configurations."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cross_backbone_common import BACKBONES, METHODS, ROOT, output_dir, sha256_file
from queue_runner import summary


def main() -> None:
    queue = summary("smoke")
    checks = {
        "queue_six_complete": queue["counts"].get("COMPLETE", 0) == 6
            and queue["counts"].get("FAILED", 0) == 0
            and queue["counts"].get("PENDING", 0) == 0
            and queue["counts"].get("RUNNING", 0) == 0,
    }
    rows = []
    for backbone in BACKBONES:
        for method in METHODS:
            directory = output_dir("smoke", backbone, method, 0)
            manifest_path = directory / "run_manifest.json"
            row = {"backbone": backbone, "method": method, "fold": 0, "directory": str(directory.resolve())}
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                artifacts = [directory / "dfag_predictions.npz"] if method == "dfag" else [
                    directory / "stage1_predictions.npz", directory / "stage2_predictions.npz"
                ]
                if method == "ours_ft":
                    artifacts.append(directory / "alpha0_5_predictions.npz")
                artifact_checks = []
                for path in artifacts:
                    with np.load(path, allow_pickle=False) as data:
                        keys = set(data.files)
                        n = len(data["sample_id"])
                        artifact_checks.append({
                            "file": path.name,
                            "exists": True,
                            "n": n,
                            "schema": {"sample_id", "fold", "y_true", "y_pred", "logits", "probabilities"}.issubset(keys),
                            "finite": bool(np.isfinite(data["logits"]).all() and np.isfinite(data["probabilities"]).all()),
                            "unique_sample_ids": len(np.unique(data["sample_id"])) == n,
                            "sha256": sha256_file(path),
                        })
                row.update({
                    "manifest_complete": manifest.get("status") == "COMPLETE",
                    "amp": manifest.get("technical_checks", {}).get("amp") is True,
                    "checkpoint_restore": manifest.get("technical_checks", {}).get("checkpoint_restore") is True,
                    "bn_adaptation_50": manifest.get("bn_adaptation_batches_completed") == 50,
                    "batch_size_64": manifest.get("batch_size") == 64,
                    "artifacts": artifact_checks,
                    "artifacts_pass": all(a["schema"] and a["finite"] and a["unique_sample_ids"] and a["n"] > 0 for a in artifact_checks),
                    "dfag_stage1_reused": method != "dfag" or manifest.get("dfag_stage1_source", {}).get("stage1_retrained") is False,
                    "technical_checks": manifest.get("technical_checks", {}),
                })
                row["status"] = "PASS" if all(row[key] for key in (
                    "manifest_complete", "amp", "checkpoint_restore", "bn_adaptation_50",
                    "batch_size_64", "artifacts_pass", "dfag_stage1_reused"
                )) else "FAIL"
            except Exception as exc:
                row.update({"status": "FAIL", "error": repr(exc)})
            rows.append(row)
    checks["all_six_technical_pass"] = len(rows) == 6 and all(row["status"] == "PASS" for row in rows)
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {"status": status, "created_unix": time.time(), "checks": checks, "queue": queue, "rows": rows,
               "accuracy_used_as_gate": False}
    (ROOT / "smoke_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "rows": [{"backbone": r["backbone"], "method": r["method"], "status": r["status"]} for r in rows]}), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
