"""Smoke gate with explicit evidence of at least one optimizer update per stage."""
import json
from pathlib import Path

import pandas as pd

import validate_smoke
from cross_backbone_common import BACKBONES, METHODS, ROOT, output_dir


def main():
    validate_smoke.main()
    rows = []
    passed = True
    for backbone in BACKBONES:
        for method in METHODS:
            history = pd.read_csv(output_dir("smoke", backbone, method, 0) / "training_history.csv")
            stage_rows = []
            expected_stages = (2,) if method == "dfag" else (1, 2)
            for stage in expected_stages:
                selected = history[history["stage"] == stage]
                overflow = int(selected["amp_overflow_batches"].sum())
                attempted = 16 * len(selected)
                observed_update = overflow < attempted
                stage_rows.append({"stage": stage, "attempted_batches": attempted,
                                   "overflow_batches": overflow, "optimizer_update_observed": observed_update})
                passed = passed and observed_update
            rows.append({"backbone": backbone, "method": method, "stages": stage_rows})
    payload = json.loads((ROOT / "smoke_audit.json").read_text(encoding="utf-8"))
    payload["optimizer_update_evidence"] = rows
    payload["checks"]["at_least_one_optimizer_update_each_stage"] = passed
    payload["status"] = "PASS" if all(payload["checks"].values()) else "FAIL"
    (ROOT / "smoke_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "optimizer_update_evidence": rows}), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
