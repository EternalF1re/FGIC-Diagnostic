"""Generate ten alpha=0.5 comparator folds through the shared GPU queue."""
from __future__ import annotations

import json
import sys
import time

from dfag_common import EXP_ROOT, FOLDS, REPO_ROOT, SEEDS


VALIDATION_ROOT = EXP_ROOT.parent
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))
from dynamic_gpu_queue import DynamicGpuQueue, SchedulerPaths, TaskSpec


def main() -> None:
    static = json.loads((EXP_ROOT / "manifests" / "static_preflight_audit.json").read_text(encoding="utf-8"))
    if static.get("status") != "PASS":
        raise RuntimeError("static preflight missing or failed")
    runner = EXP_ROOT / "scripts" / "run_alpha05_fold.py"
    tasks = []
    for fold in FOLDS:
        for seed in SEEDS:
            tasks.append(
                TaskSpec(
                    task_id=f"alpha05_seed{seed}_fold{fold}",
                    command=(
                        "{python}", str(runner), "--seed", str(seed), "--fold", str(fold),
                        "--device", "{device}",
                    ),
                    estimated_seconds=180.0,
                    metadata={
                        "seed": seed,
                        "fold": fold,
                        "inference_only": True,
                        "endpoint_fidelity_gate": True,
                        "training": False,
                    },
                )
            )
    started = time.time()
    result = DynamicGpuQueue(
        tasks,
        devices=("cuda:0", "cuda:1"),
        paths=SchedulerPaths.under(EXP_ROOT / "comparator_scheduler"),
        cwd=REPO_ROOT,
        poll_seconds=15.0,
        longest_first=True,
    ).run()
    summary = {
        "status": "ALL_10_COMPARATORS_COMPLETE",
        "tasks": 10,
        "training": False,
        "started_unix": started,
        "completed_unix": time.time(),
        "scheduler_summary": result,
    }
    (EXP_ROOT / "comparators" / "orchestrator_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
