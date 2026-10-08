"""Run all 25 Round2A formal folds through the shared dynamic GPU queue."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROUND_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROUND_ROOT.parent
REPO_ROOT = VALIDATION_ROOT.parent
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from dynamic_gpu_queue import DynamicGpuQueue, SchedulerPaths, TaskSpec
from round2a_common import ROUND_IDS, output_dir


def build_tasks() -> tuple[list[TaskSpec], list[str]]:
    runner = ROUND_ROOT / "scripts" / "train_one_round2a.py"
    tasks = []
    completed = []
    # Interleave fixed configurations by fold so baseline confirmation runs in
    # parallel with MHSA rather than being selected after seeing MHSA results.
    for fold in range(5):
        for run_id in ROUND_IDS:
            task_id = f"{run_id}_fold{fold}"
            destination = output_dir(run_id, fold)
            if destination.exists():
                manifest_path = destination / "run_manifest.json"
                if not manifest_path.exists():
                    raise RuntimeError(f"partial output requires human review: {destination}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "COMPLETE":
                    raise RuntimeError(f"non-complete output requires human review: {destination}")
                completed.append(task_id)
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    command=(
                        "{python}",
                        str(runner),
                        "--run-id",
                        run_id,
                        "--fold",
                        str(fold),
                        "--device",
                        "{device}",
                    ),
                    estimated_seconds=15_000.0,
                    metadata={"run_id": run_id, "fold": fold, "formal_training": True},
                )
            )
    return tasks, completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=("cuda:0", "cuda:1"))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    preflight_path = ROUND_ROOT / "manifests" / "preflight_audit.json"
    if not preflight_path.exists():
        raise RuntimeError("Round2A preflight audit is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("all_passed") or preflight.get("formal_runs") != 25:
        raise RuntimeError("Round2A preflight audit failed")
    top_summary = ROUND_ROOT / "orchestrator_summary.json"
    if top_summary.exists():
        raise RuntimeError(f"refusing to overwrite {top_summary}")

    tasks, completed = build_tasks()
    started = time.time()
    scheduler = DynamicGpuQueue(
        tasks,
        devices=args.devices,
        paths=SchedulerPaths.under(ROUND_ROOT / "scheduler"),
        cwd=REPO_ROOT,
        poll_seconds=args.poll_seconds,
        longest_first=True,
        completed_task_ids=completed,
    )
    result = scheduler.run()
    summary = {
        "status": "ALL_25_COMPLETE",
        "formal_runs": 25,
        "mhsa_runs": 15,
        "baseline_runs": 10,
        "shared_dynamic_queue": True,
        "started_unix": started,
        "completed_unix": time.time(),
        "scheduler_summary": result,
        "stop_gate": True,
    }
    top_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
