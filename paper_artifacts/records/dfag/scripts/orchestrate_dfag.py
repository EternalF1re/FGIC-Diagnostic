"""Run exactly 15 frozen Phase2F jobs through the shared dynamic GPU queue."""
from __future__ import annotations

import argparse
import json
import sys
import time

from dfag_common import EXP_ROOT, FOLDS, REPO_ROOT, SEEDS, output_dir


VALIDATION_ROOT = EXP_ROOT.parent
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))
from dynamic_gpu_queue import DynamicGpuQueue, SchedulerPaths, TaskSpec


def build_tasks() -> tuple[list[TaskSpec], list[str]]:
    runner = EXP_ROOT / "scripts" / "train_one_dfag.py"
    tasks, completed = [], []
    for fold in FOLDS:
        for seed in SEEDS:
            task_id = f"seed{seed}_fold{fold}"
            destination = output_dir(seed, fold)
            if destination.exists():
                manifest_path = destination / "run_manifest.json"
                if not manifest_path.exists():
                    raise RuntimeError(f"partial output requires human review: {destination}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "COMPLETE":
                    raise RuntimeError(f"non-complete output requires human review: {destination}")
                completed.append(task_id)
            tasks.append(TaskSpec(
                task_id=task_id,
                command=("{python}", str(runner), "--seed", str(seed), "--fold", str(fold), "--device", "{device}"),
                estimated_seconds=8_000.0,
                metadata={"seed": seed, "fold": fold, "formal_training": True,
                          "standalone_dynamic_dfag": True, "fixed_g_training": False,
                          "anchor_sweep": False, "gating_source_ablation": False, "joint_model": False},
            ))
    return tasks, completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=("cuda:0", "cuda:1"))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    preflight = json.loads((EXP_ROOT / "manifests" / "preflight_audit.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("formal_jobs") != 15:
        raise RuntimeError("Phase2F preflight missing or failed")
    if (EXP_ROOT / "orchestrator_summary.json").exists():
        raise FileExistsError("refusing to overwrite orchestrator summary")
    tasks, completed = build_tasks()
    started = time.time()
    scheduler = DynamicGpuQueue(tasks, devices=args.devices, paths=SchedulerPaths.under(EXP_ROOT / "scheduler"),
                                cwd=REPO_ROOT, poll_seconds=args.poll_seconds, longest_first=True,
                                completed_task_ids=completed)
    result = scheduler.run()
    summary = {"status": "ALL_15_COMPLETE", "formal_runs": 15, "shared_dynamic_queue": True,
               "existing_comparators_retrained": False, "fixed_g_training_launched": False,
               "anchor_sweep_launched": False, "gating_source_ablation_launched": False,
               "joint_model_launched": False, "seed45_46_launched": False,
               "started_unix": started, "completed_unix": time.time(), "scheduler_summary": result,
               "stop_after_phase2f": True}
    (EXP_ROOT / "orchestrator_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
