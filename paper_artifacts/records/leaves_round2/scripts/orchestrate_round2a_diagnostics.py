"""Run all checkpoint-only Round2A diagnostics through the shared GPU queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROUND_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROUND_ROOT.parent
REPO_ROOT = VALIDATION_ROOT.parent
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from dynamic_gpu_queue import DynamicGpuQueue, SchedulerPaths, TaskSpec
from round2a_common import MHSA_IDS, RUN_SPECS


def lambda_key(run_id: str) -> str:
    return f"lambda_{str(float(RUN_SPECS[run_id]['lambda'])).replace('.', '_')}"


def destination(run_id: str, stage: int, fold: int) -> Path:
    return ROUND_ROOT / "diagnostics" / "fold_outputs" / lambda_key(run_id) / f"stage{stage}" / f"fold_{fold}"


def build_tasks() -> tuple[list[TaskSpec], list[str]]:
    extractor = ROUND_ROOT / "scripts" / "extract_round2a_diagnostics.py"
    tasks: list[TaskSpec] = []
    completed: list[str] = []
    for run_id in MHSA_IDS:
        for stage in (1, 2):
            for fold in range(5):
                task_id = f"{run_id}_stage{stage}_fold{fold}"
                output = destination(run_id, stage, fold)
                manifest_path = output / "manifest.json"
                if output.exists():
                    required = (manifest_path, output / "stages.npz", output / "attention.npz")
                    if not all(path.is_file() for path in required):
                        raise RuntimeError(f"partial diagnostic output requires review: {output}")
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("status") != "PASS" or manifest.get("training_performed") is not False:
                        raise RuntimeError(f"invalid diagnostic manifest: {manifest_path}")
                    completed.append(task_id)
                tasks.append(
                    TaskSpec(
                        task_id=task_id,
                        command=(
                            "{python}", str(extractor),
                            "--run-id", run_id,
                            "--stage", str(stage),
                            "--fold", str(fold),
                            "--device", "{device}",
                        ),
                        estimated_seconds=240.0,
                        metadata={"run_id": run_id, "stage": stage, "fold": fold, "training": False},
                    )
                )
    return tasks, completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=("cuda:0", "cuda:1"))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--skip-finalize", action="store_true")
    args = parser.parse_args()

    summary = json.loads((ROUND_ROOT / "orchestrator_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "ALL_25_COMPLETE" or int(summary.get("formal_runs", -1)) != 25:
        raise RuntimeError("Round2A formal training is not complete")
    preflight = json.loads((ROUND_ROOT / "manifests" / "preflight_audit.json").read_text(encoding="utf-8"))
    if not preflight.get("all_passed"):
        raise RuntimeError("Round2A preflight is absent or failed")

    tasks, completed = build_tasks()
    scheduler = DynamicGpuQueue(
        tasks,
        devices=args.devices,
        paths=SchedulerPaths.under(ROUND_ROOT / "diagnostics" / "scheduler"),
        cwd=REPO_ROOT,
        poll_seconds=args.poll_seconds,
        longest_first=False,
        completed_task_ids=completed,
    )
    result = scheduler.run()
    print(json.dumps({"status": "ROUND2A_DIAGNOSTICS_COMPLETE", **result}, ensure_ascii=False), flush=True)
    if not args.skip_finalize:
        subprocess.run(
            [sys.executable, str(ROUND_ROOT / "scripts" / "finalize_round2a.py")],
            cwd=str(REPO_ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
