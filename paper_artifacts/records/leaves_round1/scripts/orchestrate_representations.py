"""Extract all Round1 stage representations through the shared GPU queue."""

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


LAMBDA_KEYS = ("lambda_0_1", "lambda_0_7", "lambda_0_9", "lambda_1_0")


def output_dir(lambda_key: str, stage: int, fold: int) -> Path:
    return ROUND_ROOT / "representation" / "fold_features" / lambda_key / f"stage{stage}" / f"fold_{fold}"


def build_tasks() -> tuple[list[TaskSpec], list[str]]:
    extractor = ROUND_ROOT / "scripts" / "extract_representation.py"
    tasks: list[TaskSpec] = []
    completed: list[str] = []
    for lambda_key in LAMBDA_KEYS:
        for stage in (1, 2):
            for fold in range(5):
                task_id = f"{lambda_key}_stage{stage}_fold{fold}"
                destination = output_dir(lambda_key, stage, fold)
                manifest_path = destination / "manifest.json"
                feature_path = destination / "stages.npz"
                if destination.exists():
                    if not manifest_path.exists() or not feature_path.exists():
                        raise RuntimeError(f"partial representation artifact requires review: {destination}")
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("status") != "PASS":
                        raise RuntimeError(f"non-PASS representation manifest: {manifest_path}")
                    completed.append(task_id)
                tasks.append(
                    TaskSpec(
                        task_id=task_id,
                        command=(
                            "{python}",
                            str(extractor),
                            "--lambda-key",
                            lambda_key,
                            "--stage",
                            str(stage),
                            "--fold",
                            str(fold),
                            "--device",
                            "{device}",
                        ),
                        estimated_seconds=180.0,
                        metadata={"lambda_key": lambda_key, "stage": stage, "fold": fold},
                    )
                )
    return tasks, completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=("cuda:0", "cuda:1"))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    orchestration = json.loads((ROUND_ROOT / "orchestrator_summary.json").read_text(encoding="utf-8"))
    if orchestration.get("status") != "ALL_20_COMPLETE":
        raise RuntimeError("Round1 formal training is not complete")
    postflight = json.loads((ROUND_ROOT / "manifests" / "postflight_audit.json").read_text(encoding="utf-8"))
    if not postflight.get("all_passed"):
        raise RuntimeError("Round1 postflight audit is absent or failed")

    tasks, completed = build_tasks()
    scheduler_root = ROUND_ROOT / "representation" / "scheduler"
    scheduler = DynamicGpuQueue(
        tasks,
        devices=args.devices,
        paths=SchedulerPaths.under(scheduler_root),
        cwd=REPO_ROOT,
        poll_seconds=args.poll_seconds,
        longest_first=True,
        completed_task_ids=completed,
    )
    summary = scheduler.run()
    print(json.dumps({"status": "REPRESENTATION_EXTRACTION_COMPLETE", **summary}), flush=True)

    if not args.skip_analysis:
        command = [sys.executable, str(ROUND_ROOT / "scripts" / "analyze_representations.py")]
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
