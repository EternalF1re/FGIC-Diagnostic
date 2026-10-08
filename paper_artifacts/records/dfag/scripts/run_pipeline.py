"""Run the fixed 15-job queue, complexity profile, and finalizer in order."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parent.parent
STATE = EXP_ROOT / "pipeline_state.json"


def write(payload: dict) -> None:
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(EXP_ROOT / "scripts" / script), *args], cwd=REPO_ROOT, check=True)


def main() -> None:
    if STATE.exists():
        raise FileExistsError(STATE)
    started = time.time()
    state = {"status": "RUNNING_TRAINING", "started_unix": started, "formal_jobs": 15,
             "queue": "shared_dynamic_gpu_queue", "devices": ["cuda:0", "cuda:1"],
             "seed45_46": False, "fixed_g_training": False, "anchor_sweep": False,
             "gating_source_ablation": False, "ssph_joint": False}
    write(state)
    try:
        run("orchestrate_dfag.py", "--devices", "cuda:0", "cuda:1", "--poll-seconds", "30")
        state["status"] = "RUNNING_COMPLEXITY"; write(state)
        run("profile_complexity.py")
        state["status"] = "FINALIZING"; write(state)
        run("finalize_dfag.py")
        state.update({"status": "COMPLETE", "completed_unix": time.time(), "elapsed_seconds": time.time() - started})
        write(state)
    except Exception as exc:
        state.update({"status": "FAILED", "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
                      "error": repr(exc), "traceback": traceback.format_exc()})
        write(state)
        raise


if __name__ == "__main__":
    main()
