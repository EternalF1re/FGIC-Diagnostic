"""Fail-closed seed45/46 comparator, preflight, training, and finalization pipeline."""
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


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(EXP_ROOT / "scripts" / script), *arguments],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    if STATE.exists():
        raise FileExistsError(STATE)
    started = time.time()
    state = {
        "status": "RUNNING_STATIC_PREFLIGHT",
        "started_unix": started,
        "target_seeds": [45, 46],
        "formal_jobs": 10,
        "queue": "shared_dynamic_gpu_queue",
        "devices": ["cuda:0", "cuda:1"],
        "only_new_training": "seed45/46 unified standalone dynamic DFAG",
        "seed47_plus": False,
        "fixed_g_training": False,
        "anchor_sweep": False,
        "gating_source_ablation": False,
        "ssph_joint": False,
        "progressive_head_training": False,
    }
    write(state)
    try:
        run("prepare_static.py")
        state["status"] = "RUNNING_ALPHA05_ENDPOINT_FIDELITY"
        write(state)
        run("orchestrate_alpha05.py")
        state["status"] = "FINALIZING_PREFLIGHT"
        write(state)
        run("finalize_preflight.py")
        state["status"] = "RUNNING_TRAINING"
        write(state)
        run("orchestrate_dfag.py", "--devices", "cuda:0", "cuda:1", "--poll-seconds", "30")
        state["status"] = "FINALIZING_FIVE_SEED_RESULTS"
        write(state)
        run("finalize_confirmation.py")
        decision = json.loads((EXP_ROOT / "final_decision.json").read_text(encoding="utf-8"))
        state.update(
            {
                "status": "COMPLETE",
                "classification": decision["classification"],
                "completed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "automatic_followup_launched": False,
            }
        )
        write(state)
    except Exception as exc:
        state.update(
            {
                "status": "FAILED",
                "completed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "automatic_followup_launched": False,
            }
        )
        write(state)
        raise


if __name__ == "__main__":
    main()
