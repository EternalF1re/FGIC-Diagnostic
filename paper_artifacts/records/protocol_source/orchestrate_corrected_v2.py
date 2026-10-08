"""Final corrected orchestrator: exactly two GPU0 smoke jobs, finalize, STOP."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import protocol_core


def main() -> None:
    root = protocol_core.ROOT
    scripts = Path(__file__).resolve().parent
    started = time.time()
    state = {
        "status": "RUNNING", "started_unix": started, "device": "cuda:0",
        "corrected_smoke": True, "previous_smoke_reused": False,
        "jobs": [], "formal_jobs_launched": 0,
    }
    status_path = root / "orchestrator_status.json"
    status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for run_id in ("ours_ft", "progressive_lambda_0_7"):
        job = {"run_id": run_id, "status": "RUNNING", "started_unix": time.time()}
        state["jobs"].append(job)
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        command = [sys.executable, str(scripts / "train_corrected_smoke_v2.py"), "--run-id", run_id, "--device", "cuda:0"]
        result = subprocess.run(command, cwd=str(root), check=False)
        job.update({"status": "COMPLETE" if result.returncode == 0 else "FAILED", "returncode": result.returncode,
                    "completed_unix": time.time()})
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            state.update({"status": "FAILED", "completed_unix": time.time(), "elapsed_seconds": time.time() - started})
            status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(result.returncode)
    final = subprocess.run([sys.executable, str(scripts / "finalize_corrected.py")], cwd=str(root), check=False)
    state.update({
        "status": "COMPLETE" if final.returncode == 0 else "FINALIZE_FAILED",
        "finalize_returncode": final.returncode, "completed_unix": time.time(),
        "elapsed_seconds": time.time() - started, "formal_jobs_launched": 0,
        "stopped_after_corrected_smoke": True,
    })
    status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(final.returncode)


if __name__ == "__main__":
    main()
