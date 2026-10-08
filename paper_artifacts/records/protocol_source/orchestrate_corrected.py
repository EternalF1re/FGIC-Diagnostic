"""Run exactly the two corrected smoke jobs sequentially on GPU0, finalize, STOP."""

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
    ledger = {
        "status": "RUNNING", "started_unix": started, "device": "cuda:0",
        "corrected_smoke": True, "previous_smoke_reused": False,
        "jobs": [], "formal_jobs_launched": 0,
    }
    path = root / "orchestrator_status.json"
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for run_id in ("ours_ft", "progressive_lambda_0_7"):
        job = {"run_id": run_id, "status": "RUNNING", "started_unix": time.time()}
        ledger["jobs"].append(job)
        path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        command = [sys.executable, str(scripts / "train_corrected_smoke.py"), "--run-id", run_id, "--device", "cuda:0"]
        result = subprocess.run(command, cwd=str(root), check=False)
        job.update({"status": "COMPLETE" if result.returncode == 0 else "FAILED", "returncode": result.returncode,
                    "completed_unix": time.time()})
        path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            ledger.update({"status": "FAILED", "completed_unix": time.time(), "elapsed_seconds": time.time() - started})
            path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(result.returncode)
    result = subprocess.run([sys.executable, str(scripts / "finalize_corrected.py")], cwd=str(root), check=False)
    ledger.update({
        "status": "COMPLETE" if result.returncode == 0 else "FINALIZE_FAILED",
        "finalize_returncode": result.returncode, "completed_unix": time.time(),
        "elapsed_seconds": time.time() - started, "formal_jobs_launched": 0,
        "stopped_after_corrected_smoke": True,
    })
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
