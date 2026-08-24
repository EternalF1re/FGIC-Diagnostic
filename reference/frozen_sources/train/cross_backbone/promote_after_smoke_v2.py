"""Promote the strengthened smoke gate to the unchanged formal queue."""
from __future__ import annotations

import json
import subprocess
import sys
import time

from cross_backbone_common import ROOT
from queue_runner import initialize, summary


STATE = ROOT / "pipeline_state.json"

def save(payload):
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    save({"status": "SMOKE16_RUNNING", "time": time.time(), "technical_train_batches_per_stage": 16})
    while True:
        counts = summary("smoke")["counts"]
        if counts.get("FAILED", 0):
            save({"status": "SMOKE16_FAILED", "time": time.time(), "counts": counts})
            raise SystemExit(2)
        if counts.get("COMPLETE", 0) == 6 and not counts.get("PENDING", 0) and not counts.get("RUNNING", 0):
            break
        time.sleep(30)
    validation = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_smoke_v2.py")])
    if validation.returncode:
        save({"status": "SMOKE16_AUDIT_FAILED", "time": time.time(), "returncode": validation.returncode})
        raise SystemExit(validation.returncode)
    initialize("formal")
    logs = ROOT / "logs" / "formal_workers"
    logs.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    workers = []
    for gpu in (0, 1):
        out = (logs / f"gpu{gpu}.out.log").open("w", encoding="utf-8")
        err = (logs / f"gpu{gpu}.err.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "queue_runner.py"), "work", "--run-type", "formal", "--device", f"cuda:{gpu}"],
            cwd=ROOT.parents[1], stdout=out, stderr=err, creationflags=flags,
        )
        workers.append({"gpu": gpu, "pid": process.pid, "stdout": out.name, "stderr": err.name})
    save({"status": "FORMAL_RUNNING", "time": time.time(), "smoke_status": "PASS",
          "smoke_batches_per_stage": 16, "formal_jobs": 30,
          "shared_queue": str((ROOT / "formal_queue.sqlite").resolve()), "workers": workers})

if __name__ == "__main__":
    main()
