"""Wait for the formal shared queue and run fail-closed postflight finalization."""
from __future__ import annotations

import json
import subprocess
import sys
import time

from cross_backbone_common import ROOT
from queue_runner import db_path, summary


STATE = ROOT / "postflight_state.json"

def save(payload):
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    save({"status": "WAITING_FOR_FORMAL_QUEUE", "time": time.time()})
    while not db_path("formal").exists():
        pipeline = ROOT / "pipeline_state.json"
        if pipeline.exists():
            state = json.loads(pipeline.read_text(encoding="utf-8"))
            if "FAILED" in state.get("status", ""):
                save({"status": "UPSTREAM_FAILED", "time": time.time(), "pipeline": state})
                raise SystemExit(2)
        time.sleep(30)
    while True:
        counts = summary("formal")["counts"]
        save({"status": "FORMAL_RUNNING", "time": time.time(), "counts": counts})
        if counts.get("FAILED", 0):
            save({"status": "FORMAL_FAILED", "time": time.time(), "counts": counts})
            raise SystemExit(2)
        if counts.get("COMPLETE", 0) == 30 and not counts.get("PENDING", 0) and not counts.get("RUNNING", 0):
            break
        time.sleep(60)
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "finalize.py")])
    if completed.returncode:
        save({"status": "FINALIZATION_FAILED", "time": time.time(), "returncode": completed.returncode})
        raise SystemExit(completed.returncode)
    save({"status": "COMPLETE", "time": time.time(), "formal_jobs": 30, "outputs_generated": True})

if __name__ == "__main__":
    main()
