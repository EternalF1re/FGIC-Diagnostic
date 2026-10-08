"""Two-GPU fail-closed scheduler for exactly 20 Phase2D Round1 runs."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from round1_common import ROUND_ROOT, output_dir


QUEUES = {
    "cuda:0": [(run, fold) for run in ("baseline_seed43", "lambda_0_7_seed42") for fold in range(5)],
    "cuda:1": [(run, fold) for run in ("baseline_seed44", "lambda_0_9_seed42") for fold in range(5)],
}
ALL_TASKS = [task for device in ("cuda:0","cuda:1") for task in QUEUES[device]]


def read_manifest(run_id: str, fold: int) -> dict:
    path = output_dir(run_id, fold) / "run_manifest.json"
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {"status":"UNREADABLE"}


def write_ledger(running: dict, queues: dict, halted: bool) -> None:
    queued = {task for q in queues.values() for task in q}
    rows=[]
    for run_id,fold in ALL_TASKS:
        info=read_manifest(run_id,fold); key=(run_id,fold); status=info.get("status","PENDING"); device=""; pid=""
        if key in running:
            status="RUNNING"; device=running[key]["device"]; pid=running[key]["process"].pid
        elif key in queued: status="HALTED_AFTER_FAILURE" if halted else "QUEUED"
        rows.append({"run_id":run_id,"fold":fold,"status":status,"device":device,"pid":pid,
                     "split_id":info.get("split_id",f"skf42_fold{fold}"),"training_seed":info.get("training_seed",""),
                     "shortcut_lambda":info.get("shortcut_lambda",""),
                     "best_stage1_epoch_zero_based":info.get("best_stage1_epoch_zero_based",""),"best_stage1_accuracy":info.get("best_stage1_accuracy",""),
                     "best_stage2_epoch_zero_based":info.get("best_stage2_epoch_zero_based",""),"best_stage2_accuracy":info.get("best_stage2_accuracy",""),
                     "elapsed_seconds":info.get("elapsed_seconds",""),"error":info.get("error","")})
    with (ROUND_ROOT/"run_ledger.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    audit_path=ROUND_ROOT/"manifests"/"preflight_audit.json"
    if not audit_path.exists() or not json.loads(audit_path.read_text(encoding="utf-8"))["all_passed"]:
        raise RuntimeError("preflight audit missing/failed")
    queues={device:[] for device in QUEUES}
    for device,tasks in QUEUES.items():
        for task in tasks:
            info=read_manifest(*task)
            if info.get("status")=="COMPLETE": continue
            if info: raise RuntimeError(f"partial/non-complete artifact requires human review: {task} status={info.get('status')}")
            queues[device].append(task)
    running={}; halted=False; logs=ROUND_ROOT/"logs"; events=logs/"orchestrator_events.jsonl"
    if events.exists(): raise RuntimeError(f"refusing to overwrite {events}")
    write_ledger(running,queues,halted)
    with events.open("x",encoding="utf-8",buffering=1) as event_handle:
        while any(queues.values()) or running:
            for device in ("cuda:0","cuda:1"):
                if halted or not queues[device] or any(job["device"]==device for job in running.values()): continue
                run_id,fold=queues[device].pop(0); log_path=logs/f"{run_id}_fold_{fold}.log"
                if log_path.exists(): raise RuntimeError(f"refusing to overwrite {log_path}")
                handle=log_path.open("x",encoding="utf-8",buffering=1)
                cmd=[sys.executable,str(ROUND_ROOT/"scripts"/"train_one_round1.py"),"--run-id",run_id,"--fold",str(fold),"--device",device]
                process=subprocess.Popen(cmd,cwd=str(ROUND_ROOT.parent.parent),stdout=handle,stderr=subprocess.STDOUT)
                running[(run_id,fold)]={"process":process,"device":device,"handle":handle,"log":str(log_path)}
                event={"event":"START","time":time.time(),"run_id":run_id,"fold":fold,"device":device,"pid":process.pid}
                event_handle.write(json.dumps(event)+"\n"); print(json.dumps(event),flush=True)
            time.sleep(30)
            for key,job in list(running.items()):
                code=job["process"].poll()
                if code is None: continue
                job["handle"].close(); del running[key]
                event={"event":"EXIT","time":time.time(),"run_id":key[0],"fold":key[1],"device":job["device"],"exit_code":code,"log":job["log"]}
                event_handle.write(json.dumps(event)+"\n"); print(json.dumps(event),flush=True)
                if code!=0: halted=True
            write_ledger(running,queues,halted)
            if halted and not running: break
    summary={"status":"FAILED_HALTED" if halted else "ALL_20_COMPLETE","formal_runs":20,"stop_gate":True,"completed_unix":time.time()}
    (ROUND_ROOT/"orchestrator_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary),flush=True)
    if halted: raise SystemExit(1)


if __name__ == "__main__":
    main()
