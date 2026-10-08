"""Fail-closed dynamic two-GPU scheduler for the frozen 60-job plan."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any


ROOT=Path(__file__).resolve().parents[1]; RUN_ROOT=ROOT/"formal_run"; LOG_ROOT=RUN_ROOT/"logs"; RUN_ROOT.mkdir(parents=True,exist_ok=True); LOG_ROOT.mkdir(parents=True,exist_ok=True)
PLAN_PATH=ROOT/"CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json"; AUTH_PATH=ROOT/"FORMAL_RUN_AUTHORIZATION.json"; LEDGER_PATH=RUN_ROOT/"job_ledger.json"; EVENTS_PATH=RUN_ROOT/"scheduler_events.jsonl"; STATUS_PATH=RUN_ROOT/"formal_orchestrator_status.json"; LOCK_PATH=RUN_ROOT/"orchestrator.lock"
lock=threading.Lock(); stop_event=threading.Event(); ledger:dict[str,dict[str,Any]]={}


def atomic_json(path:Path,payload:dict[str,Any])->None:
    temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); temporary.replace(path)


def event(kind:str,**payload)->None:
    row={"time":time.time(),"event":kind,**payload}
    with lock:
        with EVENTS_PATH.open("a",encoding="utf-8") as handle: handle.write(json.dumps(row,ensure_ascii=False)+"\n")


def save_ledger()->None:
    atomic_json(LEDGER_PATH,{"schema_version":1,"jobs":list(ledger.values())})


def update_status(status:str,**extra)->None:
    counts={name:sum(1 for row in ledger.values() if row["status"]==name) for name in ("PENDING","RUNNING","COMPLETE","FAILED")}
    atomic_json(STATUS_PATH,{"status":status,"updated_unix":time.time(),"counts":counts,**extra})


def update_root_status(started:int,status:str)->None:
    path=ROOT/"CONTROLLED_REIMPLEMENTATION_FINAL_STATUS.json"; payload=json.loads(path.read_text(encoding="utf-8")); payload.update({"status":status,"training_authorized":True,"formal_jobs_started":started,"formal_outputs_created":sum(1 for row in ledger.values() if row["status"]=="COMPLETE"),"stop_point_observed":False}); atomic_json(path,payload)


def command_for(job:dict[str,Any],gpu:str)->tuple[list[str],dict[str,str]]:
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu; env["PYTHONPATH"]="./wsl_python_deps:.:./scripts"; env["HF_HUB_OFFLINE"]="1"; env["TRANSFORMERS_OFFLINE"]="1"; env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    command=[sys.executable,"scripts/formal_single_job.py","--job-id",job["job_id"],"--device","cuda:0"]
    return command,env


def run_one(job:dict[str,Any],gpu:str)->int:
    job_id=job["job_id"]; stdout_path=LOG_ROOT/f"{job_id}.stdout.log"; stderr_path=LOG_ROOT/f"{job_id}.stderr.log"
    with lock:
        ledger[job_id].update({"status":"RUNNING","gpu":gpu,"started_unix":time.time(),"stdout":str(stdout_path),"stderr":str(stderr_path)}); save_ledger(); update_status("FORMAL_RUNNING"); update_root_status(sum(1 for row in ledger.values() if row["status"] in {"RUNNING","COMPLETE","FAILED"}),"FORMAL_RUNNING")
    event("JOB_STARTED",job_id=job_id,gpu=gpu)
    command,env=command_for(job,gpu)
    with stdout_path.open("w",encoding="utf-8") as stdout, stderr_path.open("w",encoding="utf-8") as stderr:
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    with lock:
        status="COMPLETE" if completed.returncode==0 else "FAILED"; ledger[job_id].update({"status":status,"returncode":completed.returncode,"completed_unix":time.time()}); save_ledger(); update_status("FORMAL_RUNNING" if completed.returncode==0 else "FAIL_CLOSED")
    event("JOB_FINISHED",job_id=job_id,gpu=gpu,returncode=completed.returncode)
    if completed.returncode!=0: stop_event.set()
    return completed.returncode


def dynamic_phase(jobs:list[dict[str,Any]],phase:str)->None:
    queue=deque(jobs); event("PHASE_STARTED",phase=phase,jobs=len(jobs))
    def worker(gpu:str)->None:
        while not stop_event.is_set():
            with lock:
                if not queue: return
                job=queue.popleft()
            if run_one(job,gpu)!=0: return
    threads=[threading.Thread(target=worker,args=(str(gpu),),name=f"gpu-{gpu}") for gpu in (0,1)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if stop_event.is_set(): raise RuntimeError(f"fail-closed after failure in {phase}")
    event("PHASE_COMPLETE",phase=phase)


def cal_phase(jobs:list[dict[str,Any]])->None:
    event("PHASE_STARTED",phase="cal_exclusive",jobs=len(jobs))
    for job in jobs:
        if stop_event.is_set(): raise RuntimeError("fail-closed before CAL")
        job_id=job["job_id"]; stdout_path=LOG_ROOT/f"{job_id}.stdout.log"; stderr_path=LOG_ROOT/f"{job_id}.stderr.log"
        with lock:
            ledger[job_id].update({"status":"RUNNING","gpu":"0,1 exclusive","started_unix":time.time(),"stdout":str(stdout_path),"stderr":str(stderr_path)}); save_ledger(); update_status("FORMAL_RUNNING"); update_root_status(sum(1 for row in ledger.values() if row["status"] in {"RUNNING","COMPLETE","FAILED"}),"FORMAL_RUNNING")
        event("JOB_STARTED",job_id=job_id,gpu="0,1 exclusive",topology="2-rank NCCL SyncBN 2x8")
        env=os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES":"0,1","PYTHONPATH":"./wsl_python_deps:.:./scripts","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"})
        command=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2","scripts/formal_cal_job_v2.py","--job-id",job_id]
        with stdout_path.open("w",encoding="utf-8") as stdout, stderr_path.open("w",encoding="utf-8") as stderr: completed=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        with lock:
            status="COMPLETE" if completed.returncode==0 else "FAILED"; ledger[job_id].update({"status":status,"returncode":completed.returncode,"completed_unix":time.time()}); save_ledger(); update_status("FORMAL_RUNNING" if completed.returncode==0 else "FAIL_CLOSED")
        event("JOB_FINISHED",job_id=job_id,gpu="0,1 exclusive",returncode=completed.returncode)
        if completed.returncode!=0: stop_event.set(); raise RuntimeError(f"CAL job failed: {job_id}")
    event("PHASE_COMPLETE",phase="cal_exclusive")


def main()->None:
    descriptor=os.open(LOCK_PATH,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(descriptor,str(os.getpid()).encode()); os.close(descriptor)
    try:
        auth=json.loads(AUTH_PATH.read_text(encoding="utf-8"))
        if auth.get("status")!="PASS" or auth.get("training_authorized") is not True: raise RuntimeError("authorization gate not PASS")
        plan=json.loads(PLAN_PATH.read_text(encoding="utf-8")); jobs=plan["jobs"]
        if len(jobs)!=60: raise RuntimeError("formal plan is not 60 jobs")
        if LEDGER_PATH.exists() or EVENTS_PATH.exists(): raise RuntimeError("refusing duplicate/restart over an existing formal ledger")
        for job in jobs: ledger[job["job_id"]]={"job_id":job["job_id"],"ordinal":job["ordinal"],"method":job["method"],"dataset":job["dataset"],"fold":job["fold"],"status":"PENDING"}
        save_ledger(); update_status("FORMAL_RUNNING",training_authorized=True,formal_jobs=60); update_root_status(0,"FORMAL_RUNNING"); event("FORMAL_60_JOB_RUN_STARTED",plan_sha256=auth["plan_sha256"],pid=os.getpid())
        independent=[job for job in jobs if job["method"] not in {"cal","dfag"}]; dfag=[job for job in jobs if job["method"]=="dfag"]; cal=[job for job in jobs if job["method"]=="cal"]
        dynamic_phase(independent,"single_gpu_independent"); dynamic_phase(dfag,"dfag_after_same_fold_ours"); cal_phase(cal)
        subprocess.run([sys.executable,"scripts/finalize_formal_results.py"],cwd=ROOT,check=True)
        update_status("FORMAL_RUN_COMPLETE",training_authorized=True,formal_jobs=60); update_root_status(60,"FORMAL_RUN_COMPLETE"); event("FORMAL_RUN_COMPLETE",completed_jobs=60,failed_jobs=0)
    except Exception as exc:
        update_status("FAIL_CLOSED",error=repr(exc),traceback=traceback.format_exc()); update_root_status(sum(1 for row in ledger.values() if row["status"] in {"RUNNING","COMPLETE","FAILED"}),"FAIL_CLOSED"); event("FORMAL_RUN_FAIL_CLOSED",error=repr(exc)); raise
    finally:
        if LOCK_PATH.exists(): LOCK_PATH.unlink()


if __name__=="__main__": main()
