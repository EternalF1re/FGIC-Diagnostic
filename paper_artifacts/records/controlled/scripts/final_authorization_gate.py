"""Last immutable-identity gate; writes authorization only when every check passes."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from common import core, runtime  # noqa: E402

EXPECTED_PLAN_SHA="1e4af9ba95615f58a08fe01371f9f0714ed4baf0e5768b5522ba1a28a5935e91"; ARTIFACT_SHA="773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb"; BACKBONE_SHA="45179af731d5ea5470ba5906fadf0b55b884338ca64d2b7f53f46096665ec6c6"


def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def atomic(path:Path,payload:dict[str,Any])->None:
    temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); temporary.replace(path)


def git_head(path:Path)->str: return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()


def live_fold_hashes(dataset:str)->tuple[str,str]:
    records=runtime.records(dataset); labels=np.asarray([int(row["label"]) for row in records]); heldout=np.full(len(records),-1,dtype=np.int64)
    for fold in range(5):
        _,indices=core.split_indices(labels,fold); heldout[indices]=fold
    assignments=[{"sample_id":str(row["sample_id"]),"label":int(row["label"]),"heldout_fold":int(heldout[index])} for index,row in enumerate(records)]
    if dataset=="cub":
        root=Path(runtime.external_config()["datasets"]["cub"]["root"]); cmap={}
        for line in (root/"classes.txt").read_text(encoding="utf-8").splitlines():
            index,name=line.split(maxsplit=1); cmap[str(int(index)-1)]=name
    else:
        by_label={}
        for row in records: by_label.setdefault(int(row["label"]),set()).add(Path(row["path"]).parent.name)
        if any(len(names)!=1 for names in by_label.values()): raise RuntimeError("Cars class-map ambiguity")
        cmap={str(label):next(iter(by_label[label])) for label in sorted(by_label)}
    return core.canonical_sha(assignments),core.canonical_sha(cmap)


def main()->None:
    if (ROOT/"FORMAL_RUN_AUTHORIZATION.json").exists(): raise RuntimeError("authorization file already exists")
    if (ROOT/"formal_run").exists(): raise RuntimeError("formal run directory already exists")
    plan_path=ROOT/"CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json"; plan_sha=sha(plan_path); plan=json.loads(plan_path.read_text(encoding="utf-8")); source_lock=json.loads((ROOT/"CONTROLLED_REIMPLEMENTATION_SOURCE_LOCK_MANIFEST.json").read_text(encoding="utf-8")); source_gate=json.loads((ROOT/"preflight"/"final_gate"/"source_and_table_gate.json").read_text(encoding="utf-8"))
    checks:dict[str,bool]={"plan_sha":plan_sha==EXPECTED_PLAN_SHA,"planned_copy_sha":sha(ROOT/"planned_jobs"/"CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json")==EXPECTED_PLAN_SHA,"job_count":len(plan["jobs"])==60,"source_gate":source_gate.get("status")=="PASS","l2_source_trace":source_gate.get("l2sp_source_trace")=="PASS","mc_inference_fidelity":source_gate.get("mcloss_inference_fidelity")=="PASS","adaptation_table_frozen":source_gate.get("final_adaptation_table")=="FROZEN" and (ROOT/"CONTROLLED_REIMPLEMENTATION_FINAL_ADAPTATION_TABLE.csv").is_file()}
    for job in plan["jobs"]: checks[f"config_{job['job_id']}"]=sha(ROOT/job["config_path"])==job["config_sha256"]
    for relative,expected in source_lock["implementation_files_sha256"].items(): checks[f"implementation_{relative}"]=sha(ROOT/relative)==expected
    source_root=Path("<LOCAL_PATH>"); checks["l2_upstream_commit"]=git_head(source_root/"l2sp")==source_lock["upstream"]["l2_sp"]["commit"]; checks["mc_upstream_commit"]=git_head(source_root/"mcloss")==source_lock["upstream"]["mc_loss"]["commit"]; checks["cal_upstream_commit"]=git_head(source_root/"cal")==source_lock["upstream"]["cal"]["commit"]
    artifact=Path(os.environ["HF_HOME"])/"hub"/"models--timm--resnet50.a1_in1k"/"blobs"/ARTIFACT_SHA; checks["pretrained_artifact"]=artifact.is_file() and sha(artifact)==ARTIFACT_SHA; checks["loaded_backbone_state_preflight"]=source_lock["pretrained"]["loaded_backbone_state_dict_sha256"]==BACKBONE_SHA
    live={}
    for dataset in ("cub","cars"):
        fold_sha,class_sha=live_fold_hashes(dataset); live[dataset]={"fold_manifest_sha256":fold_sha,"class_map_sha256":class_sha}; checks[f"{dataset}_fold_manifest"]=fold_sha==source_lock["fold_manifest_sha256"][dataset]; checks[f"{dataset}_class_map"]=class_sha==source_lock["class_map_sha256"][dataset]
    smoke_paths={"l2_sp":ROOT/"preflight"/"final_gate"/"l2_sp_optimization_smoke"/"summary.json","mc_loss":ROOT/"preflight"/"final_gate"/"mc_loss_optimization_smoke"/"summary.json","cal":ROOT/"preflight"/"final_gate"/"cal_optimization_smoke"/"summary.json"}; smokes={name:json.loads(path.read_text(encoding="utf-8")) for name,path in smoke_paths.items()}
    for name,payload in smokes.items(): checks[f"{name}_optimization_smoke"]=payload.get("status")=="PASS" and payload.get("formal_result") is False and payload.get("heldout_expected_count")==1199 and payload.get("heldout_exported_count")==1199 and payload.get("heldout_alignment",{}).get("all_pass") is True
    cal_cub=json.loads((ROOT/"preflight"/"wsl2_final"/"cub"/"summary.json").read_text(encoding="utf-8")); cal_cars=json.loads((ROOT/"preflight"/"wsl2_final"/"cars"/"summary.json").read_text(encoding="utf-8")); checks["cal_environment_closed"]=cal_cub.get("status")=="PASS" and cal_cars.get("status")=="PASS"
    runners=["scripts/formal_single_job.py","scripts/formal_cal_job.py","scripts/formal_cal_job_v2.py","scripts/formal_orchestrator.py","scripts/finalize_formal_results.py"]; subprocess.run([sys.executable,"-m","py_compile",*[str(ROOT/item) for item in runners]],check=True); checks["formal_runners_compile"]=True
    status="PASS" if all(checks.values()) else "FAIL"; retries={"l2_sp":[path.name for path in sorted((ROOT/"preflight"/"final_gate").glob("l2_sp_optimization_smoke_attempt*"))],"mc_loss":[path.name for path in sorted((ROOT/"preflight"/"final_gate").glob("mc_loss_optimization_smoke_attempt*"))],"final_attempt":"PASS"}
    payload={"schema_version":1,"status":status,"ready_for_formal_run":status=="PASS","training_authorized":status=="PASS","formal_jobs_started":0,"authorized_unix":time.time() if status=="PASS" else None,"plan_sha256":plan_sha,"checks":checks,"live_data_identity":live,"optimization_smokes":smokes,"prelaunch_retries":retries,"runner_sha256":{item:sha(ROOT/item) for item in runners},"fail_closed":True,"authorization_scope":"exact frozen 60-job plan only"}
    atomic(ROOT/"FORMAL_RUN_AUTHORIZATION.json",payload)
    report=f"# Final Controlled Reimplementation Authorization Gate\n\n- `READY_FOR_FORMAL_RUN = {status}`\n- `training_authorized = {str(status=='PASS').lower()}`\n- plan SHA: `{plan_sha}`\n- L2-SP source trace: `{source_gate['l2sp_source_trace']}`\n- MC-Loss inference fidelity: `{source_gate['mcloss_inference_fidelity']}`\n- final adaptation table: `{source_gate['final_adaptation_table']}`\n- L2-SP optimization smoke: `{smokes['l2_sp']['status']}`\n- MC-Loss optimization smoke: `{smokes['mc_loss']['status']}`\n- CAL optimization smoke: `{smokes['cal']['status']}`\n- CAL 2-rank NCCL/SyncBN environment: `CLOSED`\n\nAll retry artifacts are retained in `preflight/final_gate`; smoke accuracy is technical-only and was not used to alter any frozen hyperparameter.\n"
    (ROOT/"CONTROLLED_REIMPLEMENTATION_FINAL_AUTHORIZATION_GATE.md").write_text(report,encoding="utf-8")
    root_status_path=ROOT/"CONTROLLED_REIMPLEMENTATION_FINAL_STATUS.json"; root_status=json.loads(root_status_path.read_text(encoding="utf-8")); root_status.update({"status":"READY_FOR_FORMAL_RUN" if status=="PASS" else "FAIL_CLOSED","training_authorized":status=="PASS","formal_jobs_started":0,"formal_outputs_created":0,"stop_point_observed":status!="PASS","final_gate":"PASS" if status=="PASS" else "FAIL","final_gate_manifest":"FORMAL_RUN_AUTHORIZATION.json"}); atomic(root_status_path,root_status)
    print(json.dumps({"status":status,"training_authorized":status=="PASS","plan_sha256":plan_sha,"checks":len(checks),"failed_checks":[key for key,value in checks.items() if not value]},ensure_ascii=False))
    if status!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
