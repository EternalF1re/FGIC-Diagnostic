"""Formal exclusive two-rank NCCL/SyncBN runner for one CAL fold."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import core, runtime  # noqa: E402
from methods.cal.model import CALFeatureCenter, CALModel, build_optimizer, set_fractional_lr, training_objective  # noqa: E402


EXPECTED_PLAN_SHA = "1e4af9ba95615f58a08fe01371f9f0714ed4baf0e5768b5522ba1a28a5935e91"
ARTIFACT_SHA = "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb"
BACKBONE_SHA = "45179af731d5ea5470ba5906fadf0b55b884338ca64d2b7f53f46096665ec6c6"
RUN_ROOT = ROOT / "formal_run"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def load_backbone(model: CALModel) -> str:
    artifact = Path(os.environ["HF_HOME"]) / "hub" / "models--timm--resnet50.a1_in1k" / "blobs" / ARTIFACT_SHA
    if not artifact.is_file() or file_sha(artifact) != ARTIFACT_SHA: raise RuntimeError("pretrained artifact SHA mismatch")
    full = load_file(str(artifact), device="cpu")
    model.backbone.load_state_dict({key: value for key, value in full.items() if not key.startswith("fc.")}, strict=True)
    digest = core.state_digest(model.backbone.state_dict())
    if digest != BACKBONE_SHA: raise RuntimeError("loaded backbone SHA mismatch")
    return digest


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, fold: int) -> dict[str, Any]:
    model.eval(); logits=[]; labels=[]; indices=[]; ids=[]
    for images, batch_labels, batch_indices, batch_ids in loader:
        current = model(images.to(device))["causal_logits"]
        logits.append(current.detach().float().cpu().numpy()); labels.append(batch_labels.numpy()); indices.append(batch_indices.numpy()); ids.extend(str(item) for item in batch_ids)
    result: dict[str, Any] = {"logits": np.concatenate(logits), "labels": np.concatenate(labels).astype(np.int64), "dataset_indices": np.concatenate(indices).astype(np.int64), "sample_ids": np.asarray(ids)}
    result["predictions"] = result["logits"].argmax(1).astype(np.int64); result["fold_ids"] = np.full(len(result["labels"]), fold, dtype=np.int64)
    result["metrics"] = {"accuracy": float(accuracy_score(result["labels"], result["predictions"])), "macro_f1": float(f1_score(result["labels"], result["predictions"], average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(result["labels"], result["predictions"]))}
    return result


def validate(job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    auth = json.loads((ROOT / "FORMAL_RUN_AUTHORIZATION.json").read_text(encoding="utf-8"))
    if auth.get("training_authorized") is not True or auth.get("status") != "PASS": raise RuntimeError("formal run not authorized")
    plan_path = ROOT / "CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json"
    if file_sha(plan_path) != EXPECTED_PLAN_SHA: raise RuntimeError("plan SHA drift")
    plan = json.loads(plan_path.read_text(encoding="utf-8")); job = next((row for row in plan["jobs"] if row["job_id"] == job_id), None)
    if job is None or job["method"] != "cal": raise ValueError(job_id)
    config_path = ROOT / job["config_path"]
    if file_sha(config_path) != job["config_sha256"]: raise RuntimeError("config SHA drift")
    config = json.loads(config_path.read_text(encoding="utf-8")); topology = config["training"]["topology"]
    if topology != {"global_batch": 16, "gpus": 2, "ddp_processes": 2, "per_process_batch": 8, "sync_batch_norm": True}: raise RuntimeError("CAL topology drift")
    if job["seed"] != 42 or config["evaluation"]["tta"] or config["evaluation"]["cross_fold_ensemble"]: raise RuntimeError("seed/evaluation drift")
    return job, config


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--job-id", required=True); args=parser.parse_args()
    rank=int(os.environ["RANK"]); local_rank=int(os.environ["LOCAL_RANK"]); world_size=int(os.environ["WORLD_SIZE"])
    if world_size != 2: raise RuntimeError("CAL requires exactly two ranks")
    job, config = validate(args.job_id); output=RUN_ROOT/"jobs"/args.job_id; manifest_path=output/"run_manifest.json"; started=time.time()
    if rank == 0:
        if output.exists(): raise RuntimeError(f"refusing to reuse output: {output}")
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(manifest_path, {"status":"RUNNING","formal_result":True,"job":job,"started_unix":started,"topology":{"world_size":2,"backend":"nccl","per_rank_batch":8,"global_batch":16,"sync_batch_norm":True},"primary_logits":"causal_effect_p_minus_p_cf","no_official_test_access":True,"tta":False,"fold_ensemble":False,"pin_memory":False,"runner_sha256":file_sha(Path(__file__)),"launcher_sha256":file_sha(ROOT/"scripts"/"formal_cal_job_v2.py"),"runtime_sha256":file_sha(Path(runtime.__file__)),"protocol_core_sha256":file_sha(Path(core.external_protocol.__file__)),"cars_lookup_implementation":runtime.CARS_LOOKUP_IMPLEMENTATION})
    dist.init_process_group("nccl"); torch.cuda.set_device(local_rank); device=torch.device("cuda",local_rank)
    try:
        torch.manual_seed(42); torch.cuda.manual_seed_all(42); dataset=job["dataset"]; fold=int(job["fold"])
        train_loader, heldout_loader, development, train_idx, heldout_idx = runtime.loaders(dataset,"cal",fold,8,num_workers=4,distributed=True,rank=rank,world_size=2)
        train_loader.pin_memory=False; heldout_loader.pin_memory=False; expected=runtime.expected_heldout(development,heldout_idx,fold)
        model=CALModel(int(config["num_classes"]),pretrained=False); initial_sha=load_backbone(model); hashes=[None,None]; dist.all_gather_object(hashes,initial_sha)
        if hashes != [BACKBONE_SHA,BACKBONE_SHA]: raise RuntimeError("rank backbone SHA mismatch")
        model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device); syncbn_count=sum(isinstance(module,torch.nn.SyncBatchNorm) for module in model.modules())
        ddp=DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=True); center=CALFeatureCenter(int(config["num_classes"])).to(device); optimizer=build_optimizer(ddp.module)
        best=-1.0; best_epoch=-1; checkpoint=output/"best_stage1.pth"; history=[]
        for epoch in range(160):
            ddp.train(); train_loader.sampler.set_epoch(epoch); losses=[]
            for batch_index,(images,labels,_,_) in enumerate(train_loader):
                images,labels=images.to(device),labels.to(device)
                if images.shape[0] != 8 and batch_index < len(train_loader)-1: raise RuntimeError("CAL per-rank batch degradation")
                lr=set_fractional_lr(optimizer,epoch,batch_index/max(1,len(train_loader))); optimizer.zero_grad(set_to_none=True); objective=training_objective(ddp,center,images,labels); loss=objective["loss"]
                if not torch.isfinite(loss): raise FloatingPointError(f"non-finite CAL loss epoch={epoch} batch={batch_index}")
                loss.backward(); optimizer.step(); reduced=loss.detach().clone(); dist.all_reduce(reduced); losses.append(float((reduced/2).cpu()))
            dist.barrier(); accuracy_tensor=torch.zeros((),device=device); current_metrics=None
            if rank == 0:
                result=evaluate(ddp.module,heldout_loader,device,fold); runtime.verify_export(result,expected); current_metrics=result["metrics"]; accuracy_tensor.fill_(current_metrics["accuracy"])
                if current_metrics["accuracy"] > best:
                    best=current_metrics["accuracy"]; best_epoch=epoch; atomic_checkpoint(checkpoint,{"model_state_dict":ddp.module.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"feature_center_state_dict":center.state_dict(),"method":"cal","dataset":dataset,"fold":fold,"seed":42,"stage":1,"epoch":epoch,"selection_metric":best,"config_sha256":job["config_sha256"],"formal_result":True,"world_size":2,"per_rank_batch":8,"global_batch":16,"sync_batch_norm":True})
                history.append({"epoch":epoch,"loss":float(np.mean(losses)),"lr":lr,"heldout_accuracy":current_metrics["accuracy"],"best_accuracy":best})
                (output/"training_history.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
            dist.broadcast(accuracy_tensor,src=0); dist.barrier()
        rank_report={"rank":rank,"local_rank":local_rank,"gpu":torch.cuda.get_device_name(local_rank),"world_size":2,"backend":"nccl","syncbn_count":syncbn_count,"center_updates":center.update_count,"status":"PASS"}; (output/f"rank_{rank}.json").write_text(json.dumps(rank_report,indent=2)+"\n",encoding="utf-8"); dist.barrier()
        if rank == 0:
            selected=torch.load(checkpoint,map_location=device,weights_only=False); ddp.module.load_state_dict(selected["model_state_dict"],strict=True); final=evaluate(ddp.module,heldout_loader,device,fold); alignment=runtime.verify_export(final,expected); export=output/"heldout_predictions.npz"; np.savez_compressed(export,**{key:value for key,value in final.items() if key!="metrics"},formal_result=np.asarray([True]))
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")); manifest.update({"status":"COMPLETE","completed_unix":time.time(),"elapsed_seconds":time.time()-started,"train_count":len(train_idx),"heldout_count":len(heldout_idx),"best_epoch":best_epoch,"best_checkpoint":str(checkpoint),"best_checkpoint_sha256":file_sha(checkpoint),"final_export":{"path":str(export),"sha256":file_sha(export),"alignment":alignment,"metrics":final["metrics"]},"all_finite":True,"checkpoint_roundtrip":True,"syncbn_count":syncbn_count}); atomic_json(manifest_path,manifest); print(json.dumps({"status":"COMPLETE","job_id":args.job_id,"metrics":final["metrics"]}),flush=True)
        dist.barrier()
    except Exception as exc:
        failure={"status":"FAILED","rank":rank,"error":repr(exc),"traceback":traceback.format_exc()}; (output/f"rank_{rank}_failure.json").write_text(json.dumps(failure,indent=2)+"\n",encoding="utf-8")
        if rank==0:
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")); manifest.update(failure); atomic_json(manifest_path,manifest)
        raise
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__": main()
