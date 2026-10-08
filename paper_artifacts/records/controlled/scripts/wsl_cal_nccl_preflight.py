"""Frozen CAL 2x8 NCCL/SyncBN topology preflight for WSL2/Linux only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import timm
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.cal.model import CALFeatureCenter, CALModel, build_optimizer, training_objective  # noqa: E402


ARTIFACT = Path("<LOCAL_PATH> Face/模型数据/10118/huggingface/hub/models--timm--resnet50.a1_in1k/blobs/773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb")
EXPECTED_ARTIFACT_SHA = "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb"
EXPECTED_BACKBONE_SHA = "45179af731d5ea5470ba5906fadf0b55b884338ca64d2b7f53f46096665ec6c6"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_frozen_backbone(model: CALModel) -> str:
    if file_sha(ARTIFACT) != EXPECTED_ARTIFACT_SHA:
        raise RuntimeError("pretrained artifact SHA mismatch")
    full_state = load_file(str(ARTIFACT), device="cpu")
    backbone_state = {key: value for key, value in full_state.items() if not key.startswith("fc.")}
    model.backbone.load_state_dict(backbone_state, strict=True)
    digest = state_sha(model.backbone.state_dict())
    if digest != EXPECTED_BACKBONE_SHA:
        raise RuntimeError(f"loaded backbone SHA mismatch: {digest}")
    return digest


def validate_config(dataset: str) -> tuple[dict, dict]:
    config_path = ROOT / "configs" / f"cal_{dataset}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json").read_text(encoding="utf-8"))
    job = next(row for row in plan["jobs"] if row["method"] == "cal" and row["dataset"] == dataset and row["fold"] == 0)
    topology = config["training"]["topology"]
    required = {
        "global_batch": 16, "gpus": 2, "ddp_processes": 2,
        "per_process_batch": 8, "sync_batch_norm": True,
    }
    if topology != required:
        raise RuntimeError(f"frozen topology mismatch: {topology}")
    if file_sha(config_path) != job["config_sha256"]:
        raise RuntimeError("config SHA mismatch")
    if config["training_seed_every_fold"] != 42 or config["backbone"]["input"] != [3, 299, 299]:
        raise RuntimeError("seed/input config mismatch")
    if config["training"]["primary_logits"] != "causal_effect_p_minus_p_cf":
        raise RuntimeError("CAL primary-logit semantics mismatch")
    if config["training_authorized"] is not False or config["formal_status"] != "PLANNED_NOT_AUTHORIZED":
        raise RuntimeError("formal authorization invariant mismatch")
    return config, job


def checkpoint_roundtrip(ddp: DistributedDataParallel, optimizer: torch.optim.Optimizer, center: CALFeatureCenter, dataset: str, config_sha: str) -> dict:
    before_sha = state_sha(ddp.module.state_dict())
    handle = tempfile.NamedTemporaryFile(prefix=f"cal_{dataset}_preflight_", suffix=".pth", dir="/tmp", delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        torch.save({
            "technical_preflight_only": True,
            "dataset": dataset,
            "config_sha256": config_sha,
            "model_state_dict": ddp.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature_center_state_dict": center.state_dict(),
        }, path)
        checkpoint_sha = file_sha(path)
        checkpoint_bytes = path.stat().st_size
        loaded = torch.load(path, map_location=torch.device("cuda", torch.cuda.current_device()), weights_only=False)
        if loaded["dataset"] != dataset or loaded["config_sha256"] != config_sha or loaded["technical_preflight_only"] is not True:
            raise RuntimeError("checkpoint provenance mismatch")
        with torch.no_grad():
            ddp.module.classifier.weight.view(-1)[0].add_(1.0)
        ddp.module.load_state_dict(loaded["model_state_dict"], strict=True)
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        center.load_state_dict(loaded["feature_center_state_dict"], strict=True)
        after_sha = state_sha(ddp.module.state_dict())
        if before_sha != after_sha:
            raise RuntimeError("checkpoint model roundtrip state mismatch")
        return {"pass": True, "transient_checkpoint_sha256": checkpoint_sha, "transient_checkpoint_bytes": checkpoint_bytes, "model_state_sha256": after_sha, "transient_checkpoint_deleted": True}
    finally:
        path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cub", "cars"), required=True)
    args = parser.parse_args()
    dataset = args.dataset
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    output_dir = ROOT / "preflight" / "wsl2_final" / dataset
    result = {"rank": rank, "local_rank": local_rank, "world_size": world_size, "dataset": dataset, "status": "FAIL"}
    initialized = False
    started = time.time()
    try:
        config, job = validate_config(dataset)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("exactly two visible CUDA GPUs are required")
        if not dist.is_available() or not dist.is_nccl_available():
            raise RuntimeError("NCCL backend is required and unavailable")
        if world_size != 2 or rank not in (0, 1) or local_rank not in (0, 1):
            raise RuntimeError("torchrun must create exactly two ranks with local ranks 0/1")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=15))
        initialized = True
        if dist.get_backend() != "nccl" or dist.get_world_size() != 2:
            raise RuntimeError("NCCL/world-size invariant failed")

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats(device)

        num_classes = int(config["num_classes"])
        model = CALModel(num_classes=num_classes, pretrained=False)
        initial_backbone_sha = load_frozen_backbone(model)
        hashes: list[str | None] = [None, None]
        dist.all_gather_object(hashes, initial_backbone_sha)
        if hashes != [EXPECTED_BACKBONE_SHA, EXPECTED_BACKBONE_SHA]:
            raise RuntimeError(f"two-rank initial backbone hashes differ: {hashes}")

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        syncbn_count = sum(isinstance(module, torch.nn.SyncBatchNorm) for module in model.modules())
        remaining_batchnorm = sum(isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)) for module in model.modules())
        if syncbn_count == 0 or remaining_batchnorm != 0:
            raise RuntimeError(f"SyncBN conversion failed: sync={syncbn_count}, remaining={remaining_batchnorm}")
        model = model.to(device)
        ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True)
        center = CALFeatureCenter(num_classes).to(device)
        optimizer = build_optimizer(ddp.module)

        local_batch = 8
        images = torch.randn(local_batch, 3, 299, 299, device=device)
        labels = torch.arange(local_batch, device=device, dtype=torch.long) % num_classes
        center_before = center.value.clone()
        optimizer.zero_grad(set_to_none=True)
        objective = training_objective(ddp, center, images, labels)
        if objective["primary_logits"].shape != (local_batch, num_classes) or not torch.isfinite(objective["primary_logits"]).all():
            raise RuntimeError("causal-effect logits invalid")
        if int(objective["second_forward_batch"].item()) != 2 * local_batch:
            raise RuntimeError("crop/drop second forward did not use local batch 16")
        if torch.equal(center_before, center.value) or center.update_count != 1:
            raise RuntimeError("feature-center update failed")
        if not torch.isfinite(objective["loss"]):
            raise RuntimeError("non-finite loss")
        objective["loss"].backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in ddp.parameters()):
            raise RuntimeError("non-finite gradient")
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in ddp.parameters()):
            raise RuntimeError("non-finite parameter after optimizer step")

        ddp.eval()
        center_before_eval = center.value.clone()
        with torch.inference_mode():
            evaluation = ddp(images)
        if evaluation["causal_logits"].shape != (local_batch, num_classes) or not torch.isfinite(evaluation["causal_logits"]).all():
            raise RuntimeError("evaluation causal-effect logits invalid")
        if evaluation["selected_attention"].shape[1] != 1 or not torch.equal(center_before_eval, center.value):
            raise RuntimeError("evaluation triggered training-only attention/center behavior")

        checkpoint_info: list[dict | None] = [None]
        if rank == 0:
            checkpoint_info[0] = checkpoint_roundtrip(ddp, optimizer, center, dataset, job["config_sha256"])
        dist.broadcast_object_list(checkpoint_info, src=0)
        if not checkpoint_info[0] or not checkpoint_info[0]["pass"]:
            raise RuntimeError("checkpoint roundtrip failed")
        dist.barrier()

        checks = {
            "two_rank_nccl_initialized": True,
            "rank_bound_to_expected_gpu": torch.cuda.current_device() == local_rank,
            "sync_batch_norm": syncbn_count > 0 and remaining_batchnorm == 0,
            "forward": True,
            "causal_effect_logits": True,
            "feature_center_update": True,
            "crop_drop_second_forward": True,
            "backward": True,
            "optimizer_step": True,
            "no_deadlock_device_mismatch_nan_inf": True,
            "global_batch_16": local_batch * world_size == 16,
            "checkpoint_save_load_roundtrip": bool(checkpoint_info[0]["pass"]),
            "heldout_original_view_no_training_augmentation": True,
            "two_rank_initial_backbone_hash_equal": hashes == [EXPECTED_BACKBONE_SHA, EXPECTED_BACKBONE_SHA],
        }
        if not all(checks.values()):
            raise RuntimeError(f"one or more final checks failed: {checks}")
        result.update({
            "status": "PASS",
            "backend": dist.get_backend(),
            "device_index": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(local_rank),
            "local_batch": local_batch,
            "effective_global_batch": local_batch * world_size,
            "syncbn_module_count": syncbn_count,
            "initial_backbone_sha256": initial_backbone_sha,
            "config_sha256": job["config_sha256"],
            "primary_logits": "p-p_cf",
            "checks": checks,
            "checkpoint_roundtrip": checkpoint_info[0],
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            "elapsed_seconds": time.time() - started,
            "accuracy_computed_or_reported": False,
            "formal_training": False,
        })
        write_json(output_dir / f"rank_{rank}.json", result)
        gathered: list[dict | None] = [None, None]
        dist.all_gather_object(gathered, result)
        if rank == 0:
            summary = {
                "schema_version": 1,
                "status": "PASS" if all(row and row["status"] == "PASS" for row in gathered) else "FAIL",
                "dataset": dataset,
                "topology": "NCCL 2 ranks / rank0-GPU0 / rank1-GPU1 / per-rank 8 / global 16 / SyncBN",
                "training_seed": 42,
                "input": [3, 299, 299],
                "ranks": gathered,
                "formal_training": False,
                "accuracy_computed_or_reported": False,
            }
            write_json(output_dir / "summary.json", summary)
        dist.barrier()
        return 0
    except Exception as error:
        result.update({"error": repr(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started})
        write_json(output_dir / f"rank_{rank if rank >= 0 else 'unknown'}.json", result)
        raise
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
