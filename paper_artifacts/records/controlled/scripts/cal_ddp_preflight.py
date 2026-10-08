"""Two-process, two-GPU CAL DDP/SyncBN executable preflight.

Launch only through torch.distributed.run with nproc_per_node=2. This is a
single synthetic optimizer step and never reads held-out labels or writes a
formal checkpoint.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core import canonical_sha, seed_everything, state_digest  # noqa: E402
from methods.cal.model import CALFeatureCenter, CALModel, build_optimizer, training_objective  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    output_dir = ROOT / "preflight" / "cal_ddp"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"rank": rank, "local_rank": local_rank, "world_size": world_size, "status": "FAIL"}
    initialized = False
    try:
        if world_size != 2 or rank not in (0, 1) or local_rank not in (0, 1):
            raise RuntimeError("CAL preflight requires exactly two torchrun processes")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("CAL preflight requires two visible CUDA devices")
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=10))
        initialized = True
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        seed_everything(42)

        model = CALModel(num_classes=200, pretrained=True)
        initial_backbone_sha = state_digest(model.backbone.state_dict())
        gathered_hashes: list[str | None] = [None] * world_size
        dist.all_gather_object(gathered_hashes, initial_backbone_sha)
        if len(set(gathered_hashes)) != 1:
            raise RuntimeError(f"initial backbone hashes differ: {gathered_hashes}")

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        syncbn_count = sum(isinstance(module, torch.nn.SyncBatchNorm) for module in model.modules())
        remaining_bn = sum(isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)) for module in model.modules())
        if syncbn_count == 0 or remaining_bn != 0:
            raise RuntimeError(f"SyncBN conversion incomplete: sync={syncbn_count}, remaining_bn={remaining_bn}")
        model = model.to(device)
        ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True)
        center = CALFeatureCenter(200).to(device)
        optimizer = build_optimizer(ddp.module)

        local_batch = 8
        images = torch.randn(local_batch, 3, 299, 299, device=device)
        labels = torch.arange(local_batch, device=device) % 200
        center_before = center.value.clone()
        optimizer.zero_grad(set_to_none=True)
        objective = training_objective(ddp, center, images, labels)
        objective["loss"].backward()
        gradient_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in ddp.parameters())
        if not gradient_finite:
            raise RuntimeError("non-finite CAL gradient")
        optimizer.step()
        if not torch.isfinite(objective["loss"]):
            raise RuntimeError("non-finite CAL loss")
        if objective["primary_logits"].shape != (local_batch, 200):
            raise RuntimeError("causal-effect logit shape mismatch")
        if torch.equal(center_before, center.value) or center.update_count != 1:
            raise RuntimeError("feature center did not update exactly once")
        if int(objective["second_forward_batch"].item()) != 2 * local_batch:
            raise RuntimeError("crop/drop second-forward batch mismatch")

        ddp.eval()
        center_before_eval = center.value.clone()
        with torch.inference_mode():
            evaluation = ddp(images)
        if evaluation["selected_attention"].shape[1] != 1:
            raise RuntimeError("held-out evaluation selected training attention paths")
        if not torch.equal(center_before_eval, center.value):
            raise RuntimeError("held-out evaluation modified feature center")

        independent_center = CALFeatureCenter(200).to(device)
        if independent_center.value.data_ptr() == center.value.data_ptr() or torch.count_nonzero(independent_center.value):
            raise RuntimeError("fold-local feature-center isolation assertion failed")
        checks = {
            "ddp_two_process_initialized": dist.get_world_size() == 2,
            "syncbn_conversion": syncbn_count > 0 and remaining_bn == 0,
            "effective_global_batch_16": local_batch * world_size == 16,
            "forward_backward_optimizer_step": True,
            "causal_effect_logits": evaluation["causal_logits"].shape == (local_batch, 200),
            "feature_center_update": center.update_count == 1,
            "crop_drop_second_forward": int(objective["second_forward_batch"].item()) == 16,
            "heldout_no_crop_or_tta": evaluation["selected_attention"].shape[1] == 1,
            "process_initial_backbone_hash_equal": len(set(gathered_hashes)) == 1,
            "fold_state_isolation": True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"CAL preflight check failed: {checks}")
        result.update({
            "status": "PASS", "backend": backend, "device": torch.cuda.get_device_name(local_rank),
            "local_batch": local_batch, "effective_global_batch": local_batch * world_size,
            "syncbn_module_count": syncbn_count, "initial_backbone_sha256": initial_backbone_sha,
            "loss": float(objective["loss"].detach().cpu()), "checks": checks,
        })
        write_json(output_dir / f"rank_{rank}.json", result)
        gathered_results: list[dict | None] = [None] * world_size
        dist.all_gather_object(gathered_results, result)
        if rank == 0:
            summary = {
                "schema_version": 1,
                "status": "PASS" if all(row and row["status"] == "PASS" for row in gathered_results) else "FAIL",
                "topology": "2 GPUs / 2 DDP processes / per-process batch 8 / global batch 16 / SyncBN",
                "ranks": gathered_results,
            }
            summary["result_sha256"] = canonical_sha(summary["ranks"])
            write_json(output_dir / "summary.json", summary)
        dist.barrier()
        return 0
    except Exception as error:
        result.update({"error": repr(error), "traceback": traceback.format_exc()})
        write_json(output_dir / f"rank_{rank if rank >= 0 else 'unknown'}.json", result)
        raise
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
