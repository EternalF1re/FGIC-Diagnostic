from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .baselines.cal import CALFeatureCenter, CALModel, build_optimizer, set_fractional_lr, training_objective
from .data import build_loaders, seed_everything
from .engine import evaluate, save_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-required two-rank CAL runner")
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("The controlled CAL protocol requires exactly two ranks")
    if not torch.cuda.is_available():
        raise RuntimeError("CAL DDP requires two CUDA GPUs")
    config = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    if config["method"] != "cal":
        raise ValueError("Expected a CAL experiment config")
    topology = config["training"]["topology"]
    expected = {"global_batch": 16, "gpus": 2, "ddp_processes": 2, "per_process_batch": 8, "sync_batch_norm": True}
    if topology != expected:
        raise ValueError(f"CAL topology differs from the frozen protocol: {topology}")
    output = Path(args.output).resolve()
    started = time.time()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "run_manifest.json").write_text(json.dumps({
            "status": "RUNNING", "method": "cal", "dataset": config["dataset"], "fold": args.fold,
            "seed": 42, "topology": expected, "primary_logits": "causal_effect_p_minus_p_cf",
            "tta": False, "cross_fold_ensemble": False,
        }, indent=2) + "\n", encoding="utf-8")
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        seed_everything(42)
        train_loader, heldout_loader, _, train_idx, heldout_idx = build_loaders(
            args.dataset_config, config["dataset"], "cal", args.fold, 8, args.workers,
            distributed=True, rank=rank, world_size=world_size,
        )
        model = CALModel(int(config["num_classes"]), pretrained=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True)
        center = CALFeatureCenter(int(config["num_classes"])).to(device)
        optimizer = build_optimizer(ddp.module)
        epochs = 1 if args.smoke else 160
        best = -1.0
        best_epoch = -1
        checkpoint = output / "best_stage1.pth"
        history = []
        for epoch in range(epochs):
            ddp.train()
            train_loader.sampler.set_epoch(epoch)
            losses = []
            for batch_index, (images, labels, _, _) in enumerate(train_loader):
                if images.shape[0] != 8:
                    raise RuntimeError("CAL per-rank batch size degraded below 8")
                images, labels = images.to(device), labels.to(device)
                lr = set_fractional_lr(optimizer, epoch, batch_index / max(1, len(train_loader)))
                optimizer.zero_grad(set_to_none=True)
                objective = training_objective(ddp, center, images, labels)
                loss = objective["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite CAL loss at epoch {epoch}, batch {batch_index}")
                loss.backward()
                optimizer.step()
                reduced = loss.detach().clone()
                dist.all_reduce(reduced)
                losses.append(float((reduced / world_size).cpu()))
            dist.barrier()
            if rank == 0:
                current = evaluate(ddp.module, "cal", heldout_loader, device, args.fold)
                accuracy = current["metrics"]["accuracy"]
                if accuracy > best:
                    best, best_epoch = accuracy, epoch
                    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
                    torch.save({
                        "model_state_dict": ddp.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "feature_center_state_dict": center.state_dict(),
                        "method": "cal", "dataset": config["dataset"], "fold": args.fold,
                        "seed": 42, "stage": 1, "epoch": epoch, "selection_metric": accuracy,
                        "world_size": 2, "per_rank_batch": 8, "global_batch": 16, "sync_batch_norm": True,
                    }, temporary)
                    temporary.replace(checkpoint)
                history.append({"epoch": epoch, "loss": float(np.mean(losses)), "lr": lr, "heldout_accuracy": accuracy, "best_accuracy": best})
                (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            dist.barrier()
        if rank == 0:
            selected = torch.load(checkpoint, map_location=device, weights_only=False)
            ddp.module.load_state_dict(selected["model_state_dict"], strict=True)
            final = evaluate(ddp.module, "cal", heldout_loader, device, args.fold)
            save_predictions(output / "heldout_predictions.npz", final)
            manifest = {
                "status": "COMPLETE", "method": "cal", "dataset": config["dataset"], "fold": args.fold,
                "seed": 42, "topology": expected, "best_epoch": best_epoch, "best_checkpoint": str(checkpoint),
                "metrics": final["metrics"], "train_count": len(train_idx), "heldout_count": len(heldout_idx),
                "elapsed_seconds": time.time() - started, "smoke": args.smoke,
            }
            (output / "metrics.json").write_text(json.dumps(final["metrics"], indent=2) + "\n", encoding="utf-8")
            (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(manifest, indent=2), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
