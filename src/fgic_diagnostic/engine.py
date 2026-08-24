from __future__ import annotations

import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .baselines.l2_sp import build_optimizer as l2_optimizer
from .baselines.l2_sp import set_iteration_lr
from .baselines.mc_loss import build_optimizer as mc_optimizer
from .baselines.mc_loss import set_epoch_lr
from .data import build_loaders, seed_everything
from .metrics import classification_metrics
from .models import DFAGModel, build_model


def amp_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def batch_mix(images: torch.Tensor, labels: torch.Tensor, epoch: int, batch_index: int, epochs: int):
    scale = max(0.0, 1.0 - (epoch - epochs * 0.7) / (epochs * 0.3)) if epoch > epochs * 0.7 else 1.0
    rng = np.random.default_rng(42_000_000 + epoch * 10_000 + batch_index)
    choice = float(rng.random())
    generator = torch.Generator(device=images.device).manual_seed(84_000_000 + epoch * 10_000 + batch_index)
    permutation = torch.randperm(images.shape[0], generator=generator, device=images.device)
    if choice < 0.5 and scale > 0:
        lam = float(rng.beta(scale, scale))
        height, width = images.shape[-2:]
        ratio = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(width * ratio), int(height * ratio)
        cx, cy = int(rng.integers(width)), int(rng.integers(height))
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (height * width))
        return mixed, labels, labels[permutation], lam, "cutmix", scale
    if choice < 0.9 and scale > 0:
        alpha = 0.4 * scale
        lam = float(rng.beta(alpha, alpha))
        return lam * images + (1.0 - lam) * images[permutation], labels, labels[permutation], lam, "mixup", scale
    return images, labels, labels, 1.0, "none", scale


def logits_for(model: nn.Module, method: str, images: torch.Tensor) -> torch.Tensor:
    if method in {"l2_sp", "mc_loss"}:
        return model(images)
    if method == "cal":
        return model(images)["causal_logits"]
    return model(images)[0]


@torch.inference_mode()
def evaluate(model: nn.Module, method: str, loader, device: torch.device, fold: int) -> dict[str, Any]:
    model.eval()
    logits, labels, indices, sample_ids = [], [], [], []
    for images, batch_labels, batch_indices, batch_ids in loader:
        current = logits_for(model, method, images.to(device)).detach().float().cpu().numpy()
        logits.append(current)
        labels.append(batch_labels.numpy())
        indices.append(batch_indices.numpy())
        sample_ids.extend(str(item) for item in batch_ids)
    logits_array = np.concatenate(logits)
    labels_array = np.concatenate(labels).astype(np.int64)
    predictions = logits_array.argmax(1).astype(np.int64)
    return {
        "logits": logits_array,
        "labels": labels_array,
        "predictions": predictions,
        "dataset_indices": np.concatenate(indices).astype(np.int64),
        "sample_ids": np.asarray(sample_ids),
        "fold_ids": np.full(len(labels_array), fold, dtype=np.int64),
        "metrics": classification_metrics(labels_array, predictions),
    }


def save_checkpoint(path: Path, model: nn.Module, method: str, dataset: str, fold: int, stage: int, epoch: int, accuracy: float) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(), "method": method, "dataset": dataset, "fold": fold,
        "seed": 42, "stage": stage, "epoch": epoch, "selection_metric": accuracy,
        "selection_metric_type": "deterministic_original_view_heldout_accuracy",
    }, temporary)
    temporary.replace(path)


def save_predictions(path: Path, result: dict[str, Any]) -> None:
    np.savez_compressed(path, **{key: value for key, value in result.items() if key != "metrics"})


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train_l2(model, train_loader, heldout_loader, device, dataset, fold, output, history, iterations: int = 9000):
    optimizer = l2_optimizer(model)
    iterator = iter(train_loader)
    for iteration in range(iterations):
        try:
            images, labels, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels, _, _ = next(iterator)
        model.train()
        images, labels = images.to(device), labels.to(device)
        lr = set_iteration_lr(optimizer, iteration)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            objective = model.objective(images, labels, 0.1, 0.01)
        objective["loss"].backward()
        optimizer.step()
        if iteration % 100 == 0 or iteration == iterations - 1:
            history.append({"stage": 1, "iteration": iteration, "loss": float(objective["loss"].detach()), "lr": lr})
    result = evaluate(model, "l2_sp", heldout_loader, device, fold)
    checkpoint = output / "best_stage1.pth"
    save_checkpoint(checkpoint, model, "l2_sp", dataset, fold, 1, iterations - 1, result["metrics"]["accuracy"])
    return checkpoint, result


def train_mc(model, train_loader, heldout_loader, device, dataset, fold, output, history, epochs: int = 300):
    optimizer = mc_optimizer(model)
    best = -1.0
    checkpoint = output / "best_stage1.pth"
    for epoch in range(epochs):
        model.train()
        rates = set_epoch_lr(optimizer, epoch)
        losses = []
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                objective = model(images, labels)
            objective["loss"].backward()
            optimizer.step()
            losses.append(float(objective["loss"].detach()))
        result = evaluate(model, "mc_loss", heldout_loader, device, fold)
        accuracy = result["metrics"]["accuracy"]
        if accuracy > best:
            best = accuracy
            save_checkpoint(checkpoint, model, "mc_loss", dataset, fold, 1, epoch, accuracy)
        history.append({"stage": 1, "epoch": epoch, "loss": float(np.mean(losses)), "heldout_accuracy": accuracy, "best_accuracy": best, "backbone_lr": rates[0], "classifier_lr": rates[1]})
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    return checkpoint, evaluate(model, "mc_loss", heldout_loader, device, fold)


def train_stage1(model, method, train_loader, heldout_loader, device, dataset, fold, output, history, epochs: int = 150, class_weights=None):
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 3), eta_min=1e-5)
    best = -1.0
    checkpoint = output / "best_stage1.pth"
    for epoch in range(epochs):
        model.train()
        losses = []
        if epoch < 3:
            factor = (epoch + 1) / 3.0
            optimizer.param_groups[0]["lr"] = 1e-4 * factor
            optimizer.param_groups[1]["lr"] = 1e-3 * factor
        scale = 1.0
        for batch_index, (images, labels, _, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images, label_a, label_b, lam, _, scale = batch_mix(images, labels, epoch, batch_index, epochs)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = model(images)[0]
                loss = lam * F.cross_entropy(logits, label_a, weight=class_weights, label_smoothing=0.05) + (1.0 - lam) * F.cross_entropy(logits, label_b, weight=class_weights, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        result = evaluate(model, method, heldout_loader, device, fold)
        accuracy = result["metrics"]["accuracy"]
        if accuracy > best:
            best = accuracy
            save_checkpoint(checkpoint, model, method, dataset, fold, 1, epoch, accuracy)
        history.append({"stage": 1, "epoch": epoch, "loss": float(np.mean(losses)), "heldout_accuracy": accuracy, "best_accuracy": best, "augmentation_scale": scale})
        if epoch >= 3:
            scheduler.step()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    return checkpoint, evaluate(model, method, heldout_loader, device, fold)


@torch.no_grad()
def recalibrate_bn(model: nn.Module, train_loader, device: torch.device, batches: int = 50) -> dict[str, Any]:
    model.train()
    completed = 0
    for images, _, _, _ in train_loader:
        model(images.to(device))
        completed += 1
        if completed == batches:
            break
    if completed != batches:
        raise RuntimeError(f"BN recalibration requires {batches} batches, loader supplied {completed}")
    return {"reset": False, "whole_model_train": True, "no_grad": True, "batches": completed}


def train_stage2(model, method, train_loader, heldout_loader, device, dataset, fold, output, history, epochs: int = 60, class_weights=None):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best = -1.0
    checkpoint = output / "best_stage2.pth"
    for epoch in range(epochs):
        model.train()
        losses = []
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = model(images)[0]
                loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=0.03)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Stage-2 loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        result = evaluate(model, method, heldout_loader, device, fold)
        accuracy = result["metrics"]["accuracy"]
        if accuracy > best:
            best = accuracy
            save_checkpoint(checkpoint, model, method, dataset, fold, 2, epoch, accuracy)
        history.append({"stage": 2, "epoch": epoch, "loss": float(np.mean(losses)), "heldout_accuracy": accuracy, "best_accuracy": best, "lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    return checkpoint, evaluate(model, method, heldout_loader, device, fold)


def run_experiment(
    experiment_config: str | Path,
    dataset_config: str | Path,
    fold: int,
    output_dir: str | Path,
    device_name: str = "cuda:0",
    workers: int = 4,
    stage1_checkpoint: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    started = time.time()
    config_path = Path(experiment_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    method = str(config["method"])
    dataset = str(config["dataset"])
    if method == "cal":
        raise ValueError("CAL uses the protocol-required DDP entry point: fgic-cal-ddp")
    seed_everything(int(config.get("training_seed_every_fold", 42)))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage1_config = config["training"].get("stage1")
    if method == "l2_sp":
        batch_size = 64
    elif isinstance(stage1_config, dict):
        batch_size = int(stage1_config.get("batch_size", 32))
    else:
        batch_size = int(config["training"]["stage2"].get("batch_size", 32))
    train_loader, heldout_loader, records, train_idx, heldout_idx = build_loaders(
        dataset_config, dataset, method, fold, batch_size, workers,
    )
    backbone_name = config.get("backbone", {}).get("timm_model_name", "resnet50")
    model = build_model(method, dataset, int(config["num_classes"]), pretrained=method != "dfag", backbone_name=backbone_name).to(device)
    class_weights = None
    if config.get("class_weights") == "leaves_inverse_frequency_v1":
        labels = np.asarray([row["label"] for row in records], dtype=np.int64)
        counts = np.bincount(labels[train_idx], minlength=int(config["num_classes"])).astype(np.float64)
        weights = len(train_idx) / (int(config["num_classes"]) * counts)
        weights[counts < np.median(counts)] *= 1.1
        weights = np.clip(weights, 0.2, 5.0)
        weights /= weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    history: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "status": "RUNNING", "method": method, "dataset": dataset, "fold": fold, "seed": 42,
        "experiment_config": str(config_path), "device": str(device), "original_view": True,
        "tta": False, "cross_fold_ensemble": False, "train_count": len(train_idx), "heldout_count": len(heldout_idx),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if method == "l2_sp":
        checkpoint, final = train_l2(model, train_loader, heldout_loader, device, dataset, fold, output, history, 20 if smoke else 9000)
    elif method == "mc_loss":
        checkpoint, final = train_mc(model, train_loader, heldout_loader, device, dataset, fold, output, history, 1 if smoke else 300)
    elif method in {"ours_ft", "progressive"}:
        _, stage1 = train_stage1(model, method, train_loader, heldout_loader, device, dataset, fold, output, history, 1 if smoke else 150, class_weights)
        save_predictions(output / "stage1_heldout_predictions.npz", stage1)
        bn_record = recalibrate_bn(model, train_loader, device, min(50, len(train_loader)) if smoke else 50)
        checkpoint, final = train_stage2(model, method, train_loader, heldout_loader, device, dataset, fold, output, history, 1 if smoke else 60, class_weights)
        manifest["bn_recalibration"] = bn_record
    elif method == "dfag":
        if stage1_checkpoint is None:
            raise ValueError("DFAG requires --stage1-checkpoint from same-dataset same-fold Ours-FT")
        payload = torch.load(Path(stage1_checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(model, DFAGModel):
            raise TypeError(type(model))
        model.load_stage1(payload, dataset, fold)
        model.to(device)
        bn_record = recalibrate_bn(model, train_loader, device, min(50, len(train_loader)) if smoke else 50)
        checkpoint, final = train_stage2(model, "dfag", train_loader, heldout_loader, device, dataset, fold, output, history, 1 if smoke else 60, class_weights)
        manifest["stage1_source"] = str(Path(stage1_checkpoint).resolve())
        manifest["bn_recalibration"] = bn_record
    else:
        raise ValueError(method)
    save_predictions(output / "heldout_predictions.npz", final)
    write_history(output / "training_history.csv", history)
    manifest.update({
        "status": "COMPLETE", "elapsed_seconds": time.time() - started,
        "best_checkpoint": str(checkpoint), "metrics": final["metrics"], "smoke": smoke,
    })
    (output / "metrics.json").write_text(json.dumps(final["metrics"], indent=2) + "\n", encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def evaluate_checkpoint(
    experiment_config: str | Path,
    dataset_config: str | Path,
    checkpoint_path: str | Path,
    fold: int,
    output_dir: str | Path,
    device_name: str = "cuda:0",
    workers: int = 4,
) -> dict[str, Any]:
    config = json.loads(Path(experiment_config).read_text(encoding="utf-8"))
    method, dataset = config["method"], config["dataset"]
    device = torch.device(device_name)
    batch = 64 if dataset == "classifyleaves" else 32
    _, heldout, _, _, _ = build_loaders(dataset_config, dataset, method, fold, batch, workers)
    model = build_model(
        method, dataset, int(config["num_classes"]), pretrained=False,
        backbone_name=config.get("backbone", {}).get("timm_model_name", "resnet50"),
    ).to(device)
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    result = evaluate(model, method, heldout, device, fold)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_predictions(output / "heldout_predictions.npz", result)
    (output / "metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n", encoding="utf-8")
    return result["metrics"]


def interpolate_checkpoints(stage1_path: str | Path, stage2_path: str | Path, output_path: str | Path, alpha: float = 0.5) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    stage1 = torch.load(Path(stage1_path), map_location="cpu", weights_only=False)
    stage2 = torch.load(Path(stage2_path), map_location="cpu", weights_only=False)
    for field in ("method", "dataset", "fold"):
        if stage1.get(field) != stage2.get(field):
            raise ValueError(f"Checkpoint {field} mismatch")
    state1, state2 = stage1["model_state_dict"], stage2["model_state_dict"]
    if state1.keys() != state2.keys():
        raise ValueError("Checkpoint state keys differ")
    interpolated = {}
    for key in state1:
        first, second = state1[key], state2[key]
        if first.shape != second.shape:
            raise ValueError(f"Shape mismatch for {key}")
        interpolated[key] = alpha * first + (1.0 - alpha) * second if first.is_floating_point() else second.clone()
    payload = {
        "model_state_dict": interpolated, "method": stage1["method"], "dataset": stage1["dataset"],
        "fold": stage1["fold"], "seed": stage1.get("seed", 42), "alpha": alpha,
        "formula": "alpha*Stage1 + (1-alpha)*Stage2", "training": False, "bn_refresh": False,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return {key: value for key, value in payload.items() if key != "model_state_dict"}
