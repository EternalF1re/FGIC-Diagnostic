"""Train one isolated cross-backbone configuration/fold job."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import sklearn
import timm
import torch
import torchvision
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor, nn

from cross_backbone_common import (
    BACKBONES,
    BLOCKS,
    CONFIG_PATH,
    FOLDS,
    METHODS,
    ROOT,
    CrossBackboneModel,
    StandaloneDFAG,
    batch_mix,
    build_loaders,
    ensure_preflight_pass,
    fold_class_weights,
    initialize_head,
    interpolate_complete_state,
    load_config,
    output_dir,
    parameter_counts,
    seed_everything,
    selected_stage1_path,
    sha256_file,
    state_digest,
)


def environment(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": props.name,
        "gpu_memory_bytes": props.total_memory,
    }


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    labels_parts, ids_parts, logits_parts, names = [], [], [], []
    with torch.inference_mode():
        for images, labels, indices, batch_names in loader:
            logits, _ = model(images.to(device, non_blocking=True))
            logits_parts.append(logits.float().cpu().numpy())
            labels_parts.append(labels.numpy())
            ids_parts.append(indices.numpy())
            names.extend(str(name) for name in batch_names)
    logits = np.concatenate(logits_parts).astype(np.float32)
    labels = np.concatenate(labels_parts).astype(np.int64)
    sample_ids = np.concatenate(ids_parts).astype(np.int64)
    predictions = logits.argmax(axis=1).astype(np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted, dtype=np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        **metrics(labels, predictions),
        "sample_id": sample_ids,
        "y_true": labels,
        "y_pred": predictions,
        "logits": logits,
        "probabilities": probabilities.astype(np.float32),
        "image_name": np.asarray(names),
    }


def save_prediction(path: Path, result: dict[str, Any], fold: int, backbone: str, configuration: str, stage: str) -> None:
    n = len(result["sample_id"])
    arrays = {
        "sample_id": result["sample_id"],
        "fold": np.full(n, fold, dtype=np.int64),
        "y_true": result["y_true"],
        "y_pred": result["y_pred"],
        "logits": result["logits"],
        "probabilities": result["probabilities"],
        "image_name": result["image_name"],
        "backbone": np.full(n, backbone),
        "configuration": np.full(n, configuration),
        "stage": np.full(n, stage),
        "training_seed": np.full(n, 42, dtype=np.int64),
    }
    if not all(np.isfinite(value).all() for value in (arrays["logits"], arrays["probabilities"])):
        raise FloatingPointError("Non-finite prediction artifact")
    np.savez_compressed(path, **arrays)


def checkpoint_payload(model: nn.Module, backbone: str, method: str, fold: int, stage: int, epoch: int, accuracy: float) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "backbone": backbone,
        "method": method,
        "fold": fold,
        "stage": stage,
        "epoch": epoch,
        "metric": accuracy,
        "metric_type": "heldout_original_view_accuracy",
        "split_random_state": 42,
        "training_seed": 42,
        "config_sha256": sha256_file(CONFIG_PATH),
    }


def adapt_batch_norm(model: nn.Module, train_loader, device: torch.device, batches: int, is_dfag: bool) -> int:
    model.train()
    if is_dfag:
        model.anchor.eval()
    completed = 0
    with torch.no_grad():
        for images, _, _, _ in train_loader:
            model(images.to(device, non_blocking=True))
            completed += 1
            if completed >= batches:
                break
    return completed


def train_stage1(model, train_loader, val_loader, class_weights, device, config, backbone, method, fold, output, smoke_batches):
    protocol = config["common_training_protocol"]
    formal_epochs = int(protocol["stage1_epochs"])
    epochs = 1 if smoke_batches is not None else formal_epochs
    warmup = int(protocol["stage1_warmup_epochs"])
    base_lrs = [float(protocol["stage1_backbone_lr"]), float(protocol["stage1_head_lr"])]
    optimizer = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": base_lrs[0]}, {"params": model.head.parameters(), "lr": base_lrs[1]}],
        betas=tuple(protocol["adam_betas"]), eps=float(protocol["adam_eps"]), weight_decay=float(protocol["stage1_weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=formal_epochs - warmup, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=float(protocol["stage1_label_smoothing"]))
    scaler = torch.amp.GradScaler(device.type, init_scale=float(protocol["amp_grad_scaler_init_scale"]), enabled=True)
    checkpoint = output / "best_stage1.pth"
    best_accuracy, best_epoch = -1.0, -1
    history = []
    for epoch in range(epochs):
        model.train()
        if epoch < warmup:
            factor = (epoch + 1) / warmup
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * factor
        started, losses, overflow = time.perf_counter(), [], 0
        for batch_index, (images, labels, _, _) in enumerate(train_loader):
            if smoke_batches is not None and batch_index >= smoke_batches:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images, label_a, label_b, lam, _ = batch_mix(images, labels, epoch, batch_index, formal_epochs, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits, _ = model(images)
                loss_a = criterion(logits, label_a)
                loss_b = criterion(logits, label_b)
                loss = lam * loss_a + (1.0 - lam) * loss_b
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage1 loss epoch={epoch} batch={batch_index}")
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                overflow += 1
            elif not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite Stage1 gradient not handled by GradScaler")
            losses.append(float(loss.detach().cpu()))
        result = evaluate(model, val_loader, device)
        val_accuracy = result["accuracy"]
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save(checkpoint_payload(model, backbone, method, fold, 1, epoch, val_accuracy), checkpoint)
        history.append({"stage": 1, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_accuracy,
                        "best_val_accuracy": best_accuracy, "backbone_lr": optimizer.param_groups[0]["lr"],
                        "head_lr": optimizer.param_groups[1]["lr"], "amp_overflow_batches": overflow,
                        "elapsed_seconds": time.perf_counter() - started})
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(json.dumps({"backbone": backbone, "method": method, "fold": fold, "stage": 1, "epoch": epoch + 1,
                          "epochs": epochs, "val_accuracy": val_accuracy, "best": best_accuracy}), flush=True)
        if epoch >= warmup:
            scheduler.step()
    return checkpoint, best_epoch, best_accuracy, history


def train_stage2(model, train_loader, val_loader, class_weights, device, config, backbone, method, fold, output, history, smoke_batches, is_dfag):
    protocol = config["common_training_protocol"]
    formal_epochs = int(protocol["stage2_epochs"])
    epochs = 1 if smoke_batches is not None else formal_epochs
    parameters = model.optimized_parameters() if is_dfag else list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(protocol["stage2_all_parameter_lr"]),
                                  betas=tuple(protocol["adam_betas"]), eps=float(protocol["adam_eps"]),
                                  weight_decay=float(protocol["stage2_weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=formal_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=float(protocol["stage2_label_smoothing"]))
    scaler = torch.amp.GradScaler(device.type, init_scale=float(protocol["amp_grad_scaler_init_scale"]), enabled=True)
    checkpoint = output / "best_stage2.pth"
    best_accuracy, best_epoch = -1.0, -1
    gate_grad_nonzero = 0
    for epoch in range(epochs):
        model.train()
        if is_dfag:
            model.anchor.eval()
        started, losses, overflow = time.perf_counter(), [], 0
        for batch_index, (images, labels, _, _) in enumerate(train_loader):
            if smoke_batches is not None and batch_index >= smoke_batches:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits, _ = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage2 loss epoch={epoch} batch={batch_index}")
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if is_dfag:
                if any(parameter.grad is not None for parameter in model.anchor.parameters()):
                    raise RuntimeError("DFAG anchor gradient present")
                if any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() for parameter in model.dfag_gate.parameters()):
                    gate_grad_nonzero += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(protocol["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                overflow += 1
            elif not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite Stage2 gradient not handled by GradScaler")
            losses.append(float(loss.detach().cpu()))
        result = evaluate(model, val_loader, device)
        val_accuracy = result["accuracy"]
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            payload = ({**model.checkpoint_payload_state()} if is_dfag else {"model_state_dict": model.state_dict()})
            payload.update({"backbone": backbone, "method": method, "fold": fold, "stage": 2, "epoch": epoch,
                            "metric": val_accuracy, "metric_type": "heldout_original_view_accuracy",
                            "split_random_state": 42, "training_seed": 42, "config_sha256": sha256_file(CONFIG_PATH)})
            torch.save(payload, checkpoint)
        history.append({"stage": 2, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_accuracy,
                        "best_val_accuracy": best_accuracy, "lr": optimizer.param_groups[0]["lr"],
                        "amp_overflow_batches": overflow, "elapsed_seconds": time.perf_counter() - started})
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(json.dumps({"backbone": backbone, "method": method, "fold": fold, "stage": 2, "epoch": epoch + 1,
                          "epochs": epochs, "val_accuracy": val_accuracy, "best": best_accuracy}), flush=True)
        scheduler.step()
    return checkpoint, best_epoch, best_accuracy, gate_grad_nonzero


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--fold", required=True, type=int, choices=FOLDS)
    parser.add_argument("--device", required=True, choices=("cuda:0", "cuda:1"))
    parser.add_argument("--run-type", required=True, choices=("smoke", "formal"))
    parser.add_argument("--smoke-train-batches", type=int, default=2)
    args = parser.parse_args()
    ensure_preflight_pass()
    config = load_config()
    backbone, method, fold = args.backbone, args.method, args.fold
    smoke_batches = int(args.smoke_train_batches) if args.run_type == "smoke" else None
    output = output_dir(args.run_type, backbone, method, fold)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "COMPLETE":
        print(json.dumps({"status": "ALREADY_COMPLETE", "backbone": backbone, "method": method, "fold": fold}), flush=True)
        return
    if output.exists():
        raise RuntimeError(f"refusing partial output reuse: {output}")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(CONFIG_PATH, output / "config.json")
    device = torch.device(args.device)
    seed_everything(42)
    started = time.time()
    manifest: dict[str, Any] = {
        "status": "RUNNING", "run_type": args.run_type, "backbone": backbone, "method": method, "fold": fold,
        "split_id": f"skf42_fold{fold}", "training_seed": 42, "device": str(device), "started_unix": started,
        "config_sha256": sha256_file(CONFIG_PATH), "runner_sha256": sha256_file(Path(__file__).resolve()),
        "old_cross_backbone_checkpoint_used": False, "joint_configuration": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        train_loader, val_loader, dataset, train_idx, val_idx = build_loaders(fold, 1 if method != "dfag" else 2, config, backbone)
        np.savez_compressed(output / "split_indices.npz", train_indices=train_idx, validation_indices=val_idx,
                            split_random_state=np.asarray([42]), training_seed=np.asarray([42]))
        class_weights = fold_class_weights(dataset.labels, train_idx, NUM_CLASSES)
        np.save(output / "class_weights.npy", class_weights.numpy())
        history: list[dict[str, Any]] = []
        source_info = None
        if method in {"ours_ft", "progressive"}:
            model = CrossBackboneModel(backbone, method, pretrained=True)
            initialize_head(model.head)
            counts = parameter_counts(model, method)
            model.to(device)
            stage1_path, stage1_epoch, stage1_accuracy, history = train_stage1(
                model, train_loader, val_loader, class_weights, device, config, backbone, method, fold, output, smoke_batches
            )
            stage1_payload = torch.load(stage1_path, map_location=device, weights_only=False)
            model.load_state_dict(stage1_payload["model_state_dict"], strict=True)
            stage1_result = evaluate(model, val_loader, device)
            if not np.array_equal(stage1_result["sample_id"], val_idx):
                raise ValueError("Stage1 heldout sample order mismatch")
            save_prediction(output / "stage1_predictions.npz", stage1_result, fold, backbone, method, "stage1")
            del train_loader, val_loader
            train_loader, val_loader, dataset, train_idx2, val_idx2 = build_loaders(fold, 2, config, backbone)
            if not np.array_equal(train_idx, train_idx2) or not np.array_equal(val_idx, val_idx2):
                raise ValueError("Stage1/Stage2 split mismatch")
            bn_batches = adapt_batch_norm(model, train_loader, device, int(config["common_training_protocol"]["bn_adaptation_batches_before_stage2"]), False)
            stage2_path, stage2_epoch, stage2_accuracy, _ = train_stage2(
                model, train_loader, val_loader, class_weights, device, config, backbone, method, fold, output, history, smoke_batches, False
            )
            stage2_payload = torch.load(stage2_path, map_location=device, weights_only=False)
            model.load_state_dict(stage2_payload["model_state_dict"], strict=True)
            stage2_result = evaluate(model, val_loader, device)
            if not np.array_equal(stage2_result["sample_id"], val_idx):
                raise ValueError("Stage2 heldout sample order mismatch")
            save_prediction(output / "stage2_predictions.npz", stage2_result, fold, backbone, method, "stage2")
            interpolation = None
            if method == "ours_ft":
                interpolated_state, interpolation = interpolate_complete_state(
                    stage1_payload["model_state_dict"], stage2_payload["model_state_dict"], 0.5
                )
                model.load_state_dict(interpolated_state, strict=True)
                alpha_result = evaluate(model, val_loader, device)
                save_prediction(output / "alpha0_5_predictions.npz", alpha_result, fold, backbone, method, "alpha0.5")
            metrics_payload = {
                "stage1": {key: stage1_result[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
                "stage2": {key: stage2_result[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
            }
            if method == "ours_ft":
                metrics_payload["alpha0.5"] = {key: alpha_result[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")}
        else:
            source = selected_stage1_path(args.run_type, backbone, fold)
            source_manifest_path = output_dir(args.run_type, backbone, "ours_ft", fold) / "run_manifest.json"
            if not source.is_file() or not source_manifest_path.is_file():
                raise RuntimeError("DFAG requires completed same-run Ours-FT Stage1 source")
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            if source_manifest.get("status") != "COMPLETE":
                raise RuntimeError("DFAG Ours-FT dependency is not complete")
            source_hash = sha256_file(source)
            if source_hash != source_manifest["best_stage1_sha256"]:
                raise RuntimeError("DFAG Stage1 source SHA mismatch")
            source_info = {"path": str(source.resolve()), "sha256": source_hash, "stage1_retrained": False}
            model = StandaloneDFAG(backbone)
            model.load_common_stage1(source)
            counts = parameter_counts(model, method)
            expected_gate = 2 * model.dimension * (model.dimension // 16)
            if counts["gate_parameters"] != expected_gate:
                raise ValueError("adaptive gate parameter count mismatch")
            anchor_before = state_digest(model.anchor.state_dict())
            gate_before = state_digest(model.dfag_gate.state_dict())
            model.to(device)
            bn_batches = adapt_batch_norm(model, train_loader, device, int(config["common_training_protocol"]["bn_adaptation_batches_before_stage2"]), True)
            stage2_path, stage2_epoch, stage2_accuracy, gate_grad_nonzero = train_stage2(
                model, train_loader, val_loader, class_weights, device, config, backbone, method, fold, output, history, smoke_batches, True
            )
            selected = torch.load(stage2_path, map_location=device, weights_only=False)
            model.load_payload_state(selected)
            result = evaluate(model, val_loader, device)
            if not np.array_equal(result["sample_id"], val_idx):
                raise ValueError("DFAG heldout sample order mismatch")
            save_prediction(output / "dfag_predictions.npz", result, fold, backbone, method, "dfag")
            anchor_after = state_digest(model.anchor.state_dict())
            gate_after = state_digest(model.dfag_gate.state_dict())
            if anchor_before != anchor_after or gate_before == gate_after or gate_grad_nonzero <= 0:
                raise RuntimeError("DFAG frozen-anchor/gate-update sanity failed")
            stage1_epoch = source_manifest["best_stage1_epoch_zero_based"]
            stage1_accuracy = source_manifest["best_stage1_accuracy"]
            metrics_payload = {"dfag": {key: result[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")}}
            interpolation = None

        (output / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
        manifest.update({
            "status": "COMPLETE", "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
            "environment": environment(device), "parameter_counts": counts, "batch_size": config["backbones"][backbone]["batch_size"],
            "bn_adaptation_batches_completed": bn_batches, "best_stage1_epoch_zero_based": stage1_epoch,
            "best_stage1_accuracy": stage1_accuracy, "best_stage1_checkpoint": str((selected_stage1_path(args.run_type, backbone, fold) if method == "dfag" else stage1_path).resolve()),
            "best_stage1_sha256": sha256_file(selected_stage1_path(args.run_type, backbone, fold) if method == "dfag" else stage1_path),
            "best_stage2_epoch_zero_based": stage2_epoch, "best_stage2_accuracy": stage2_accuracy,
            "best_stage2_checkpoint": str(stage2_path.resolve()), "best_stage2_sha256": sha256_file(stage2_path),
            "dfag_stage1_source": source_info, "interpolation": interpolation, "metrics": metrics_payload,
            "technical_checks": {"checkpoint_restore": True, "prediction_schema": True, "all_finite": True, "amp": True},
        })
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "COMPLETE", "run_type": args.run_type, "backbone": backbone, "method": method, "fold": fold, "metrics": metrics_payload}), flush=True)
    except Exception as exc:
        manifest.update({"status": "FAILED", "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
                         "error": repr(exc), "traceback": traceback.format_exc()})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
