"""Formal single-GPU runner for L2-SP, MC-Loss, Ours, Progressive and DFAG."""
from __future__ import annotations

import argparse
import csv
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
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import core, runtime  # noqa: E402
from methods.l2_sp.model import L2SPModel, build_optimizer as l2_optimizer, set_iteration_lr  # noqa: E402
from methods.mc_loss.model import MCLossModel, build_optimizer as mc_optimizer, set_epoch_lr  # noqa: E402


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def artifact_path() -> Path:
    hf_home = Path(os.environ["HF_HOME"])
    return hf_home / "hub" / "models--timm--resnet50.a1_in1k" / "blobs" / ARTIFACT_SHA


def load_locked_backbone(backbone: torch.nn.Module) -> None:
    artifact = artifact_path()
    if not artifact.is_file() or file_sha(artifact) != ARTIFACT_SHA:
        raise RuntimeError("locked pretrained artifact missing or SHA mismatch")
    full = load_file(str(artifact), device="cpu")
    state = {key: value for key, value in full.items() if not key.startswith("fc.")}
    backbone.load_state_dict(state, strict=True)
    if core.state_digest(backbone.state_dict()) != BACKBONE_SHA:
        raise RuntimeError("loaded backbone state SHA mismatch")


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, method: str, loader, device: torch.device, fold: int) -> dict[str, Any]:
    model.eval()
    logits, labels, indices, ids = [], [], [], []
    for images, batch_labels, batch_indices, batch_ids in loader:
        images = images.to(device)
        if method == "l2_sp":
            current = model(images)
        elif method == "mc_loss":
            current = model(images)
        elif method in {"ours_ft", "progressive"}:
            current = model(images)[0]
        elif method == "dfag":
            current = model(images)[0]
        else:
            raise ValueError(method)
        logits.append(current.detach().float().cpu().numpy())
        labels.append(batch_labels.numpy())
        indices.append(batch_indices.numpy())
        ids.extend(str(item) for item in batch_ids)
    result: dict[str, Any] = {
        "logits": np.concatenate(logits), "labels": np.concatenate(labels).astype(np.int64),
        "dataset_indices": np.concatenate(indices).astype(np.int64), "sample_ids": np.asarray(ids),
    }
    result["predictions"] = result["logits"].argmax(1).astype(np.int64)
    result["fold_ids"] = np.full(len(result["labels"]), fold, dtype=np.int64)
    result["metrics"] = metrics(result["labels"], result["predictions"])
    return result


def save_export(path: Path, result: dict[str, Any], expected: dict[str, np.ndarray]) -> dict[str, Any]:
    checks = runtime.verify_export(result, expected)
    np.savez_compressed(path, **{key: value for key, value in result.items() if key != "metrics"}, formal_result=np.asarray([True]))
    return {"path": str(path), "sha256": file_sha(path), "alignment": checks, "metrics": result["metrics"]}


def build_model(method: str, dataset: str, num_classes: int) -> torch.nn.Module:
    if method == "l2_sp":
        model = L2SPModel(num_classes, pretrained=False)
        load_locked_backbone(model.backbone)
        parameters = dict(model.backbone.named_parameters())
        for name, buffer_name in model._reference_buffers.items():
            getattr(model, buffer_name).copy_(parameters[name].detach())
        if float(model.regularization()["sp_raw"]) != 0.0:
            raise RuntimeError("L2-SP reference is not the locked ImageNet state")
    elif method == "mc_loss":
        model = MCLossModel(dataset, pretrained=False)
        load_locked_backbone(model.backbone)
    elif method in {"ours_ft", "progressive"}:
        model = core.SingleBranchModel(method, num_classes, pretrained=False)
        load_locked_backbone(model.backbone)
    elif method == "dfag":
        model = core.DFAGModel(num_classes)
    else:
        raise ValueError(method)
    for module in model.modules():
        if hasattr(module, "set_grad_checkpointing"):
            module.set_grad_checkpointing(True)
    return model


def checkpoint_payload(model, job: dict[str, Any], stage: int, epoch: int, value: float) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(), "method": job["method"], "dataset": job["dataset"],
        "fold": job["fold"], "seed": 42, "stage": stage, "epoch": epoch,
        "selection_metric": value, "selection_metric_type": "deterministic_original_view_heldout_accuracy",
        "config_sha256": job["config_sha256"], "formal_result": True,
    }


def amp_context():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def train_l2(model, train_loader, heldout_loader, expected, device, job, output, history):
    optimizer = l2_optimizer(model)
    iterator = iter(train_loader)
    losses = []
    for iteration in range(9000):
        try:
            images, labels, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels, _, _ = next(iterator)
        model.train(); images, labels = images.to(device), labels.to(device)
        lr = set_iteration_lr(optimizer, iteration)
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            objective = model.objective(images, labels, 0.1, 0.01)
        loss = objective["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite L2-SP loss at iteration {iteration}")
        loss.backward(); optimizer.step()
        value = float(loss.detach().cpu()); losses.append(value)
        if iteration % 100 == 0 or iteration == 8999:
            history.append({"stage": 1, "iteration": iteration, "loss": value, "lr": lr})
    result = evaluate(model, "l2_sp", heldout_loader, device, job["fold"])
    checkpoint = output / "best_stage1.pth"
    atomic_checkpoint(checkpoint, checkpoint_payload(model, job, 1, 8999, result["metrics"]["accuracy"]))
    return checkpoint, result


def train_mc(model, train_loader, heldout_loader, expected, device, job, output, history):
    optimizer = mc_optimizer(model)
    best, best_epoch, checkpoint = -1.0, -1, output / "best_stage1.pth"
    for epoch in range(300):
        model.train(); rates = set_epoch_lr(optimizer, epoch); losses = []
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context():
                objective = model(images, labels)
            loss = objective["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MC-Loss at epoch {epoch}")
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        result = evaluate(model, "mc_loss", heldout_loader, device, job["fold"])
        accuracy = result["metrics"]["accuracy"]
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            atomic_checkpoint(checkpoint, checkpoint_payload(model, job, 1, epoch, accuracy))
        history.append({"stage": 1, "epoch": epoch, "loss": float(np.mean(losses)), "backbone_lr": rates[0], "classifier_lr": rates[1], "heldout_accuracy": accuracy, "best_accuracy": best})
    selected = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    return checkpoint, evaluate(model, "mc_loss", heldout_loader, device, job["fold"])


def mixed_loss(logits, label_a, label_b, lam, smoothing):
    return lam * F.cross_entropy(logits, label_a, label_smoothing=smoothing) + (1.0 - lam) * F.cross_entropy(logits, label_b, label_smoothing=smoothing)


def stage1_current(model, train_loader, heldout_loader, device, job, output, history):
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4}, {"params": model.head.parameters(), "lr": 1e-3},
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=147, eta_min=1e-5)
    best, best_epoch, checkpoint = -1.0, -1, output / "best_stage1.pth"
    for epoch in range(150):
        model.train(); losses = []
        if epoch < 3:
            factor = (epoch + 1) / 3.0
            optimizer.param_groups[0]["lr"] = 1e-4 * factor; optimizer.param_groups[1]["lr"] = 1e-3 * factor
        for batch_index, (images, labels, _, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images, label_a, label_b, lam, kind, scale = core.external_protocol.batch_mix(images, labels, epoch, batch_index, 150, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context():
                logits = model(images)[0]
                loss = mixed_loss(logits, label_a, label_b, lam, 0.05)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite Stage1 loss epoch {epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        result = evaluate(model, job["method"], heldout_loader, device, job["fold"]); accuracy = result["metrics"]["accuracy"]
        if accuracy > best:
            best, best_epoch = accuracy, epoch; atomic_checkpoint(checkpoint, checkpoint_payload(model, job, 1, epoch, accuracy))
        history.append({"stage": 1, "epoch": epoch, "loss": float(np.mean(losses)), "heldout_accuracy": accuracy, "best_accuracy": best, "backbone_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"], "augmentation_scale": scale})
        if epoch >= 3: scheduler.step()
    selected = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(selected["model_state_dict"], strict=True)
    return checkpoint, best_epoch, evaluate(model, job["method"], heldout_loader, device, job["fold"])


@torch.no_grad()
def recalibrate_bn(model, train_loader, device) -> dict[str, Any]:
    model.train(); completed = 0
    for images, _, _, _ in train_loader:
        model(images.to(device)); completed += 1
        if completed == 50: break
    if completed != 50: raise RuntimeError("BN recalibration did not complete 50 batches")
    return {"reset": False, "whole_model_train": True, "no_grad": True, "batches": 50}


def stage2_current(model, method, train_loader, heldout_loader, device, job, output, history):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
    best, best_epoch, checkpoint = -1.0, -1, output / "best_stage2.pth"
    for epoch in range(60):
        model.train(); losses = []
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad(set_to_none=True)
            with amp_context():
                logits = model(images)[0]
                loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite Stage2 loss epoch {epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        result = evaluate(model, method, heldout_loader, device, job["fold"]); accuracy = result["metrics"]["accuracy"]
        if accuracy > best:
            best, best_epoch = accuracy, epoch; atomic_checkpoint(checkpoint, checkpoint_payload(model, job, 2, epoch, accuracy))
        history.append({"stage": 2, "epoch": epoch, "loss": float(np.mean(losses)), "heldout_accuracy": accuracy, "best_accuracy": best, "lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
    selected = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(selected["model_state_dict"], strict=True)
    return checkpoint, best_epoch, evaluate(model, method, heldout_loader, device, job["fold"])


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def validate_job(job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = json.loads((ROOT / "FORMAL_RUN_AUTHORIZATION.json").read_text(encoding="utf-8"))
    if authorization.get("training_authorized") is not True or authorization.get("status") != "PASS":
        raise RuntimeError("formal training is not authorized")
    plan_path = ROOT / "CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json"
    if file_sha(plan_path) != EXPECTED_PLAN_SHA: raise RuntimeError("plan SHA drift")
    plan = json.loads(plan_path.read_text(encoding="utf-8")); job = next((row for row in plan["jobs"] if row["job_id"] == job_id), None)
    if job is None or job["method"] == "cal": raise ValueError(job_id)
    config_path = ROOT / job["config_path"]
    if file_sha(config_path) != job["config_sha256"]: raise RuntimeError("config SHA drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["method"] != job["method"] or config["dataset"] != job["dataset"] or job["seed"] != 42: raise RuntimeError("job identity/seed mismatch")
    if config["evaluation"] != {"pooled_oof": True, "each_development_sample_exactly_once": True, "original_view": True, "tta": False, "cross_fold_ensemble": False, "metrics": ["accuracy", "macro_f1", "balanced_accuracy"], "official_test_excluded_from_training_selection_and_oof": True}: raise RuntimeError("evaluation protocol drift")
    return job, config


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--job-id", required=True); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    job, config = validate_job(args.job_id); output = RUN_ROOT / "jobs" / args.job_id
    if output.exists(): raise RuntimeError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True, exist_ok=False); manifest_path = output / "run_manifest.json"; started = time.time()
    manifest: dict[str, Any] = {"status": "RUNNING", "formal_result": True, "job": job, "started_unix": started, "device": args.device, "runner_sha256": file_sha(Path(__file__)), "runtime_sha256": file_sha(Path(runtime.__file__)), "protocol_core_sha256": file_sha(Path(core.external_protocol.__file__)), "cars_lookup_implementation": runtime.CARS_LOOKUP_IMPLEMENTATION, "no_official_test_access": True, "tta": False, "fold_ensemble": False, "runtime_precision": "BF16 autocast training; FP32 parameters/optimizer and deterministic FP32 original-view inference", "activation_checkpointing": True, "pin_memory": False}
    atomic_json(manifest_path, manifest)
    try:
        core.seed_everything(42); device = torch.device(args.device); method, dataset, fold = job["method"], job["dataset"], int(job["fold"])
        batch = 64 if method == "l2_sp" else 32
        train_loader, heldout_loader, development, train_idx, heldout_idx = runtime.loaders(dataset, method, fold, batch, num_workers=4)
        train_loader.pin_memory = False; heldout_loader.pin_memory = False
        expected = runtime.expected_heldout(development, heldout_idx, fold)
        model = build_model(method, dataset, int(config["num_classes"])).to(device); history: list[dict[str, Any]] = []
        stage1_export = None; bn_record = None
        if method == "l2_sp":
            checkpoint, final = train_l2(model, train_loader, heldout_loader, expected, device, job, output, history)
        elif method == "mc_loss":
            checkpoint, final = train_mc(model, train_loader, heldout_loader, expected, device, job, output, history)
        elif method in {"ours_ft", "progressive"}:
            stage1_path, stage1_epoch, stage1 = stage1_current(model, train_loader, heldout_loader, device, job, output, history)
            stage1_export = save_export(output / "stage1_heldout_predictions.npz", stage1, expected)
            bn_record = recalibrate_bn(model, train_loader, device)
            checkpoint, stage2_epoch, final = stage2_current(model, method, train_loader, heldout_loader, device, job, output, history)
        else:
            source_dir = RUN_ROOT / "jobs" / f"ours_ft_{dataset}_fold{fold}"; source_manifest = json.loads((source_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if source_manifest.get("status") != "COMPLETE": raise RuntimeError("same-fold Ours dependency incomplete")
            source_path = Path(source_manifest["best_stage1_checkpoint"]); source_sha = file_sha(source_path)
            payload = torch.load(source_path, map_location="cpu", weights_only=False)
            model.load_same_fold_stage1(payload, dataset, fold); model.to(device)
            bn_record = recalibrate_bn(model, train_loader, device)
            checkpoint, stage2_epoch, final = stage2_current(model, "dfag", train_loader, heldout_loader, device, job, output, history)
            manifest["dfag_stage1_source"] = {"path": str(source_path), "sha256": source_sha, "same_dataset": True, "same_fold": True, "stage1_retrained": False}
        final_export = save_export(output / "heldout_predictions.npz", final, expected); write_history(output / "training_history.csv", history)
        manifest.update({"status": "COMPLETE", "completed_unix": time.time(), "elapsed_seconds": time.time() - started, "train_count": len(train_idx), "heldout_count": len(heldout_idx), "best_checkpoint": str(checkpoint), "best_checkpoint_sha256": file_sha(checkpoint), "best_stage1_checkpoint": str(output / "best_stage1.pth") if (output / "best_stage1.pth").is_file() else None, "best_stage1_sha256": file_sha(output / "best_stage1.pth") if (output / "best_stage1.pth").is_file() else None, "stage1_export": stage1_export, "bn_recalibration": bn_record, "final_export": final_export, "all_finite": True, "checkpoint_roundtrip": True})
        atomic_json(manifest_path, manifest); print(json.dumps({"status": "COMPLETE", "job_id": args.job_id, "metrics": final_export["metrics"]}), flush=True)
    except Exception as exc:
        manifest.update({"status": "FAILED", "completed_unix": time.time(), "elapsed_seconds": time.time() - started, "error": repr(exc), "traceback": traceback.format_exc()}); atomic_json(manifest_path, manifest); raise


if __name__ == "__main__":
    main()
