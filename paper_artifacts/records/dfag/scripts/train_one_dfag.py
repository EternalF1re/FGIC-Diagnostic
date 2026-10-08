"""Train one authorized seed/fold standalone dynamic DFAG Stage2 job."""
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import timm
import torch
import torchvision
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from dfag_common import (
    EXP_ROOT, FEATURE_DIM, FOLDS, GATE_PARAMETERS, SEEDS, UnifiedStandaloneDFAG,
    bn_state, config_path, load_config, output_dir, parameter_counts, sha256_file,
    source_checkpoint, source_dir, state_digest,
)
from screen_core import (
    LeafDataset, build_loaders, eval_transform, fold_class_weights, seed_everything,
    seed_worker,
)


MODES = ("dynamic", "mean_vector", "mean_scalar", "constant_0_5", "anchor_forced", "plastic_forced")


def environment(device: torch.device) -> dict:
    data = {"python_executable": sys.executable, "python_version": platform.python_version(),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
            "numpy": np.__version__, "sklearn": sklearn.__version__, "device": str(device),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device.index or 0)
        data.update({"gpu_name": props.name, "gpu_memory_bytes": props.total_memory})
    return data


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions))}


def tensor_norm(parameters) -> float:
    return math.sqrt(sum(float(torch.sum(parameter.detach().float() ** 2).cpu()) for parameter in parameters))


def gradient_norm(parameters) -> float:
    return math.sqrt(sum(float(torch.sum(parameter.grad.detach().float() ** 2).cpu())
                         for parameter in parameters if parameter.grad is not None))


def initial_gate_stats(model, loader, device) -> dict[str, float]:
    model.eval()
    images = next(iter(loader))[0].to(device, non_blocking=True)
    with torch.inference_mode():
        result = model.forward_components(images)
    gate = result["gate"].float()
    return {"mean": float(gate.mean().cpu()), "std": float(gate.std(unbiased=True).cpu()),
            "min": float(gate.min().cpu()), "max": float(gate.max().cpu()), "n": gate.numel()}


def adapt_batch_norm(model, train_loader, device, batches: int) -> int:
    model.train()
    model.anchor.eval()
    completed = 0
    with torch.no_grad():
        for images, _, _, _ in train_loader:
            model(images.to(device, non_blocking=True))
            completed += 1
            if completed >= batches:
                break
    return completed


def deterministic_training_loader(config: dict, train_idx: np.ndarray) -> DataLoader:
    data_cfg, protocol = config["dataset"], config["common_training_protocol"]
    dataset = LeafDataset(Path(data_cfg["train_csv"]), Path(data_cfg["root"]), eval_transform())
    return DataLoader(Subset(dataset, train_idx.tolist()), shuffle=False, drop_last=False,
                      batch_size=int(protocol["batch_size"]), num_workers=int(protocol["num_workers"]),
                      pin_memory=True, worker_init_fn=seed_worker,
                      persistent_workers=int(protocol["num_workers"]) > 0)


def training_mean_gate(model, loader, device) -> tuple[np.ndarray, int]:
    model.eval()
    total = np.zeros(FEATURE_DIM, dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for images, _, _, _ in loader:
            result = model.forward_components(images.to(device, non_blocking=True))
            gate = result["gate"].float().cpu().numpy()
            total += gate.sum(axis=0, dtype=np.float64)
            count += len(gate)
    return (total / count).astype(np.float32), count


def evaluate_all(model, loader, device, mean_vector: np.ndarray) -> tuple[dict, dict[str, dict[str, float]]]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in (
        "sample_id", "label", "image_name", "f_anc", "f_spec", "f_fuse", "gate")}
    logits = {mode: [] for mode in MODES}
    vector = torch.from_numpy(mean_vector).to(device).reshape(1, FEATURE_DIM)
    scalar = float(mean_vector.mean())
    with torch.inference_mode():
        for images, labels, indices, names in loader:
            images = images.to(device, non_blocking=True)
            result = model.forward_components(images)
            f_anc, f_spec, gate = result["f_anc"], result["f_spec"], result["gate"]
            fused = {
                "dynamic": result["f_fuse"],
                "mean_vector": vector * f_anc + (1.0 - vector) * f_spec,
                "mean_scalar": scalar * f_anc + (1.0 - scalar) * f_spec,
                "constant_0_5": 0.5 * f_anc + 0.5 * f_spec,
                "anchor_forced": f_anc,
                "plastic_forced": f_spec,
            }
            for mode, feature in fused.items():
                mode_logits, _ = model.plastic.head(feature)
                logits[mode].append(mode_logits.float().cpu().numpy())
            arrays["sample_id"].append(indices.numpy())
            arrays["label"].append(labels.numpy())
            arrays["image_name"].append(np.asarray([str(name) for name in names]))
            arrays["f_anc"].append(f_anc.float().cpu().numpy())
            arrays["f_spec"].append(f_spec.float().cpu().numpy())
            arrays["f_fuse"].append(result["f_fuse"].float().cpu().numpy())
            arrays["gate"].append(gate.float().cpu().numpy())
    merged = {key: np.concatenate(values) for key, values in arrays.items()}
    for mode in MODES:
        merged[f"logits_{mode}"] = np.concatenate(logits[mode]).astype(np.float32)
        merged[f"prediction_{mode}"] = merged[f"logits_{mode}"].argmax(axis=1).astype(np.int64)
    metric_table = {mode: metrics(merged["label"], merged[f"prediction_{mode}"]) for mode in MODES}
    return merged, metric_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--fold", type=int, required=True, choices=FOLDS)
    parser.add_argument("--device", required=True, choices=("cuda:0", "cuda:1"))
    args = parser.parse_args()
    seed, fold, device = args.seed, args.fold, torch.device(args.device)
    preflight = json.loads((EXP_ROOT / "manifests" / "preflight_audit.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("Phase2F preflight missing or failed")
    output = output_dir(seed, fold)
    if output.exists():
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "COMPLETE":
            print(json.dumps({"status": "ALREADY_COMPLETE", "seed": seed, "fold": fold}), flush=True)
            return
        raise RuntimeError(f"refusing to reuse partial output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "run_manifest.json"
    config = load_config(seed)
    seed_everything(seed)
    source = source_checkpoint(seed, fold)
    source_hash = sha256_file(source)
    source_manifest = json.loads((source_dir(seed, fold) / "run_manifest.json").read_text(encoding="utf-8"))
    if source_hash != source_manifest["best_stage1_sha256"]:
        raise RuntimeError("selected Stage1 SHA256 mismatch")
    started = time.time()
    manifest = {"status": "RUNNING", "phase": "Phase2F unified standalone dynamic DFAG",
                "seed": seed, "fold": fold, "split_id": f"skf42_fold{fold}", "split_random_state": 42,
                "device": str(device), "formal_training": True, "stage1_retrained": False,
                "stage2_only": True, "dfag": "standalone_dynamic", "ssph": False, "mhsa": False,
                "terminal": False, "fixed_g_training": False, "anchor_sweep": False,
                "gating_source_ablation": False, "joint_model": False,
                "source_stage1_checkpoint": str(source.resolve()), "source_stage1_sha256": source_hash,
                "source_stage1_epoch_zero_based": source_manifest["best_stage1_epoch_zero_based"],
                "config_sha256": sha256_file(config_path(seed)), "runner_sha256": sha256_file(Path(__file__).resolve()),
                "started_unix": started}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(config_path(seed), output / "config.json")
    try:
        train_loader, val_loader, dataset, train_idx, val_idx = build_loaders(fold, 2, config)
        source_split = np.load(source_dir(seed, fold) / "split_indices.npz")
        if not np.array_equal(train_idx, source_split["train_indices"]) or not np.array_equal(val_idx, source_split["validation_indices"]):
            raise ValueError("DFAG/source split mismatch")
        np.savez_compressed(output / "split_indices.npz", train_indices=train_idx, validation_indices=val_idx,
                            split_random_state=np.asarray([42]), training_seed=np.asarray([seed]))
        class_weights = fold_class_weights(dataset.labels, train_idx, int(config["dataset"]["num_classes"]))
        expected_weights = np.load(source_dir(seed, fold) / "class_weights.npy")
        if not np.array_equal(class_weights.numpy(), expected_weights):
            raise ValueError("DFAG/source class weights mismatch")
        np.save(output / "class_weights.npy", class_weights.numpy())

        model = UnifiedStandaloneDFAG(int(config["dataset"]["num_classes"]))
        source_payload = model.load_common_stage1(source)
        counts = parameter_counts(model)
        if counts["gate_parameters"] != GATE_PARAMETERS:
            raise ValueError("gate parameter count mismatch")
        anchor_initial_digest = state_digest(model.anchor.state_dict())
        anchor_bn_initial_digest = state_digest(bn_state(model.anchor))
        gate_initial_digest = state_digest(model.dfag_gate.state_dict())
        gate_initial_parameter_norm = tensor_norm(model.dfag_gate.parameters())
        model.to(device)
        initial_stats = initial_gate_stats(model, val_loader, device)

        protocol = config["common_training_protocol"]
        bn_completed = adapt_batch_norm(model, train_loader, device, int(protocol["bn_adaptation_batches_before_stage2"]))
        optimized = model.optimized_parameters()
        optimized_ids = {id(parameter) for parameter in optimized}
        if any(id(parameter) in optimized_ids for parameter in model.anchor.parameters()):
            raise RuntimeError("anchor present in optimizer parameter set")
        optimizer = torch.optim.AdamW(optimized, lr=float(protocol["stage2_all_parameter_lr"]),
                                      betas=tuple(protocol["adam_betas"]), eps=float(protocol["adam_eps"]),
                                      weight_decay=float(protocol["stage2_weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(protocol["stage2_epochs"]), eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=float(protocol["stage2_label_smoothing"]))
        scaler = torch.amp.GradScaler(device.type, init_scale=float(protocol["amp_grad_scaler_init_scale"]),
                                      enabled=device.type == "cuda" and bool(protocol["amp"]))
        history = []
        best_accuracy, best_epoch = -1.0, -1
        checkpoint_path = output / "best_stage2.pth"
        gate_grad_nonzero_batches = 0
        gate_grad_observed_batches = 0
        epochs = int(protocol["stage2_epochs"])
        for epoch in range(epochs):
            model.train(); model.anchor.eval()
            losses, gate_gradients = [], []
            amp_overflow_batches = 0
            epoch_started = time.perf_counter()
            for batch_index, (images, labels, _, _) in enumerate(train_loader):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda" and bool(protocol["amp"])):
                    logits, _ = model(images)
                    loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss epoch={epoch} batch={batch_index}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(parameter.grad is not None for parameter in model.anchor.parameters()):
                    raise RuntimeError("anchor gradient unexpectedly present")
                gate_grad = gradient_norm(model.dfag_gate.parameters())
                gate_gradients.append(gate_grad); gate_grad_observed_batches += 1
                if gate_grad > 0:
                    gate_grad_nonzero_batches += 1
                total_grad = torch.nn.utils.clip_grad_norm_(optimized, float(protocol["gradient_clip_norm"]))
                if not torch.isfinite(total_grad):
                    if not scaler.is_enabled():
                        raise FloatingPointError("non-finite FP32 gradient")
                    scaler.step(optimizer); scaler.update(); amp_overflow_batches += 1
                    continue
                scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
            model.eval()
            correct = total = 0
            with torch.inference_mode():
                for images, labels, _, _ in val_loader:
                    logits, _ = model(images.to(device, non_blocking=True))
                    correct += int((logits.argmax(1).cpu() == labels).sum()); total += len(labels)
            val_accuracy = correct / total
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch = val_accuracy, epoch
                payload = {**model.checkpoint_payload_state(), "seed": seed, "fold": fold, "stage": 2, "epoch": epoch,
                           "metric": val_accuracy, "metric_type": "heldout_original_view_accuracy",
                           "source_stage1_checkpoint": str(source.resolve()), "source_stage1_sha256": source_hash,
                           "config_sha256": sha256_file(config_path(seed)), "gate_input": "f_spec only",
                           "fusion": "g*f_anc + (1-g)*f_spec"}
                torch.save(payload, checkpoint_path)
            row = {"stage": 2, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_accuracy,
                   "best_val_accuracy": best_accuracy, "lr": optimizer.param_groups[0]["lr"],
                   "gate_gradient_norm_mean": float(np.mean(gate_gradients)), "gate_gradient_norm_max": float(np.max(gate_gradients)),
                   "amp_overflow_batches": amp_overflow_batches, "elapsed_seconds": time.perf_counter() - epoch_started}
            history.append(row)
            pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
            print(json.dumps({"seed": seed, "fold": fold, "stage": 2, "epoch": epoch + 1, "epochs": epochs,
                              "loss": row["train_loss"], "val_accuracy": val_accuracy, "best": best_accuracy,
                              "gate_grad": row["gate_gradient_norm_mean"]}), flush=True)
            scheduler.step()

        selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_payload_state(selected)
        deterministic_loader = deterministic_training_loader(config, train_idx)
        mean_vector, mean_count = training_mean_gate(model, deterministic_loader, device)
        np.save(output / "training_side_mean_gate_vector.npy", mean_vector)
        arrays, metric_table = evaluate_all(model, val_loader, device, mean_vector)
        if not np.array_equal(arrays["sample_id"], val_idx):
            raise ValueError("held-out sample order mismatch")
        np.savez_compressed(output / "heldout_dfag_artifacts.npz", **arrays)
        selected_gate_digest = state_digest(model.dfag_gate.state_dict())
        gate_values = arrays["gate"]
        selected_gate_stats = {"mean": float(gate_values.mean()), "std": float(gate_values.std(ddof=1)),
                               "min": float(gate_values.min()), "max": float(gate_values.max()), "n": int(gate_values.size)}
        anchor_final_digest = state_digest(model.anchor.state_dict())
        anchor_bn_final_digest = state_digest(bn_state(model.anchor))
        max_branch_difference = float(np.max(np.abs(arrays["f_anc"] - arrays["f_spec"])))
        sanity = {"gate_initial_parameter_norm": gate_initial_parameter_norm,
                  "gate_selected_parameter_norm": tensor_norm(model.dfag_gate.parameters()),
                  "gate_initial_output": initial_stats, "gate_selected_output": selected_gate_stats,
                  "gate_initial_state_sha256": gate_initial_digest, "gate_selected_state_sha256": selected_gate_digest,
                  "gate_parameters_changed": gate_initial_digest != selected_gate_digest,
                  "gate_gradient_observed_batches": gate_grad_observed_batches,
                  "gate_gradient_nonzero_batches": gate_grad_nonzero_batches,
                  "gate_gradients_not_permanently_zero": gate_grad_nonzero_batches > 0,
                  "anchor_state_sha256_before": anchor_initial_digest, "anchor_state_sha256_after": anchor_final_digest,
                  "anchor_unchanged": anchor_initial_digest == anchor_final_digest,
                  "anchor_bn_sha256_before": anchor_bn_initial_digest, "anchor_bn_sha256_after": anchor_bn_final_digest,
                  "anchor_bn_unchanged": anchor_bn_initial_digest == anchor_bn_final_digest,
                  "anchor_gradients_absent": all(parameter.grad is None for parameter in model.anchor.parameters()),
                  "max_abs_fanc_minus_fspec": max_branch_difference,
                  "branches_not_bitwise_identical": max_branch_difference > 0,
                  "all_outputs_finite": all(np.isfinite(value).all() for key, value in arrays.items() if value.dtype.kind in "fc")}
        if not all(sanity[key] for key in ("gate_parameters_changed", "gate_gradients_not_permanently_zero", "anchor_unchanged",
                                           "anchor_bn_unchanged", "anchor_gradients_absent", "branches_not_bitwise_identical", "all_outputs_finite")):
            raise RuntimeError(f"implementation sanity failure: {sanity}")
        metrics_payload = {"seed": seed, "fold": fold, "n": len(val_idx), "selected_epoch_zero_based": best_epoch,
                           "selected_accuracy": best_accuracy, "modes": metric_table,
                           "training_side_mean_gate_count": mean_count,
                           "training_side_mean_gate_scalar": float(mean_vector.mean()), "sanity": sanity}
        (output / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.update({"status": "COMPLETE", "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
                         "environment": environment(device), "parameter_counts": counts,
                         "bn_adaptation_batches_completed": bn_completed,
                         "bn_refresh_modules": ["plastic.backbone BatchNorm", "plastic.head.bn1", "plastic.head.bn2"],
                         "optimizer_modules": ["plastic.backbone", "plastic.head", "dfag_gate"],
                         "anchor_in_optimizer": False, "best_checkpoint": str(checkpoint_path.resolve()),
                         "best_checkpoint_sha256": sha256_file(checkpoint_path), "selected_epoch_zero_based": best_epoch,
                         "selected_accuracy": best_accuracy, "history_epochs": len(history),
                         "heldout_artifact": "heldout_dfag_artifacts.npz",
                         "training_side_mean_gate_vector": "training_side_mean_gate_vector.npy",
                         "metrics": metric_table, "sanity": sanity})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "COMPLETE", "seed": seed, "fold": fold, "dynamic": metric_table["dynamic"]}), flush=True)
    except Exception as exc:
        manifest.update({"status": "FAILED", "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
                         "error": repr(exc), "traceback": traceback.format_exc()})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
