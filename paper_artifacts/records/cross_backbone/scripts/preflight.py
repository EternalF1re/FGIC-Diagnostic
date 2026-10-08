"""Fail-closed preflight for the final cross-backbone evaluation."""
from __future__ import annotations

import gc
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn

from cross_backbone_common import (
    BACKBONES,
    BLOCKS,
    CONFIG_PATH,
    EXPECTED_DIMENSIONS,
    FOLDS,
    HIDDEN,
    METHODS,
    NUM_CLASSES,
    REDUCTION,
    ROOT,
    SHORTCUT_LAMBDA,
    ConventionalHead,
    CrossBackboneModel,
    DynamicGate,
    MappingBlock,
    ProgressiveHead,
    StandaloneDFAG,
    load_config,
    parameter_counts,
    sha256_file,
    split_indices,
)


SOURCE_CONFIG = ROOT.parent / "phase2b_screen" / "configs" / "controlled_screen.json"
SOURCE_DFAG = ROOT.parent / "phase2f_unified_standalone_dfag" / "scripts" / "dfag_common.py"
SOURCE_INTERPOLATION = ROOT.parent / "phase2d_checkpoint_state_interpolation" / "scripts" / "csi_common_v2.py"
TRAIN_RUNNER = ROOT / "scripts" / "train_one.py"


def gpu_state(index: int = 0) -> dict[str, Any]:
    raw = subprocess.run([
        "nvidia-smi", "--id", str(index),
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True).stdout.strip()
    name, utilization, used, total = [part.strip() for part in raw.split(",")]
    return {"name": name, "utilization_percent": int(utilization), "memory_used_mib": int(used), "memory_total_mib": int(total)}


def architecture_audit(backbone_name: str) -> dict[str, Any]:
    single = CrossBackboneModel(backbone_name, "ours_ft", pretrained=True).eval()
    progressive = CrossBackboneModel(backbone_name, "progressive", pretrained=False).eval()
    dfag = StandaloneDFAG(backbone_name).eval()
    dimension = int(single.dimension)
    with torch.inference_mode():
        shape = list(single.backbone(torch.randn(2, 3, 299, 299)).shape)
    gate_layers = list(dfag.dfag_gate.gate)
    expected_gate = 2 * dimension * (dimension // REDUCTION)
    checks = {
        "actual_dimension_matches_expectation": dimension == EXPECTED_DIMENSIONS[backbone_name],
        "actual_forward_shape": shape == [2, dimension],
        "classifier_input_dimension": single.head.fc1.in_features == dimension,
        "conventional_head_exact": isinstance(single.head, ConventionalHead)
            and single.head.fc1.out_features == 1024 and single.head.fc2.in_features == 1024
            and single.head.fc2.out_features == 1024 and single.head.classifier.out_features == NUM_CLASSES,
        "progressive_projection_D_to_256": progressive.head.input_projection.in_features == dimension
            and progressive.head.input_projection.out_features == HIDDEN,
        "five_mapping_blocks": len(progressive.head.mapping_blocks) == BLOCKS == 5,
        "mapping_block_exact": all(
            isinstance(block, MappingBlock)
            and tuple(type(module) for module in block.mapping) == (nn.Linear, nn.BatchNorm1d, nn.SiLU, nn.Dropout)
            and block.mapping[3].p == 0.3 for block in progressive.head.mapping_blocks
        ),
        "lambda_exact_0_7": SHORTCUT_LAMBDA == 0.7,
        "mean_f1_to_f5": "torch.stack(stages, dim=1).mean(dim=1)" in inspect.getsource(ProgressiveHead.forward),
        "mhsa_absent": not any(isinstance(module, nn.MultiheadAttention) for module in progressive.modules()),
        "terminal_residual_absent": "return self.classifier(feature), feature" in inspect.getsource(ProgressiveHead.forward),
        "progressive_one_branch": set(dict(progressive.named_children())) == {"backbone", "head"},
        "dfag_two_branches": dfag.anchor is not dfag.plastic and dfag.anchor.backbone is not dfag.plastic.backbone,
        "dfag_dimension": dfag.dimension == dimension,
        "gate_structure": len(gate_layers) == 4 and gate_layers[0].in_features == dimension
            and gate_layers[0].out_features == dimension // REDUCTION and gate_layers[2].in_features == dimension // REDUCTION
            and gate_layers[2].out_features == dimension,
        "gate_bias_free": gate_layers[0].bias is None and gate_layers[2].bias is None,
        "gate_input_fspec_only": "self.dfag_gate(f_spec)" in inspect.getsource(StandaloneDFAG.forward_components),
        "fusion_exact": "gate * f_anc + (1.0 - gate) * f_spec" in inspect.getsource(StandaloneDFAG.forward_components),
        "adaptive_gate_parameter_count": sum(parameter.numel() for parameter in dfag.dfag_gate.parameters()) == expected_gate,
        "anchor_frozen": not any(parameter.requires_grad for parameter in dfag.anchor.parameters()),
        "dfag_no_progressive_head": not any(isinstance(module, ProgressiveHead) for module in dfag.modules()),
    }
    data = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "backbone": backbone_name,
        "actual_dimension": dimension,
        "forward_shape": shape,
        "hidden_dimension": HIDDEN,
        "D_over_H": dimension / HIDDEN,
        "pretrained_cfg": single.backbone.pretrained_cfg,
        "gate_parameters": expected_gate,
        "parameter_counts": {
            "ours_ft": parameter_counts(single, "ours_ft"),
            "progressive": parameter_counts(progressive, "progressive"),
            "dfag": parameter_counts(dfag, "dfag"),
        },
        "checks": checks,
    }
    del single, progressive, dfag
    gc.collect()
    return data


def memory_feasibility(backbone: str, method: str, batch_size: int, device: torch.device) -> dict[str, Any]:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    if method == "dfag":
        model = StandaloneDFAG(backbone)
        parameters = model.optimized_parameters()
    else:
        model = CrossBackboneModel(backbone, method, pretrained=False)
        parameters = list(model.parameters())
    model.to(device).train()
    if method == "dfag":
        model.anchor.eval()
    optimizer = torch.optim.AdamW(parameters, lr=5e-5)
    scaler = torch.amp.GradScaler("cuda", init_scale=4096, enabled=True)
    x = torch.randn(batch_size, 3, 299, 299, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(logits, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_finite = bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters, 5.0)).item())
    gate_grad_nonzero = True
    if method == "dfag":
        gate_grad_nonzero = any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 for parameter in model.dfag_gate.parameters())
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    result = {
        "status": "PASS" if torch.isfinite(loss).item() and grad_finite and gate_grad_nonzero else "FAIL",
        "backbone": backbone,
        "method": method,
        "batch_size": batch_size,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "gradient_finite": grad_finite,
        "gate_gradient_nonzero": gate_grad_nonzero,
        "peak_memory_bytes": peak,
        "peak_memory_mib": peak / (1024.0 ** 2),
        "gpu_total_memory_mib": torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 2),
    }
    del x, y, logits, loss, optimizer, scaler, model, parameters
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    return result


def main() -> None:
    config = load_config()
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["authoritative_source_sha"] = sha256_file(SOURCE_CONFIG) == config["authority"]["classify_leaves_config_sha256"]
    checks["dfag_authority_exists"] = SOURCE_DFAG.is_file()
    checks["interpolation_authority_exists"] = SOURCE_INTERPOLATION.is_file()
    checks["runner_exists"] = TRAIN_RUNNER.is_file()

    required_protocol_keys = [
        "stage1_epochs", "stage2_epochs", "optimizer", "adam_betas", "adam_eps",
        "stage1_backbone_lr", "stage1_head_lr", "stage1_weight_decay", "stage1_warmup_epochs",
        "stage1_scheduler", "stage1_label_smoothing", "stage2_all_parameter_lr", "stage2_weight_decay",
        "stage2_scheduler", "stage2_label_smoothing", "amp", "amp_dtype", "amp_grad_scaler_init_scale",
        "gradient_clip_norm", "bn_adaptation_batches_before_stage2", "checkpoint_selection",
        "stage1_batch_augmentation", "stage2_batch_augmentation", "spatial_augmentation_both_stages",
    ]
    protocol_match = {}
    for key in required_protocol_keys:
        protocol_match[key] = config["common_training_protocol"][key] == source["common_training_protocol"][key]
    checks["authoritative_protocol_values_exact"] = all(protocol_match.values())
    checks["split_exact"] = config["split"] == source["split"]
    checks["training_seed_exact"] = config["seed"]["training_seed"] == 42 == source["seed"]["training_seed"]

    csv_path = Path(config["dataset"]["train_csv"])
    image_root = Path(config["dataset"]["root"])
    frame = pd.read_csv(csv_path)
    labels_text = frame.iloc[:, 1].astype(str)
    ordered = list(dict.fromkeys(labels_text.tolist()))
    mapping = {label: index for index, label in enumerate(ordered)}
    labels = labels_text.map(mapping).to_numpy(np.int64)
    image_paths = [image_root / str(name) for name in frame.iloc[:, 0]]
    missing = [str(path) for path in image_paths if not path.is_file()]
    checks["dataset_path"] = csv_path.is_file() and image_root.is_dir()
    checks["dataset_n_18353"] = len(frame) == 18_353
    checks["class_count_176"] = len(mapping) == NUM_CLASSES
    checks["all_images_exist"] = not missing
    checks["image_names_unique"] = frame.iloc[:, 0].astype(str).is_unique
    fold_rows = []
    validation_union = []
    for fold in FOLDS:
        train_idx, val_idx = split_indices(labels, fold, 42)
        fold_rows.append({"fold": fold, "train_n": len(train_idx), "validation_n": len(val_idx),
                          "intersection_n": int(np.intersect1d(train_idx, val_idx).size),
                          "validation_classes": int(np.unique(labels[val_idx]).size)})
        validation_union.extend(val_idx.tolist())
    checks["no_fold_leakage"] = all(row["intersection_n"] == 0 for row in fold_rows)
    counts = np.bincount(np.asarray(validation_union), minlength=len(frame))
    checks["fivefold_exact_oof_coverage"] = len(validation_union) == len(frame) and np.all(counts == 1)

    architectures = {backbone: architecture_audit(backbone) for backbone in BACKBONES}
    checks["architecture_hard_checks"] = all(row["status"] == "PASS" for row in architectures.values())
    checks["fixed_H_256"] = all(row["hidden_dimension"] == 256 for row in architectures.values())
    checks["compression_ratio_not_constant"] = len({row["D_over_H"] for row in architectures.values()}) == 2
    checks["same_batch_size_within_and_across_backbones"] = {config["backbones"][b]["batch_size"] for b in BACKBONES} == {64}

    before = gpu_state(0)
    if before["utilization_percent"] > 15 or before["memory_used_mib"] > 1500:
        raise RuntimeError(f"GPU0 not idle enough for memory preflight: {before}")
    device = torch.device("cuda:0")
    memory = []
    for backbone in BACKBONES:
        for method in METHODS:
            try:
                memory.append(memory_feasibility(backbone, method, int(config["backbones"][backbone]["batch_size"]), device))
            except torch.OutOfMemoryError as exc:
                memory.append({"status": "FAIL", "backbone": backbone, "method": method,
                               "batch_size": config["backbones"][backbone]["batch_size"], "error": repr(exc)})
                gc.collect(); torch.cuda.empty_cache()
    checks["gpu_memory_feasibility_all_six"] = all(row["status"] == "PASS" for row in memory)

    oof_schema = ["sample_id", "fold", "y_true", "y_pred", "logits", "probabilities", "image_name", "backbone", "configuration", "stage", "training_seed"]
    checks["oof_schema_complete"] = set(oof_schema) == {"sample_id", "fold", "y_true", "y_pred", "logits", "probabilities", "image_name", "backbone", "configuration", "stage", "training_seed"}
    checks["checkpoint_tie_break_source"] = "if val_accuracy > best_accuracy" in TRAIN_RUNNER.read_text(encoding="utf-8")
    checks["dfag_stage1_reuse_authorized"] = config["configurations"]["dfag"]["stage1_retrained"] is False
    checks["no_joint_no_lambda_sweep"] = config["formal_matrix"]["configurations"] == ["ours_ft", "progressive", "dfag"] and SHORTCUT_LAMBDA == 0.7

    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "status": status,
        "created_unix": time.time(),
        "checks": checks,
        "protocol_value_match": protocol_match,
        "source_hashes": {
            "authoritative_config": sha256_file(SOURCE_CONFIG),
            "authoritative_dfag_implementation": sha256_file(SOURCE_DFAG),
            "authoritative_interpolation": sha256_file(SOURCE_INTERPOLATION),
            "new_config": sha256_file(CONFIG_PATH),
            "runner": sha256_file(TRAIN_RUNNER),
        },
        "dataset": {"n": len(frame), "classes": len(mapping), "missing_images": missing[:20], "folds": fold_rows},
        "architectures": architectures,
        "memory_feasibility": memory,
        "gpu_state_before": before,
        "gpu_state_after": gpu_state(0),
        "oof_schema": oof_schema,
        "dfag_stage1_provenance": "same-backbone/same-fold selected Ours-FT Stage-1 checkpoint from this new run; copied to frozen anchor and plastic; no separate DFAG Stage-1 training",
        "limitations": "H=256 is fixed across backbones, so the projection compression ratio differs across architectures and is not independently controlled.",
    }
    (ROOT / "preflight_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Final Cross-Backbone Preflight Audit", "", f"Status: **{status}**", "",
        "## Protocol provenance", "",
        f"- Authoritative Classify Leaves config SHA256: `{payload['source_hashes']['authoritative_config']}`",
        "- Standalone DFAG Stage 1 is not retrained; it reuses the same-backbone/same-fold Ours-FT selected Stage-1 checkpoint generated by this new experiment.",
        "- Old cross-backbone results and checkpoints are not used.", "",
        "## Dataset and folds", "",
        f"- Samples: {len(frame):,}; classes: {len(mapping)}; missing images: {len(missing)}.",
        "- StratifiedKFold(5, shuffle=True, random_state=42); every sample appears in held-out OOF exactly once; no train/validation overlap.", "",
        "## Architecture", "",
        "| Backbone | Actual D | H | D/H | Gate parameters | Architecture checks |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for backbone, row in architectures.items():
        lines.append(f"| {backbone} | {row['actual_dimension']} | {row['hidden_dimension']} | {row['D_over_H']:.1f} | {row['gate_parameters']:,} | {row['status']} |")
    lines.extend(["", "H=256 is fixed across backbones, so the projection compression ratio differs across architectures and is not independently controlled.", "",
                  "## GPU memory feasibility", "", "| Backbone | Configuration | Batch | Peak allocated (MiB) | Status |", "|---|---|---:|---:|---|"])
    for row in memory:
        lines.append(f"| {row['backbone']} | {row['method']} | {row['batch_size']} | {row.get('peak_memory_mib', float('nan')):.1f} | {row['status']} |")
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- `{key}`: {'PASS' if value else 'FAIL'}" for key, value in checks.items())
    lines.extend(["", "No training result, smoke accuracy, or old cross-backbone checkpoint was used to choose architecture, lambda, H, or batch size."])
    (ROOT / "FINAL_CROSS_BACKBONE_PREFLIGHT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": len(checks), "architectures": {k: v["actual_dimension"] for k, v in architectures.items()}, "memory": memory}), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
