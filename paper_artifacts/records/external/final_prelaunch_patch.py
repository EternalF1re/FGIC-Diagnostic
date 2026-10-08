"""Apply the final protocol-only patch without launching formal training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
VALIDATION_ROOT = HERE.parent
CORRECTED_ROOT = VALIDATION_ROOT / "corrected_external_protocol_smoke"
CORRECTED_SCRIPTS = CORRECTED_ROOT / "scripts"
if str(CORRECTED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTED_SCRIPTS))

import protocol_core  # noqa: E402
import corrected_transform  # noqa: E402


corrected_transform.install()

INSTRUCTION_SHA256 = "B70F752377437D7D76EAA4CEBF17CEEA436B7A49EB5F7950D5341075C7E0C93B"
PREVIOUS_CONFIG_SHA256 = "a4a975bf1636e91e27621bd666d91e8bab3b5f960e7d964bbf50fd50f1e990fc"
SUPERSESSION_REASON = (
    "Stage1 label smoothing corrected from manuscript-recorded 0.03 "
    "to audited controlled-protocol value 0.05 before formal training."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def build_final_config() -> tuple[dict, Path, str]:
    previous_path = CORRECTED_ROOT / "config.json"
    observed_previous_hash = sha256(previous_path)
    if observed_previous_hash != PREVIOUS_CONFIG_SHA256:
        raise ValueError(f"previous config hash mismatch: {observed_previous_hash}")
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    final = deepcopy(previous)
    final["schema_version"] = 3
    final["experiment_id"] = "FINAL_NEWLY_FROZEN_UNIFIED_CROSS_DATASET_VALIDATION_PROTOCOL"
    final["instruction_sha256"] = INSTRUCTION_SHA256
    final["training"]["stage1"]["label_smoothing"] = 0.05
    final["training"]["stage2"]["label_smoothing"] = 0.03
    final["protocol_supersession"] = {
        "previous_config_path": str(previous_path.resolve()),
        "previous_config_sha256": PREVIOUS_CONFIG_SHA256,
        "reason_for_supersession": SUPERSESSION_REASON,
        "formal_results_under_previous_config": 0,
        "formal_results_under_final_config_at_creation": 0,
        "provenance_change_not_result_tuning": True,
    }
    final["batch_size_provenance"] = {
        "classify_leaves_controlled_batch_size": 64,
        "new_external_validation_batch_size": 32,
        "within_each_external_dataset_same_across_compared_configurations": True,
        "explicit_protocol_difference_not_error": True,
    }
    final["stage2_epoch_index_invariant"] = (
        "zero-based Stage2 epoch 0 denotes validation performed after the first "
        "completed gradient-based Stage2 training epoch"
    )
    final["future_cross_dataset_lambda_reporting_rule"] = {
        "common_leaves_cub_points": [0.1, 0.7, 1.0],
        "leaves_only_descriptive_point": 0.9,
        "four_point_leaves_and_three_point_cub_not_fully_matched": True,
        "changes_formal_jobs": False,
    }
    path = HERE / "final_external_protocol_config.json"
    write_json(path, final)
    return final, path, sha256(path)


def build_formal_ledger(config_sha256: str) -> tuple[Path, list[dict]]:
    specs = [
        ("CUB-200-2011", "Ours-FT", ""),
        ("CUB-200-2011", "Progressive Head", "0.7"),
        ("CUB-200-2011", "Progressive Head", "0.1"),
        ("CUB-200-2011", "Progressive Head", "1.0"),
        ("Stanford Cars", "Ours-FT", ""),
        ("Stanford Cars", "Progressive Head", "0.7"),
        ("Oxford Flowers-102", "Ours-FT", ""),
        ("Oxford Flowers-102", "Progressive Head", "0.7"),
    ]
    rows = []
    for dataset, method, shortcut_lambda in specs:
        for fold in range(5):
            rows.append({
                "dataset": dataset,
                "method": method,
                "lambda": shortcut_lambda,
                "training_seed": 42,
                "split_random_state": 42,
                "fold": fold,
                "formal_status": "PENDING",
                "protocol_config_sha256": config_sha256,
            })
    path = HERE / "formal_job_ledger.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def verify_epoch0_semantics() -> dict:
    runner = CORRECTED_SCRIPTS / "train_cub_smoke.py"
    lines = runner.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("def stage2_train"))
    end = next(index for index, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("def "))
    section = lines[start:end]

    def relative(fragment: str) -> int:
        return next(index for index, line in enumerate(section) if fragment in line)

    positions = {
        "epoch_loop": relative("for epoch in range(epochs):"),
        "training_batch_loop": relative("for batch_index, (images, labels"),
        "loss_backward": relative("scaler.scale(loss).backward()"),
        "optimizer_step": relative("scaler.step(optimizer)"),
        "validation": relative("val_accuracy = float(evaluate(model, val_loader, device)"),
        "checkpoint_condition": relative("if val_accuracy > best_accuracy:"),
        "checkpoint_save": relative("torch.save(checkpoint_payload"),
    }
    ordered = list(positions.values()) == sorted(positions.values())
    corrected_history = {}
    for run_id in ("ours_ft", "progressive_lambda_0_7"):
        history = pd.read_csv(CORRECTED_ROOT / "runs" / run_id / "fold_0" / "stage2_history.csv")
        epoch0 = history.loc[history["epoch_zero_based"] == 0]
        corrected_history[run_id] = {
            "epoch0_row_count": int(len(epoch0)),
            "epoch0_finite_train_loss": bool(len(epoch0) == 1 and np.isfinite(epoch0.iloc[0]["train_loss"])),
            "epoch0_finite_validation_accuracy": bool(len(epoch0) == 1 and np.isfinite(epoch0.iloc[0]["val_accuracy"])),
            "epoch0_amp_overflow_batches": int(epoch0.iloc[0]["amp_overflow_batches"]),
        }
    confirmed = ordered and all(
        row["epoch0_row_count"] == 1
        and row["epoch0_finite_train_loss"]
        and row["epoch0_finite_validation_accuracy"]
        and row["epoch0_amp_overflow_batches"] == 0
        for row in corrected_history.values()
    )
    return {
        "confirmed": confirmed,
        "invariant": "zero-based Stage2 epoch 0 denotes validation after the first completed gradient-based Stage2 epoch",
        "runner_path": str(runner.resolve()),
        "runner_sha256": sha256(runner),
        "stage2_function_one_based_lines": {key: start + value + 1 for key, value in positions.items()},
        "control_flow_order_verified": ordered,
        "corrected_smoke_epoch0_rows": corrected_history,
    }


def export_stage2_curves() -> tuple[Path, Path, list[dict]]:
    rows = []
    summaries = []
    for run_id in ("ours_ft", "progressive_lambda_0_7"):
        run_root = CORRECTED_ROOT / "runs" / run_id / "fold_0"
        history = pd.read_csv(run_root / "stage2_history.csv").sort_values("epoch_zero_based")
        metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
        best_epoch = int(metrics["stage2"]["best_epoch_zero_based"])
        if len(history) != 60 or history["epoch_zero_based"].tolist() != list(range(60)):
            raise ValueError(f"invalid 60-point Stage2 curve for {run_id}")
        values = history["val_accuracy"].to_numpy(dtype=np.float64)
        changes = np.diff(values)
        for row in history.itertuples(index=False):
            rows.append({
                "run_id": run_id,
                "epoch_zero_based": int(row.epoch_zero_based),
                "validation_accuracy": float(row.val_accuracy),
                "train_loss": float(row.train_loss),
                "learning_rate": float(row.backbone_lr),
                "selected_best": int(row.epoch_zero_based) == best_epoch,
            })
        summaries.append({
            "run_id": run_id,
            "epoch0_validation_accuracy": float(values[0]),
            "minimum_validation_accuracy": float(values.min()),
            "maximum_validation_accuracy": float(values.max()),
            "final_epoch_accuracy": float(values[-1]),
            "best_epoch_zero_based": best_epoch,
            "higher_than_previous": int(np.sum(changes > 0)),
            "lower_than_previous": int(np.sum(changes < 0)),
            "equal_to_previous": int(np.sum(changes == 0)),
            "strictly_monotonic_decreasing": bool(np.all(changes < 0)),
            "local_recoveries_after_epoch0": bool(np.any(changes > 0)),
        })
    curve_path = HERE / "corrected_smoke_stage2_curves.csv"
    pd.DataFrame(rows).to_csv(curve_path, index=False)
    audit_path = HERE / "CORRECTED_SMOKE_STAGE2_CURVE_AUDIT.md"
    table = []
    for row in summaries:
        table.append(
            f"| {row['run_id']} | {row['epoch0_validation_accuracy']:.8f} | "
            f"{row['minimum_validation_accuracy']:.8f} | {row['maximum_validation_accuracy']:.8f} | "
            f"{row['final_epoch_accuracy']:.8f} | {row['best_epoch_zero_based']} | "
            f"{row['higher_than_previous']} | {row['lower_than_previous']} | {row['equal_to_previous']} | "
            f"{'YES' if row['strictly_monotonic_decreasing'] else 'NO'} | "
            f"{'YES' if row['local_recoveries_after_epoch0'] else 'NO'} |"
        )
    audit = f"""# Corrected Smoke Stage2 Curve Audit

This is a read-only export of the two already completed corrected CUB fold0 smoke runs. No training was repeated. These curves are descriptive technical diagnostics, not cross-dataset scientific conclusions and not a basis for model or lambda selection.

| Run | Epoch0 val acc | Minimum | Maximum | Final epoch | Best epoch | Higher | Lower | Equal | Strict monotonic decrease | Local recoveries after epoch0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(table)}

`epoch_zero_based=0` is the validation recorded after the first completed gradient-based Stage2 training epoch. Higher/lower/equal counts compare each of epochs 1-59 with its immediately preceding epoch.
"""
    audit_path.write_text(audit, encoding="utf-8")
    return curve_path, audit_path, summaries


def lightweight_sanity(final_config: dict, ledger_path: Path) -> dict:
    ledger_hash_before = sha256(ledger_path)
    checkpoint_before = sorted(str(path) for path in HERE.rglob("*.pth"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    protocol_core.seed_everything(42)
    train_loader, _, _, _, _, _, _ = protocol_core.build_cub_loaders(0, 1, final_config)
    images, labels, _, sample_ids, _ = next(iter(train_loader))
    if len(images) != 32 or len(labels) != 32 or len(sample_ids) != 32:
        raise ValueError("sanity batch is not a legal batch_size=32 training batch")
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    mixed_images, labels_a, labels_b, lam, augmentation_type, augmentation_scale = protocol_core.batch_mix(
        images, labels, epoch=0, batch_index=0, epochs=150, device=device
    )
    records = {}
    for run_id in ("ours_ft", "progressive_lambda_0_7"):
        protocol_core.seed_everything(42)
        model = protocol_core.ExternalModel(run_id, 200, pretrained=True)
        protocol_core.initialize_head(model)
        model.to(device).train()
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": 1e-4},
                {"params": model.head.parameters(), "lr": 1e-3},
            ],
            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        scaler = torch.amp.GradScaler(device.type, init_scale=4096, enabled=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, _, _ = model(mixed_images)
            loss = protocol_core.mixed_loss(criterion, logits, labels_a, labels_b, lam)
        finite_loss = bool(torch.isfinite(loss).item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite_gradients = bool(gradients) and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
        nonzero_gradients = sum(int(bool(torch.count_nonzero(gradient).item())) for gradient in gradients)
        records[run_id] = {
            "batch_size": len(images),
            "criterion": "CrossEntropyLoss(label_smoothing=0.05)",
            "class_weights": None,
            "augmentation_type": augmentation_type,
            "augmentation_scale": augmentation_scale,
            "finite_loss": finite_loss,
            "gradient_tensor_count": len(gradients),
            "finite_gradients": finite_gradients,
            "nonzero_gradient_tensor_count": nonzero_gradients,
            "backward_pass_completed": True,
            "optimizer_step_performed": False,
            "checkpoint_saved": False,
            "accuracy_computed": False,
        }
        del model, optimizer, criterion, scaler, logits, loss, gradients
        if device.type == "cuda":
            torch.cuda.empty_cache()
    checkpoint_after = sorted(str(path) for path in HERE.rglob("*.pth"))
    ledger_hash_after = sha256(ledger_path)
    passed = (
        all(row["finite_loss"] and row["finite_gradients"] and row["backward_pass_completed"]
            and not row["optimizer_step_performed"] and not row["checkpoint_saved"]
            and not row["accuracy_computed"] for row in records.values())
        and checkpoint_before == checkpoint_after
        and ledger_hash_before == ledger_hash_after
    )
    return {
        "pass": passed,
        "technical_check_only_not_experiment": True,
        "device": str(device),
        "legal_training_batch_size": len(images),
        "records": records,
        "checkpoint_files_before": checkpoint_before,
        "checkpoint_files_after": checkpoint_after,
        "formal_ledger_sha256_before": ledger_hash_before,
        "formal_ledger_sha256_after": ledger_hash_after,
        "formal_ledger_unchanged": ledger_hash_before == ledger_hash_after,
    }


def main() -> None:
    final_config, config_path, config_hash = build_final_config()
    ledger_path, ledger = build_formal_ledger(config_hash)
    epoch0 = verify_epoch0_semantics()
    curve_path, curve_audit_path, curve_summaries = export_stage2_curves()
    sanity = lightweight_sanity(final_config, ledger_path)
    write_json(HERE / "lightweight_sanity_check.json", sanity)

    unique = {(row["dataset"], row["method"], row["lambda"], row["fold"]) for row in ledger}
    checks = {
        "previous_config_hash_exact": sha256(CORRECTED_ROOT / "config.json") == PREVIOUS_CONFIG_SHA256,
        "stage1_label_smoothing_0_05": final_config["training"]["stage1"]["label_smoothing"] == 0.05,
        "stage2_label_smoothing_0_03": final_config["training"]["stage2"]["label_smoothing"] == 0.03,
        "external_batch_size_32": final_config["training"]["batch_size"] == 32,
        "ledger_rows_40": len(ledger) == 40,
        "ledger_unique_40": len(unique) == 40,
        "ledger_pending_40": sum(row["formal_status"] == "PENDING" for row in ledger) == 40,
        "formal_jobs_started_0": True,
        "ledger_seed42": {row["training_seed"] for row in ledger} == {42},
        "ledger_split42": {row["split_random_state"] for row in ledger} == {42},
        "ledger_new_hash_every_row": {row["protocol_config_sha256"] for row in ledger} == {config_hash},
        "epoch0_semantics_confirmed": epoch0["confirmed"],
        "lightweight_sanity_pass": sanity["pass"],
        "curve_rows_120": len(pd.read_csv(curve_path)) == 120,
        "no_checkpoint_created": not any(HERE.rglob("*.pth")),
    }
    ready = all(checks.values())
    final_manifest = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "instruction_sha256": INSTRUCTION_SHA256,
        "protocol_identity": "FINAL NEWLY FROZEN UNIFIED CROSS-DATASET VALIDATION PROTOCOL",
        "historical_protocol_reproduction": False,
        "previous_config_sha256": PREVIOUS_CONFIG_SHA256,
        "reason_for_supersession": SUPERSESSION_REASON,
        "new_config_path": str(config_path.resolve()),
        "new_config_sha256": config_hash,
        "formal_results_under_previous_config": 0,
        "formal_results_under_new_config": 0,
        "final_config": final_config,
        "formal_ledger": {
            "path": str(ledger_path.resolve()), "sha256": sha256(ledger_path),
            "rows": len(ledger), "pending": sum(row["formal_status"] == "PENDING" for row in ledger),
            "started": 0,
        },
        "lightweight_sanity": sanity,
        "epoch0_semantics": epoch0,
        "corrected_smoke_stage2_curve_export": {
            "csv": str(curve_path.resolve()), "csv_sha256": sha256(curve_path),
            "audit": str(curve_audit_path.resolve()), "audit_sha256": sha256(curve_audit_path),
            "summaries": curve_summaries,
        },
        "final_checks": checks,
        "STAGE1_LABEL_SMOOTHING": 0.05,
        "STAGE2_LABEL_SMOOTHING": 0.03,
        "EXTERNAL_BATCH_SIZE": 32,
        "NEW_CONFIG_SHA256": config_hash,
        "FORMAL_LEDGER_ROWS": 40,
        "FORMAL_LEDGER_PENDING": 40,
        "FORMAL_JOBS_STARTED": 0,
        "EPOCH0_SEMANTICS_CONFIRMED": "YES" if epoch0["confirmed"] else "NO",
        "LIGHTWEIGHT_SANITY_PASS": "YES" if sanity["pass"] else "NO",
        "FINAL_40_JOB_RUN_READY": "YES" if ready else "NO",
        "stop_before_formal_training": True,
    }
    json_path = HERE / "final_external_protocol_manifest.json"
    write_json(json_path, final_manifest)

    md = f"""# Final External Protocol Manifest

## Final pre-launch status

- `STAGE1_LABEL_SMOOTHING = 0.05`
- `STAGE2_LABEL_SMOOTHING = 0.03`
- `EXTERNAL_BATCH_SIZE = 32`
- `NEW_CONFIG_SHA256 = {config_hash}`
- `FORMAL_LEDGER_ROWS = 40`
- `FORMAL_LEDGER_PENDING = 40`
- `FORMAL_JOBS_STARTED = 0`
- `EPOCH0_SEMANTICS_CONFIRMED = {'YES' if epoch0['confirmed'] else 'NO'}`
- `LIGHTWEIGHT_SANITY_PASS = {'YES' if sanity['pass'] else 'NO'}`
- `FINAL_40_JOB_RUN_READY = {'YES' if ready else 'NO'}`

Formal training remains stopped pending manual approval.

## Superseded protocol identity

- Previous config SHA-256: `{PREVIOUS_CONFIG_SHA256}`
- Reason: {SUPERSESSION_REASON}
- Formal results under the previous config: **0**
- Formal results under this final config: **0**

This correction comes from protocol provenance audit, not smoke accuracy or result-dependent tuning.

## Frozen protocol delta

The only scientific protocol correction is Stage1 label smoothing `0.03 -> 0.05`. Stage2 remains `0.03`. It applies identically to all 40 CUB/Cars/Flowers jobs and all methods/lambdas.

External batch size remains 32. Classify Leaves controlled analyses use 64; this is an explicit protocol difference, not an error. Every configuration within an external dataset uses the same batch size.

All other data pools, folds, transforms, optimization, augmentation, BN recalibration, checkpoint selection, evaluation, architectures, lambdas, and per-fold CKA/cosine aggregation semantics remain frozen.

## Lightweight sanity check

Both CUB Ours-FT and Progressive lambda=0.7 constructed the corrected legal batch-size-32 Stage1 path, used `CrossEntropyLoss(label_smoothing=0.05)`, produced finite loss, and completed one isolated backward pass with finite gradients. No optimizer step, checkpoint, accuracy, result, or ledger-status change was produced. This was a technical configuration check, not an experiment.

## Stage2 epoch-0 invariant

Confirmed directly from `{epoch0['runner_path']}` (`{epoch0['runner_sha256']}`): the runner completes the training batch loop, loss backward, and optimizer step before validation and checkpoint selection. Therefore zero-based Stage2 epoch 0 is validation after the first completed gradient-based Stage2 epoch. Behavior was not changed.

## Formal ledger and reporting

`formal_job_ledger.csv` contains exactly 40 unique PENDING jobs with the new config hash. Started jobs: 0.

Future Leaves-vs-CUB shortcut comparison uses only common lambda points `0.1, 0.7, 1.0`. Leaves `0.9` is an additional Leaves-only descriptive point; the four-point Leaves curve and three-point CUB curve are not fully matched.
"""
    (HERE / "FINAL_EXTERNAL_PROTOCOL_MANIFEST.md").write_text(md, encoding="utf-8")
    print(json.dumps({key: final_manifest[key] for key in (
        "STAGE1_LABEL_SMOOTHING", "STAGE2_LABEL_SMOOTHING", "EXTERNAL_BATCH_SIZE",
        "NEW_CONFIG_SHA256", "FORMAL_LEDGER_ROWS", "FORMAL_LEDGER_PENDING",
        "FORMAL_JOBS_STARTED", "EPOCH0_SEMANTICS_CONFIRMED",
        "LIGHTWEIGHT_SANITY_PASS", "FINAL_40_JOB_RUN_READY"
    )}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
