"""Finalize the corrected two-job smoke and stop before formal training."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

import corrected_transform
import protocol_core


corrected_transform.install()


def main() -> None:
    root = protocol_core.ROOT
    config = protocol_core.load_config()
    preflight_path = root / "manifests" / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    ledger_path = root / "formal_job_ledger.csv"
    with ledger_path.open("r", newline="", encoding="utf-8-sig") as handle:
        ledger = list(csv.DictReader(handle))

    run_manifests = {}
    for run_id in protocol_core.RUN_IDS:
        path = root / "runs" / run_id / "fold_0" / "run_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        run_manifests[run_id] = json.loads(path.read_text(encoding="utf-8"))

    progressive_root = root / "runs" / "progressive_lambda_0_7" / "fold_0"
    stage1_stages = np.load(progressive_root / "stage1_validation_post_shortcut_stages.npy", mmap_mode="r")
    stage2_stages = np.load(progressive_root / "stage2_validation_post_shortcut_stages.npy", mmap_mode="r")
    finite_metrics = all(
        math.isfinite(float(manifest["metrics"][stage][split_name][metric]))
        for manifest in run_manifests.values()
        for stage in ("stage1", "stage2")
        for split_name in ("validation", "secondary_official_test")
        for metric in ("accuracy", "macro_f1", "balanced_accuracy")
    )
    ledger_unique = {(row["dataset"], row["method"], row["lambda"], row["fold"]) for row in ledger}
    expected_counts = {
        ("CUB-200-2011", "Ours-FT", ""): 5,
        ("CUB-200-2011", "Progressive Head", "0.7"): 5,
        ("CUB-200-2011", "Progressive Head", "0.1"): 5,
        ("CUB-200-2011", "Progressive Head", "1.0"): 5,
        ("Stanford Cars", "Ours-FT", ""): 5,
        ("Stanford Cars", "Progressive Head", "0.7"): 5,
        ("Oxford Flowers-102", "Ours-FT", ""): 5,
        ("Oxford Flowers-102", "Progressive Head", "0.7"): 5,
    }
    actual_counts = {}
    for row in ledger:
        key = (row["dataset"], row["method"], row["lambda"])
        actual_counts[key] = actual_counts.get(key, 0) + 1

    transforms = corrected_transform.serialized_transforms()
    ledger_checks = {
        "row_count_exactly_40": len(ledger) == 40,
        "unique_job_count_exactly_40": len(ledger_unique) == 40,
        "exact_dataset_method_lambda_counts": actual_counts == expected_counts,
        "all_status_pending": {row["formal_status"] for row in ledger} == {"PENDING"},
        "all_training_seed_42": {row["training_seed"] for row in ledger} == {"42"},
        "all_split_random_state_42": {row["split_random_state"] for row in ledger} == {"42"},
        "all_config_hash_current": {row["protocol_config_sha256"] for row in ledger} == {protocol_core.sha256_file(protocol_core.CONFIG_PATH)},
    }
    smoke_checks = {
        "preflight_pass": preflight.get("status") == "PASS",
        "all_preflight_hard_checks_pass": all(row["pass"] for row in preflight["checks"]),
        "both_new_runs_complete": len(run_manifests) == 2 and all(row.get("status") == "COMPLETE" for row in run_manifests.values()),
        "old_smoke_not_reused": preflight.get("previous_smoke_reused") is False and config["smoke_scope"]["reuse_previous_smoke"] is False,
        "fold0_seed42_split42_only": all(row["fold"] == 0 and row["training_seed"] == 42 and row["split_random_state"] == 42 for row in run_manifests.values()),
        "counts_exact": all(row["train_count"] == 4795 and row["validation_count"] == 1199 and row["official_test_count"] == 5794 for row in run_manifests.values()),
        "no_leakage": all(row["no_train_validation_overlap"] and row["no_development_test_overlap"] for row in run_manifests.values()),
        "no_class_weights": all(row["class_weights"] is None for row in run_manifests.values()),
        "original_view_no_tta_no_ensemble": all(not row["primary_tta"] and not row["fold_ensemble"] for row in run_manifests.values()),
        "official_test_not_used_for_selection": all(not row["official_test_labels_used_for_selection"] for row in run_manifests.values()),
        "stage2_from_selected_stage1": all(row["stage2_started_from_selected_stage1"] for row in run_manifests.values()),
        "bn_recalibration_exact": all(row["bn_recalibration"]["batches"] == 50 and not row["bn_recalibration"]["reset_running_stats"] and row["bn_recalibration"]["whole_model_train_mode"] and row["bn_recalibration"]["no_grad"] and row["bn_recalibration"]["dropout_training"] and not row["bn_recalibration"]["optimizer_created_before_recalibration"] for row in run_manifests.values()),
        "history_150_60": all(row["history_counts"] == {"stage1": 150, "stage2": 60} for row in run_manifests.values()),
        "finite_metrics": finite_metrics,
        "cub_rotation_absent": not transforms["cub"]["random_rotation_present"],
        "cars_rotation_absent": not transforms["cars"]["random_rotation_present"],
        "flowers_rotation_present": transforms["flowers"]["random_rotation_present"],
        "vertical_flip_absent_all": all(not transforms[key]["vertical_flip_present"] for key in transforms),
        "affine_degrees_zero_all": all(transforms[key]["random_affine_degrees"] == [0.0, 0.0] for key in transforms),
        "progressive_stage1_shape_N_5_256": tuple(stage1_stages.shape) == (1199, 5, 256),
        "progressive_stage2_shape_N_5_256": tuple(stage2_stages.shape) == (1199, 5, 256),
        "progressive_lambda_exact_0_7": run_manifests["progressive_lambda_0_7"]["shortcut_lambda"] == 0.7,
        "progressive_diagnostics_both_stages": set(run_manifests["progressive_lambda_0_7"]["metrics"]["representation_diagnostics"]) == {"stage1", "stage2"},
        "no_formal_job_launched": preflight.get("formal_jobs_launched") == 0 and all(not row["full_40_job_run_authorized"] for row in run_manifests.values()),
    }
    ledger_frozen = all(ledger_checks.values())
    smoke_pass = all(smoke_checks.values())
    full_ready = ledger_frozen and smoke_pass
    final = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "protocol_identity": "NEWLY FROZEN UNIFIED CROSS-DATASET VALIDATION PROTOCOL",
        "historical_protocol_reproduction": False,
        "instruction_sha256": config["instruction_sha256"],
        "config_path": str(protocol_core.CONFIG_PATH.resolve()),
        "config_sha256": protocol_core.sha256_file(protocol_core.CONFIG_PATH),
        "corrected_transform_source": str(Path(corrected_transform.__file__).resolve()),
        "corrected_transform_sha256": protocol_core.sha256_file(Path(corrected_transform.__file__)),
        "exact_protocol": config,
        "preflight": preflight,
        "formal_job_ledger": {
            "path": str(ledger_path.resolve()), "sha256": protocol_core.sha256_file(ledger_path),
            "row_count": len(ledger), "checks": ledger_checks,
        },
        "smoke_runs": run_manifests,
        "smoke_checks": smoke_checks,
        "CORRECTED_SMOKE_PASS": "YES" if smoke_pass else "NO",
        "FORMAL_40_JOB_LEDGER_FROZEN": "YES" if ledger_frozen else "NO",
        "FULL_40_JOB_RUN_READY": "YES" if full_ready else "NO",
        "automatic_formal_jobs_launched": 0,
        "stop_after_smoke": True,
    }
    json_path = root / "corrected_external_protocol_manifest.json"
    json_path.write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    protocol_md = f"""# Corrected External Protocol Manifest

## Frozen identity

- Protocol: **NEWLY FROZEN UNIFIED CROSS-DATASET VALIDATION PROTOCOL**
- Historical reproduction: **NO**
- Training seed / split random state: `42 / 42`
- Primary evaluation: deterministic original-view five-fold OOF; no TTA; no fold ensemble
- Official-test results: secondary only
- Config SHA-256: `{final['config_sha256']}`

## Corrected dataset-specific training transforms

### CUB-200-2011

```text
{transforms['cub']['train_repr']}
```

`RandomRotation` absent; `RandomVerticalFlip` absent; `RandomAffine.degrees=[0.0,0.0]`.

### Stanford Cars

```text
{transforms['cars']['train_repr']}
```

`RandomRotation` absent; `RandomVerticalFlip` absent; `RandomAffine.degrees=[0.0,0.0]`.

### Oxford Flowers-102

```text
{transforms['flowers']['train_repr']}
```

`RandomRotation(-180,180)` present with white fill; `RandomVerticalFlip` absent; `RandomAffine.degrees=[0.0,0.0]`.

### Validation / official test, all datasets

```text
{transforms['cub']['validation_repr']}
```

Deterministic, original view only.

## Frozen model and training

Ours-FT uses Inception-ResNet-v2 GAP 1536 followed by the 1024/1024 classification head. Progressive Head uses a 1536-to-256 projection, five mappings with `f_i=F_i(f_(i-1))+0.7*f_(i-1)`, arithmetic mean of post-shortcut `f1...f5`, Dropout(0.2), and the classifier. MHSA, terminal residual, and DFAG are absent.

Stage 1 is 150 epochs, batch 32, AdamW, backbone/head LR `1e-4/1e-3`, weight decay `1e-3`, three-epoch warm-up, cosine annealing, label smoothing 0.03, Mixup 0.4/p0.4, CutMix 1.0/p0.5, no class weights. Epoch 105 is full strength; epochs 106-149 decay. Stage 2 reloads the selected Stage-1 checkpoint, applies the exact 50-batch no-reset BN recalibration, then trains 60 epochs with AdamW LR `5e-5`, weight decay `5e-4`, cosine scheduler and label smoothing 0.03; Mixup/CutMix/class weights are off.

## CUB representation aggregation

Cosine and centered-linear CKA are computed independently using each fold model and its own held-out samples. All five fold values are retained, then combined by an unweighted arithmetic mean. Feature matrices from different fold models must never be concatenated for a global CKA. Authorized lambdas remain exactly 0.1, 0.7, and 1.0.

## Formal ledger

`formal_job_ledger.csv` contains exactly 40 unique PENDING rows: CUB 20, Cars 10, Flowers 10. No formal job was launched.
"""
    (root / "CORRECTED_EXTERNAL_PROTOCOL_MANIFEST.md").write_text(protocol_md, encoding="utf-8")

    result_rows = []
    for run_id, manifest in run_manifests.items():
        for stage in ("stage1", "stage2"):
            row = manifest["metrics"][stage]
            result_rows.append(
                f"| {run_id} | {stage.title()} | {row['best_epoch_zero_based']} | "
                f"{row['validation']['accuracy']:.6f} | {row['validation']['macro_f1']:.6f} | "
                f"{row['validation']['balanced_accuracy']:.6f} | {row['secondary_official_test']['accuracy']:.6f} |"
            )
    diag = run_manifests["progressive_lambda_0_7"]["metrics"]["representation_diagnostics"]
    failed = [name for name, passed in {**ledger_checks, **smoke_checks}.items() if not passed]
    audit_md = f"""# Corrected External Smoke Audit

## Final decisions

- **CORRECTED_SMOKE_PASS = {'YES' if smoke_pass else 'NO'}**
- **FORMAL_40_JOB_LEDGER_FROZEN = {'YES' if ledger_frozen else 'NO'}**
- **FULL_40_JOB_RUN_READY = {'YES' if full_ready else 'NO'}**

Manual approval is still required. Formal jobs launched: **0**.

## Hard-check summary

- Preflight: `{preflight['status']}`; {len(preflight['checks'])} checks; {len([row for row in preflight['checks'] if not row['pass']])} failed.
- Corrected smoke jobs: two new CUB fold0/seed42 runs; old smoke checkpoints, predictions, and metrics were not reused.
- Each run completed Stage1 150 epochs, exact 50-batch no-reset BN recalibration, and Stage2 60 epochs.
- CUB fold0 counts: train 4,795; validation 1,199; official test 5,794; no overlap.
- Original-view deterministic evaluation only; no TTA, no ensemble, no class weights.
- Progressive post-shortcut tensors: Stage1 `{list(stage1_stages.shape)}`, Stage2 `{list(stage2_stages.shape)}`.
- Formal ledger: {len(ledger)} unique rows, all PENDING, current config hash on every row.

## Corrected fold0 smoke metrics

These are technical smoke metrics, not final five-fold OOF results. They are not compared with the old smoke for selection.

| Run | Stage | Best epoch (zero-based) | Val Accuracy | Val Macro-F1 | Val Balanced Acc. | Secondary official-test Accuracy |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(result_rows)}

## Progressive representation diagnostics

| Stage | Mean off-diagonal cosine | Centered-linear CKA |
|---|---:|---:|
| Stage 1 | {diag['stage1']['mean_off_diagonal_cosine']:.8f} | {diag['stage1']['mean_centered_linear_cka']:.8f} |
| Stage 2 | {diag['stage2']['mean_off_diagonal_cosine']:.8f} | {diag['stage2']['mean_centered_linear_cka']:.8f} |

These fold0 values describe similarity only. Future five-fold reporting must retain fold-specific values and use their unweighted arithmetic mean.

## Failed checks

{failed if failed else 'None.'}

## STOP

The corrected smoke is complete. No protocol adjustment was made after observing results, and no formal job was launched.
"""
    (root / "CORRECTED_EXTERNAL_SMOKE_AUDIT.md").write_text(audit_md, encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE",
        "CORRECTED_SMOKE_PASS": final["CORRECTED_SMOKE_PASS"],
        "FORMAL_40_JOB_LEDGER_FROZEN": final["FORMAL_40_JOB_LEDGER_FROZEN"],
        "FULL_40_JOB_RUN_READY": final["FULL_40_JOB_RUN_READY"],
        "formal_jobs_launched": 0,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

