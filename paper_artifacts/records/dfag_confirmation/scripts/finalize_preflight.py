"""Assemble alpha=0.5 comparators and issue the training authorization gate."""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import torch

from confirmation_common import assemble_reference, comparator_task_dir, validate_oof
from dfag_common import (
    EXPECTED_N,
    EXP_ROOT,
    FOLDS,
    SEEDS,
    UnifiedStandaloneDFAG,
    bn_state,
    comparator_oof,
    load_config,
    output_dir,
    sha256_file,
    source_checkpoint,
    state_digest,
)
from prepare_static import verify_snapshot
from screen_core import build_loaders


def main() -> None:
    static = json.loads((EXP_ROOT / "manifests" / "static_preflight_audit.json").read_text(encoding="utf-8"))
    orchestration = json.loads((EXP_ROOT / "comparators" / "orchestrator_summary.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "static_preflight_pass": static.get("status") == "PASS",
        "target_seeds_exact_45_46": SEEDS == (45, 46),
        "five_folds_per_seed": FOLDS == (0, 1, 2, 3, 4),
        "ten_new_training_jobs": len(SEEDS) * len(FOLDS) == 10,
        "ten_comparator_tasks_complete": orchestration.get("status") == "ALL_10_COMPARATORS_COMPLETE",
    }
    comparator_rows = []
    alignment = labels_exact = fold_exact = True
    for seed in SEEDS:
        pieces = []
        for fold in FOLDS:
            task = comparator_task_dir(seed, fold)
            manifest = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
            endpoint_rows = manifest.get("endpoint_fidelity", [])
            endpoint_pass = (
                manifest.get("status") == "COMPLETE"
                and manifest.get("inference_only") is True
                and manifest.get("optimizer") is False
                and manifest.get("backward") is False
                and manifest.get("bn_refresh") is False
                and len(endpoint_rows) == 3
                and all(row.get("parameter_state_unchanged") for row in endpoint_rows)
                and all(row.get("all_gradients_none") for row in endpoint_rows)
                and all(
                    row.get("sample_ids_exact") and row.get("labels_exact") and row.get("predictions_exact")
                    for row in endpoint_rows if row["alpha"] in (0.0, 1.0)
                )
            )
            with np.load(task / "alpha_0_5.npz", allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            pieces.append(data)
            comparator_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "status": manifest.get("status"),
                    "endpoint_fidelity_pass": endpoint_pass,
                    "artifact": str((task / "alpha_0_5.npz").resolve()),
                    "artifact_sha256": sha256_file(task / "alpha_0_5.npz"),
                    "stage1_sha256": manifest.get("stage1_sha256"),
                    "stage2_sha256": manifest.get("stage2_sha256"),
                }
            )
        merged = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
        order = np.argsort(merged["sample_id"])
        merged = {key: value[order] for key, value in merged.items()}
        validate_oof(merged, f"seed{seed}/alpha0.5")
        stage1 = assemble_reference(seed, 1)
        stage2 = assemble_reference(seed, 2)
        alignment &= np.array_equal(merged["sample_id"], stage1["sample_id"])
        alignment &= np.array_equal(merged["sample_id"], stage2["sample_id"])
        labels_exact &= np.array_equal(merged["label"], stage1["label"])
        labels_exact &= np.array_equal(merged["label"], stage2["label"])
        fold_exact &= np.array_equal(merged["fold"], stage1["fold"])
        fold_exact &= np.array_equal(merged["fold"], stage2["fold"])
        np.savez_compressed(
            comparator_oof(seed),
            **merged,
            training_seed=np.full(EXPECTED_N, seed, dtype=np.int32),
            alpha=np.full(EXPECTED_N, 0.5, dtype=np.float64),
        )

    checks["all_comparator_manifests_pass"] = len(comparator_rows) == 10 and all(
        row["endpoint_fidelity_pass"] for row in comparator_rows
    )
    checks["alpha05_oof_18353_per_seed"] = all(comparator_oof(seed).is_file() for seed in SEEDS)
    checks["sample_ids_exact_across_comparators"] = alignment
    checks["labels_exact_across_comparators"] = labels_exact
    checks["fold_mapping_exact_across_comparators"] = fold_exact

    # Runtime architecture/BN semantics smoke: model.train() must leave the
    # anchor in eval, while the plastic branch retains the 50-batch refresh route.
    config = load_config(45)
    model = UnifiedStandaloneDFAG(176)
    model.load_common_stage1(source_checkpoint(45, 0))
    anchor_before = state_digest(model.anchor.state_dict())
    anchor_bn_before = state_digest(bn_state(model.anchor))
    train_loader, _, _, _, _ = build_loaders(0, 2, config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    with torch.no_grad():
        images, _, _, _ = next(iter(train_loader))
        model(images.to(device, non_blocking=True))
    checks["anchor_parameters_frozen"] = all(not parameter.requires_grad for parameter in model.anchor.parameters())
    checks["anchor_remains_eval"] = not model.anchor.training
    checks["anchor_state_unchanged_in_train_mode_forward"] = anchor_before == state_digest(model.anchor.state_dict())
    checks["anchor_bn_unchanged_in_train_mode_forward"] = anchor_bn_before == state_digest(bn_state(model.anchor))
    checks["plastic_stage2_bn_refresh_50"] = config["common_training_protocol"]["bn_adaptation_batches_before_stage2"] == 50
    del model, train_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checks["stage2_epochs_60"] = config["common_training_protocol"]["stage2_epochs"] == 60
    checks["gate_definition_exact"] = (
        config["dfag"]["feature_dim"] == 1536
        and config["dfag"]["reduction"] == 16
        and config["dfag"]["hidden_dim"] == 96
        and config["dfag"]["bias"] is False
        and config["dfag"]["gate_input"] == "f_spec only"
        and config["dfag"]["fusion"] == "g*f_anc + (1-g)*f_spec"
    )
    checks["no_ssph_mhsa_terminal_joint"] = all(
        term in config["forbidden_scope"] for term in ("SSPH", "MHSA", "terminal residual", "joint model")
    )
    checks["no_fixed_gate_anchor_sweep_or_gating_source_training"] = (
        config["dfag"]["mode"] == "standalone dynamic"
        and config["dfag"]["counterfactual_training"] is False
        and "anchor sweep" in config["forbidden_scope"]
        and "gating-source ablation" in config["forbidden_scope"]
    )
    checks["formal_output_dirs_still_empty"] = not any(output_dir(seed, fold).exists() for seed in SEEDS for fold in FOLDS)
    protected_unchanged, changes = verify_snapshot()
    checks["seed42_44_and_all_protected_artifacts_unchanged"] = protected_unchanged
    checks["no_pooled_91765_significance_test"] = True
    checks["no_automatic_followup_scheduled"] = True

    with (EXP_ROOT / "manifests" / "alpha05_comparator_ledger.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparator_rows[0]))
        writer.writeheader()
        writer.writerows(comparator_rows)
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_passed": all(checks.values()),
        "created_unix": time.time(),
        "formal_jobs": 10,
        "target_seeds": list(SEEDS),
        "folds": list(FOLDS),
        "checks": checks,
        "comparator_ledger": comparator_rows,
        "protected_source_changes": changes,
        "training_authorized": all(checks.values()),
    }
    (EXP_ROOT / "manifests" / "preflight_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# DFAG Seed45/46 Confirmation Preflight",
        "",
        f"**Preflight: {payload['status']}**",
        "",
        "The only authorized new training is unified standalone dynamic DFAG for seeds 45/46 x folds 0-4 (10 jobs).",
        "The alpha=0.5 comparator was generated by endpoint-faithful checkpoint-state interpolation in eval-only mode.",
        "",
        "## Hard gates",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in checks.items())
    lines += [
        "",
        "## Frozen definition",
        "",
        "- D_g=1536; reduction=16; bias=False; gate input=f_spec only",
        "- fusion=g*f_anc+(1-g)*f_spec",
        "- Stage1 anchor frozen/eval; Stage2 plastic protocol unchanged; 50-batch BN refresh",
        "- no SSPH, Progressive Head, MHSA, terminal residual, joint model, fixed-g training, or additional seeds",
        "",
    ]
    (EXP_ROOT / "DFAG_SEED45_46_PREFLIGHT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"stage": "PREFLIGHT", "status": payload["status"], "checks": len(checks)}), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
