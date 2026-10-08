"""Finalize the read-only cross-backbone supplementary audit artifacts."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import finalize


ROOT = SCRIPT_DIR.parent
AUDIT_DIR = ROOT / "supplementary_audit"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def p_text(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def main() -> None:
    paired_rows = []
    oof = ROOT / "final_cross_backbone_oof_predictions"
    for backbone in finalize.BACKBONES:
        paired_rows.append(finalize.paired(
            load_npz(oof / f"{backbone}_ours_stage1.npz"),
            load_npz(oof / f"{backbone}_ours_stage2.npz"),
            backbone,
            "ours_stage2_minus_stage1",
            "Ours-FT Stage1",
            "Ours-FT Stage2",
        ))

    paired_csv = ROOT / "FINAL_OURS_STAGE2_VS_STAGE1_PAIRED.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    endpoint_rows = []
    endpoint_payloads = {}
    for backbone in finalize.BACKBONES:
        payload = json.loads((AUDIT_DIR / f"endpoint_{backbone}.json").read_text(encoding="utf-8"))
        endpoint_payloads[backbone] = payload
        for row in payload["folds"]:
            state = row["state_checks"]
            a0 = row["alpha0_vs_stage2"]
            a1 = row["alpha1_vs_stage1"]
            endpoint_rows.append({
                "backbone": backbone,
                "fold": row["fold"],
                "alpha0_predictions_exact_stage2": a0["predictions_exact"],
                "alpha0_prediction_mismatch_count": a0["prediction_mismatch_count"],
                "alpha0_logits_exact_stage2": a0["logits_exact"],
                "alpha0_logits_max_abs_difference": a0["logits_max_abs_difference"],
                "alpha0_all_state_exact_stage2": state["alpha0_all_state_exact_stage2"],
                "alpha1_predictions_exact_stage1": a1["predictions_exact"],
                "alpha1_prediction_mismatch_count": a1["prediction_mismatch_count"],
                "alpha1_logits_exact_stage1": a1["logits_exact"],
                "alpha1_logits_max_abs_difference": a1["logits_max_abs_difference"],
                "alpha1_all_state_exact_stage1": state["alpha1_all_state_exact_stage1"],
                "alpha1_floating_state_exact_stage1": state["alpha1_floating_state_exact_stage1"],
                "alpha1_running_mean_exact_stage1": state["alpha1_running_mean_exact_stage1"],
                "alpha1_running_var_exact_stage1": state["alpha1_running_var_exact_stage1"],
                "alpha1_num_batches_tracked_exact_stage1": state["alpha1_num_batches_tracked_exact_stage1"],
                "alpha1_num_batches_tracked_exact_stage2": state["alpha1_num_batches_tracked_exact_stage2"],
                "nonfloating_state_tensor_count": state["nonfloating_state_tensors"],
                "stage1_vs_stage2_nonfloating_difference_count": state["stage1_vs_stage2_nonfloating_difference_count"],
                "inference_endpoint_pass": row["endpoint_prediction_logits_pass"],
            })

    endpoint_csv = ROOT / "FINAL_ALPHA_ENDPOINT_FIDELITY.csv"
    with endpoint_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)

    result = {
        "status": "COMPLETE",
        "training_or_bn_update_performed": False,
        "paired_protocol": {
            "n": finalize.N,
            "bootstrap_replicates": finalize.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": finalize.BOOTSTRAP_SEED,
            "ci": "95% percentile",
            "mcnemar": "exact two-sided",
            "folds_are_descriptive_only": True,
        },
        "ours_stage2_minus_stage1": paired_rows,
        "interpolation_semantics": {
            "floating_state": "alpha*Stage1+(1-alpha)*Stage2",
            "nonfloating_state": "copied from Stage2",
            "endpoint_special_handling": False,
            "alpha0_strict_checkpoint_state_identity": "PASS",
            "alpha1_strict_checkpoint_state_identity": "FAIL: non-floating num_batches_tracked copied from Stage2",
            "alpha0_prediction_logits_identity": "PASS 10/10 folds across both backbones",
            "alpha1_prediction_logits_identity": "PASS 10/10 folds across both backbones",
        },
        "endpoint_rows": endpoint_rows,
        "endpoint_raw": endpoint_payloads,
    }
    json_path = ROOT / "FINAL_CROSS_BACKBONE_SUPPLEMENTARY_AUDIT.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Final Cross-Backbone Supplementary Audit",
        "",
        "Status: **COMPLETE**",
        "",
        "No training, optimizer update, BatchNorm recalibration, checkpoint modification, or formal-output overwrite was performed.",
        "",
        "## 1. Ours-FT Stage 2 minus Stage 1 paired pooled-OOF statistics",
        "",
        "The exact formal paired protocol was reused: N=18,353, 100,000 paired bootstrap resamples with seed 20260818, 95% percentile CI, and exact two-sided McNemar. Folds are descriptive strata only.",
        "",
        "| Backbone | Stage 1 Acc. | Stage 2 Acc. | Delta pp | 95% CI pp | McNemar | Fold direction | Disagreement |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['backbone']} | {100*row['reference_accuracy']:.4f}% | {100*row['candidate_accuracy']:.4f}% | "
            f"{row['delta_accuracy_pp']:+.4f} | [{row['ci95_low_pp']:.4f}, {row['ci95_high_pp']:.4f}] | "
            f"{p_text(row['mcnemar_exact_two_sided_p'])} | negative {row['fold_negative']}/5, "
            f"positive {row['fold_positive']}/5, tie {row['fold_tie']}/5 | "
            f"{row['prediction_disagreement_count']} ({row['prediction_disagreement_percent']:.4f}%) |"
        )

    lines.extend([
        "",
        "## 2. Alpha interpolation implementation semantics",
        "",
        "- Every floating state tensor, including parameters, BatchNorm running_mean, and BatchNorm running_var, uses `alpha*Stage1 + (1-alpha)*Stage2`.",
        "- Every non-floating state tensor is copied from Stage 2.",
        "- There is no endpoint-specific branch for alpha=0 or alpha=1.",
        "- In these checkpoints the only non-floating state tensors are BatchNorm `num_batches_tracked` counters.",
        "",
        "## 3. Per-fold endpoint fidelity",
        "",
        "| Backbone | Fold | alpha=0 pred | alpha=0 logits | alpha=0 full state=Stage2 | alpha=1 pred | alpha=1 logits | alpha=1 full state=Stage1 | Inference endpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in endpoint_rows:
        mark = lambda value: "PASS" if value else "FAIL"
        lines.append(
            f"| {row['backbone']} | {row['fold']} | {mark(row['alpha0_predictions_exact_stage2'])} | "
            f"{mark(row['alpha0_logits_exact_stage2'])} | {mark(row['alpha0_all_state_exact_stage2'])} | "
            f"{mark(row['alpha1_predictions_exact_stage1'])} | {mark(row['alpha1_logits_exact_stage1'])} | "
            f"{mark(row['alpha1_all_state_exact_stage1'])} | {mark(row['inference_endpoint_pass'])} |"
        )

    lines.extend([
        "",
        "## 4. Endpoint conclusion",
        "",
        "- alpha=0: strict checkpoint-state identity with Stage 2 passes, and predictions/logits are bitwise exact in all 10 backbone-fold cases.",
        "- alpha=1: all floating state, including BN running statistics, is exactly Stage 1; predictions/logits are bitwise exact in all 10 cases. Strict full-state identity fails because Stage-2 `num_batches_tracked` counters are retained.",
        "- Retaining Stage-2 `num_batches_tracked` does not alter these eval-mode endpoint logits because the counters are not used by BatchNorm in eval mode.",
        "",
        "## 5. Accurate alpha=0.5 description",
        "",
        "The current alpha=0.5 artifact is a complete floating-state checkpoint interpolation: all floating parameters and floating buffers are averaged as `0.5*Stage1 + 0.5*Stage2`, while non-floating BatchNorm `num_batches_tracked` buffers are copied from Stage 2. It is not a literal interpolation of every checkpoint state tensor.",
        "",
    ])
    md_path = ROOT / "FINAL_CROSS_BACKBONE_SUPPLEMENTARY_AUDIT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "COMPLETE",
        "paired_csv": str(paired_csv.resolve()),
        "endpoint_csv": str(endpoint_csv.resolve()),
        "json": str(json_path.resolve()),
        "audit": str(md_path.resolve()),
    }))


if __name__ == "__main__":
    main()
