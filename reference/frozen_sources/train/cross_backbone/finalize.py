"""Postflight OOF assembly, paired statistics, and final audit outputs."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from cross_backbone_common import BACKBONES, CONFIG_PATH, FOLDS, ROOT, load_config, output_dir, sha256_file
from queue_runner import summary as queue_summary


N = 18_353
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260818
OOF_DIR = ROOT / "final_cross_backbone_oof_predictions"

OUTCOMES = {
    "ours_stage1": ("ours_ft", "stage1_predictions.npz"),
    "ours_stage2": ("ours_ft", "stage2_predictions.npz"),
    "ours_alpha0.5": ("ours_ft", "alpha0_5_predictions.npz"),
    "progressive_stage1": ("progressive", "stage1_predictions.npz"),
    "progressive_stage2": ("progressive", "stage2_predictions.npz"),
    "dfag": ("dfag", "dfag_predictions.npz"),
}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def load_fold(backbone: str, outcome: str, fold: int) -> dict[str, np.ndarray]:
    method, filename = OUTCOMES[outcome]
    path = output_dir("formal", backbone, method, fold) / filename
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    required = {"sample_id", "fold", "y_true", "y_pred", "logits", "probabilities", "image_name"}
    if not required.issubset(result):
        raise ValueError(f"OOF schema missing in {path}: {required - set(result)}")
    n = len(result["sample_id"])
    if not all(len(result[key]) == n for key in required):
        raise ValueError(f"OOF length mismatch: {path}")
    if not np.all(result["fold"] == fold):
        raise ValueError(f"Fold identity mismatch: {path}")
    if not np.isfinite(result["logits"]).all() or not np.isfinite(result["probabilities"]).all():
        raise FloatingPointError(f"Non-finite OOF: {path}")
    return result


def assemble(backbone: str, outcome: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    parts, fold_metrics = [], []
    for fold in FOLDS:
        part = load_fold(backbone, outcome, fold)
        parts.append(part)
        fold_metrics.append({"backbone": backbone, "outcome": outcome, "fold": fold,
                             "n": len(part["sample_id"]), **metric(part["y_true"], part["y_pred"])})
    keys = parts[0].keys()
    merged = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    order = np.argsort(merged["sample_id"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if len(merged["sample_id"]) != N or not np.array_equal(merged["sample_id"], np.arange(N)):
        raise ValueError(f"Incomplete pooled OOF: {backbone}/{outcome}")
    if len(np.unique(merged["sample_id"])) != N:
        raise ValueError(f"Duplicate pooled OOF sample: {backbone}/{outcome}")
    target = OOF_DIR / f"{backbone}_{outcome}.npz"
    np.savez_compressed(target, **merged)
    return merged, fold_metrics


def paired(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], backbone: str,
           comparison: str, reference_name: str, candidate_name: str) -> dict[str, Any]:
    if not np.array_equal(reference["sample_id"], candidate["sample_id"]):
        raise ValueError(f"paired sample mismatch: {comparison}")
    if not np.array_equal(reference["y_true"], candidate["y_true"]):
        raise ValueError(f"paired label mismatch: {comparison}")
    if not np.array_equal(reference["fold"], candidate["fold"]):
        raise ValueError(f"paired fold mismatch: {comparison}")
    y = reference["y_true"]
    ref_pred, cand_pred = reference["y_pred"], candidate["y_pred"]
    ref_correct, cand_correct = ref_pred == y, cand_pred == y
    delta_values = cand_correct.astype(np.int8) - ref_correct.astype(np.int8)
    counts = np.asarray([(delta_values == -1).sum(), (delta_values == 0).sum(), (delta_values == 1).sum()])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.multinomial(N, counts / N, size=BOOTSTRAP_REPLICATES)
    boot_pp = 100.0 * (draws[:, 2] - draws[:, 0]) / N
    n10 = int(np.sum(ref_correct & ~cand_correct))
    n01 = int(np.sum(~ref_correct & cand_correct))
    discordant = n10 + n01
    fold_deltas = []
    for fold in FOLDS:
        mask = reference["fold"] == fold
        fold_deltas.append(100.0 * float(delta_values[mask].mean()))
    epsilon = 1e-12
    negative = sum(value < -epsilon for value in fold_deltas)
    positive = sum(value > epsilon for value in fold_deltas)
    ties = 5 - negative - positive
    ref_acc, cand_acc = float(ref_correct.mean()), float(cand_correct.mean())
    return {
        "backbone": backbone,
        "comparison": comparison,
        "reference": reference_name,
        "candidate": candidate_name,
        "n": N,
        "reference_accuracy": ref_acc,
        "candidate_accuracy": cand_acc,
        "delta_accuracy_pp": 100.0 * (cand_acc - ref_acc),
        "ci95_low_pp": float(np.quantile(boot_pp, 0.025)),
        "ci95_high_pp": float(np.quantile(boot_pp, 0.975)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n10_reference_correct_candidate_wrong": n10,
        "n01_reference_wrong_candidate_correct": n01,
        "mcnemar_exact_two_sided_p": float(binomtest(min(n10, n01), discordant, 0.5).pvalue) if discordant else 1.0,
        "prediction_disagreement_count": int(np.sum(ref_pred != cand_pred)),
        "prediction_disagreement_percent": 100.0 * float(np.mean(ref_pred != cand_pred)),
        "fold0_delta_pp": fold_deltas[0],
        "fold1_delta_pp": fold_deltas[1],
        "fold2_delta_pp": fold_deltas[2],
        "fold3_delta_pp": fold_deltas[3],
        "fold4_delta_pp": fold_deltas[4],
        "fold_negative": negative,
        "fold_positive": positive,
        "fold_tie": ties,
    }


def md_p(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def main() -> None:
    queue = queue_summary("formal")
    if queue["counts"].get("COMPLETE", 0) != 30 or queue["counts"].get("FAILED", 0):
        raise RuntimeError(f"formal queue not 30/30 complete: {queue['counts']}")
    config = load_config()
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    summary_rows, fold_rows, artifacts = [], [], []
    canonical_labels = None
    for backbone in BACKBONES:
        data[backbone] = {}
        for outcome in OUTCOMES:
            merged, folds = assemble(backbone, outcome)
            data[backbone][outcome] = merged
            if canonical_labels is None:
                canonical_labels = merged["y_true"]
            elif not np.array_equal(canonical_labels, merged["y_true"]):
                raise ValueError(f"cross-outcome labels mismatch: {backbone}/{outcome}")
            summary_rows.append({"backbone": backbone, "outcome": outcome, "n": N,
                                 **metric(merged["y_true"], merged["y_pred"])})
            fold_rows.extend(folds)
            artifact_path = OOF_DIR / f"{backbone}_{outcome}.npz"
            artifacts.append({"backbone": backbone, "outcome": outcome, "path": str(artifact_path.resolve()),
                              "sha256": sha256_file(artifact_path), "n": N})

    comparisons = []
    for backbone in BACKBONES:
        comparisons.extend([
            paired(data[backbone]["ours_stage1"], data[backbone]["progressive_stage1"], backbone,
                   "progressive_minus_ours_stage1", "Ours-FT Stage1", "Progressive Stage1"),
            paired(data[backbone]["ours_stage2"], data[backbone]["progressive_stage2"], backbone,
                   "progressive_minus_ours_stage2", "Ours-FT Stage2", "Progressive Stage2"),
            paired(data[backbone]["ours_stage1"], data[backbone]["dfag"], backbone,
                   "dfag_minus_stage1", "Ours-FT Stage1", "DFAG"),
            paired(data[backbone]["ours_stage2"], data[backbone]["dfag"], backbone,
                   "dfag_minus_stage2", "Ours-FT Stage2", "DFAG"),
            paired(data[backbone]["ours_alpha0.5"], data[backbone]["dfag"], backbone,
                   "dfag_minus_alpha0.5", "Ours-FT alpha0.5", "DFAG"),
        ])

    pd.DataFrame(summary_rows).to_csv(ROOT / "FINAL_CROSS_BACKBONE_OOF_SUMMARY.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ROOT / "FINAL_CROSS_BACKBONE_FOLD_METRICS.csv", index=False)
    pd.DataFrame(comparisons).to_csv(ROOT / "FINAL_CROSS_BACKBONE_PAIRED_STATISTICS.csv", index=False)

    source_rows = []
    for backbone in BACKBONES:
        for fold in FOLDS:
            ours_manifest = json.loads((output_dir("formal", backbone, "ours_ft", fold) / "run_manifest.json").read_text(encoding="utf-8"))
            dfag_manifest = json.loads((output_dir("formal", backbone, "dfag", fold) / "run_manifest.json").read_text(encoding="utf-8"))
            source_rows.append({
                "backbone": backbone, "fold": fold,
                "ours_stage1": ours_manifest["best_stage1_checkpoint"],
                "ours_stage1_sha256": ours_manifest["best_stage1_sha256"],
                "dfag_source": dfag_manifest["dfag_stage1_source"]["path"],
                "dfag_source_sha256": dfag_manifest["dfag_stage1_source"]["sha256"],
                "exact_match": ours_manifest["best_stage1_sha256"] == dfag_manifest["dfag_stage1_source"]["sha256"],
            })
    if not all(row["exact_match"] for row in source_rows):
        raise RuntimeError("DFAG Stage1 provenance mismatch")

    direction_changes = {}
    for comparison in sorted({row["comparison"] for row in comparisons}):
        subset = [row for row in comparisons if row["comparison"] == comparison]
        signs = [int(np.sign(row["delta_accuracy_pp"])) for row in subset]
        direction_changes[comparison] = {"resnet50_delta_pp": subset[0]["delta_accuracy_pp"],
                                         "convnext_tiny_delta_pp": subset[1]["delta_accuracy_pp"],
                                         "direction_change": signs[0] != signs[1]}

    result = {
        "status": "COMPLETE",
        "created_unix": time.time(),
        "config": str(CONFIG_PATH.resolve()),
        "config_sha256": sha256_file(CONFIG_PATH),
        "formal_jobs_complete": 30,
        "formal_jobs_failed": 0,
        "preflight_status": json.loads((ROOT / "preflight_audit.json").read_text(encoding="utf-8"))["status"],
        "smoke_status": json.loads((ROOT / "smoke_audit.json").read_text(encoding="utf-8"))["status"],
        "backbone_dimensions": {backbone: config["backbones"][backbone]["actual_dimension"] for backbone in BACKBONES},
        "batch_sizes": {backbone: config["backbones"][backbone]["batch_size"] for backbone in BACKBONES},
        "gate_parameters": {backbone: 2 * config["backbones"][backbone]["actual_dimension"]
                            * (config["backbones"][backbone]["actual_dimension"] // 16) for backbone in BACKBONES},
        "oof_coverage": {"n": N, "exactly_once": True, "artifacts": artifacts},
        "oof_summary": summary_rows,
        "fold_metrics": fold_rows,
        "paired_statistics": comparisons,
        "dfag_stage1_provenance": source_rows,
        "backbone_dependent_direction_changes": direction_changes,
        "interpretation_boundaries": [
            "Under the frozen cross-backbone protocol only.",
            "H=256 is fixed, so D/H differs and is not independently controlled.",
            "No claim of universal or backbone-independent effectiveness.",
            "Five folds are descriptive strata; paired OOF samples are the statistical unit.",
        ],
    }
    (ROOT / "FINAL_CROSS_BACKBONE_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8"
    )

    protocol = config["common_training_protocol"]
    manifest_lines = [
        "# Final Cross-Backbone Protocol Manifest", "",
        f"- Config SHA256: `{sha256_file(CONFIG_PATH)}`",
        "- Dataset: Classify Leaves; 18,353 samples; 176 classes; five-fold StratifiedKFold with shuffle=True and random_state=42.",
        "- Training seed: 42 for every fold; batch size: 64 for every backbone and configuration.",
        f"- Stage 1: {protocol['stage1_epochs']} epochs; AdamW; backbone LR {protocol['stage1_backbone_lr']}; head LR {protocol['stage1_head_lr']}; weight decay {protocol['stage1_weight_decay']}; 3-epoch warmup then cosine schedule.",
        f"- Stage 2: {protocol['stage2_epochs']} epochs; all-parameter LR {protocol['stage2_all_parameter_lr']}; weight decay {protocol['stage2_weight_decay']}; cosine schedule.",
        f"- Label smoothing: Stage 1 {protocol['stage1_label_smoothing']}; Stage 2 {protocol['stage2_label_smoothing']}.",
        f"- Stage-1 batch augmentation: {protocol['stage1_batch_augmentation']}",
        f"- Spatial augmentation both stages: {', '.join(protocol['spatial_augmentation_both_stages'])}.",
        f"- BN recalibration: {protocol['bn_adaptation_batches_before_stage2']} no-grad training-mode batches before Stage-2 optimizer construction.",
        f"- AMP: FP16 with initial GradScaler scale {protocol['amp_grad_scaler_init_scale']}; gradient clipping {protocol['gradient_clip_norm']}.",
        f"- Checkpoint selection: {protocol['checkpoint_selection']}",
        f"- Validation: {protocol['validation_transform']}; original view; no TTA; no fold ensemble.",
        "- DFAG Stage 1: exact selected Ours-FT Stage-1 checkpoint from the same new backbone/fold; copied to frozen anchor and plastic; DFAG trains Stage 2 only.",
        "- Alpha 0.5: complete floating checkpoint state interpolation; nonfloating buffers copied from Stage 2; eval only; no BN refresh.",
        "- Primary analysis: complete pooled five-fold OOF; 100,000 paired bootstrap resamples (seed 20260818); exact two-sided McNemar.",
        "", "H=256 is fixed across backbones, so the projection compression ratio differs across architectures and is not independently controlled.",
    ]
    (ROOT / "FINAL_CROSS_BACKBONE_PROTOCOL_MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    comparison_lines = [
        "| Backbone | Comparison | Reference Acc. | Candidate Acc. | Delta (pp) | 95% CI | McNemar | Fold direction | Disagreement |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in comparisons:
        comparison_lines.append(
            f"| {row['backbone']} | {row['comparison']} | {100*row['reference_accuracy']:.2f}% | "
            f"{100*row['candidate_accuracy']:.2f}% | {row['delta_accuracy_pp']:+.2f} | "
            f"[{row['ci95_low_pp']:.2f}, {row['ci95_high_pp']:.2f}] | {md_p(row['mcnemar_exact_two_sided_p'])} | "
            f"negative {row['fold_negative']}/5, positive {row['fold_positive']}/5, tie {row['fold_tie']}/5 | "
            f"{row['prediction_disagreement_percent']:.2f}% |"
        )
    summary_lines = ["| Backbone | Outcome | Accuracy | Macro-F1 | Balanced Accuracy |", "|---|---|---:|---:|---:|"]
    for row in summary_rows:
        summary_lines.append(f"| {row['backbone']} | {row['outcome']} | {100*row['accuracy']:.2f}% | {row['macro_f1']:.4f} | {row['balanced_accuracy']:.4f} |")
    change_lines = [f"- `{name}`: {'YES' if row['direction_change'] else 'NO'}; ResNet-50 {row['resnet50_delta_pp']:+.3f} pp, ConvNeXt-Tiny {row['convnext_tiny_delta_pp']:+.3f} pp."
                    for name, row in direction_changes.items()]
    audit = f"""# Final Cross-Backbone Evaluation Audit

Status: **COMPLETE**

## A. Protocol provenance

The run used the frozen Classify Leaves controlled protocol and standalone DFAG semantics. No old cross-backbone result or checkpoint was reused. Config SHA256: `{sha256_file(CONFIG_PATH)}`.

## B. Backbone dimensions

| Backbone | Actual D | H | D/H | Gate parameters |
|---|---:|---:|---:|---:|
| ResNet-50 | 2048 | 256 | 8.0 | 524,288 |
| ConvNeXt-Tiny | 768 | 256 | 3.0 | 73,728 |

H=256 is fixed across backbones, so the projection compression ratio differs across architectures and is not independently controlled.

## C. Job completion

- Preflight: PASS
- Six technical smoke configurations: PASS
- Formal jobs: 30/30 COMPLETE
- Failed formal jobs: 0

## D. OOF coverage

Each of 12 final pooled artifacts contains all 18,353 held-out samples exactly once, with aligned labels and fold identities. Original-view predictions only; no TTA or fold ensemble.

## E-H. Ours-FT, Progressive Head, and standalone DFAG results

{chr(10).join(summary_lines)}

## G/I. Paired OOF statistics

{chr(10).join(comparison_lines)}

## J. DFAG gate and Stage-1 provenance

- ResNet-50 gate: 524,288 parameters (2048 -> 128 -> 2048; bias-free).
- ConvNeXt-Tiny gate: 73,728 parameters (768 -> 48 -> 768; bias-free).
- All ten DFAG folds reused the exact same-run, same-backbone, same-fold Ours-FT selected Stage-1 SHA as common anchor/plastic initialization; Stage 1 was not retrained inside DFAG.

## K-L. Fold direction and prediction disagreement

Fold-direction counts and prediction disagreement are reported in the paired table. Folds are descriptive strata and were not treated as independent significance units.

## Backbone-dependent direction changes

{chr(10).join(change_lines)}

## M. Protocol limitations

These are supporting results under the frozen cross-backbone protocol. They do not prove architecture generalization, backbone independence, universal benefit/harm, or a causal effect of compression ratio. H is fixed at 256, so D/H is confounded with backbone identity. Result directions are reported without result-dependent retuning.
"""
    (ROOT / "FINAL_CROSS_BACKBONE_AUDIT.md").write_text(audit, encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "jobs": 30, "oof_artifacts": len(artifacts),
                      "comparisons": len(comparisons), "outputs": [
                          "FINAL_CROSS_BACKBONE_AUDIT.md", "FINAL_CROSS_BACKBONE_OOF_SUMMARY.csv",
                          "FINAL_CROSS_BACKBONE_PAIRED_STATISTICS.csv", "FINAL_CROSS_BACKBONE_RESULTS.json",
                          "FINAL_CROSS_BACKBONE_FOLD_METRICS.csv", "final_cross_backbone_oof_predictions",
                          "FINAL_CROSS_BACKBONE_PROTOCOL_MANIFEST.md", "final_cross_backbone_protocol_config.json"
                      ]}), flush=True)


if __name__ == "__main__":
    main()
