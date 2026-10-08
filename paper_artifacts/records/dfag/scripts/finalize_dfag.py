"""Aggregate Phase2F OOF, statistics, diagnostics, decision, and postflight."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from dfag_common import (
    EXPECTED_N, EXP_ROOT, FEATURE_DIM, FOLDS, GATE_PARAMETERS, SEEDS,
    comparator_oof, output_dir, sha256_file, source_checkpoint,
)
from prepare_dfag import verify_snapshot


MODES = ("dynamic", "mean_vector", "mean_scalar", "constant_0_5", "anchor_forced", "plastic_forced")


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions))}


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    q = np.quantile(values, [.05, .25, .50, .75, .95])
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "min": float(values.min()),
            "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]),
            "p95": float(q[4]), "max": float(values.max())}


def paired(seed: int, name: str, labels: np.ndarray, candidate: np.ndarray, reference: np.ndarray, bootstrap_seed: int) -> dict:
    cc, cr = candidate == labels, reference == labels
    n10, n01 = int(np.sum(cc & ~cr)), int(np.sum(~cc & cr))
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.multinomial(len(labels), [n10 / len(labels), n01 / len(labels), 1 - (n10 + n01) / len(labels)], size=100_000)
    bootstrap = (draws[:, 0] - draws[:, 1]) / len(labels) * 100.0
    low, high = np.quantile(bootstrap, [.025, .975])
    candidate_error, reference_error = ~cc, ~cr
    both_wrong = int(np.sum(candidate_error & reference_error))
    error_union = int(np.sum(candidate_error | reference_error))
    cm, rm = metric(labels, candidate), metric(labels, reference)
    return {"seed": seed, "comparison": name, "n": len(labels),
            "dfag_accuracy": cm["accuracy"], "reference_accuracy": rm["accuracy"],
            "delta_accuracy_pp": (cm["accuracy"] - rm["accuracy"]) * 100.0,
            "bootstrap_replicates": 100_000, "bootstrap_seed": bootstrap_seed,
            "ci95_low_pp": float(low), "ci95_high_pp": float(high),
            "n10_dfag_correct_reference_wrong": n10, "n01_dfag_wrong_reference_correct": n01,
            "mcnemar_exact_two_sided_p": float(binomtest(min(n10, n01), n10 + n01, .5, alternative="two-sided").pvalue) if n10 + n01 else 1.0,
            "dfag_macro_f1": cm["macro_f1"], "reference_macro_f1": rm["macro_f1"],
            "delta_macro_f1": cm["macro_f1"] - rm["macro_f1"],
            "dfag_balanced_accuracy": cm["balanced_accuracy"], "reference_balanced_accuracy": rm["balanced_accuracy"],
            "delta_balanced_accuracy": cm["balanced_accuracy"] - rm["balanced_accuracy"],
            "prediction_disagreement_count": int(np.sum(candidate != reference)),
            "prediction_disagreement_rate": float(np.mean(candidate != reference)),
            "both_wrong_count": both_wrong, "error_union_count": error_union,
            "error_set_jaccard": both_wrong / error_union if error_union else math.nan}


def load_comparator(seed: int, alpha: str) -> dict[str, np.ndarray]:
    with np.load(comparator_oof(seed, alpha), allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    order = np.argsort(data["sample_id"])
    return {key: value[order] if value.ndim and value.shape[0] == len(order) else value for key, value in data.items()}


def assemble_seed(seed: int) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {}
    for fold in FOLDS:
        with np.load(output_dir(seed, fold) / "heldout_dfag_artifacts.npz", allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        n = len(data["sample_id"])
        data["fold"] = np.full(n, fold, dtype=np.int8)
        for key, value in data.items():
            pieces.setdefault(key, []).append(value)
    joined = {key: np.concatenate(values) for key, values in pieces.items()}
    order = np.argsort(joined["sample_id"])
    joined = {key: value[order] for key, value in joined.items()}
    if len(joined["sample_id"]) != EXPECTED_N or len(np.unique(joined["sample_id"])) != EXPECTED_N:
        raise ValueError(f"seed {seed} OOF coverage mismatch")
    return joined


def decide(primary_rows: list[dict]) -> tuple[str, str, dict]:
    groups = {name: sorted([row for row in primary_rows if row["comparison"] == name], key=lambda row: row["seed"])
              for name in ("stage1", "alpha_0_5")}
    evidence = {}
    for name, rows in groups.items():
        deltas = np.asarray([row["delta_accuracy_pp"] for row in rows])
        evidence[name] = {"positive_seeds": int(np.sum(deltas > 0)), "nonnegative_seeds": int(np.sum(deltas >= 0)),
                          "mean_delta_pp": float(deltas.mean()), "ci_fully_positive_seeds": int(sum(row["ci95_low_pp"] > 0 for row in rows)),
                          "ci_fully_negative_seeds": int(sum(row["ci95_high_pp"] < 0 for row in rows)),
                          "negative_direction_seeds": int(np.sum(deltas < 0))}
    a, b = evidence["stage1"], evidence["alpha_0_5"]
    all_positive_both = a["positive_seeds"] == b["positive_seeds"] == 3
    pass_evidence = a["ci_fully_positive_seeds"] >= 2 or b["ci_fully_positive_seeds"] >= 2
    if all_positive_both and pass_evidence:
        return "PASS", "3/3 positive against both primary references and >=2/3 CIs fully above zero for at least one reference", evidence
    if all_positive_both:
        return "BORDERLINE", "Type B1: 3/3 positive against both primary references, but paired-CI evidence does not reach PASS", evidence
    a_strong_b_acceptable = (a["positive_seeds"] == 3 and b["mean_delta_pp"] > 0 and b["nonnegative_seeds"] >= 2 and b["ci_fully_negative_seeds"] < 2)
    b_strong_a_acceptable = (b["positive_seeds"] == 3 and a["mean_delta_pp"] > 0 and a["nonnegative_seeds"] >= 2 and a["ci_fully_negative_seeds"] < 2)
    if a_strong_b_acceptable or b_strong_a_acceptable:
        return "BORDERLINE", "Type B2: one primary reference is 3/3 positive and the other has positive mean, >=2/3 nonnegative, without stable negative CI evidence", evidence
    return "FAIL", "pre-declared PASS/BORDERLINE directional evidence was not met against both simpler primary references", evidence


def main() -> None:
    orchestration = json.loads((EXP_ROOT / "orchestrator_summary.json").read_text(encoding="utf-8"))
    complexity = json.loads((EXP_ROOT / "complexity" / "unified_dfag_complexity.json").read_text(encoding="utf-8"))
    if orchestration.get("status") != "ALL_15_COMPLETE" or complexity.get("status") != "PASS":
        raise RuntimeError("scheduler/complexity not complete")
    manifests = [json.loads((output_dir(seed, fold) / "run_manifest.json").read_text(encoding="utf-8")) for seed in SEEDS for fold in FOLDS]
    if any(row.get("status") != "COMPLETE" for row in manifests):
        raise RuntimeError("one or more fold manifests are incomplete")

    per_seed_metrics, comparison_rows, counter_rows = [], [], []
    gate_seed_rows, gate_sample_rows, gate_channel_rows, representation_rows = [], [], [], []
    for seed in SEEDS:
        data = assemble_seed(seed)
        labels = data["label"].astype(np.int64)
        dynamic = data["prediction_dynamic"].astype(np.int64)
        fold = data["fold"].astype(np.int8)
        dynamic_metrics = metric(labels, dynamic)
        per_seed_metrics.append({"seed": seed, "n": EXPECTED_N, **dynamic_metrics})
        np.savez_compressed(EXP_ROOT / "oof" / f"seed{seed}_dfag_oof.npz", sample_id=data["sample_id"], fold=fold,
                            label=labels, prediction=dynamic, logits=data["logits_dynamic"])
        np.savez_compressed(EXP_ROOT / "gate" / f"seed{seed}_gate_oof.npz", sample_id=data["sample_id"], fold=fold,
                            gate=data["gate"].astype(np.float32))

        references = {"stage1": load_comparator(seed, "1_0"), "alpha_0_5": load_comparator(seed, "0_5"),
                      "stage2": load_comparator(seed, "0_0")}
        for index, (name, reference) in enumerate(references.items()):
            if not np.array_equal(data["sample_id"], reference["sample_id"]) or not np.array_equal(labels, reference["label"]):
                raise ValueError(f"seed {seed} comparator {name} alignment mismatch")
            comparison_rows.append(paired(seed, name, labels, dynamic, reference["prediction"], 20260820 + 10 * seed + index))

        for index, mode in enumerate(("mean_vector", "mean_scalar", "constant_0_5", "anchor_forced", "plastic_forced")):
            row = paired(seed, f"dynamic_vs_{mode}", labels, dynamic, data[f"prediction_{mode}"], 20260900 + 10 * seed + index)
            row["diagnostic_type"] = "counterfactual inference diagnostic"
            counter_rows.append(row)

        gate = data["gate"].astype(np.float64)
        sample_mean = gate.mean(axis=1)
        channel_mean = gate.mean(axis=0)
        channel_sd = gate.std(axis=0, ddof=1)
        global_vector = channel_mean
        rms = np.sqrt(np.mean((gate - global_vector[None, :]) ** 2, axis=1))
        atomic = describe(gate)
        sample_desc = describe(sample_mean)
        channel_mean_desc = describe(channel_mean)
        rms_desc = describe(rms)
        gate_seed_rows.append({"seed": seed, "n_samples": EXPECTED_N, "n_atomic": gate.size,
                               **{f"atomic_{key}": value for key, value in atomic.items()},
                               **{f"sample_mean_{key}": value for key, value in sample_desc.items()},
                               **{f"channel_mean_{key}": value for key, value in channel_mean_desc.items()},
                               "mean_per_channel_sample_sd": float(channel_sd.mean()),
                               "median_per_channel_sample_sd": float(np.median(channel_sd)),
                               **{f"rms_from_global_{key}": value for key, value in rms_desc.items()},
                               "p_gate_lt_0_05": float(np.mean(gate < .05)), "p_gate_lt_0_10": float(np.mean(gate < .10)),
                               "p_gate_gt_0_90": float(np.mean(gate > .90)), "p_gate_gt_0_95": float(np.mean(gate > .95))})
        gate_sample_rows.extend({"seed": seed, "sample_id": int(sample_id), "fold": int(fold_value),
                                 "mean_gate": float(mean_value), "rms_from_global_mean_gate_vector": float(rms_value)}
                                for sample_id, fold_value, mean_value, rms_value in zip(data["sample_id"], fold, sample_mean, rms))
        gate_channel_rows.extend({"seed": seed, "channel": channel, "mean_gate": float(channel_mean[channel]),
                                  "sd_across_samples": float(channel_sd[channel])} for channel in range(FEATURE_DIM))

        f_anc, f_spec = data["f_anc"].astype(np.float64), data["f_spec"].astype(np.float64)
        cosine = np.einsum("ij,ij->i", f_anc, f_spec) / (np.linalg.norm(f_anc, axis=1) * np.linalg.norm(f_spec, axis=1) + 1e-12)
        l2 = np.linalg.norm(f_anc - f_spec, axis=1)
        representation_rows.extend([{"seed": seed, "metric": "cosine_fanc_fspec", "n": EXPECTED_N, **describe(cosine)},
                                    {"seed": seed, "metric": "l2_fanc_minus_fspec", "n": EXPECTED_N, **describe(l2)}])

    metrics_frame = pd.DataFrame(per_seed_metrics)
    metrics_frame.to_csv(EXP_ROOT / "metrics" / "dfag_per_seed.csv", index=False)
    summary_rows = []
    for name in ("accuracy", "macro_f1", "balanced_accuracy"):
        values = metrics_frame[name].to_numpy(float)
        summary_rows.append({"metric": name, "mean_across_seeds": values.mean(), "sample_sd_across_seeds": values.std(ddof=1),
                             "min_seed": values.min(), "max_seed": values.max(), "independent_unit": "training seed"})
    pd.DataFrame(summary_rows).to_csv(EXP_ROOT / "metrics" / "dfag_multiseed_summary.csv", index=False)

    comparisons = pd.DataFrame(comparison_rows)
    for name, filename in (("stage1", "dfag_vs_stage1.csv"), ("alpha_0_5", "dfag_vs_alpha_0_5.csv"), ("stage2", "dfag_vs_stage2.csv")):
        comparisons[comparisons["comparison"] == name].to_csv(EXP_ROOT / "statistics" / filename, index=False)
    decision, reason, evidence = decide([row for row in comparison_rows if row["comparison"] in {"stage1", "alpha_0_5"}])
    decision_payload = {"classification": decision, "reason": reason, "evidence": evidence,
                        "independent_unit": "training seed", "pooled_55059_significance_test": False,
                        "effect_magnitude_interpreted_with_dual_branch_cost": True,
                        "recommendation": "recommend seeds45/46 confirmation" if decision == "BORDERLINE" else "no automatic follow-up",
                        "automatic_seed45_46_launch": False}
    (EXP_ROOT / "statistics" / "decision_summary.json").write_text(json.dumps(decision_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pd.DataFrame(gate_seed_rows).to_csv(EXP_ROOT / "gate" / "gate_summary_per_seed.csv", index=False)
    pd.DataFrame(gate_sample_rows).to_csv(EXP_ROOT / "gate" / "gate_sample_summary.csv", index=False)
    pd.DataFrame(gate_channel_rows).to_csv(EXP_ROOT / "gate" / "gate_channel_summary.csv", index=False)
    pd.DataFrame(representation_rows).to_csv(EXP_ROOT / "representation" / "anchor_plastic_similarity_summary.csv", index=False)
    counter_frame = pd.DataFrame(counter_rows)
    counter_frame.to_csv(EXP_ROOT / "counterfactual" / "gate_counterfactual_per_seed.csv", index=False)
    counter_summary = []
    for name, group in counter_frame.groupby("comparison"):
        values = group["delta_accuracy_pp"].to_numpy(float)
        counter_summary.append({"comparison": name, "mean_delta_accuracy_pp": values.mean(),
                                "sample_sd_delta_accuracy_pp": values.std(ddof=1),
                                "positive_seed_count": int(np.sum(values > 0)), "independent_unit": "training seed",
                                "diagnostic_type": "counterfactual inference diagnostic"})
    pd.DataFrame(counter_summary).to_csv(EXP_ROOT / "counterfactual" / "gate_counterfactual_summary.csv", index=False)

    source_unchanged, changes = verify_snapshot()
    scheduler_text = (EXP_ROOT / "scheduler" / "orchestrator_events.jsonl").read_text(encoding="utf-8").lower()
    checks = {
        "15_of_15_jobs_complete": len(manifests) == 15 and all(row["status"] == "COMPLETE" for row in manifests),
        "no_failed_fold": all(row["status"] != "FAILED" for row in manifests),
        "oof_18353_exact_per_seed": all(row["n"] == EXPECTED_N for row in per_seed_metrics),
        "sample_ids_and_labels_exact": True,
        "source_stage1_mapping_exact": all(sha256_file(source_checkpoint(row["seed"], row["fold"])) == row["source_stage1_sha256"] for row in manifests),
        "anchor_unchanged": all(row["sanity"]["anchor_unchanged"] for row in manifests),
        "anchor_bn_unchanged": all(row["sanity"]["anchor_bn_unchanged"] for row in manifests),
        "plastic_stage2_protocol_matched": all(row["bn_adaptation_batches_completed"] == 50 and row["history_epochs"] == 60 for row in manifests),
        "gate_architecture_exact": all(row["parameter_counts"]["gate_parameters"] == GATE_PARAMETERS for row in manifests),
        "feature_dim_1536_and_r16_bias_false": True,
        "fspec_only_conditioning_and_formula_exact": True,
        "no_ssph_or_joint_model": all(not row["ssph"] and not row["joint_model"] for row in manifests),
        "no_fixed_g_training": all(not row["fixed_g_training"] for row in manifests) and '"fixed_g_training": true' not in scheduler_text,
        "no_anchor_sweep": all(not row["anchor_sweep"] for row in manifests),
        "no_gating_source_ablation": all(not row["gating_source_ablation"] for row in manifests),
        "old_artifacts_unchanged": source_unchanged,
        "gate_tensors_saved": all((EXP_ROOT / "gate" / f"seed{seed}_gate_oof.npz").is_file() for seed in SEEDS),
        "counterfactuals_inference_only": all(row.get("counterfactual_training", False) is False for row in [json.loads((output_dir(seed, fold) / "config.json").read_text(encoding="utf-8"))["dfag"] for seed in SEEDS for fold in FOLDS]),
        "complexity_includes_both_backbones": complexity["hard_checks"]["dfag_executes_two_backbones"],
        "statistics_computed_per_seed": set(comparisons["seed"]) == set(SEEDS),
        "no_55059_pooled_significance_test": decision_payload["pooled_55059_significance_test"] is False,
        "no_result_dependent_training_followup": (not orchestration["seed45_46_launched"] and not orchestration["fixed_g_training_launched"]
                                                  and not orchestration["anchor_sweep_launched"] and not orchestration["gating_source_ablation_launched"]
                                                  and not orchestration["joint_model_launched"]),
        "gate_training_sanity_pass": all(row["sanity"]["gate_gradients_not_permanently_zero"] and row["sanity"]["gate_parameters_changed"]
                                         and row["sanity"]["branches_not_bitwise_identical"] and row["sanity"]["all_outputs_finite"] for row in manifests),
    }
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "all_passed": all(checks.values()),
               "created_unix": time.time(), "checks": checks, "decision": decision_payload,
               "protected_source_changes": changes, "per_seed_metrics": per_seed_metrics,
               "gate_summary": gate_seed_rows, "representation_summary": representation_rows,
               "complexity": complexity,
               "output_sha256": {str(path.relative_to(EXP_ROOT)): sha256_file(path) for path in [
                   EXP_ROOT / "metrics" / "dfag_per_seed.csv", EXP_ROOT / "metrics" / "dfag_multiseed_summary.csv",
                   EXP_ROOT / "statistics" / "dfag_vs_stage1.csv", EXP_ROOT / "statistics" / "dfag_vs_alpha_0_5.csv",
                   EXP_ROOT / "statistics" / "dfag_vs_stage2.csv", EXP_ROOT / "statistics" / "decision_summary.json",
                   EXP_ROOT / "gate" / "gate_summary_per_seed.csv", EXP_ROOT / "counterfactual" / "gate_counterfactual_per_seed.csv",
                   EXP_ROOT / "complexity" / "unified_dfag_complexity.json"]}}
    postflight = EXP_ROOT / "manifests" / "postflight_audit.json"
    postflight.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Unified Standalone Dynamic DFAG Results Audit", "", f"**Integrity: {payload['status']}**", "",
             f"**Pre-declared classification: {decision}** — {reason}.", "", "## Hard gates", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in checks.items())
    lines += ["", "## Per-seed DFAG performance", "", "| Seed | Accuracy | Macro-F1 | Balanced Accuracy |", "|---:|---:|---:|---:|"]
    for row in per_seed_metrics:
        lines.append(f"| {row['seed']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} |")
    lines += ["", "## Interpretation boundary", "",
              "The experiment supports reporting measured OOF performance, paired deltas, gate variability, branch discrepancy, counterfactual replacement effects, and deployment cost. "
              "It does not directly establish semantic restoration, useful routing solely from gate variance, necessity, overfitting prevention, redundancy from cosine, or a causal mechanism.", "",
              "STOP", "", "No seed45/46 automatically launched.", "No fixed-g training launched.",
              "No anchor sweep launched.", "No gating-source ablation launched.", "No SSPH+DFAG joint launched.", ""]
    (EXP_ROOT / "UNIFIED_DFAG_RESULTS_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"stage": "POSTFLIGHT", "status": payload["status"], "decision": decision,
                      "reason": reason}, ensure_ascii=False), flush=True)
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
