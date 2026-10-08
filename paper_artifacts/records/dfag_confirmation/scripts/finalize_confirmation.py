"""Finalize seed45/46 and the frozen five-seed Phase2F decision."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from confirmation_common import assemble_reference, metrics, paired, validate_oof
from dfag_common import (
    EXPECTED_N,
    EXP_ROOT,
    FEATURE_DIM,
    FIVE_SEEDS,
    FOLDS,
    GATE_PARAMETERS,
    OLD_PHASE2F,
    SEEDS,
    comparator_oof,
    output_dir,
    sha256_file,
)
from prepare_static import verify_snapshot


MODES = ("dynamic", "mean_vector", "mean_scalar", "constant_0_5", "anchor_forced", "plastic_forced")
REPLACEMENTS = ("mean_vector", "mean_scalar", "constant_0_5", "anchor_forced", "plastic_forced")


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(values.max()),
    }


def assemble_dfag(seed: int) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {}
    for fold in FOLDS:
        with np.load(output_dir(seed, fold) / "heldout_dfag_artifacts.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        data["fold"] = np.full(len(data["sample_id"]), fold, dtype=np.int8)
        for key, value in data.items():
            pieces.setdefault(key, []).append(value)
    joined = {key: np.concatenate(values) for key, values in pieces.items()}
    order = np.argsort(joined["sample_id"])
    joined = {key: value[order] for key, value in joined.items()}
    if len(joined["sample_id"]) != EXPECTED_N or not np.array_equal(joined["sample_id"], np.arange(EXPECTED_N)):
        raise RuntimeError(f"seed{seed}: DFAG OOF coverage/order mismatch")
    if not np.array_equal(joined["logits_dynamic"].argmax(axis=1), joined["prediction_dynamic"]):
        raise RuntimeError(f"seed{seed}: dynamic logits/predictions mismatch")
    return joined


def load_alpha05(seed: int) -> dict[str, np.ndarray]:
    with np.load(comparator_oof(seed), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    order = np.argsort(data["sample_id"])
    data = {key: value[order] if value.ndim and value.shape[0] == len(order) else value for key, value in data.items()}
    validate_oof(data, f"seed{seed}/alpha0.5")
    return data


def five_seed_decision(comparisons: pd.DataFrame) -> dict:
    evidence = {}
    for reference in ("stage1", "alpha_0_5"):
        rows = comparisons[comparisons["comparison"] == reference].sort_values("seed")
        if rows["seed"].astype(int).tolist() != list(FIVE_SEEDS):
            raise RuntimeError(f"five-seed comparison coverage mismatch: {reference}")
        deltas = rows["delta_accuracy_pp"].to_numpy(float)
        evidence[reference] = {
            "seed_deltas_pp": {str(int(seed)): float(delta) for seed, delta in zip(rows["seed"], deltas)},
            "positive_seed_count": int(np.sum(deltas > 0.0)),
            "negative_direction_seed_count": int(np.sum(deltas < 0.0)),
            "ci_strictly_positive_count": int(np.sum(rows["ci95_low_pp"].to_numpy(float) > 0.0)),
            "ci_fully_below_zero_count": int(np.sum(rows["ci95_high_pp"].to_numpy(float) < 0.0)),
            "mean_delta_pp": float(deltas.mean()),
            "sample_sd_delta_pp": float(deltas.std(ddof=1)),
        }
    stage1, alpha = evidence["stage1"], evidence["alpha_0_5"]
    criterion_a = stage1["positive_seed_count"] == 5
    criterion_b = alpha["positive_seed_count"] == 5
    supporting = []
    if stage1["ci_strictly_positive_count"] >= 3:
        supporting.append("stage1")
    if alpha["ci_strictly_positive_count"] >= 3:
        supporting.append("alpha_0_5")
    criterion_c = bool(supporting)
    if supporting == ["stage1"]:
        other = ["alpha_0_5"]
    elif supporting == ["alpha_0_5"]:
        other = ["stage1"]
    else:
        other = ["stage1", "alpha_0_5"]
    criterion_d = all(
        evidence[name]["negative_direction_seed_count"] == 0
        and evidence[name]["ci_fully_below_zero_count"] == 0
        for name in other
    )
    criteria = {
        "A_stage1_5_of_5_positive": criterion_a,
        "B_alpha_0_5_5_of_5_positive": criterion_b,
        "C_at_least_one_primary_has_3_of_5_strictly_positive_ci": criterion_c,
        "D_other_primary_has_no_negative_direction_or_fully_negative_ci": criterion_d,
    }
    classification = "PASS" if all(criteria.values()) else "FAIL"
    return {
        "classification": classification,
        "criteria": criteria,
        "supporting_reference_for_C": supporting,
        "reference_checked_for_D": other,
        "evidence": evidence,
        "independent_unit": "training seed",
        "pooled_91765_significance_test": False,
        "borderline_allowed": False,
        "additional_seed47_plus_authorized": False,
        "rule": "PASS iff A and B and C and D; otherwise FAIL",
    }


def main() -> None:
    orchestration = json.loads((EXP_ROOT / "orchestrator_summary.json").read_text(encoding="utf-8"))
    preflight = json.loads((EXP_ROOT / "manifests" / "preflight_audit.json").read_text(encoding="utf-8"))
    if orchestration.get("status") != "ALL_10_COMPLETE" or preflight.get("status") != "PASS":
        raise RuntimeError("training orchestration/preflight incomplete")
    manifests = [
        json.loads((output_dir(seed, fold) / "run_manifest.json").read_text(encoding="utf-8"))
        for seed in SEEDS for fold in FOLDS
    ]
    if len(manifests) != 10 or any(row.get("status") != "COMPLETE" for row in manifests):
        raise RuntimeError("one or more seed45/46 folds are incomplete")

    new_metrics, new_comparisons, new_counter = [], [], []
    new_gate, new_gate_samples, new_gate_channels, new_representation = [], [], [], []
    oof_integrity = True
    for seed in SEEDS:
        data = assemble_dfag(seed)
        labels = data["label"].astype(np.int64)
        dynamic = data["prediction_dynamic"].astype(np.int64)
        fold = data["fold"].astype(np.int8)
        stage1 = assemble_reference(seed, 1)
        stage2 = assemble_reference(seed, 2)
        alpha05 = load_alpha05(seed)
        for reference in (stage1, stage2, alpha05):
            oof_integrity &= np.array_equal(data["sample_id"], reference["sample_id"])
            oof_integrity &= np.array_equal(labels, reference["label"])
            oof_integrity &= np.array_equal(fold, reference["fold"])
        dynamic_metrics = metrics(labels, dynamic)
        new_metrics.append({"seed": seed, "n": EXPECTED_N, **dynamic_metrics})
        np.savez_compressed(
            EXP_ROOT / "oof" / f"seed{seed}_dfag_oof.npz",
            sample_id=data["sample_id"], fold=fold, label=labels,
            prediction=dynamic, logits=data["logits_dynamic"],
        )
        np.savez_compressed(
            EXP_ROOT / "gate" / f"seed{seed}_gate_oof.npz",
            sample_id=data["sample_id"], fold=fold, gate=data["gate"].astype(np.float32),
        )
        references = {"stage1": stage1, "alpha_0_5": alpha05, "stage2": stage2}
        for index, (name, reference) in enumerate(references.items()):
            new_comparisons.append(
                paired(seed, name, labels, dynamic, reference["prediction"], 20260820 + 10 * seed + index)
            )
        for index, mode in enumerate(REPLACEMENTS):
            row = paired(
                seed,
                f"dynamic_vs_{mode}",
                labels,
                dynamic,
                data[f"prediction_{mode}"],
                20260900 + 10 * seed + index,
            )
            row["diagnostic_type"] = "COUNTERFACTUAL_INFERENCE_ONLY"
            new_counter.append(row)

        gate = data["gate"].astype(np.float64)
        sample_mean = gate.mean(axis=1)
        channel_mean = gate.mean(axis=0)
        channel_sd = gate.std(axis=0, ddof=1)
        rms = np.sqrt(np.mean((gate - channel_mean[None, :]) ** 2, axis=1))
        new_gate.append(
            {
                "seed": seed,
                "n_samples": EXPECTED_N,
                "n_atomic": gate.size,
                **{f"atomic_{key}": value for key, value in describe(gate).items()},
                **{f"sample_mean_{key}": value for key, value in describe(sample_mean).items()},
                **{f"channel_mean_{key}": value for key, value in describe(channel_mean).items()},
                "mean_per_channel_sample_sd": float(channel_sd.mean()),
                "median_per_channel_sample_sd": float(np.median(channel_sd)),
                **{f"rms_from_global_{key}": value for key, value in describe(rms).items()},
                "p_gate_lt_0_05": float(np.mean(gate < 0.05)),
                "p_gate_lt_0_10": float(np.mean(gate < 0.10)),
                "p_gate_gt_0_90": float(np.mean(gate > 0.90)),
                "p_gate_gt_0_95": float(np.mean(gate > 0.95)),
            }
        )
        new_gate_samples.extend(
            {
                "seed": seed,
                "sample_id": int(sample_id),
                "fold": int(fold_value),
                "mean_gate": float(mean_value),
                "rms_from_global_mean_gate_vector": float(rms_value),
            }
            for sample_id, fold_value, mean_value, rms_value in zip(
                data["sample_id"], fold, sample_mean, rms
            )
        )
        new_gate_channels.extend(
            {
                "seed": seed,
                "channel": channel,
                "mean_gate": float(channel_mean[channel]),
                "sd_across_samples": float(channel_sd[channel]),
            }
            for channel in range(FEATURE_DIM)
        )
        f_anc = data["f_anc"].astype(np.float64)
        f_spec = data["f_spec"].astype(np.float64)
        cosine = np.einsum("ij,ij->i", f_anc, f_spec) / (
            np.linalg.norm(f_anc, axis=1) * np.linalg.norm(f_spec, axis=1) + 1e-12
        )
        l2 = np.linalg.norm(f_anc - f_spec, axis=1)
        new_representation.extend(
            [
                {"seed": seed, "metric": "cosine_fanc_fspec", "n": EXPECTED_N, **describe(cosine)},
                {"seed": seed, "metric": "l2_fanc_minus_fspec", "n": EXPECTED_N, **describe(l2)},
            ]
        )

    # Save seed45/46-only audit tables.
    pd.DataFrame(new_metrics).to_csv(EXP_ROOT / "metrics" / "dfag_seed45_46_per_seed.csv", index=False)
    pd.DataFrame(new_comparisons).to_csv(EXP_ROOT / "statistics" / "dfag_seed45_46_paired.csv", index=False)
    pd.DataFrame(new_gate).to_csv(EXP_ROOT / "gate" / "gate_summary_seed45_46.csv", index=False)
    pd.DataFrame(new_gate_samples).to_csv(EXP_ROOT / "gate" / "gate_sample_summary_seed45_46.csv", index=False)
    pd.DataFrame(new_gate_channels).to_csv(EXP_ROOT / "gate" / "gate_channel_summary_seed45_46.csv", index=False)
    pd.DataFrame(new_representation).to_csv(
        EXP_ROOT / "representation" / "anchor_plastic_similarity_seed45_46.csv", index=False
    )
    pd.DataFrame(new_counter).to_csv(
        EXP_ROOT / "counterfactual" / "gate_counterfactual_per_seed_45_46.csv", index=False
    )

    # Freeze old 42-44 evidence and append only the two new independent seeds.
    old_metrics = pd.read_csv(OLD_PHASE2F / "metrics" / "dfag_per_seed.csv")
    five_metrics = pd.concat([old_metrics, pd.DataFrame(new_metrics)], ignore_index=True).sort_values("seed")
    if five_metrics["seed"].astype(int).tolist() != list(FIVE_SEEDS):
        raise RuntimeError("five-seed metric coverage mismatch")
    five_metrics.to_csv(EXP_ROOT / "dfag_five_seed_per_seed.csv", index=False)
    metric_summary = []
    for name in ("accuracy", "macro_f1", "balanced_accuracy"):
        values = five_metrics[name].to_numpy(float)
        metric_summary.append(
            {
                "metric": name,
                "mean_across_seeds": float(values.mean()),
                "sample_sd_across_seeds": float(values.std(ddof=1)),
                "min_seed": float(values.min()),
                "max_seed": float(values.max()),
                "independent_unit": "training seed",
            }
        )
    pd.DataFrame(metric_summary).to_csv(EXP_ROOT / "dfag_five_seed_summary.csv", index=False)

    five_comparison_frames = []
    filenames = {
        "stage1": "dfag_vs_stage1_five_seed.csv",
        "alpha_0_5": "dfag_vs_alpha_0_5_five_seed.csv",
        "stage2": "dfag_vs_stage2_five_seed.csv",
    }
    new_comparison_frame = pd.DataFrame(new_comparisons)
    for name, filename in filenames.items():
        old = pd.read_csv(OLD_PHASE2F / "statistics" / f"dfag_vs_{name}.csv")
        new = new_comparison_frame[new_comparison_frame["comparison"] == name]
        combined = pd.concat([old, new], ignore_index=True).sort_values("seed")
        if combined["seed"].astype(int).tolist() != list(FIVE_SEEDS):
            raise RuntimeError(f"five-seed comparison coverage mismatch: {name}")
        combined.to_csv(EXP_ROOT / filename, index=False)
        five_comparison_frames.append(combined)
    five_comparisons = pd.concat(five_comparison_frames, ignore_index=True)
    decision = five_seed_decision(five_comparisons)
    (EXP_ROOT / "final_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    old_gate = pd.read_csv(OLD_PHASE2F / "gate" / "gate_summary_per_seed.csv")
    five_gate = pd.concat([old_gate, pd.DataFrame(new_gate)], ignore_index=True).sort_values("seed")
    five_gate.to_csv(EXP_ROOT / "gate_summary_five_seed.csv", index=False)
    old_representation = pd.read_csv(
        OLD_PHASE2F / "representation" / "anchor_plastic_similarity_summary.csv"
    )
    five_representation = pd.concat(
        [old_representation, pd.DataFrame(new_representation)], ignore_index=True
    ).sort_values(["seed", "metric"])
    five_representation.to_csv(EXP_ROOT / "anchor_plastic_similarity_five_seed.csv", index=False)

    old_counter = pd.read_csv(OLD_PHASE2F / "counterfactual" / "gate_counterfactual_per_seed.csv")
    old_counter["diagnostic_type"] = "COUNTERFACTUAL_INFERENCE_ONLY"
    five_counter = pd.concat([old_counter, pd.DataFrame(new_counter)], ignore_index=True)
    counter_summary_rows = []
    for comparison, group in five_counter.groupby("comparison"):
        group = group.sort_values("seed")
        if group["seed"].astype(int).tolist() != list(FIVE_SEEDS):
            raise RuntimeError(f"counterfactual five-seed coverage mismatch: {comparison}")
        deltas = group["delta_accuracy_pp"].to_numpy(float)
        row = {
            "comparison": comparison,
            **{f"seed{int(seed)}_delta_accuracy_pp": float(delta) for seed, delta in zip(group["seed"], deltas)},
            "five_seed_mean_delta_accuracy_pp": float(deltas.mean()),
            "five_seed_sample_sd_delta_accuracy_pp": float(deltas.std(ddof=1)),
            "positive_seed_count": int(np.sum(deltas > 0)),
            "independent_unit": "training seed",
            "diagnostic_type": "COUNTERFACTUAL_INFERENCE_ONLY",
        }
        counter_summary_rows.append(row)
    pd.DataFrame(counter_summary_rows).to_csv(EXP_ROOT / "gate_counterfactual_five_seed.csv", index=False)

    protected_unchanged, protected_changes = verify_snapshot()
    scheduler_events = (EXP_ROOT / "scheduler" / "orchestrator_events.jsonl").read_text(encoding="utf-8").lower()
    checks = {
        "ten_of_ten_complete": len(manifests) == 10 and all(row["status"] == "COMPLETE" for row in manifests),
        "zero_failed_folds": all(row["status"] != "FAILED" for row in manifests),
        "oof_18353_per_new_seed": all(row["n"] == EXPECTED_N for row in new_metrics),
        "oof_sample_ids_labels_and_folds_exact": oof_integrity,
        "anchor_unchanged": all(row["sanity"]["anchor_unchanged"] for row in manifests),
        "anchor_bn_unchanged": all(row["sanity"]["anchor_bn_unchanged"] for row in manifests),
        "plastic_stage2_protocol_matched": all(
            row["bn_adaptation_batches_completed"] == 50 and row["history_epochs"] == 60 for row in manifests
        ),
        "gate_architecture_exact": all(row["parameter_counts"]["gate_parameters"] == GATE_PARAMETERS for row in manifests),
        "seed45_46_exact": {int(row["seed"]) for row in manifests} == set(SEEDS),
        "counterfactual_inference_only": all(row["diagnostic_type"] == "COUNTERFACTUAL_INFERENCE_ONLY" for row in new_counter),
        "historical_seed42_44_artifacts_unchanged": protected_unchanged,
        "no_pooled_five_seed_significance_test": decision["pooled_91765_significance_test"] is False,
        "statistics_independent_unit_seed": decision["independent_unit"] == "training seed",
        "no_fixed_g_training": orchestration["fixed_g_training_launched"] is False,
        "no_anchor_sweep": orchestration["anchor_sweep_launched"] is False,
        "no_gating_source_ablation": orchestration["gating_source_ablation_launched"] is False,
        "no_joint_or_progressive_training": (
            orchestration["joint_model_launched"] is False
            and orchestration["progressive_head_training_launched"] is False
        ),
        "no_seed47_plus": orchestration["seed47_plus_launched"] is False,
        "scheduler_contains_only_ten_authorized_tasks": (
            scheduler_events.count('"event": "start"') == 10
            and "seed47" not in scheduler_events
            and "seed48" not in scheduler_events
        ),
        "gate_training_sanity_pass": all(
            row["sanity"]["gate_parameters_changed"]
            and row["sanity"]["gate_gradients_not_permanently_zero"]
            for row in manifests
        ),
        "final_classification_not_borderline": decision["classification"] in ("PASS", "FAIL"),
    }
    postflight = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_passed": all(checks.values()),
        "created_unix": time.time(),
        "checks": checks,
        "final_decision": decision,
        "new_seed_metrics": new_metrics,
        "protected_source_changes": protected_changes,
        "output_sha256": {},
    }
    major_outputs = [
        "dfag_five_seed_per_seed.csv",
        "dfag_five_seed_summary.csv",
        "dfag_vs_stage1_five_seed.csv",
        "dfag_vs_alpha_0_5_five_seed.csv",
        "dfag_vs_stage2_five_seed.csv",
        "gate_summary_five_seed.csv",
        "anchor_plastic_similarity_five_seed.csv",
        "gate_counterfactual_five_seed.csv",
        "final_decision.json",
    ]
    postflight["output_sha256"] = {
        name: sha256_file(EXP_ROOT / name) for name in major_outputs
    }
    (EXP_ROOT / "manifests" / "postflight_audit.json").write_text(
        json.dumps(postflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (EXP_ROOT / "manifests" / "integrity_report.json").write_text(
        json.dumps(postflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    new_report = [
        "# DFAG Seed45/46 Results Audit",
        "",
        f"**Postflight integrity: {postflight['status']}**",
        "",
        "| Seed | Accuracy | Macro-F1 | Balanced Accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for row in new_metrics:
        new_report.append(
            f"| {row['seed']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} |"
        )
    new_report += ["", "## Postflight gates", ""]
    new_report.extend(f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in checks.items())
    new_report += [
        "",
        "Only seed45/46 unified standalone dynamic DFAG was trained. No later experiment was launched.",
        "",
        "STOP",
        "",
    ]
    (EXP_ROOT / "DFAG_SEED45_46_RESULTS_AUDIT.md").write_text("\n".join(new_report), encoding="utf-8")

    five_report = [
        "# DFAG Five-Seed Final Audit",
        "",
        f"**FINAL CLASSIFICATION: {decision['classification']}**",
        "",
        "## Frozen A/B/C/D criteria",
        "",
    ]
    five_report.extend(
        f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in decision["criteria"].items()
    )
    five_report += [
        "",
        "## Five-seed DFAG performance",
        "",
        "| Seed | Accuracy | Macro-F1 | Balanced Accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for row in five_metrics.to_dict("records"):
        five_report.append(
            f"| {int(row['seed'])} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} |"
        )
    five_report += [
        "",
        "## Interpretation boundary",
        "",
        "The inference-only counterfactual analysis does not isolate an accuracy advantage attributable specifically to sample-dependent gate variation in the trained DFAG models.",
        "",
        "These counterfactuals are not independently trained fixed-gating baselines and therefore do not establish that dynamic gating was unnecessary during training or that the observed DFAG performance originates solely from dual-branch fusion.",
        "",
        "Gate variation and anchor/plastic cosine/L2 are descriptive only; they do not establish semantic restoration, useful routing, redundancy, overfitting prevention, or a causal mechanism.",
        "",
        "No pooled 91,765-sample significance test was performed. The independent unit is the training seed.",
        "",
        "STOP",
        "",
    ]
    (EXP_ROOT / "DFAG_FIVE_SEED_FINAL_AUDIT.md").write_text("\n".join(five_report), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "POSTFLIGHT",
                "status": postflight["status"],
                "classification": decision["classification"],
            }
        ),
        flush=True,
    )
    if postflight["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
