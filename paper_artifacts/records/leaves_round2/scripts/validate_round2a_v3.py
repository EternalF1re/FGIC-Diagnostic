"""Fail-closed preflight validation before Round2A formal training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from round2a_common import (
    BASELINE_IDS,
    MHSA_IDS,
    PHASE2B_ROOT,
    ROUND_IDS,
    ROUND_ROOT,
    RUN_SPECS,
    build_round_model,
    load_round_config,
)
from screen_core import LeafDataset, eval_transform, fold_class_weights, load_config, model_counts, split_indices


INSTRUCTION_SHA256 = "FC4D9CD4DA231A1117581AD0EE06F40AFA521B83A4FA770E704648DE731D4307"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    audit_path = ROUND_ROOT / "manifests" / "preflight_audit.json"
    if audit_path.exists():
        raise RuntimeError(f"refusing to overwrite {audit_path}")
    base = load_config()
    checks = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            raise RuntimeError(f"{name}: {detail}")

    instruction = ROUND_ROOT / "manifests" / f"instruction_{INSTRUCTION_SHA256}.txt"
    check("instruction_snapshot_hash", instruction.exists() and sha256(instruction).upper() == INSTRUCTION_SHA256, str(instruction))
    plan = ROUND_ROOT / "ROUND2A_ANALYSIS_PLAN.md"
    check("analysis_plan_frozen", plan.exists(), f"sha256={sha256(plan) if plan.exists() else 'missing'}")
    step1 = json.loads((ROUND_ROOT / "manifests" / "baseline_three_seed_pretraining_audit.json").read_text(encoding="utf-8"))
    check("baseline_three_seed_audit", step1.get("status") == "PASS" and step1.get("training_authorized_by_step1"), str(step1.get("stop_reasons")))

    dataset_cfg = base["dataset"]
    dataset = LeafDataset(Path(dataset_cfg["train_csv"]), Path(dataset_cfg["root"]), eval_transform())
    expected_counts = {"baseline": 57_114_448, "deep_narrow_mhsa": 55_340_880}
    for run_id in ROUND_IDS:
        config = load_round_config(run_id)
        run = config["round2a_run"]
        spec = RUN_SPECS[run_id]
        check(f"{run_id}_dataset_exact", config["dataset"] == base["dataset"], "dataset equals Phase2B")
        check(f"{run_id}_split_exact", config["split"] == base["split"], "split equals Phase2B")
        check(f"{run_id}_protocol_exact", config["common_training_protocol"] == base["common_training_protocol"], "protocol equals Phase2B")
        check(f"{run_id}_seed", config["seed"]["training_seed"] == spec["seed"] and not config["seed"]["seed_plus_fold"], str(config["seed"]))
        check(f"{run_id}_terminal_residual_off", run["terminal_residual"] is False, str(run))
        model = build_round_model(run_id, 176, pretrained=False)
        counts = model_counts(model)
        check(f"{run_id}_parameter_count", counts["total_parameters"] == expected_counts[spec["family"]], str(counts))
        dummy = torch.randn(3, 1536)
        model.head.eval()
        if run_id in BASELINE_IDS:
            logits, features, trace = model.head(dummy, return_trace=True)
            check(f"{run_id}_baseline_trace", trace["feature_pre_classifier"] == (3, 1024) and logits.shape == (3, 176) and features.shape == (3, 1024), str(trace))
            check(f"{run_id}_no_mhsa", not any(isinstance(module, nn.MultiheadAttention) for module in model.modules()), "baseline contains no MHSA")
        else:
            logits, features, extras = model.head(
                dummy,
                return_trace=True,
                return_attention=True,
                return_stages=True,
            )
            trace = extras["trace"]
            weights = extras["attention_weights"]
            stacked = extras["post_shortcut_stages"]
            fused = extras["fused_stages"]
            check(f"{run_id}_lambda", model.head.shortcut_lambda == spec["lambda"], f"lambda={model.head.shortcut_lambda}")
            check(f"{run_id}_token_trace", trace["stacked_stages"] == (3, 5, 256) and trace["fused_stages"] == (3, 5, 256), str(trace))
            check(f"{run_id}_attention_shape", weights is not None and weights.shape == (3, 4, 5, 5), str(weights.shape if weights is not None else None))
            check(f"{run_id}_attention_rows", torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6), "attention key distributions sum to one")
            check(f"{run_id}_mhsa_fields", model.head.mhsa.num_heads == 4 and model.head.mhsa.dropout == 0.1 and model.head.mhsa.batch_first, str(model.head.mhsa))
            check(f"{run_id}_two_layernorms", isinstance(model.head.pre_attention_norm, nn.LayerNorm) and isinstance(model.head.fusion_norm, nn.LayerNorm), "pre-LN and fusion-LN present")
            normalized = model.head.pre_attention_norm(stacked)
            attention_output, _ = model.head.mhsa(normalized, normalized, normalized, need_weights=False)
            expected_fused = model.head.fusion_norm(stacked + attention_output)
            max_abs_error = float((fused - expected_fused).abs().max().item())
            check(
                f"{run_id}_fusion_formula",
                torch.allclose(fused, expected_fused, atol=1e-6, rtol=1e-6),
                f"LN(S + MHSA(LN(S))); max_abs_error={max_abs_error:.3e}",
            )
            check(f"{run_id}_output_shape", logits.shape == (3, 176) and features.shape == (3, 256), f"logits={logits.shape}, features={features.shape}")
            check(f"{run_id}_f0_excluded", run["token_sequence"] == ["f1", "f2", "f3", "f4", "f5"], str(run["token_sequence"]))

    for fold in range(5):
        train_idx, val_idx = split_indices(dataset.labels, fold, 42)
        saved = np.load(PHASE2B_ROOT / "#0" / f"fold_{fold}" / "split_indices.npz")
        check(f"fold{fold}_assignment", np.array_equal(train_idx, saved["train_indices"]) and np.array_equal(val_idx, saved["validation_indices"]), f"train={len(train_idx)}, val={len(val_idx)}")
        weights = fold_class_weights(dataset.labels, train_idx, 176).numpy()
        check(f"fold{fold}_class_weights", np.array_equal(weights, np.load(PHASE2B_ROOT / "#0" / f"fold_{fold}" / "class_weights.npy")), "exact Phase2B weights")

    config_manifest = json.loads((ROUND_ROOT / "manifests" / "config_generation.json").read_text(encoding="utf-8"))
    check("formal_scope_25", config_manifest.get("formal_runs") == 25 and len(config_manifest.get("effective_configs", [])) == 5, str(config_manifest.get("formal_runs")))
    check("mhsa_scope_15", len(MHSA_IDS) == 3 and config_manifest.get("new_mhsa_formal_runs") == 15, str(MHSA_IDS))
    check("baseline_scope_10", len(BASELINE_IDS) == 2 and config_manifest.get("new_baseline_formal_runs") == 10, str(BASELINE_IDS))

    result = {
        "all_passed": True,
        "checks": checks,
        "formal_runs": 25,
        "mhsa_runs": 15,
        "baseline_runs": 10,
        "source_hashes": {
            "phase2b_train_one": sha256(PHASE2B_ROOT / "train_one.py"),
            "phase2b_screen_core": sha256(PHASE2B_ROOT / "screen_core.py"),
            "round2a_common": sha256(Path(__file__).with_name("round2a_common.py")),
            "round2a_train_one": sha256(Path(__file__).with_name("train_one_round2a.py")),
        },
        "expected_model_counts": expected_counts,
        "training_performed": False,
        "forbidden_variants_absent": True,
    }
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PREFLIGHT_PASS", "checks": len(checks), "formal_runs": 25}))


if __name__ == "__main__":
    main()
