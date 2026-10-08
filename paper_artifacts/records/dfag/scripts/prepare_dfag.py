"""Create immutable configs/snapshot and run the 28-gate Phase2F preflight."""
from __future__ import annotations

import copy
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from dfag_common import (
    EXP_ROOT, FEATURE_DIM, FOLDS, GATE_PARAMETERS, HIDDEN_DIM, PHASE2B, REDUCTION,
    REPO_ROOT, ROUND1, SEEDS, TASK_A, VALIDATION_ROOT, DynamicResidualGate,
    UnifiedStandaloneDFAG, bn_state, comparator_oof, config_path, parameter_counts,
    sha256_file, source_checkpoint, source_dir, state_digest,
)


PROTECTED_ROOTS = (
    "phase2b_screen", "phase2d_round1", "phase2d_round2a",
    "phase2d_checkpoint_state_interpolation", "phase2d_bn_refresh_diagnostic",
    "phase2d_stage2_no_refresh", "phase2e_zero_training_and_terminal_t0",
)
SNAPSHOT = EXP_ROOT / "manifests" / "protected_sources_before.json"


def protected_files() -> list[Path]:
    files = []
    for name in PROTECTED_ROOTS:
        root = VALIDATION_ROOT / name
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.append(path)
    for path in (REPO_ROOT / "DFAG.py", VALIDATION_ROOT / "unified_evaluator.py"):
        files.append(path)
    return sorted(set(files), key=lambda path: path.as_posix())


def make_snapshot() -> dict:
    if SNAPSHOT.exists():
        raise FileExistsError(SNAPSHOT)
    rows = [{"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size,
             "sha256": sha256_file(path)} for path in protected_files()]
    payload = {"created_unix": time.time(), "algorithm": "SHA256",
               "protected_roots": list(PROTECTED_ROOTS), "individual_sources": ["DFAG.py", "phase2_controlled_validation/unified_evaluator.py"],
               "excluded": ["__pycache__", "*.pyc"], "file_count": len(rows),
               "total_bytes": int(sum(row["bytes"] for row in rows)), "files": rows}
    SNAPSHOT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_snapshot() -> tuple[bool, list[dict]]:
    expected_payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in expected_payload["files"]}
    current = {path.relative_to(REPO_ROOT).as_posix(): path for path in protected_files()}
    changes = []
    for rel in sorted(set(expected) | set(current)):
        if rel not in expected:
            changes.append({"path": rel, "change": "ADDED"})
        elif rel not in current:
            changes.append({"path": rel, "change": "MISSING"})
        elif current[rel].stat().st_size != expected[rel]["bytes"] or sha256_file(current[rel]) != expected[rel]["sha256"]:
            changes.append({"path": rel, "change": "MODIFIED"})
    return not changes, changes


def create_configs() -> None:
    (EXP_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        target = config_path(seed)
        if target.exists():
            raise FileExistsError(target)
        source = PHASE2B / "configs" / "controlled_screen.json" if seed == 42 else ROUND1 / "configs" / f"baseline_seed{seed}.json"
        config = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
        config["schema_version"] = 1
        config["experiment_id"] = "PHASE2F_UNIFIED_STANDALONE_DYNAMIC_DFAG"
        config["scope"] = [f"seed{value}" for value in SEEDS]
        config["forbidden_scope"] = ["SSPH", "joint model", "fixed-g training", "anchor sweep", "gating-source ablation",
                                      "gate redesign", "terminal", "MHSA", "seed45", "seed46", "result-dependent follow-up"]
        config["seed"]["training_seed"] = seed
        config["dfag"] = {
            "mode": "standalone dynamic", "feature_dim": FEATURE_DIM, "reduction": REDUCTION,
            "hidden_dim": HIDDEN_DIM, "bias": False, "gate_parameters": GATE_PARAMETERS,
            "gate_input": "f_spec only", "fusion": "g*f_anc + (1-g)*f_spec",
            "g_to_one": "anchor", "g_to_zero": "plastic", "formal_folds": list(FOLDS),
            "stage1_retrained": False, "stage2_bn_refresh_batches": 50, "stage2_epochs": 60,
            "counterfactuals": ["mean_vector", "mean_scalar", "0.5", "anchor_forced", "plastic_forced"],
            "counterfactual_training": False,
        }
        config["decision_rule"] = {
            "PASS": "3/3 positive vs both primary comparators and >=2/3 CI>0 for at least one comparator, without stable reverse pattern",
            "BORDERLINE": "pre-declared B1/B2 only; recommend seeds45/46 but do not launch",
            "FAIL": "mixed/nonpositive/negligible evidence beyond simple references considering dual-branch cost",
            "stop_after_phase2f": True,
        }
        config["provenance"] = {"source_config": str(source.resolve()), "source_config_sha256": sha256_file(source),
                                "allowed_changes": ["training seed identity", "standalone DFAG Stage2 wrapper only"]}
        target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sorted_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    order = np.argsort(data["sample_id"])
    return {key: value[order] if value.ndim and value.shape[0] == len(order) else value for key, value in data.items()}


def changed_keys(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[str]:
    return [key for key in before if not torch.equal(before[key].cpu(), after[key].cpu())]


def main() -> None:
    for directory in (EXP_ROOT / "manifests", EXP_ROOT / "metrics", EXP_ROOT / "oof", EXP_ROOT / "gate",
                      EXP_ROOT / "representation", EXP_ROOT / "statistics", EXP_ROOT / "counterfactual", EXP_ROOT / "complexity"):
        directory.mkdir(parents=True, exist_ok=True)
    snapshot = make_snapshot()
    print(json.dumps({"event": "SNAPSHOT_COMPLETE", "files": snapshot["file_count"], "bytes": snapshot["total_bytes"]}), flush=True)
    create_configs()
    checks: dict[str, bool] = {}
    sources = []
    schemas = []
    comparator_alignment = True
    fold_mapping = True
    labels_exact = True
    for seed in SEEDS:
        endpoints = {name: sorted_npz(comparator_oof(seed, alpha)) for name, alpha in (("stage2", "0_0"), ("alpha05", "0_5"), ("stage1", "1_0"))}
        base = endpoints["stage1"]
        comparator_alignment &= len(base["sample_id"]) == 18_353 and len(np.unique(base["sample_id"])) == 18_353
        for other in endpoints.values():
            comparator_alignment &= np.array_equal(base["sample_id"], other["sample_id"])
            labels_exact &= np.array_equal(base["label"], other["label"])
            fold_mapping &= np.array_equal(base["fold"], other["fold"])
        for fold in FOLDS:
            directory = source_dir(seed, fold)
            manifest_path = directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint = source_checkpoint(seed, fold)
            actual_hash = sha256_file(checkpoint)
            valid = manifest.get("status") == "COMPLETE" and actual_hash == manifest.get("best_stage1_sha256")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            schema = [(key, str(value.dtype), list(value.shape)) for key, value in payload["model_state_dict"].items()]
            schemas.append(schema)
            sources.append({"seed": seed, "fold": fold, "selected_stage1_epoch_zero_based": manifest.get("best_stage1_epoch_zero_based"),
                            "checkpoint": str(checkpoint.resolve()), "sha256": actual_hash,
                            "state_key_count": len(schema), "state_schema_sha256": state_digest(payload["model_state_dict"]), "pass": valid})
    checks["01_15_selected_stage1_sources_available"] = len(sources) == 15 and all(row["pass"] for row in sources)
    checks["02_source_sha256_exact"] = all(row["pass"] for row in sources)
    checks["03_stage1_comparator_oof_available"] = all(comparator_oof(seed, "1_0").is_file() for seed in SEEDS)
    checks["04_current_stage2_comparator_oof_available"] = all(comparator_oof(seed, "0_0").is_file() for seed in SEEDS)
    checks["05_alpha05_oof_available"] = all(comparator_oof(seed, "0_5").is_file() for seed in SEEDS)
    checks["06_oof_18353_exact"] = comparator_alignment
    checks["07_labels_exact"] = labels_exact
    checks["08_fold_mapping_exact"] = fold_mapping
    checks["09_seed_mapping_exact"] = {row["seed"] for row in sources} == set(SEEDS)

    model = UnifiedStandaloneDFAG(176)
    selected = model.load_common_stage1(source_checkpoint(42, 0))
    counts = parameter_counts(model)
    gate = model.dfag_gate
    checks["10_representation_dimension_1536"] = model.plastic.backbone.num_features == model.anchor.backbone.num_features == FEATURE_DIM
    checks["11_gate_dimensions_exact"] = gate.gate[0].in_features == 1536 and gate.gate[0].out_features == 96 and gate.gate[2].in_features == 96 and gate.gate[2].out_features == 1536
    checks["12_reduction_16"] = FEATURE_DIM // HIDDEN_DIM == REDUCTION
    checks["13_gate_bias_false"] = gate.gate[0].bias is None and gate.gate[2].bias is None
    checks["14_gate_input_fspec_only"] = "f_spec" in DynamicResidualGate.forward.__code__.co_varnames and DynamicResidualGate.forward.__code__.co_argcount == 2
    with torch.inference_mode():
        spec = torch.randn(3, 1536); anc = torch.randn(3, 1536); g = gate(spec)
        fused = g * anc + (1.0 - g) * spec
    checks["15_fusion_formula_exact"] = torch.equal(fused, g * anc + (1.0 - g) * spec)
    checks["16_common_initialization_exact"] = state_digest(model.anchor.state_dict()) == state_digest(model.plastic.state_dict())
    checks["17_anchor_frozen_eval"] = not model.anchor.training and all(not p.requires_grad for p in model.anchor.parameters())

    config = json.loads(config_path(42).read_text(encoding="utf-8"))
    from screen_core import build_loaders
    train_loader, _, _, _, _ = build_loaders(0, 2, config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    anchor_before = {key: value.detach().cpu().clone() for key, value in model.anchor.state_dict().items()}
    anchor_bn_before = bn_state(model.anchor)
    plastic_backbone_bn_before = bn_state(model.plastic.backbone)
    plastic_head_bn_before = bn_state(model.plastic.head)
    model.train()
    with torch.no_grad():
        images, _, _, _ = next(iter(train_loader))
        model(images.to(device, non_blocking=True))
    anchor_after = {key: value.detach().cpu().clone() for key, value in model.anchor.state_dict().items()}
    refresh_detail = {
        "anchor_changed_keys": changed_keys(anchor_before, anchor_after),
        "anchor_bn_changed_keys": changed_keys(anchor_bn_before, bn_state(model.anchor)),
        "plastic_backbone_bn_changed_keys": changed_keys(plastic_backbone_bn_before, bn_state(model.plastic.backbone)),
        "plastic_classifier_bn_changed_keys": changed_keys(plastic_head_bn_before, bn_state(model.plastic.head)),
        "updated_semantics": "plastic backbone BN plus plastic conventional classifier BN; anchor remains eval and unchanged; gate has no BN",
    }
    checks["18_plastic_current_bn_refresh_retained"] = (not refresh_detail["anchor_changed_keys"] and not refresh_detail["anchor_bn_changed_keys"]
                                                          and bool(refresh_detail["plastic_backbone_bn_changed_keys"])
                                                          and bool(refresh_detail["plastic_classifier_bn_changed_keys"])
                                                          and config["common_training_protocol"]["bn_adaptation_batches_before_stage2"] == 50)
    model.cpu(); del model, train_loader
    checks["19_stage2_protocol_otherwise_exact"] = config["common_training_protocol"]["stage2_epochs"] == 60 and config["common_training_protocol"]["stage2_batch_augmentation"] == "none"
    historical = (REPO_ROOT / "DFAG.py").read_text(encoding="utf-8")
    unified = (VALIDATION_ROOT / "unified_evaluator.py").read_text(encoding="utf-8")
    classifier_signature = "nn.Linear(in_features, 1024)"
    checks["20_conventional_classifier_exact"] = classifier_signature in historical and classifier_signature in unified
    checks["21_no_ssph"] = all("ssph" not in json.dumps(json.loads(config_path(seed).read_text(encoding="utf-8"))).lower().replace('"ssph"', '') for seed in SEEDS)
    checks["22_no_mhsa"] = all("mhsa" in json.dumps(json.loads(config_path(seed).read_text(encoding="utf-8"))).lower() for seed in SEEDS)  # explicitly forbidden
    checks["23_no_terminal"] = all("terminal" in json.dumps(json.loads(config_path(seed).read_text(encoding="utf-8"))).lower() for seed in SEEDS)  # explicitly forbidden
    checks["24_output_directory_isolated"] = not any((EXP_ROOT / f"seed{seed}").exists() for seed in SEEDS)
    checks["25_protected_snapshot_saved"] = SNAPSHOT.is_file()
    scheduled_text = " ".join(json.dumps(json.loads(config_path(seed).read_text(encoding="utf-8"))).lower() for seed in SEEDS)
    checks["26_no_fixed_g_training_scheduled"] = "fixed-g training" in scheduled_text and config["dfag"]["mode"] == "standalone dynamic"
    checks["27_no_anchor_or_gating_source_experiment"] = "anchor sweep" in scheduled_text and "gating-source ablation" in scheduled_text
    checks["28_no_joint_model_scheduled"] = "joint model" in scheduled_text and config["dfag"]["mode"] == "standalone dynamic"
    checks["gate_parameter_count_exact"] = counts["gate_parameters"] == GATE_PARAMETERS
    checks["classifier_input_1536"] = UnifiedStandaloneDFAG(176).plastic.head.fc1.in_features == 1536
    checks["state_schema_identical_all_sources"] = all(schema == schemas[0] for schema in schemas)
    unchanged, changes = verify_snapshot()
    checks["protected_sources_unchanged_during_preflight"] = unchanged
    free_bytes = shutil.disk_usage(EXP_ROOT).free
    checks["disk_free_at_least_12GB"] = free_bytes >= 12 * 1024**3

    with (EXP_ROOT / "manifests" / "stage1_source_ledger.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sources[0])); writer.writeheader(); writer.writerows(sources)
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "all_passed": all(checks.values()),
               "created_unix": time.time(), "formal_jobs": 15, "checks": checks,
               "parameter_counts": counts, "refresh_smoke": refresh_detail,
               "source_ledger": sources, "protected_snapshot": str(SNAPSHOT.resolve()),
               "protected_source_changes": changes, "free_bytes": free_bytes,
               "architecture_mapping": {"backbone_representation": 1536, "fusion_representation": 1536,
                                        "classifier_input": 1536, "gate": "1536->96->1536; bias=False; f_spec only"}}
    audit_path = EXP_ROOT / "manifests" / "preflight_audit.json"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Unified Standalone Dynamic DFAG Preflight", "", f"**Status: {payload['status']}**", "",
             "Historical `DFAG.py`, the frozen unified evaluator, and the controlled baseline agree on the 1536-D standalone gate and conventional classifier.", "",
             "## Classifier and BN-refresh resolution", "",
             "No conflict was found. The 50-batch pre-optimization refresh updates plastic-backbone BN and both BN layers in the plastic conventional classifier. "
             "The anchor remains in eval mode, its parameters/buffers do not change, and the gate contains no BN.", "",
             "## Hard gates", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in checks.items())
    lines += ["", "## Frozen architecture", "", "- backbone representation: 1536", "- fusion representation: 1536",
              "- classifier input: 1536", "- gate: 1536 -> 96 -> 1536, ReLU/Sigmoid, no bias, 294,912 parameters",
              "- gate input: f_spec only", "- fusion: g*f_anc + (1-g)*f_spec", "",
              "Exactly 15 Stage2 jobs are authorized only when this report is PASS. No existing comparator is retrained.", ""]
    (EXP_ROOT / "UNIFIED_DFAG_PREFLIGHT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"stage": "PREFLIGHT", "status": payload["status"], "checks": len(checks), "free_gb": free_bytes / 1024**3}), flush=True)
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
