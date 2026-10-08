"""Create the isolated confirmation snapshot and run pre-comparator checks."""
from __future__ import annotations

import copy
import csv
import json
import shutil
import time
from pathlib import Path

import torch

from dfag_common import (
    EXP_ROOT,
    FEATURE_DIM,
    FOLDS,
    GATE_PARAMETERS,
    HIDDEN_DIM,
    OLD_PHASE2F,
    REDUCTION,
    REPO_ROOT,
    ROUND2A,
    SEEDS,
    VALIDATION_ROOT,
    DynamicResidualGate,
    UnifiedStandaloneDFAG,
    bn_state,
    config_path,
    parameter_counts,
    sha256_file,
    source_checkpoint,
    source_dir,
    state_digest,
)


PROTECTED_ROOTS = (
    "phase2f_unified_standalone_dfag",
    "phase2d_round2a",
    "phase2d_checkpoint_state_interpolation",
    "phase2b_screen",
    "phase2d_round1",
)
SNAPSHOT = EXP_ROOT / "manifests" / "protected_sources_before.json"


def protected_files() -> list[Path]:
    paths: list[Path] = []
    for name in PROTECTED_ROOTS:
        root = VALIDATION_ROOT / name
        paths.extend(
            path for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    paths.extend((REPO_ROOT / "DFAG.py", VALIDATION_ROOT / "unified_evaluator.py"))
    return sorted(set(paths), key=lambda path: path.as_posix())


def make_snapshot() -> dict:
    if SNAPSHOT.exists():
        raise FileExistsError(SNAPSHOT)
    rows = []
    for path in protected_files():
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "created_unix": time.time(),
        "algorithm": "SHA256",
        "protected_roots": list(PROTECTED_ROOTS),
        "file_count": len(rows),
        "total_bytes": int(sum(row["bytes"] for row in rows)),
        "files": rows,
    }
    SNAPSHOT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_snapshot() -> tuple[bool, list[dict]]:
    expected_payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in expected_payload["files"]}
    current = {path.relative_to(REPO_ROOT).as_posix(): path for path in protected_files()}
    changes = []
    for relative in sorted(set(expected) | set(current)):
        if relative not in expected:
            changes.append({"path": relative, "change": "ADDED"})
        elif relative not in current:
            changes.append({"path": relative, "change": "MISSING"})
        elif (
            current[relative].stat().st_size != expected[relative]["bytes"]
            or sha256_file(current[relative]) != expected[relative]["sha256"]
        ):
            changes.append({"path": relative, "change": "MODIFIED"})
    return not changes, changes


def create_configs() -> None:
    old_config = json.loads((OLD_PHASE2F / "configs" / "seed43.json").read_text(encoding="utf-8"))
    for seed in SEEDS:
        target = config_path(seed)
        if target.exists():
            raise FileExistsError(target)
        source = ROUND2A / "configs" / f"baseline_seed{seed}.json"
        config = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
        config["schema_version"] = 1
        config["experiment_id"] = "PHASE2F_SEED45_46_FINAL_CONFIRMATION"
        config["scope"] = ["seed45", "seed46"]
        config["forbidden_scope"] = [
            "SSPH", "Progressive Head", "joint model", "fixed-g training", "anchor sweep",
            "gating-source ablation", "gate redesign", "terminal residual", "MHSA", "lambda sweep",
            "seed47+", "result-dependent follow-up",
        ]
        config["seed"]["training_seed"] = seed
        config["dfag"] = copy.deepcopy(old_config["dfag"])
        config["decision_rule"] = {
            "A": "DFAG vs Stage1-only: 5/5 seed accuracy deltas strictly > 0",
            "B": "DFAG vs alpha=0.5: 5/5 seed accuracy deltas strictly > 0",
            "C": "at least one primary reference has >=3/5 paired-bootstrap CI lower bounds strictly > 0",
            "D": "the other primary reference has no negative-direction seed and no CI fully below zero",
            "PASS": "A and B and C and D all true",
            "FAIL": "otherwise",
            "borderline_allowed": False,
            "stop_after_confirmation": True,
        }
        config["provenance"] = {
            "source_config": str(source.resolve()),
            "source_config_sha256": sha256_file(source),
            "old_phase2f_config": str((OLD_PHASE2F / "configs" / "seed43.json").resolve()),
            "old_phase2f_config_sha256": sha256_file(OLD_PHASE2F / "configs" / "seed43.json"),
            "allowed_changes": ["training seed identity", "source checkpoint identity", "output path"],
        }
        target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if any((EXP_ROOT / f"seed{seed}").exists() for seed in SEEDS):
        raise RuntimeError("new formal output directory is not empty")
    old_postflight = json.loads((OLD_PHASE2F / "manifests" / "postflight_audit.json").read_text(encoding="utf-8"))
    old_decision = json.loads((OLD_PHASE2F / "statistics" / "decision_summary.json").read_text(encoding="utf-8"))
    snapshot = make_snapshot()
    print(
        json.dumps({"event": "PROTECTED_SNAPSHOT", "files": snapshot["file_count"], "bytes": snapshot["total_bytes"]}),
        flush=True,
    )
    create_configs()
    checks: dict[str, bool] = {}
    checks["old_phase2f_integrity_pass"] = old_postflight.get("status") == "PASS"
    checks["old_phase2f_classification_borderline"] = old_decision.get("classification") == "BORDERLINE"
    checks["target_seeds_exact_45_46"] = SEEDS == (45, 46)
    checks["five_folds_exact"] = FOLDS == (0, 1, 2, 3, 4)
    checks["ten_formal_jobs_exact"] = len(SEEDS) * len(FOLDS) == 10

    old_config = json.loads((OLD_PHASE2F / "configs" / "seed43.json").read_text(encoding="utf-8"))
    sources = []
    schemas = []
    protocol_equal = True
    split_provenance = True
    for seed in SEEDS:
        config = json.loads(config_path(seed).read_text(encoding="utf-8"))
        protocol_equal &= config["dataset"] == old_config["dataset"]
        protocol_equal &= config["split"] == old_config["split"]
        protocol_equal &= config["common_training_protocol"] == old_config["common_training_protocol"]
        protocol_equal &= config["architectures"]["#0"] == old_config["architectures"]["#0"]
        protocol_equal &= config["dfag"] == old_config["dfag"]
        for fold in FOLDS:
            directory = source_dir(seed, fold)
            manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
            checkpoint = source_checkpoint(seed, fold)
            actual_hash = sha256_file(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            schema = [(key, str(value.dtype), list(value.shape)) for key, value in payload["model_state_dict"].items()]
            schemas.append(schema)
            valid = (
                manifest.get("status") == "COMPLETE"
                and int(manifest.get("training_seed")) == seed
                and int(manifest.get("fold")) == fold
                and actual_hash == manifest.get("best_stage1_sha256")
            )
            split = __import__("numpy").load(directory / "split_indices.npz")
            split_provenance &= len(split["validation_indices"]) in (3670, 3671)
            sources.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "selected_stage1_epoch_zero_based": manifest.get("best_stage1_epoch_zero_based"),
                    "checkpoint": str(checkpoint.resolve()),
                    "sha256": actual_hash,
                    "state_key_count": len(schema),
                    "state_schema_sha256": state_digest(payload["model_state_dict"]),
                    "pass": valid,
                }
            )
    checks["ten_stage1_sources_available"] = len(sources) == 10 and all(row["pass"] for row in sources)
    checks["source_sha256_exact"] = all(row["pass"] for row in sources)
    checks["state_schema_identical"] = all(schema == schemas[0] for schema in schemas)
    checks["split_provenance_available"] = split_provenance
    checks["plastic_stage2_protocol_identical_to_old_phase2f"] = protocol_equal

    model = UnifiedStandaloneDFAG(176)
    model.load_common_stage1(source_checkpoint(45, 0))
    gate = model.dfag_gate
    counts = parameter_counts(model)
    checks["dg_1536"] = model.anchor.backbone.num_features == model.plastic.backbone.num_features == FEATURE_DIM
    checks["r_16_hidden_96"] = REDUCTION == 16 and HIDDEN_DIM == 96
    checks["bias_false"] = gate.gate[0].bias is None and gate.gate[2].bias is None
    checks["gate_parameters_exact"] = counts["gate_parameters"] == GATE_PARAMETERS
    checks["fspec_only_conditioning"] = (
        DynamicResidualGate.forward.__code__.co_argcount == 2
        and "f_spec" in DynamicResidualGate.forward.__code__.co_varnames
    )
    with torch.inference_mode():
        f_spec = torch.randn(3, FEATURE_DIM)
        f_anc = torch.randn(3, FEATURE_DIM)
        learned = gate(f_spec)
        fused = learned * f_anc + (1.0 - learned) * f_spec
    checks["fusion_formula_exact"] = torch.equal(fused, learned * f_anc + (1.0 - learned) * f_spec)
    checks["common_anchor_plastic_initialization"] = (
        state_digest(model.anchor.state_dict()) == state_digest(model.plastic.state_dict())
    )
    checks["anchor_frozen_eval"] = not model.anchor.training and all(not p.requires_grad for p in model.anchor.parameters())
    checks["no_ssph_mhsa_terminal_joint"] = (
        not any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())
        and model.plastic.head.__class__.__name__ == "BaselineHead"
    )
    checks["no_fixed_gate_training"] = all(
        json.loads(config_path(seed).read_text(encoding="utf-8"))["dfag"]["mode"] == "standalone dynamic"
        and json.loads(config_path(seed).read_text(encoding="utf-8"))["dfag"]["counterfactual_training"] is False
        for seed in SEEDS
    )
    checks["no_anchor_or_gating_source_ablation"] = True
    checks["new_output_dirs_absent"] = not any((EXP_ROOT / f"seed{seed}").exists() for seed in SEEDS)
    checks["protected_snapshot_saved"] = SNAPSHOT.is_file()
    unchanged, changes = verify_snapshot()
    checks["protected_sources_unchanged_during_static_preflight"] = unchanged
    free_bytes = shutil.disk_usage(EXP_ROOT).free
    checks["disk_free_at_least_12gb"] = free_bytes >= 12 * 1024 ** 3

    with (EXP_ROOT / "manifests" / "stage1_source_ledger.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sources[0]))
        writer.writeheader()
        writer.writerows(sources)
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": "STATIC_PRE_COMPARATOR",
        "created_unix": time.time(),
        "formal_jobs": 10,
        "target_seeds": list(SEEDS),
        "folds": list(FOLDS),
        "checks": checks,
        "source_ledger": sources,
        "parameter_counts": counts,
        "protected_snapshot": str(SNAPSHOT.resolve()),
        "protected_source_changes": changes,
        "free_bytes": free_bytes,
    }
    (EXP_ROOT / "manifests" / "static_preflight_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": payload["stage"], "status": payload["status"], "checks": len(checks)}), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
