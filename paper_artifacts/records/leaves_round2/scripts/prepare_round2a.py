"""Create immutable effective configs and output directories for Round2A."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from round2a_common import PHASE2B_ROOT, ROUND_IDS, ROUND_ROOT, RUN_SPECS


INSTRUCTION_SHA256 = "FC4D9CD4DA231A1117581AD0EE06F40AFA521B83A4FA770E704648DE731D4307"
SOURCE = PHASE2B_ROOT / "configs" / "controlled_screen.json"
PLAN = ROUND_ROOT / "ROUND2A_ANALYSIS_PLAN.md"
INSTRUCTION = ROUND_ROOT / "manifests" / f"instruction_{INSTRUCTION_SHA256}.txt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not PLAN.exists() or not INSTRUCTION.exists():
        raise RuntimeError("analysis plan or frozen instruction snapshot is missing")
    if sha256(INSTRUCTION).upper() != INSTRUCTION_SHA256:
        raise RuntimeError("frozen instruction snapshot hash mismatch")
    step1 = json.loads((ROUND_ROOT / "manifests" / "baseline_three_seed_pretraining_audit.json").read_text(encoding="utf-8"))
    if step1.get("status") != "PASS" or not step1.get("training_authorized_by_step1"):
        raise RuntimeError("baseline three-seed audit did not authorize training")

    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    (ROUND_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    for name in ("logs", "oof", "statistics", "representations", "attention", "manifests"):
        (ROUND_ROOT / name).mkdir(parents=True, exist_ok=True)

    created = []
    for run_id in ROUND_IDS:
        spec = RUN_SPECS[run_id]
        target = ROUND_ROOT / "configs" / f"{run_id}.json"
        if target.exists():
            raise RuntimeError(f"refusing to overwrite existing config: {target}")
        config = copy.deepcopy(base)
        config["schema_version"] = 1
        config["experiment_id"] = "PHASE2D_ROUND2A"
        config["scope"] = list(ROUND_IDS)
        config["forbidden_scope"] = [
            "lambda=0.5",
            "lambda=0.8",
            "terminal residual",
            "DFAG",
            "earlier-anchor DFAG",
            "learnable lambda",
            "different MHSA heads",
            "other unplanned ablations",
        ]
        config["seed"]["training_seed"] = spec["seed"]
        config["seed"]["same_training_seed_every_fold"] = True
        config["seed"]["seed_plus_fold"] = False
        config["architectures"]["round2a_mhsa"] = {
            "name": "deep_narrow_progressive_mapping_with_mhsa",
            "backbone": "identical to Phase2B #0/#1",
            "progressive_mapping": "f_i=F_i(f_{i-1})+lambda*f_{i-1}; five 256-D post-shortcut stages",
            "tokens": "[f1,f2,f3,f4,f5]; f0 excluded",
            "attention": "S_norm=LN(S); 4-head MHSA(dropout=0.1); S_fusion=LN(S+MHSA(S_norm))",
            "aggregation": "unweighted mean pooling of five fused stages",
            "terminal_global_identity_late_addition": False,
            "feature_extraction_location": "256-D fused mean pooling after Dropout and before classifier",
        }
        config["round2a_run"] = {
            "run_id": run_id,
            "architecture_family": spec["family"],
            "shortcut_lambda": spec["lambda"],
            "training_seed": spec["seed"],
            "folds": [0, 1, 2, 3, 4],
            "output_subdir": run_id,
            "mhsa": spec["family"] == "deep_narrow_mhsa",
            "mhsa_heads": 4 if spec["family"] == "deep_narrow_mhsa" else None,
            "attention_dropout": 0.1 if spec["family"] == "deep_narrow_mhsa" else None,
            "terminal_residual": False,
            "token_sequence": ["f1", "f2", "f3", "f4", "f5"] if spec["family"] == "deep_narrow_mhsa" else None,
        }
        allowed = ["training_seed"] if spec["family"] == "baseline" else ["declared shortcut_lambda", "declared MHSA module"]
        config["provenance"] = {
            "source_phase2b_config": str(SOURCE),
            "source_phase2b_config_sha256": sha256(SOURCE),
            "allowed_differences": allowed,
            "round2a_instruction_sha256": INSTRUCTION_SHA256,
            "round2a_analysis_plan_sha256": sha256(PLAN),
            "step1_audit_status": step1["status"],
        }
        target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        created.append({"run_id": run_id, "config": str(target), "sha256": sha256(target)})

    manifest_path = ROUND_ROOT / "manifests" / "config_generation.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite {manifest_path}")
    manifest = {
        "phase": "Phase2D Round2A",
        "source_config": str(SOURCE),
        "source_config_sha256": sha256(SOURCE),
        "instruction_sha256": INSTRUCTION_SHA256,
        "analysis_plan_sha256": sha256(PLAN),
        "effective_configs": created,
        "new_mhsa_formal_runs": 15,
        "new_baseline_formal_runs": 10,
        "formal_runs": 25,
        "training_authorized": True,
        "stop_after_round2a": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PREPARED", "configs": len(created), "formal_runs": 25}))


if __name__ == "__main__":
    main()
