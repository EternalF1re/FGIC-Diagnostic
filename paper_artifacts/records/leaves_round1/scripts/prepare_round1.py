"""Create the four immutable effective configs from the Phase2B protocol."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from round1_common import PHASE2B_ROOT, ROUND_ROOT


SOURCE = PHASE2B_ROOT / "configs" / "controlled_screen.json"
RUNS = {
    "baseline_seed43": {"seed": 43, "family": "baseline", "lambda": None, "subdir": "baseline_seed43"},
    "baseline_seed44": {"seed": 44, "family": "baseline", "lambda": None, "subdir": "baseline_seed44"},
    "lambda_0_7_seed42": {"seed": 42, "family": "deep_narrow", "lambda": 0.7, "subdir": "lambda_0_7_seed42"},
    "lambda_0_9_seed42": {"seed": 42, "family": "deep_narrow", "lambda": 0.9, "subdir": "lambda_0_9_seed42"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    created = []
    for run_id, spec in RUNS.items():
        target = ROUND_ROOT / "configs" / f"{run_id}.json"
        if target.exists():
            raise RuntimeError(f"refusing to overwrite existing config: {target}")
        config = copy.deepcopy(base)
        config["schema_version"] = 1
        config["experiment_id"] = "PHASE2D_ROUND1"
        config["scope"] = list(RUNS)
        config["forbidden_scope"] = ["#3", "#4", "#5", "#6", "DFAG retraining", "lambda=0.5", "lambda=0.8", "other lambda"]
        config["seed"]["training_seed"] = spec["seed"]
        config["seed"]["same_training_seed_every_fold"] = True
        config["seed"]["seed_plus_fold"] = False
        config["round1_run"] = {
            "run_id": run_id,
            "architecture_family": spec["family"],
            "shortcut_lambda": spec["lambda"],
            "training_seed": spec["seed"],
            "folds": [0, 1, 2, 3, 4],
            "output_subdir": spec["subdir"],
            "mhsa": False,
            "terminal_residual": False,
            "aggregation": "baseline head" if spec["family"] == "baseline" else "unweighted arithmetic mean of five post-shortcut stages",
        }
        config["provenance"] = {
            "source_phase2b_config": str(SOURCE),
            "source_phase2b_config_sha256": sha256(SOURCE),
            "allowed_differences": ["training_seed"] if spec["family"] == "baseline" else ["declared shortcut_lambda/run identifier"],
            "round1_instruction_sha256": "3674FEFF583A569EF1066206BDE9B13810C0CCD06CEE4ECF9189A036368EF3B9",
        }
        target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        created.append({"run_id": run_id, "config": str(target), "sha256": sha256(target)})
    manifest = {
        "phase": "Phase2D Round1",
        "source_config": str(SOURCE),
        "source_config_sha256": sha256(SOURCE),
        "effective_configs": created,
        "formal_runs": 20,
        "training_authorized": True,
        "stop_after_round1": True,
    }
    (ROUND_ROOT / "manifests" / "config_generation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PREPARED", "configs": len(created)}))


if __name__ == "__main__":
    main()
