"""Static/preflight validation before any Phase2D formal training starts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from round1_common import PHASE2B_ROOT, ROUND_IDS, ROUND_ROOT, build_round_model, load_round_config
from screen_core import LeafDataset, eval_transform, fold_class_weights, load_config, split_indices


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = load_config()
    checks = []
    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            raise RuntimeError(f"{name}: {detail}")

    dataset_cfg = base["dataset"]
    dataset = LeafDataset(Path(dataset_cfg["train_csv"]), Path(dataset_cfg["root"]), eval_transform())
    phase2b_baseline = build_round_model("baseline_seed43", 176, pretrained=False)
    phase2b_deep = build_round_model("lambda_0_7_seed42", 176, pretrained=False)
    # Exact counts from the Phase2B #0/#1 formal run manifests.
    expected_counts = {"baseline": 57114448, "deep_narrow": 55076688}
    for run_id in ROUND_IDS:
        config = load_round_config(run_id)
        check(f"{run_id}_dataset_exact", config["dataset"] == base["dataset"], "dataset block equals Phase2B")
        check(f"{run_id}_split_exact", config["split"] == base["split"], "split block equals Phase2B")
        check(f"{run_id}_protocol_exact", config["common_training_protocol"] == base["common_training_protocol"], "training protocol block equals Phase2B")
        expected_seed = 43 if run_id == "baseline_seed43" else 44 if run_id == "baseline_seed44" else 42
        check(f"{run_id}_seed", config["seed"]["training_seed"] == expected_seed and not config["seed"]["seed_plus_fold"], f"seed={expected_seed}, same every fold")
        model = build_round_model(run_id, 176, pretrained=False)
        family = config["round1_run"]["architecture_family"]
        total = sum(p.numel() for p in model.parameters())
        check(f"{run_id}_parameter_count", total == expected_counts[family], f"total={total}")
        check(f"{run_id}_no_mhsa", not any(isinstance(m, nn.MultiheadAttention) for m in model.modules()), "no MultiheadAttention module")
        check(f"{run_id}_no_terminal_residual", config["round1_run"]["terminal_residual"] is False, "declared false; frozen Phase2B head")
        if family == "deep_narrow":
            expected_lambda = config["round1_run"]["shortcut_lambda"]
            check(f"{run_id}_lambda", model.head.shortcut_lambda == expected_lambda and expected_lambda in (0.7, 0.9), f"lambda={expected_lambda}")
            dummy = torch.randn(3, 1536)
            model.head.eval()
            with torch.inference_mode():
                logits, features, trace = model.head(dummy, return_trace=True)
            check(f"{run_id}_trace", trace["stacked_stages"] == (3,5,256) and trace["logits"] == (3,176) and features.shape == (3,256), str(trace))
        else:
            dummy = torch.randn(3,1536); model.head.eval()
            with torch.inference_mode(): logits,features,trace=model.head(dummy,return_trace=True)
            check(f"{run_id}_trace", trace["feature_pre_classifier"]==(3,1024) and trace["logits"]==(3,176), str(trace))
    for fold in range(5):
        train_idx, val_idx = split_indices(dataset.labels, fold, 42)
        saved = np.load(PHASE2B_ROOT / "#0" / f"fold_{fold}" / "split_indices.npz")
        check(f"fold{fold}_assignment", np.array_equal(train_idx,saved["train_indices"]) and np.array_equal(val_idx,saved["validation_indices"]), f"train={len(train_idx)}, val={len(val_idx)}")
        weights = fold_class_weights(dataset.labels, train_idx, 176).numpy()
        check(f"fold{fold}_class_weights", np.array_equal(weights,np.load(PHASE2B_ROOT/"#0"/f"fold_{fold}"/"class_weights.npy")), "exact Phase2B weights")
    source_hashes = {name:sha256(PHASE2B_ROOT/name) for name in ("train_one.py","screen_core.py")}
    result = {"all_passed":True,"checks":checks,"source_hashes":source_hashes,"formal_runs":20,
              "forbidden_variants_absent":True,"training_performed":False}
    audit = ROUND_ROOT / "manifests" / "preflight_audit.json"
    if audit.exists(): raise RuntimeError(f"refusing to overwrite {audit}")
    audit.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PREFLIGHT_PASS","checks":len(checks)}))


if __name__ == "__main__":
    main()
