"""Final read-only source trace plus frozen adaptation-table generator."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
EXPECTED_PLAN_SHA = "1e4af9ba95615f58a08fe01371f9f0714ed4baf0e5768b5522ba1a28a5935e91"
SOURCE_ROOT = Path("<LOCAL_PATH>" if os.name != "nt" else r"<LOCAL_PATH>")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def contains(path: Path, snippets: list[str]) -> bool:
    text = path.read_text(encoding="utf-8")
    return all(snippet in text for snippet in snippets)


def config_rows() -> list[dict[str, Any]]:
    augmentation = {
        "l2_sp": "Resize342; official-style brightness/saturation/contrast perturbation; random horizontal flip; random crop299; ImageNet normalize",
        "mc_loss": "Resize299; random crop299 with padding4; random horizontal flip; ImageNet normalize",
        "cal": "Resize341; random crop299; random horizontal flip; ColorJitter(brightness=.126,saturation=.5); ImageNet normalize",
        "ours_ft": "Resize299; horizontal flip; ColorJitter(.1/.1/.1/.1); shear5; ImageNet normalize",
        "progressive": "Resize299; horizontal flip; ColorJitter(.1/.1/.1/.1); shear5; ImageNet normalize",
        "dfag": "Resize299; horizontal flip; ColorJitter(.1/.1/.1/.1); shear5; ImageNet normalize",
    }
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "configs").glob("*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        method, dataset, tr = cfg["method"], cfg["dataset"], cfg["training"]
        if method == "l2_sp":
            duration = f'{tr["duration_optimizer_iterations"]} optimizer iterations'
            batch = tr["batch_size"]
            optimizer = f'SGD; lr={tr["lr"]}; momentum={tr["momentum"]}; global WD={tr["global_weight_decay"]}'
            schedule = f'step at iteration {tr["lr_step_iteration"]}, x{tr["lr_multiplier"]}'
            smoothing = tr["label_smoothing"]
            coeff = "alpha_SP=0.1; beta_classifier_L2=0.01; biases and BN affine excluded"
        elif method == "mc_loss":
            duration = f'{tr["epochs"]} epochs'
            batch = tr["batch_size"]
            optimizer = f'SGD; backbone lr={tr["backbone_lr"]}; classifier lr={tr["classifier_lr"]}; momentum={tr["momentum"]}; WD={tr["weight_decay"]}'
            schedule = f'lr x{tr["lr_multiplier"]} at epochs {tr["lr_steps"]}'
            smoothing = tr["label_smoothing"]
            coeff = f'{tr["objective"]}; groups={tr["channel_groups"]}; CWA training-only'
        elif method == "cal":
            topology = tr["topology"]
            duration = f'{tr["epochs"]} epochs'
            batch = f'{topology["global_batch"]} global (2 x {topology["per_process_batch"]})'
            optimizer = f'SGD; lr={tr["lr"]}; momentum={tr["momentum"]}; WD={tr["weight_decay"]}'
            schedule = tr["schedule"]
            smoothing = 0.0
            coeff = f'beta={tr["feature_center_beta"]}; crop={tr["crop_theta"]}; drop={tr["drop_theta"]}; p-p_cf; 32 attentions; SyncBN'
        else:
            stage2 = tr["stage2"]
            if method == "dfag":
                duration = f'reuse same-fold Ours Stage1; Stage2 {stage2["epochs"]} epochs'
                batch = stage2["batch_size"]
                smoothing = f'Stage2={stage2["label_smoothing"]}'
            else:
                stage1 = tr["stage1"]
                duration = f'Stage1 {stage1["epochs"]} + Stage2 {stage2["epochs"]} epochs'
                batch = f'Stage1={stage1["batch_size"]}; Stage2={stage2["batch_size"]}'
                smoothing = f'Stage1={stage1["label_smoothing"]}; Stage2={stage2["label_smoothing"]}'
            optimizer = "AdamW; Stage1 backbone/head lr=1e-4/1e-3 WD=1e-3; Stage2 lr=5e-5 WD=5e-4"
            schedule = "Stage1 3-epoch warmup + cosine(T_max147,eta_min1e-5); Stage2 cosine(T_max60,eta_min1e-6)"
            coeff = cfg.get("architecture", "")
        rows.append({
            "method": method,
            "dataset": dataset,
            "backbone": cfg["backbone"]["provider"],
            "pretrained_initialization": f'ImageNet artifact {cfg["backbone"]["pretrained_artifact_sha256"]}; loaded state {cfg["backbone"]["loaded_backbone_state_sha256"]}',
            "batch_size": batch,
            "epochs_or_iterations": duration,
            "optimizer_lr_momentum_weight_decay": optimizer,
            "lr_schedule": schedule,
            "input_resolution": "299x299",
            "augmentation": augmentation[method],
            "label_smoothing": smoothing,
            "method_specific_coefficients": coeff,
            "checkpoint_selection": cfg["checkpoint_selection"],
            "training_seed": cfg["training_seed_every_fold"],
            "inference_protocol": "deterministic original-view pooled 5-fold OOF; no TTA; no cross-fold ensemble",
            "config_sha256": sha256(path),
        })
    return rows


def main() -> None:
    source_lock = json.loads((ROOT / "CONTROLLED_REIMPLEMENTATION_SOURCE_LOCK_MANIFEST.json").read_text(encoding="utf-8"))
    l2_repo, mc_repo = SOURCE_ROOT / "l2sp", SOURCE_ROOT / "mcloss"
    l2_impl = ROOT / "methods" / "l2_sp" / "model.py"
    mc_impl = ROOT / "methods" / "mc_loss" / "model.py"
    l2_checks = {
        "upstream_commit": git_head(l2_repo) == source_lock["upstream"]["l2_sp"]["commit"],
        "official_standard_partition": contains(l2_repo / "model" / "network_base.py", ["elif mode == 1:", "if 'weights' in v.name:", "tf.nn.l2_loss(v - pre_trained_weights)", "tf.nn.l2_loss(v)"]),
        "official_alpha_beta_command": contains(l2_repo / "run_classification" / "train.sh", ["--weight_decay_mode 1", "--weight_decay_rate 0.1", "--weight_decay_rate2 0.01"]),
        "frozen_coefficients": contains(l2_impl, ["alpha: float = 0.1", "beta: float = 0.01"]),
        "inherited_conv_linear_only": contains(l2_impl, ["module_types.get(module_name) in {nn.Conv2d, nn.Linear}", "parameter_name == \"weight\""]),
        "classifier_separate_l2": contains(l2_impl, ["classifier_raw = self.classifier.weight.square().sum()", "0.5 * beta * classifier_raw"]),
        "optimizer_no_global_wd": contains(l2_impl, ["weight_decay=0.0"]),
        "implementation_sha_locked": sha256(l2_impl) == source_lock["implementation_files_sha256"]["methods/l2_sp/model.py"],
    }
    mc_checks = {
        "upstream_commit": git_head(mc_repo) == source_lock["upstream"]["mc_loss"]["commit"],
        "official_training_only_mc": contains(mc_repo / "CUB-200-2011.py", ["if self.training:", "MC_loss = supervisor", "return x, loss, MC_loss", "return x, loss"]),
        "standard_eval_logits": contains(mc_impl, ["def standard_logits", "if not self.training or labels is None:", "return logits"]),
        "training_objective": contains(mc_impl, ["0.005 * (components[\"l_dis\"] - 10.0 * components[\"l_div\"])"]),
        "cub_groups": contains(mc_impl, ["[10] * 152 + [11] * 48"]),
        "cars_groups": contains(mc_impl, ["[10] * 108 + [11] * 88"]),
        "implementation_sha_locked": sha256(mc_impl) == source_lock["implementation_files_sha256"]["methods/mc_loss/model.py"],
    }
    plan_sha = sha256(ROOT / "CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json")
    rows = config_rows()
    table_path = ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_ADAPTATION_TABLE.csv"
    with table_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": 1,
        "formal_result": False,
        "l2sp_source_trace": "PASS" if all(l2_checks.values()) else "FAIL",
        "mcloss_inference_fidelity": "PASS" if all(mc_checks.values()) else "FAIL",
        "final_adaptation_table": "FROZEN" if len(rows) == 12 else "FAIL",
        "l2sp_checks": l2_checks,
        "mcloss_checks": mc_checks,
        "plan_sha256": plan_sha,
        "plan_sha_matches": plan_sha == EXPECTED_PLAN_SHA,
        "table_path": str(table_path),
        "table_sha256": sha256(table_path),
        "table_rows": len(rows),
        "common_protocol_note": "All methods share backbone, ImageNet initialization, folds, input, seed and OOF evaluation; method-specific training settings remain distinct.",
        "l2sp_value_note": "alpha=0.1 and beta=0.01 are source-backed, pre-specified values, not a unique universal/default optimum, and were not selected using CUB/Cars OOF results.",
    }
    result["status"] = "PASS" if result["plan_sha_matches"] and all(l2_checks.values()) and all(mc_checks.values()) and len(rows) == 12 else "FAIL"
    json_path = ROOT / "preflight" / "final_gate" / "source_and_table_gate.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# Controlled Reimplementation Final Source and Table Gate

- `L2SP_SOURCE_TRACE = {result['l2sp_source_trace']}`
- `MCLOSS_INFERENCE_FIDELITY = {result['mcloss_inference_fidelity']}`
- `FINAL_ADAPTATION_TABLE = {result['final_adaptation_table']}`
- frozen 60-job plan SHA: `{plan_sha}` ({'PASS' if result['plan_sha_matches'] else 'FAIL'})

## L2-SP source trace

The paper objective separates inherited-weight SP (`alpha`) from ordinary L2 on new parameters (`beta`). Its Figure 1 evaluates a grid rather than declaring a universal optimum. The locked official implementation uses the standard `mode == 1` partition at `model/network_base.py:138-156`; the locked official Dogs command specifies `alpha=0.1`, `beta=0.01` at `run_classification/train.sh:5`. Therefore the correct record is: **source-backed values pre-specified before the controlled experiments and not tuned on CUB/Cars OOF results**.

The controlled implementation applies SP only to inherited Conv/Linear weights, excludes bias and BN affine, applies separate ordinary L2 to `classifier.weight`, and uses optimizer weight decay zero. It does not double-count generic WD plus SP.

## MC-Loss inference fidelity

The locked official implementation computes the MC supervisor only inside `if self.training` (`CUB-200-2011.py:154-172`); evaluation returns the standard classifier output. No channel-selection, CWA, special channel aggregation, multi-branch inference, or other method-specific test-time operation is required. The controlled implementation's evaluation path is standard GAP plus classifier logits. Training retains the exact CUB/Cars 2048-channel grouping and `CE + 0.005*(L_dis - 10*L_div)`.

## Frozen table

`{table_path.name}` contains 12 dataset-specific rows (six methods x two datasets), including initialization, batch size, duration, optimizer, LR/schedule, momentum/WD, resolution, augmentation, smoothing, method coefficients, checkpoint rule, seed and inference protocol. The batch distinctions are explicit: L2-SP 64; MC-Loss 32; CAL global 16 (2 x 8); Ours-FT/Progressive/DFAG 32.
"""
    (ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_SOURCE_GATE.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
