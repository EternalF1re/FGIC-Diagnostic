"""Generate the frozen configs, manifests, plan and human-readable handoff."""
from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import timm
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core import (  # noqa: E402
    BACKBONE_ID, DATASETS, FOLDS, IMAGENET_MEAN, IMAGENET_STD, METHODS, SOURCE_COMMITS,
    canonical_sha, fold_manifest, load_external_config, locate_pretrained_artifact,
    sha256_file,
)


ARTIFACT_SHA = "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb"
BACKBONE_SHA = "45179af731d5ea5470ba5906fadf0b55b884338ca64d2b7f53f46096665ec6c6"
WORKSPACE_SEMANTICS_SHA = "076a794afb42bb853770df914b837e58cd96158097d2b3da0958316055883f0c"
EXTERNAL_PROTOCOL_SHA = "7f667c125983ca82ec36214b727ee8fbe3215f21562abd260f4018559f87e3e5"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def common_config(method: str, dataset: str) -> dict[str, Any]:
    classes = 200 if dataset == "cub" else 196
    return {
        "schema_version": 1,
        "experiment_family": "controlled_reimplementation_under_common_evaluation_protocol",
        "method": method,
        "dataset": dataset,
        "num_classes": classes,
        "backbone": {
            "provider": BACKBONE_ID,
            "timm_model_name": "resnet50",
            "pretrained_artifact_sha256": ARTIFACT_SHA,
            "loaded_backbone_state_sha256": BACKBONE_SHA,
            "classifier_rule": "load identical pretrained num_classes=0 backbone first; then attach method-specific module",
            "input": [3, 299, 299],
            "final_convolution_feature": [2048, 10, 10],
            "pooled_feature": [2048],
            "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        },
        "split": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "training_seed_every_fold": 42,
        "checkpoint_selection": "strict maximum deterministic original-view held-out accuracy; exact tie keeps earlier checkpoint",
        "evaluation": {
            "pooled_oof": True, "each_development_sample_exactly_once": True,
            "original_view": True, "tta": False, "cross_fold_ensemble": False,
            "metrics": ["accuracy", "macro_f1", "balanced_accuracy"],
            "official_test_excluded_from_training_selection_and_oof": True,
        },
        "training_authorized": False,
        "formal_status": "PLANNED_NOT_AUTHORIZED",
    }


def training_config(method: str, dataset: str) -> dict[str, Any]:
    config = common_config(method, dataset)
    if method == "l2_sp":
        config["training"] = {
            "objective": "CE + 0.1/2*||w_inherited-w0||^2 + 0.01/2*||w_classifier||^2",
            "parameter_partition": "inherited Conv/Linear weights only; classifier weight ordinary L2; biases and BN affine excluded",
            "optimizer": "SGD", "momentum": 0.9, "lr": 0.01, "batch_size": 64,
            "duration_optimizer_iterations": 9000, "lr_step_iteration": 6000, "lr_multiplier": 0.1,
            "global_weight_decay": 0.0, "classifier_init": "Xavier", "label_smoothing": 0.0,
            "augmentation_id": "l2sp_source_faithful_299_v1",
        }
    elif method == "mc_loss":
        boundary = "0-151:10,152-199:11" if dataset == "cub" else "0-107:10,108-195:11"
        config["training"] = {
            "objective": "CE + 0.005*(L_dis - 10*L_div)", "channel_total": 2048,
            "channel_groups": boundary, "feature": "final 2048-channel convolution before GAP",
            "cwa": "per class group floor(xi/2) zeros and ceil(xi/2) ones, resampled every training iteration; disabled at inference",
            "optimizer": "SGD", "momentum": 0.9, "batch_size": 32,
            "backbone_lr": 1e-4, "classifier_lr": 1e-2, "epochs": 300,
            "lr_steps": [150, 225], "lr_multiplier": 0.1, "weight_decay": 5e-4,
            "label_smoothing": 0.0, "augmentation_id": "mc_source_backed_minimal_299_v1",
            "provenance_note": "momentum and batch size are source-backed scratch implementation choices for unspecified pretrained settings",
        }
    elif method == "cal":
        config["training"] = {
            "architecture": "32 attention maps; 1x1 Conv+BN+ReLU; BAP; signed sqrt; L2 norm; bias-free classifier",
            "primary_logits": "causal_effect_p_minus_p_cf", "raw_logits": "diagnostic_only",
            "epochs": 160, "optimizer": "SGD", "momentum": 0.9, "weight_decay": 1e-5,
            "lr": 1e-3, "schedule": "1e-3*0.9^((epoch+fraction)/2)",
            "feature_center_beta": 0.05, "crop_theta": [0.4, 0.6], "drop_theta": [0.2, 0.5],
            "fake_attention_train": "Uniform(0,2)", "fake_attention_eval": "all ones",
            "topology": {"global_batch": 16, "gpus": 2, "ddp_processes": 2, "per_process_batch": 8, "sync_batch_norm": True},
            "augmentation_id": "cal_official_train_299_v1",
            "execution_status": "BLOCKED_CURRENT_WINDOWS_PYTORCH_HAS_NO_NCCL_AND_GLOO_CUDA_DEVICE_UNSUPPORTED",
        }
    else:
        stage1 = {
            "epochs": 150, "batch_size": 32, "optimizer": "AdamW", "backbone_lr": 1e-4, "head_lr": 1e-3,
            "weight_decay": 1e-3, "warmup_epochs": 3,
            "scheduler": "CosineAnnealingLR(T_max=147,eta_min=1e-5) after warmup", "label_smoothing": 0.05,
            "mixup": {"alpha": 0.4, "probability": 0.4}, "cutmix": {"alpha": 1.0, "probability": 0.5},
            "none_probability": 0.1, "annealing": "scale=1 through zero-based epoch105; epoch>105 linear to 0 at epoch150",
        }
        stage2 = {
            "epochs": 60, "batch_size": 32, "optimizer": "AdamW", "all_parameter_lr": 5e-5,
            "weight_decay": 5e-4, "scheduler": "CosineAnnealingLR(T_max=60,eta_min=1e-6)",
            "label_smoothing": 0.03, "mixup": False, "cutmix": False,
            "bn_recalibration": "no reset; 50 augmented training-fold batches; full model train mode; no_grad; before optimizer",
        }
        config["training"] = {"stage1": stage1, "stage2": stage2, "augmentation_id": "current_external_cub_cars_299_v1"}
        if method == "ours_ft":
            config["architecture"] = "2048->1024 BN SiLU Drop0.2->1024 BN SiLU Drop0.2->classes"
        elif method == "progressive":
            config["architecture"] = "2048->256; five fixed shortcut blocks f_i=F_i(f_(i-1))+0.7*f_(i-1); mean f1..f5; no learnable lambda/MHSA/terminal residual"
        elif method == "dfag":
            config["architecture"] = "frozen anchor plus plastic Ours-FT; feature-wise gate 2048->128->2048; g*f_anchor+(1-g)*f_plastic"
            config["training"] = {
                "stage1": "reuse exact same-dataset same-fold selected Ours-FT Stage-1 checkpoint; no independent Stage-1",
                "stage2": stage2, "augmentation_id": "current_external_cub_cars_299_v1",
                "provenance_assertions": ["method=ours_ft", "same dataset", "same fold", "anchor/plastic initial state hashes equal"],
            }
    return config


def augmentation_manifest() -> str:
    return """# Controlled Reimplementation Augmentation Manifest

状态：`FROZEN_BEFORE_FORMAL_TRAINING`。所有 validation/OOF 均为 `Resize(299,299) -> ToTensor -> ImageNet Normalize`，单一原图路径；no TTA、no cross-fold ensemble。

| Method | Frozen 299 training transform | Source/adaptation status |
|---|---|---|
| L2-SP | Resize 342x342 (bilinear) -> ToTensor -> random brightness additive delta [-63/255,63/255] -> random saturation factor [0.5,1.5] -> random contrast factor [0.2,1.8] -> horizontal flip p=.5 -> random crop 299 -> ImageNet normalize | Official TF order is Resize256, the three random color ops, mirror, crop224. 342 is round(299*256/224). The official `blur` flag gates these color perturbations; it is not Gaussian blur. Each enabled color op is applied with a random magnitude, so application probability is 1. Dataset-mean subtraction is replaced by the locked timm ImageNet normalization required by the common pretrained artifact. No padding. |
| MC-Loss | Resize 299x299 (bilinear) -> RandomCrop299(padding=4) -> horizontal flip p=.5 -> ToTensor -> ImageNet normalize | Pretrained paper gives resize/input but not a complete executable augmentation (`NOT_SPECIFIED`). This predeclared minimal candidate uses the official from-scratch repository's resize/crop-padding/flip evidence, with only size and common-normalization adaptations. No color, vertical flip, or arbitrary rotation. |
| CAL | Resize341x341 (bilinear) -> RandomCrop299 -> horizontal flip p=.5 -> ColorJitter(brightness=.126,saturation=.5) -> ToTensor -> ImageNet normalize; plus method-internal attention crop theta(.4,.6), drop theta(.2,.5), and second forward | Direct official transform with `int(299/.875)=341`; method-internal crop/drop retained. Official eval crop/ensemble is not used under the common evaluation protocol. |
| Ours-FT / Progressive / DFAG | Resize299 -> horizontal flip p=.5 -> ColorJitter(.1 brightness/contrast/saturation/hue) -> RandomAffine(degrees=0,shear=5) -> ToTensor -> ImageNet normalize | Exact current audited CUB/Cars controlled transform. No vertical flip and no arbitrary rotation. Stage 2 retains this ordinary transform; Mixup/CutMix are disabled in Stage 2. |

No choice above was selected using smoke accuracy, held-out OOF, or official-test results.
"""


def main() -> int:
    configs_dir = ROOT / "configs"
    manifests_dir = ROOT / "manifests"
    planned_dir = ROOT / "planned_jobs"
    for directory in (configs_dir, manifests_dir, planned_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit = json.loads((ROOT / "preflight" / "unit_test_results.json").read_text(encoding="utf-8"))
    smoke = json.loads((ROOT / "preflight" / "smoke_results.json").read_text(encoding="utf-8"))
    folds = {dataset: fold_manifest(dataset) for dataset in DATASETS}
    for dataset, payload in folds.items():
        write_json(manifests_dir / f"{dataset}_fold_and_class_manifest.json", payload)

    configs: dict[tuple[str, str], dict[str, Any]] = {}
    config_hashes: dict[tuple[str, str], str] = {}
    for method in METHODS:
        for dataset in DATASETS:
            payload = training_config(method, dataset)
            path = configs_dir / f"{method}_{dataset}.json"
            write_json(path, payload)
            configs[(method, dataset)] = payload
            config_hashes[(method, dataset)] = sha256_file(path)

    implementation_files = [
        ROOT / "common" / "core.py",
        ROOT / "methods" / "l2_sp" / "model.py",
        ROOT / "methods" / "mc_loss" / "model.py",
        ROOT / "methods" / "cal" / "model.py",
        ROOT / "scripts" / "run_unit_tests.py",
        ROOT / "scripts" / "run_smoke.py",
        ROOT / "scripts" / "cal_ddp_preflight.py",
    ]
    upstream = {
        "l2_sp": {"remote": "http<LOCAL_PATH>", "commit": SOURCE_COMMITS["l2_sp"], "key_file_sha256": {"database/dataset_reader.py": "3995ea12798fbe7cfeb809c14b4154c6f4f8288aa1ebba208e9ea6c7951cda4e"}},
        "mc_loss": {"remote": "http<LOCAL_PATH>", "commit": SOURCE_COMMITS["mc_loss"], "key_file_sha256": {"CUB-200-2011.py": "3e19179e3437865f67c39aa63307c50e0edbfc3b733967e3dbbfd3c0faee9771", "CUB-200-2011_ResNet18.py": "cf52b28b9a92cb2f2623f69576297f786521232a03f350a0e945360b13a92c3f", "my_pooling.py": "b2d526efa446a205d272610e06ed2ac117d3d6084042ec6fe38b7f6504bdaa54"}},
        "cal": {"remote": "http<LOCAL_PATH>", "commit": SOURCE_COMMITS["cal"], "key_file_sha256": {"fgvc/models/cal.py": "ecc9c453a527c765daf2437ed479d64a1a3f76a380444265a38607e6213f4749", "fgvc/utils.py": "e94929f5aaa42df209eb177ed3092fedca585409e6b9bc33a4a20bffb8201748", "fgvc/train_distributed.py": "7f9242e68b319ef4a8ab050b6816b97cba5aff1f4aa41e85b9d253b55cd226ad"}},
    }
    artifact = locate_pretrained_artifact()
    source_lock = {
        "schema_version": 1,
        "status": "LOCKED",
        "upstream": upstream,
        "workspace_semantics": {
            "source_commit": f"workspace_snapshot_sha256:{WORKSPACE_SEMANTICS_SHA}",
            "cross_backbone_common_sha256": WORKSPACE_SEMANTICS_SHA,
            "protocol_core_sha256": "a4ec8e36e4850f6710d4b9854b315e77ff3e084ccd2bb28c3cac80c5906e6f42",
            "corrected_transform_sha256": "20e1b49193cfda2b3981ab1d58967390675c02c2f02c25c700da915b554b1758",
            "external_protocol_config_sha256": sha256_file(ROOT.parent / "final_external_protocol" / "final_external_protocol_config.json"),
        },
        "pretrained": {
            "provider": BACKBONE_ID, "timm_version": timm.__version__,
            "url": "http<LOCAL_PATH>",
            "hf_hub_id": "timm/resnet50.a1_in1k", "revision": "artifact-content-addressed-by-sha256",
            "local_artifact": str(artifact), "artifact_sha256": sha256_file(artifact),
            "loaded_backbone_state_dict_sha256": BACKBONE_SHA,
            "dummy_forward": {"input": [1, 3, 299, 299], "final_conv": [1, 2048, 10, 10], "pooled": [1, 2048]},
        },
        "implementation_files_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in implementation_files},
        "fold_manifest_sha256": {dataset: folds[dataset]["fold_manifest_sha256"] for dataset in DATASETS},
        "class_map_sha256": {dataset: folds[dataset]["class_map_sha256"] for dataset in DATASETS},
    }
    write_json(ROOT / "CONTROLLED_REIMPLEMENTATION_SOURCE_LOCK_MANIFEST.json", source_lock)

    jobs = []
    job_number = 0
    for method in METHODS:
        for dataset in DATASETS:
            for fold in FOLDS:
                job_number += 1
                source_commit = SOURCE_COMMITS.get(method, f"workspace_snapshot_sha256:{WORKSPACE_SEMANTICS_SHA}")
                dependency = f"ours_ft_{dataset}_fold{fold}" if method == "dfag" else None
                jobs.append({
                    "job_id": f"{method}_{dataset}_fold{fold}", "ordinal": job_number,
                    "method": method, "dataset": dataset, "fold": fold, "seed": 42,
                    "config_path": f"configs/{method}_{dataset}.json", "config_sha256": config_hashes[(method, dataset)],
                    "source_commit": source_commit, "pretrained_artifact_sha256": ARTIFACT_SHA,
                    "initial_backbone_state_sha256": BACKBONE_SHA,
                    "fold_manifest_sha256": folds[dataset]["fold_manifest_sha256"],
                    "class_map_sha256": folds[dataset]["class_map_sha256"],
                    "input_resolution": [299, 299],
                    "training_protocol_identifier": f"controlled_reimplementation_{method}_v1",
                    "evaluation_protocol_identifier": "pooled_oof_original_view_no_tta_no_fold_ensemble_v1",
                    "depends_on": dependency,
                    "resources": {"gpu_count": 2 if method == "cal" else 1, "ddp_processes": 2 if method == "cal" else 1, "per_process_batch": 8 if method == "cal" else None, "device_affinity_required": method == "cal"},
                    "execution_blocker": "CAL_DDP_SYNCBN_ENVIRONMENT" if method == "cal" else None,
                    "status": "PLANNED_NOT_AUTHORIZED",
                })
    plan = {
        "schema_version": 1, "experiment": "final_controlled_reimplementation",
        "formal_job_count": len(jobs), "training_authorized": False, "formal_jobs_started": 0,
        "scheduler_policy": "single-GPU independent jobs use shared dynamic queue; DFAG waits for same-fold Ours-FT; CAL requires exclusive two-GPU topology lease",
        "fail_closed": True, "jobs": jobs,
    }
    write_json(ROOT / "CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json", plan)
    write_json(planned_dir / "CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json", plan)

    hyper_rows = []
    for method in METHODS:
        for dataset in DATASETS:
            c = configs[(method, dataset)]
            hyper_rows.append({
                "method": method, "dataset": dataset, "backbone": BACKBONE_ID, "input": "299x299", "seed": 42,
                "duration": json.dumps(c["training"], ensure_ascii=False, sort_keys=True),
                "augmentation_id": c["training"].get("augmentation_id", ""),
                "checkpoint": c["checkpoint_selection"], "evaluation": "pooled OOF; original-view; no TTA; no ensemble",
                "config_sha256": config_hashes[(method, dataset)],
            })
    table_path = ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_HYPERPARAMETER_TABLE.csv"
    with table_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hyper_rows[0]))
        writer.writeheader()
        writer.writerows(hyper_rows)

    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_AUGMENTATION_MANIFEST.md", augmentation_manifest())

    audit = f"""# Controlled Reimplementation Final Method Audit

审计日期：2026-08-20。定位：controlled reimplementation under a common evaluation protocol，不是原论文数字 reproduction。正式训练未启动。

## 总判定

`NOT_READY_FOR_FORMAL_RUN`

代码、source locks、统一初始化、fold/class map、augmentation、unit tests、12/12 technical smoke 和 60-job plan 均已完成；但 CAL 的批准目标 `2 GPU / 2 DDP process / per-process 8 / global 16 / SyncBN` 在当前环境不可执行，因此 `all_blockers_resolved=false`。

## Phase 0 blocker 关闭情况

| Item | Result |
|---|---|
| Common ResNet-50 | PASS: timm {timm.__version__}, artifact `{ARTIFACT_SHA}`, loaded state `{BACKBONE_SHA}` |
| Same-backbone Ours/Progressive/DFAG | PASS: exact audited 2048-D heads reused; Progressive fixed lambda=.7, five blocks, no MHSA/terminal residual; DFAG same-fold Ours Stage-1 provenance and frozen anchor asserted |
| L2-SP choices | PASS: alpha=.1, beta=.01, standard weight-only partition, iteration 6000 LR step, exact 299 manifest |
| MC-Loss choices | PASS: exact nonuniform 2048 groups, pretrained objective, source-backed batch/momentum declaration, golden tests |
| CAL causal primary | PASS: `p-p_cf` is checkpoint/held-out/OOF primary; raw diagnostic only; training crop/drop retained |
| CAL DDP/SyncBN topology | **BLOCKED**: torch {torch.__version__} reports NCCL unavailable; Gloo CUDA process-group construction fails with `unsupported gloo device` before model forward |

## Method implementation and verification

| Method | Implementation | Unit | Smoke CUB/Cars | Formal readiness |
|---|---|---|---|---|
| L2-SP | `methods/l2_sp/model.py` | PASS | PASS/PASS | READY, not authorized |
| MC-Loss | `methods/mc_loss/model.py` | PASS | PASS/PASS | READY, not authorized |
| CAL | `methods/cal/model.py` | PASS | PASS/PASS | CODE READY; topology blocked |
| Ours-FT | `common/core.py` | PASS | PASS/PASS | READY, not authorized |
| Progressive | `common/core.py` | PASS | PASS/PASS | READY, not authorized |
| DFAG | `common/core.py` | PASS | PASS/PASS | READY after same-fold Ours Stage-1, not authorized |

L2-SP w0 is cloned/detached immediately after common backbone construction; 53 inherited Conv/Linear weights are SP-regularized, bias and BN affine are excluded, classifier L2 is separate, optimizer WD is zero. MC inference returns only standard GAP/classifier logits. CAL unit/single-GPU smoke verifies 32 maps, BAP, fake-attention modes, causal logits, train-only center update, crop/drop second forward and original-view evaluation.

No smoke accuracy is reported or used. Official test was not accessed for training, selection, OOF, or parameter decisions.
"""
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_METHOD_AUDIT.md", audit)

    protocol = f"""# Controlled Reimplementation Frozen Protocol

Status: `FROZEN_BUT_EXECUTION_BLOCKED`; `training_authorized=false`; `formal_jobs_started=0`.

## Common evaluation lock

- CUB development 5,994; Cars development 8,144. Official test is excluded from training, tuning, checkpoint selection, and OOF.
- SKF5, shuffle=true, random_state=42. Training seed is exactly 42 in every fold, never seed+fold.
- Input 299x299. Common pretrained artifact `{ARTIFACT_SHA}` and loaded backbone state `{BACKBONE_SHA}`.
- Deterministic original-view held-out inference; strict max accuracy, exact tie keeps earlier checkpoint.
- Complete pooled OOF with each development sample exactly once. Metrics: Accuracy, Macro-F1, Balanced Accuracy. No TTA or cross-fold ensemble.

## Method locks

- L2-SP: `CE + .1/2||w_inherited-w0||^2 + .01/2||w_classifier||^2`; SGD .01, momentum .9, batch64, 9000 optimizer iterations, x.1 at 6000, WD0, Xavier, no smoothing.
- MC-Loss: exact CUB/Cars contiguous Table-II groups covering 2048; `CE+.005*(L_dis-10L_div)`; SGD, batch32, momentum.9, backbone/FC LR 1e-4/1e-2, 300 epochs, steps150/225, WD5e-4, no smoothing.
- CAL: 32-map full method, causal primary, 160 epochs, SGD1e-3 momentum.9 WD1e-5, fractional schedule, beta.05, crop/drop and second forward; target topology 2x8 DDP+SyncBN.
- Ours/Progressive: current two-stage CUB/Cars controlled training semantics, batch32; Progressive is fixed lambda .7 five-block final architecture.
- DFAG: no independent Stage-1; exact same-dataset/fold selected Ours checkpoint initializes anchor/plastic; anchor frozen; feature-wise gate; Stage2 current semantics.

Augmentation is locked separately in `CONTROLLED_REIMPLEMENTATION_AUGMENTATION_MANIFEST.md`. Per-job exact JSON configs and SHA values are under `configs/` and in the 60-job plan.
"""
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_FROZEN_PROTOCOL.md", protocol)

    risk = """# Controlled Reimplementation Final Risk Register

| ID | Severity | State | Risk / evidence | Required action |
|---|---|---|---|---|
| R1 | BLOCKER | OPEN | Current Windows torch build has `nccl=False`; both default and interface-pinned Gloo CUDA DDP initialization fail with `unsupported gloo device`. CAL 2x8 SyncBN checks 1-10 cannot complete. | Preferred minimal non-protocol change: use a Linux/WSL2/native-Linux PyTorch build with CUDA NCCL and the same two GPUs, preserve every config/source/data/init SHA, and rerun `cal_ddp_preflight.py`. Do not authorize training before PASS. |
| R2 | HIGH | NOT APPROVED | Single-GPU global16 with ordinary BN would be executable but changes the approved SyncBN topology. | Only use after explicit human protocol amendment; generate new config SHA. |
| R3 | MEDIUM | CONTROLLED | MC pretrained augmentation was not completely specified by the paper. | Frozen minimal source-backed candidate is declared `NOT_SPECIFIED` adaptation; never select a variant from OOF. |
| R4 | MEDIUM | CONTROLLED | L2-SP official TensorFlow uses dataset-mean subtraction and 224 geometry. | Only dimensional geometry and common ImageNet normalization were adapted; exact operation order/ranges are manifest-locked. |
| R5 | MEDIUM | CONTROLLED | Workspace is not a Git worktree, so Ours/Progressive/DFAG have no native commit ID. | The exact audited semantics file and all new implementation files are locked by SHA-256 in source manifest. |
| R6 | HIGH | CONTROLLED | DFAG can leak/mismatch if it uses another fold's Stage-1 checkpoint. | Hard method/dataset/fold assertions plus identical anchor/plastic initial state hashes; job dependency encoded. |
| R7 | HIGH | CONTROLLED | Smoke checkpoints are technical and predictions are not valid research results. | Transient checkpoints deleted after roundtrip; report contains no accuracy and `formal_result=false`. |
"""
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_RISK_REGISTER.md", risk)

    preflight = f"""# Controlled Reimplementation Preflight Report

## Outcome

`FAIL_CLOSED / NOT_READY_FOR_FORMAL_RUN`

- Python: `{platform.python_version()}`; torch `{torch.__version__}`; timm `{timm.__version__}`.
- CUDA devices: 2 x NVIDIA GeForce RTX 5070 Ti detected.
- Common initialization/dummy forward: PASS.
- Unit tests: `{unit['status']}`.
- Technical smoke: `{smoke['status']}` (12/12).
- Formal jobs started: 0.

## CAL topology attempt

1. `torchrun --standalone --nproc_per_node=2`: launcher failed because elastic rendezvous requested libuv not present in this Windows build.
2. `USE_LIBUV=0` with torchrun: elastic c10d path still requested libuv.
3. PyTorch `multiprocessing.spawn` with normal Windows env rendezvous: passed rendezvous layer, then Gloo returned `makeDeviceForHostname(): unsupported gloo device`.
4. Same spawn with explicit physical Ethernet IPv4/interface: Gloo returned `makeDeviceForInterface(): unsupported gloo device`.

`torch.distributed.is_nccl_available()` is false. These failures occur during process-group creation, before CAL forward. No alternative topology was substituted. Rank evidence is retained under `preflight/cal_ddp/`.

## Minimum verifiable resolution

Use an NCCL-enabled Linux CUDA PyTorch environment, retain the exact configs, data/fold/class maps, pretrained artifact and source hashes, then rerun the two-rank script. A PASS must confirm all ten requested checks before authorization. Single-GPU BN is a protocol change and remains unapproved.
"""
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_PREFLIGHT_REPORT.md", preflight)

    unit_report = f"""# Controlled Reimplementation Unit Test Report

Overall: `{unit['status']}`; result SHA `{unit['result_sha256']}`.

| Suite | Status | Key facts |
|---|---|---|
| Common | {unit['tests']['common']['status']} | CUB/Cars IDs, fold membership, no overlap, class maps, deterministic 299 eval, pretrained SHA/state, pooled OOF semantics |
| L2-SP | {unit['tests']['l2_sp']['status']} | immutable w0; 53 inherited weights; SP value/gradient zero at w0; classifier L2 separate; WD0 |
| MC-Loss | {unit['tests']['mc_loss']['status']} | exact totals/boundaries; CWA counts/reproducibility; golden L_dis={unit['tests']['mc_loss']['details']['golden_l_dis']:.9f}; golden L_div={unit['tests']['mc_loss']['details']['golden_l_div']:.9f}; sign and inference tests |
| CAL | {unit['tests']['cal']['status']} | 32 maps, 65,536-D BAP, train/eval fake attention, causal logits, train-only center, second forward, no eval augmentation leakage |
| Ours/Progressive/DFAG | {unit['tests']['ours_progressive_dfag']['status']} | fixed .7/5 blocks, no obsolete modules, same-fold provenance, frozen anchor/trainable plastic, 2048-D gate |

Machine-readable details: `preflight/unit_test_results.json`.
"""
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_UNIT_TEST_REPORT.md", unit_report)

    smoke_lines = [
        "# Controlled Reimplementation Smoke Report", "", f"Overall: `{smoke['status']}`; 12/12 method x dataset technical runs passed.", "",
        "No smoke accuracy is reported or interpreted. Each row used one training batch and one original-view held-out inference; every transient full checkpoint was deleted after save/load verification.", "",
        "| Method | Dataset | Peak MiB | Forward/backward | Checkpoint/resume | Held-out/OOF schema |", "|---|---|---:|---|---|---|",
    ]
    for row in smoke["results"]:
        smoke_lines.append(f"| {row['method']} | {row['dataset']} | {row['peak_gpu_memory_mib']:.1f} | PASS | PASS | PASS |")
    smoke_lines.extend(["", f"Machine result SHA: `{smoke['result_sha256']}`. Details: `preflight/smoke_results.json`."])
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_SMOKE_REPORT.md", "\n".join(smoke_lines))

    status = {
        "schema_version": 1,
        "status": "NOT_READY_FOR_FORMAL_RUN",
        "all_blockers_resolved": False,
        "blockers": [{
            "id": "CAL_DDP_SYNCBN_ENVIRONMENT",
            "summary": "Current Windows PyTorch has no NCCL and cannot construct a CUDA Gloo process group; approved 2x8 DDP/SyncBN topology did not initialize.",
            "evidence": "preflight/cal_ddp/rank_0.json and rank_1.json",
        }],
        "method_status": {"l2_sp": "READY_NOT_AUTHORIZED", "mc_loss": "READY_NOT_AUTHORIZED", "cal": "CODE_READY_TOPOLOGY_BLOCKED", "ours_ft": "READY_NOT_AUTHORIZED", "progressive": "READY_NOT_AUTHORIZED", "dfag": "READY_NOT_AUTHORIZED_DEPENDS_ON_OURS_STAGE1"},
        "common_protocol_status": "PASS_LOCKED",
        "source_lock_status": "PASS_LOCKED",
        "augmentation_manifest_status": "PASS_FROZEN",
        "unit_tests_status": unit["status"],
        "smoke_status": smoke["status"],
        "cal_syncbn_topology_status": "BLOCKED",
        "formal_job_count": len(jobs),
        "training_authorized": False,
        "formal_jobs_started": 0,
        "formal_outputs_created": 0,
        "stop_point_observed": True,
    }
    write_json(ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_STATUS.json", status)
    print(json.dumps({"status": status["status"], "formal_job_count": len(jobs), "deliverables": 11, "training_authorized": False, "formal_jobs_started": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
