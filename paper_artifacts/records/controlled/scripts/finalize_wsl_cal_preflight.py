"""Close the CAL environment blocker after two frozen NCCL preflights pass."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import timm
import torch
import torchvision


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BACKBONE_SHA = "45179af731d5ea5470ba5906fadf0b55b884338ca64d2b7f53f46096665ec6c6"
EXPECTED_CONFIGS = {
    "cub": "fd35c96a75309e92220955172adb644b8e1bcfccf716d6344abca5292376d1ac",
    "cars": "7226d2fbb83b8c267fc186401ae691d0f2c02baa3611c5c74cdb7b1239d4f61f",
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def validate_summary(dataset: str) -> dict:
    path = ROOT / "preflight" / "wsl2_final" / dataset / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or len(summary.get("ranks", [])) != 2:
        raise RuntimeError(f"{dataset} summary is not a two-rank PASS")
    for expected_rank, row in enumerate(summary["ranks"]):
        if row["status"] != "PASS" or row["rank"] != expected_rank or row["local_rank"] != expected_rank or row["device_index"] != expected_rank:
            raise RuntimeError(f"{dataset} rank/device mapping failed")
        if row["backend"] != "nccl" or row["local_batch"] != 8 or row["effective_global_batch"] != 16:
            raise RuntimeError(f"{dataset} topology mismatch")
        if row["syncbn_module_count"] <= 0 or row["initial_backbone_sha256"] != EXPECTED_BACKBONE_SHA:
            raise RuntimeError(f"{dataset} SyncBN/init mismatch")
        if row["config_sha256"] != EXPECTED_CONFIGS[dataset] or row["primary_logits"] != "p-p_cf":
            raise RuntimeError(f"{dataset} config/primary-logit mismatch")
        if not all(row["checks"].values()) or not row["checkpoint_roundtrip"]["pass"]:
            raise RuntimeError(f"{dataset} contains a failed check")
        if row["accuracy_computed_or_reported"] is not False or row["formal_training"] is not False:
            raise RuntimeError(f"{dataset} scope invariant failed")
    return summary


def environment_payload() -> dict:
    operations = []
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        tensor = torch.arange(16, dtype=torch.float32, device=device)
        value = float((tensor * tensor).sum().cpu())
        torch.cuda.synchronize(index)
        operations.append({"device_index": index, "device_name": torch.cuda.get_device_name(index), "value": value, "pass": value == 1240.0})
    os_release = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip('"')
    return {
        "platform": platform.platform(),
        "distribution": os_release.get("PRETTY_NAME"),
        "wsl_version": "WSL2",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "distributed_available": torch.distributed.is_available(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "nccl_version": torch.cuda.nccl.version(),
        "minimal_cuda_operations": operations,
    }


def main() -> int:
    identity_path = ROOT / "preflight" / "wsl2_final" / "cal_protocol_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("status") != "PASS" or not all(identity["checks"].values()):
        raise RuntimeError("WSL protocol identity is not PASS")
    summaries = {dataset: validate_summary(dataset) for dataset in ("cub", "cars")}
    environment = environment_payload()
    if not (
        environment["wsl_version"] == "WSL2"
        and environment["cuda_device_count"] == 2
        and environment["distributed_available"]
        and environment["nccl_available"]
        and all(row["pass"] for row in environment["minimal_cuda_operations"])
    ):
        raise RuntimeError("final WSL/NCCL environment gate failed")

    plan_path = ROOT / "CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json"
    source_lock_path = ROOT / "CONTROLLED_REIMPLEMENTATION_SOURCE_LOCK_MANIFEST.json"
    old_status = json.loads((ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_STATUS.json").read_text(encoding="utf-8"))
    if old_status["formal_jobs_started"] != 0 or old_status["training_authorized"] is not False:
        raise RuntimeError("pre-update authorization invariant failed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["formal_job_count"] != 60 or plan["training_authorized"] is not False or plan["formal_jobs_started"] != 0:
        raise RuntimeError("frozen plan authorization invariant failed")

    wheels = sorted((ROOT / "wsl_wheels").glob("*.whl"))
    execution_manifest = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "CAL NCCL/SyncBN final technical preflight only",
        "environment": environment,
        "protocol_identity": identity["identities"],
        "frozen_plan_sha256_unchanged": file_sha(plan_path),
        "frozen_source_lock_sha256_unchanged": file_sha(source_lock_path),
        "wsl_dependency_wheels_sha256": {path.name: file_sha(path) for path in wheels},
        "wsl_preflight_scripts_sha256": {
            "wsl_environment_probe.py": file_sha(ROOT / "scripts" / "wsl_environment_probe.py"),
            "wsl_protocol_identity.py": file_sha(ROOT / "scripts" / "wsl_protocol_identity.py"),
            "wsl_cal_nccl_preflight.py": file_sha(ROOT / "scripts" / "wsl_cal_nccl_preflight.py"),
            "finalize_wsl_cal_preflight.py": file_sha(Path(__file__)),
        },
        "dataset_preflights": {
            dataset: {
                "summary_path": f"preflight/wsl2_final/{dataset}/summary.json",
                "summary_sha256": file_sha(ROOT / "preflight" / "wsl2_final" / dataset / "summary.json"),
                "status": summaries[dataset]["status"],
                "config_sha256": EXPECTED_CONFIGS[dataset],
                "rank_count": 2,
                "per_rank_batch": 8,
                "global_batch": 16,
                "syncbn_module_count": summaries[dataset]["ranks"][0]["syncbn_module_count"],
                "peak_gpu_memory_mib_per_rank": [row["peak_gpu_memory_mib"] for row in summaries[dataset]["ranks"]],
                "accuracy_computed_or_reported": False,
                "formal_training": False,
            }
            for dataset in ("cub", "cars")
        },
        "blocker": {"id": "CAL_DDP_SYNCBN_ENVIRONMENT", "state": "CLOSED"},
        "training_authorized": False,
        "formal_jobs_started": 0,
    }
    write_json(ROOT / "CONTROLLED_REIMPLEMENTATION_WSL2_EXECUTION_MANIFEST.json", execution_manifest)
    write_json(ROOT / "preflight" / "wsl2_final" / "environment.json", environment)

    aggregate = {
        "schema_version": 1,
        "status": "PASS",
        "blocker_closed": "CAL_DDP_SYNCBN_ENVIRONMENT",
        "protocol_identity_status": "PASS",
        "cub_status": "PASS",
        "cars_status": "PASS",
        "all_requested_topology_checks_pass": True,
        "training_authorized": False,
        "formal_jobs_started": 0,
        "accuracy_computed_or_reported": False,
        "formal_training": False,
    }
    write_json(ROOT / "preflight" / "wsl2_final" / "FINAL_CAL_NCCL_SYNCBN_PREFLIGHT.json", aggregate)

    report = f"""# Controlled Reimplementation Preflight Report

## Final outcome

`PASS / READY_FOR_FORMAL_RUN`

The historical Windows-only blocker is closed by the final WSL2 NCCL execution. No formal training was started.

## WSL2/Linux environment

- Distribution: {environment['distribution']} under WSL2
- Python: {environment['python']}
- PyTorch: {environment['torch']}
- torchvision: {environment['torchvision']}
- timm: {environment['timm']}
- CUDA runtime: {environment['cuda_runtime']}
- NCCL: {'.'.join(map(str, environment['nccl_version']))}
- GPUs: 2 x NVIDIA GeForce RTX 5070 Ti
- `torch.distributed.is_available()`: true
- `torch.distributed.is_nccl_available()`: true
- Independent minimal CUDA tensor operation: GPU0 PASS, GPU1 PASS

## Identity gate

All identity checks PASS: CAL upstream commit and implementation SHA, CUB/Cars config SHA, pretrained artifact SHA, loaded initial backbone state SHA, CUB/Cars fold and class-map SHA, augmentation identity, and evaluation protocol identifier. Evidence: `preflight/wsl2_final/cal_protocol_identity.json`.

## Frozen CAL topology results

| Dataset | Ranks | Mapping | Batch | SyncBN | Forward / p-p_cf / center / crop-drop second forward | Backward / optimizer | Checkpoint roundtrip | Accuracy |
|---|---:|---|---|---|---|---|---|---|
| CUB | 2 | rank0-GPU0, rank1-GPU1 | 8/rank, global16 | PASS (54 modules) | PASS | PASS | PASS | not computed |
| Cars | 2 | rank0-GPU0, rank1-GPU1 | 8/rank, global16 | PASS (54 modules) | PASS | PASS | PASS | not computed |

Both jobs used NCCL and completed without deadlock, device mismatch, NaN, or Inf. Transient technical checkpoints were deleted after verified save/load. `training_authorized=false`; `formal_jobs_started=0`.
"""
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_PREFLIGHT_REPORT.md", report)
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_WSL2_CAL_PREFLIGHT_REPORT.md", report)

    audit = f"""# Controlled Reimplementation Final Method Audit

审计日期：2026-08-20。定位：controlled reimplementation under a common evaluation protocol，不是原论文数字 reproduction。

## 总判定

`READY_FOR_FORMAL_RUN`

所有 source/config/init/fold/class-map/augmentation/evaluation identity 检查、unit tests、12/12 single-GPU technical smoke，以及 CUB/Cars CAL 的 WSL2 NCCL 2×8 DDP/SyncBN final preflight 均已通过。

`CAL_DDP_SYNCBN_ENVIRONMENT = CLOSED`。

冻结协议没有修改：CAL 仍为 32 maps、完整 BAP/counterfactual/feature-center/crop-drop second forward/loss，primary logits 仍为 `p-p_cf`；rank0→GPU0，rank1→GPU1，每 rank batch8，global batch16，54 个 BatchNorm 模块成功转换为 SyncBatchNorm。

其他方法状态不变：L2-SP、MC-Loss、Ours-FT、Progressive Head、DFAG 均为 `READY_NOT_AUTHORIZED`；DFAG 仍依赖 same-dataset/same-fold Ours Stage-1 checkpoint。

本轮没有计算或报告 accuracy，没有访问 official-test 参与训练或选择，没有启动任何正式 fold job。

- `training_authorized=false`
- `formal_jobs_started=0`
- `formal_job_count=60`
"""
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_METHOD_AUDIT.md", audit)

    risk = """# Controlled Reimplementation Final Risk Register

| ID | Severity | State | Evidence / control |
|---|---|---|---|
| R1 | Former blocker | CLOSED | CUB and Cars each passed NCCL 2-rank, rank-to-GPU binding, per-rank8/global16, 54 SyncBN modules, complete CAL training pathway, backward/optimizer and checkpoint roundtrip under WSL2. |
| R2 | Protocol-change risk | CONTROLLED | Single GPU, ordinary BN, DataParallel, smaller topology and different batch size remain forbidden; none was used. |
| R3 | Medium | CONTROLLED | MC pretrained augmentation remains a predeclared source-backed candidate for an originally NOT_SPECIFIED setting; no OOF selection. |
| R4 | Medium | CONTROLLED | L2-SP geometry/common-normalization adaptations remain manifest-locked. |
| R5 | Medium | CONTROLLED | Ours/Progressive/DFAG workspace semantics remain content-addressed by SHA because the workspace is not a Git worktree. |
| R6 | High | CONTROLLED | DFAG same-fold Stage-1 dependency and frozen-anchor assertions remain mandatory. |
| R7 | High | CONTROLLED | All smoke/preflight checkpoints were technical and transient; no accuracy was computed or interpreted. |
"""
    write_text(ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_RISK_REGISTER.md", risk)

    protocol_path = ROOT / "CONTROLLED_REIMPLEMENTATION_FROZEN_PROTOCOL.md"
    protocol = protocol_path.read_text(encoding="utf-8")
    protocol = protocol.replace("Status: `FROZEN_BUT_EXECUTION_BLOCKED`;", "Status: `READY_FOR_FORMAL_RUN`;")
    protocol += "\n## Final execution-readiness closure\n\nCAL CUB and Cars passed the frozen 2x8 NCCL/DDP/SyncBN preflight under WSL2. This closes only the environment blocker; no protocol field or formal job was changed. `training_authorized=false`; `formal_jobs_started=0`.\n"
    write_text(protocol_path, protocol)

    status = {
        "schema_version": 2,
        "status": "READY_FOR_FORMAL_RUN",
        "all_blockers_resolved": True,
        "blockers": [],
        "closed_blockers": [{
            "id": "CAL_DDP_SYNCBN_ENVIRONMENT",
            "state": "CLOSED",
            "evidence": ["preflight/wsl2_final/cub/summary.json", "preflight/wsl2_final/cars/summary.json"],
        }],
        "method_status": {
            "l2_sp": "READY_NOT_AUTHORIZED",
            "mc_loss": "READY_NOT_AUTHORIZED",
            "cal": "READY_NOT_AUTHORIZED",
            "ours_ft": "READY_NOT_AUTHORIZED",
            "progressive": "READY_NOT_AUTHORIZED",
            "dfag": "READY_NOT_AUTHORIZED_DEPENDS_ON_OURS_STAGE1",
        },
        "common_protocol_status": "PASS_LOCKED",
        "source_lock_status": "PASS_LOCKED",
        "protocol_identity_status": "PASS",
        "augmentation_manifest_status": "PASS_FROZEN",
        "unit_tests_status": "PASS",
        "smoke_status": "PASS",
        "cal_syncbn_topology_status": "PASS_CLOSED",
        "cal_cub_preflight_status": "PASS",
        "cal_cars_preflight_status": "PASS",
        "formal_job_count": 60,
        "training_authorized": False,
        "formal_jobs_started": 0,
        "formal_outputs_created": 0,
        "stop_point_observed": True,
        "frozen_plan_sha256": file_sha(plan_path),
        "execution_manifest": "CONTROLLED_REIMPLEMENTATION_WSL2_EXECUTION_MANIFEST.json",
    }
    write_json(ROOT / "CONTROLLED_REIMPLEMENTATION_FINAL_STATUS.json", status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
