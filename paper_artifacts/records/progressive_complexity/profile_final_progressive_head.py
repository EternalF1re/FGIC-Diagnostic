"""Independent inference-complexity profile for the locked Progressive Head.

This script performs no training, backward pass, BN refresh, checkpoint write,
DFAG execution, TTA, or ensemble evaluation.  It imports the frozen Phase2B
head implementation and profiles Ours-FT and the final Progressive Head in the
same process with the historical THOP 2x-MAC FLOPs convention.
"""
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import thop
import torch
from thop import profile
from torch import Tensor, nn


OUTPUT = Path(__file__).resolve().parent
VALIDATION_ROOT = OUTPUT.parent
PHASE2B_ROOT = VALIDATION_ROOT / "phase2b_screen"
if str(PHASE2B_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2B_ROOT))

from screen_core import (  # noqa: E402
    BaselineHead,
    MappingBlock,
    MatchedShortcutHead,
    ScreenModel,
)


BASELINE_PARAMS_REFERENCE = 57_114_448
BASELINE_FLOPS_G_REFERENCE = 26.314805824
BASELINE_LATENCY_MS_REFERENCE = 13.614704036712647
BASELINE_PEAK_MIB_REFERENCE = 259_919_360 / (1024.0 ** 2)
HISTORICAL_PROGRESSIVE_PARAMS_APPROX = 55_076_688
NUM_CLASSES = 176
INPUT_SHAPE = (1, 3, 299, 299)
WARMUP = 10
REPEATS = 30


class LogitsOnly(nn.Module):
    """Expose the exact deployed logits route while retaining the frozen core."""

    def __init__(self, core: ScreenModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, images: Tensor) -> Tensor:
        logits, _ = self.core(images)
        return logits


def build_core(configuration: str) -> ScreenModel:
    if configuration == "Ours-FT":
        return ScreenModel("#0", NUM_CLASSES, pretrained=False)
    if configuration == "Progressive Head":
        core = ScreenModel("#1", NUM_CLASSES, pretrained=False)
        core.head.shortcut_lambda = 0.7
        return core
    raise ValueError(configuration)


def count_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def expected_parameter_formulas() -> Dict[str, int]:
    baseline_head = (
        (1536 * 1024 + 1024) + 2 * 1024
        + (1024 * 1024 + 1024) + 2 * 1024
        + (1024 * NUM_CLASSES + NUM_CLASSES)
    )
    progressive_head = (
        (1536 * 256 + 256)
        + 5 * ((256 * 256 + 256) + 2 * 256)
        + (256 * NUM_CLASSES + NUM_CLASSES)
    )
    return {"baseline_head": baseline_head, "progressive_head": progressive_head}


def architecture_audit() -> Dict[str, Any]:
    baseline = build_core("Ours-FT").eval()
    progressive = build_core("Progressive Head").eval()
    formulas = expected_parameter_formulas()
    checks: Dict[str, bool] = {}

    checks["backbone_is_inception_resnet_v2"] = (
        progressive.backbone.__class__.__name__ == "InceptionResnetV2"
    )
    checks["backbone_pooled_dimension_1536"] = progressive.backbone.num_features == 1536
    checks["single_backbone_branch"] = set(dict(progressive.named_children())) == {"backbone", "head"}
    checks["head_is_frozen_matched_shortcut_implementation"] = isinstance(
        progressive.head, MatchedShortcutHead
    )
    checks["projection_1536_to_256"] = (
        progressive.head.input_projection.in_features == 1536
        and progressive.head.input_projection.out_features == 256
    )
    checks["five_mapping_blocks"] = len(progressive.head.mapping_blocks) == 5
    checks["shortcut_lambda_0_7"] = progressive.head.shortcut_lambda == 0.7
    expected_types = (nn.Linear, nn.BatchNorm1d, nn.SiLU, nn.Dropout)
    checks["mapping_sequence_linear_bn_silu_dropout"] = all(
        isinstance(block, MappingBlock)
        and tuple(type(module) for module in block.mapping) == expected_types
        and block.mapping[0].in_features == 256
        and block.mapping[0].out_features == 256
        and block.mapping[3].p == 0.3
        for block in progressive.head.mapping_blocks
    )
    checks["aggregation_dropout_0_2"] = progressive.head.aggregation_dropout.p == 0.2
    checks["classifier_256_to_176"] = (
        progressive.head.classifier.in_features == 256
        and progressive.head.classifier.out_features == NUM_CLASSES
    )
    checks["mhsa_absent"] = not any(
        isinstance(module, nn.MultiheadAttention) for module in progressive.modules()
    )
    checks["mhsa_layernorm_absent"] = not any(
        isinstance(module, nn.LayerNorm) for module in progressive.head.modules()
    )

    # Functional route audit: the returned feature/logits must equal mean(f1..f5)
    # followed only by aggregation dropout (identity in eval) and classifier.
    torch.manual_seed(42)
    x0 = torch.randn(2, 1536)
    with torch.no_grad():
        actual_logits, actual_feature = progressive.head(x0)
        current = progressive.head.input_projection(x0)
        stages = []
        for block in progressive.head.mapping_blocks:
            current = block(current, progressive.head.shortcut_lambda)
            stages.append(current)
        expected_feature = progressive.head.aggregation_dropout(torch.stack(stages, dim=1).mean(dim=1))
        expected_logits = progressive.head.classifier(expected_feature)
    checks["mean_f1_to_f5_exact"] = torch.equal(actual_feature, expected_feature)
    checks["terminal_residual_absent"] = torch.equal(actual_logits, expected_logits)
    forward_source = inspect.getsource(MatchedShortcutHead.forward)
    checks["forward_source_uses_stage_mean"] = "stacked.mean(dim=1)" in forward_source

    baseline_head_params = count_parameters(baseline.head)
    progressive_head_params = count_parameters(progressive.head)
    baseline_backbone_params = count_parameters(baseline.backbone)
    progressive_backbone_params = count_parameters(progressive.backbone)
    baseline_total_params = count_parameters(baseline)
    progressive_total_params = count_parameters(progressive)
    checks["baseline_head_formula_match"] = baseline_head_params == formulas["baseline_head"]
    checks["progressive_head_formula_match"] = progressive_head_params == formulas["progressive_head"]
    checks["same_backbone_parameter_count"] = baseline_backbone_params == progressive_backbone_params
    checks["baseline_total_reference_exact"] = baseline_total_params == BASELINE_PARAMS_REFERENCE
    checks["progressive_total_accounting_exact"] = (
        progressive_total_params == progressive_backbone_params + progressive_head_params
    )

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "architecture": {
            "backbone": "Inception-ResNet-v2",
            "backbone_pooled_dimension": 1536,
            "projection": "1536 -> 256",
            "hidden_dimension": 256,
            "mapping_blocks": 5,
            "mapping_block": "Linear -> BatchNorm1d -> SiLU -> Dropout(p=0.3)",
            "shortcut_formula": "f_i = F_i(f_(i-1)) + 0.7 * f_(i-1)",
            "aggregation": "mean(f1,...,f5) -> Dropout(p=0.2)",
            "classifier": "256 -> 176",
            "mhsa": False,
            "terminal_residual": False,
            "dfag": False,
            "branches": 1,
        },
        "counts": {
            "backbone_parameters": progressive_backbone_params,
            "baseline_head_parameters": baseline_head_params,
            "baseline_total_parameters": baseline_total_params,
            "progressive_head_parameters": progressive_head_params,
            "progressive_total_parameters": progressive_total_params,
            "historical_progressive_sanity_reference": HISTORICAL_PROGRESSIVE_PARAMS_APPROX,
        },
        "checks": checks,
    }
    del baseline, progressive
    gc.collect()
    return result


def query_gpu(index: int) -> Tuple[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi", "--id", str(index),
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.strip().splitlines()[0].split(",")
    return int(values[0].strip()), int(values[1].strip())


def wait_for_pipeline(path: Path, poll_seconds: int) -> None:
    last_report = 0.0
    while True:
        state = json.loads(path.read_text(encoding="utf-8"))
        status = state.get("status")
        now = time.time()
        if status == "COMPLETE":
            print(json.dumps({"event": "UPSTREAM_COMPLETE", "time": now}), flush=True)
            return
        if status == "FAILED":
            raise RuntimeError(f"upstream pipeline failed: {state}")
        if now - last_report >= 900:
            print(json.dumps({"event": "WAITING_FOR_UPSTREAM", "status": status, "time": now}), flush=True)
            last_report = now
        time.sleep(poll_seconds)


def wait_for_idle_gpu(index: int, poll_seconds: int, required_samples: int = 3) -> None:
    consecutive = 0
    while consecutive < required_samples:
        utilization, memory_mib = query_gpu(index)
        idle = utilization <= 10 and memory_mib <= 1500
        consecutive = consecutive + 1 if idle else 0
        print(
            json.dumps(
                {
                    "event": "GPU_IDLE_CHECK",
                    "gpu": index,
                    "utilization_percent": utilization,
                    "memory_used_mib": memory_mib,
                    "idle": idle,
                    "consecutive_idle_samples": consecutive,
                }
            ),
            flush=True,
        )
        if consecutive < required_samples:
            time.sleep(poll_seconds)


def latency_and_memory(model: nn.Module, device: torch.device) -> Dict[str, Any]:
    dummy = torch.randn(*INPUT_SHAPE, device=device, dtype=torch.float32)
    model.eval().float()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(dummy)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        elapsed = []
        for _ in range(REPEATS):
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(dummy)
            end.record()
            torch.cuda.synchronize(device)
            elapsed.append(float(start.elapsed_time(end)))
        peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "latency_mean_ms": float(statistics.fmean(elapsed)),
        "latency_sd_ms": float(statistics.stdev(elapsed)),
        "latency_samples_ms": elapsed,
        "peak_memory_bytes": peak,
        "peak_memory_mib": peak / (1024.0 ** 2),
        "allocated_before_timed_forward_bytes": allocated_before,
        "warmup": WARMUP,
        "measured_forwards": REPEATS,
    }


def profile_model(configuration: str, device: torch.device) -> Dict[str, Any]:
    core = build_core(configuration).eval().requires_grad_(False).float().to(device)
    deployed = LogitsOnly(core).eval().requires_grad_(False)
    resident = count_parameters(deployed)
    head = count_parameters(core.head)
    backbone = count_parameters(core.backbone)
    dummy = torch.randn(*INPUT_SHAPE, device=device, dtype=torch.float32)
    with torch.no_grad():
        macs, thop_parameters = profile(deployed, inputs=(dummy,), verbose=False)
    runtime = latency_and_memory(deployed, device)
    row = {
        "configuration": configuration,
        "branches": 1,
        "resident_params_exact": resident,
        "resident_params_m": resident / 1e6,
        "backbone_params_exact": backbone,
        "head_params_exact": head,
        "head_params_m": head / 1e6,
        "thop_reported_params": int(thop_parameters),
        "macs": float(macs),
        "macs_g": float(macs / 1e9),
        "flops": float(2.0 * macs),
        "flops_g": float(2.0 * macs / 1e9),
        **runtime,
    }
    del dummy, deployed, core
    torch.cuda.empty_cache()
    gc.collect()
    return row


def difference(progressive: Dict[str, Any], baseline: Dict[str, Any], field: str) -> Dict[str, float]:
    absolute = float(progressive[field] - baseline[field])
    return {
        "absolute": absolute,
        "percent": 100.0 * absolute / float(baseline[field]),
    }


def write_outputs(audit: Dict[str, Any], baseline: Dict[str, Any], progressive: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    relative = {
        "total_params": difference(progressive, baseline, "resident_params_exact"),
        "head_params": difference(progressive, baseline, "head_params_exact"),
        "flops": difference(progressive, baseline, "flops_g"),
        "latency": difference(progressive, baseline, "latency_mean_ms"),
        "peak_memory": difference(progressive, baseline, "peak_memory_mib"),
    }
    sanity = {
        "params_exact": baseline["resident_params_exact"] == BASELINE_PARAMS_REFERENCE,
        "flops_within_0_1_percent": abs(
            baseline["flops_g"] / BASELINE_FLOPS_G_REFERENCE - 1.0
        ) <= 0.001,
        "latency_within_25_percent": abs(
            baseline["latency_mean_ms"] / BASELINE_LATENCY_MS_REFERENCE - 1.0
        ) <= 0.25,
        "peak_memory_within_15_percent": abs(
            baseline["peak_memory_mib"] / BASELINE_PEAK_MIB_REFERENCE - 1.0
        ) <= 0.15,
    }
    integrity = audit["status"] == "PASS" and all(sanity.values())
    payload: Dict[str, Any] = {
        "configuration": "Progressive Head",
        "branches": 1,
        "resident_params_exact": progressive["resident_params_exact"],
        "resident_params_m": progressive["resident_params_m"],
        "head_params_exact": progressive["head_params_exact"],
        "head_params_m": progressive["head_params_m"],
        "flops_g": progressive["flops_g"],
        "latency_mean_ms": progressive["latency_mean_ms"],
        "latency_sd_ms": progressive["latency_sd_ms"],
        "peak_memory_mib": progressive["peak_memory_mib"],
        "input_resolution": [299, 299],
        "batch_size": 1,
        "precision": "FP32",
        "warmup": WARMUP,
        "measured_forwards": REPEATS,
        "architecture_audit": audit,
        "ours_ft_sanity": {
            "reference": {
                "resident_params_exact": BASELINE_PARAMS_REFERENCE,
                "flops_g": BASELINE_FLOPS_G_REFERENCE,
                "latency_mean_ms": BASELINE_LATENCY_MS_REFERENCE,
                "peak_memory_mib": BASELINE_PEAK_MIB_REFERENCE,
            },
            "measured": baseline,
            "checks": sanity,
            "status": "PASS" if all(sanity.values()) else "FAIL",
        },
        "progressive_profile": progressive,
        "relative_to_ours_ft": relative,
        "environment": {
            "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_index": device.index,
            "thop": thop.__version__,
            "profiler_call": "thop.profile(model, inputs=(1x3x299x299 FP32,), verbose=False)",
            "flops_convention": "2 x THOP MACs, identical to the historical complexity table",
            "profiler_limitation": "THOP module hooks may omit unsupported functional operations; both rows use the identical route and version.",
            "latency_protocol": "torch.no_grad; 10 warm-up; 30 forwards; CUDA synchronize immediately before and after every measured forward",
            "memory_protocol": "absolute torch.cuda.max_memory_allocated after reset_peak_memory_stats",
        },
        "forbidden_actions": {
            "training": False,
            "optimizer": False,
            "backward": False,
            "bn_refresh": False,
            "checkpoint_modification": False,
            "dfag": False,
            "mhsa": False,
            "terminal_residual": False,
            "tta": False,
            "five_fold_ensemble": False,
        },
        "profiling_integrity": "PASS" if integrity else "FAIL",
    }
    (OUTPUT / "progressive_head_complexity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = [
        "configuration", "branches", "resident_params_exact", "resident_params_m",
        "backbone_params_exact", "head_params_exact", "head_params_m", "thop_reported_params",
        "macs", "macs_g", "flops", "flops_g", "latency_mean_ms", "latency_sd_ms",
        "peak_memory_bytes", "peak_memory_mib", "warmup", "measured_forwards",
        "relative_to_oursft_absolute", "relative_to_oursft_percent",
    ]
    csv_rows = []
    for row in (baseline, progressive):
        clean = {key: row.get(key) for key in fields}
        if row["configuration"] == "Ours-FT":
            clean["relative_to_oursft_absolute"] = 0.0
            clean["relative_to_oursft_percent"] = 0.0
        else:
            clean["relative_to_oursft_absolute"] = relative["total_params"]["absolute"]
            clean["relative_to_oursft_percent"] = relative["total_params"]["percent"]
        csv_rows.append(clean)
    with (OUTPUT / "progressive_head_complexity.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    def delta_text(key: str, unit: str = "") -> str:
        value = relative[key]
        return f"{value['absolute']:+,.6f}{unit} ({value['percent']:+.3f}%)"

    report = f"""# Final Progressive Head Complexity Audit

Architecture audit: **{audit['status']}**

- Backbone: Inception-ResNet-v2; pooled dimension 1536
- Projection: 1536 -> 256; H=256; N=5; shortcut lambda=0.7
- Mapping block: Linear -> BatchNorm1d -> SiLU -> Dropout(p=0.3)
- Aggregation: mean(f1,...,f5) -> Dropout(p=0.2) -> Linear(256,176)
- MHSA absent; terminal residual absent; DFAG absent; branch count=1
- Backbone parameters: {audit['counts']['backbone_parameters']:,}
- Progressive head parameters: {progressive['head_params_exact']:,}
- Total resident parameters: {progressive['resident_params_exact']:,}

## Ours-FT sanity

- Resident Params: {baseline['resident_params_exact']:,} ({baseline['resident_params_m']:.6f} M)
- Head Params: {baseline['head_params_exact']:,} ({baseline['head_params_m']:.6f} M)
- FLOPs: {baseline['flops_g']:.9f} G
- Latency: {baseline['latency_mean_ms']:.6f} +/- {baseline['latency_sd_ms']:.6f} ms
- Peak Mem: {baseline['peak_memory_mib']:.6f} MiB
- Sanity status: **{payload['ours_ft_sanity']['status']}**

## Progressive Head

- Resident Params: {progressive['resident_params_exact']:,} ({progressive['resident_params_m']:.6f} M)
- Head Params: {progressive['head_params_exact']:,} ({progressive['head_params_m']:.6f} M)
- FLOPs: {progressive['flops_g']:.9f} G
- Latency mean +/- sample SD: {progressive['latency_mean_ms']:.6f} +/- {progressive['latency_sd_ms']:.6f} ms
- Peak Mem: {progressive['peak_memory_mib']:.6f} MiB

## Relative to Ours-FT (Progressive minus Ours-FT)

- Total Params: {relative['total_params']['absolute']:+,.0f} ({relative['total_params']['percent']:+.3f}%)
- Head Params: {relative['head_params']['absolute']:+,.0f} ({relative['head_params']['percent']:+.3f}%)
- FLOPs: {delta_text('flops', ' G')}
- Latency: {delta_text('latency', ' ms')}
- Peak Mem: {delta_text('peak_memory', ' MiB')}

## Profiling integrity

Status: **{payload['profiling_integrity']}**

- Hardware: {payload['environment']['gpu']}
- Input: 1 x 3 x 299 x 299, FP32, eval mode, torch.no_grad
- Latency: 10 warm-up and 30 measured forwards, synchronized before and after every forward
- FLOPs: THOP {thop.__version__}, reported as 2 x MACs to match the historical table
- Memory: absolute peak allocated memory from torch.cuda.max_memory_allocated
- No training, optimizer, backward, BN refresh, checkpoint modification, DFAG, MHSA, terminal residual, TTA, or ensemble was run.
"""
    (OUTPUT / "FINAL_PROGRESSIVE_HEAD_COMPLEXITY_AUDIT.md").write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--wait-for-pipeline", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    state_path = OUTPUT / "profiling_state.json"
    try:
        audit = architecture_audit()
        (OUTPUT / "architecture_preflight.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "ARCHITECTURE_AUDIT", **audit}, ensure_ascii=False), flush=True)
        if audit["status"] != "PASS":
            raise RuntimeError("architecture audit failed; profiling stopped")
        if args.static_only:
            state_path.write_text(
                json.dumps({"status": "STATIC_PREFLIGHT_PASS", "time": time.time()}, indent=2) + "\n",
                encoding="utf-8",
            )
            return

        state_path.write_text(
            json.dumps({"status": "WAITING_FOR_EXCLUSIVE_GPU", "time": time.time()}, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.wait_for_pipeline is not None:
            wait_for_pipeline(args.wait_for_pipeline.resolve(), args.poll_seconds)
        device = torch.device(args.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("an explicit CUDA device such as cuda:0 is required")
        wait_for_idle_gpu(device.index, max(5, min(args.poll_seconds, 60)))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5070 Ti":
            raise RuntimeError(f"unexpected GPU: {torch.cuda.get_device_name(device)}")
        torch.manual_seed(42)
        np.random.seed(42)
        state_path.write_text(
            json.dumps({"status": "RUNNING_PROFILE", "time": time.time(), "device": str(device)}, indent=2) + "\n",
            encoding="utf-8",
        )
        baseline = profile_model("Ours-FT", device)
        # Fail closed before measuring the new row if structural FLOPs/params do
        # not reproduce the historical baseline route.
        if baseline["resident_params_exact"] != BASELINE_PARAMS_REFERENCE:
            raise RuntimeError(f"baseline parameter mismatch: {baseline['resident_params_exact']}")
        if abs(baseline["flops_g"] / BASELINE_FLOPS_G_REFERENCE - 1.0) > 0.001:
            raise RuntimeError(f"baseline FLOPs mismatch: {baseline['flops_g']}")
        progressive = profile_model("Progressive Head", device)
        payload = write_outputs(audit, baseline, progressive, device)
        status = "COMPLETE_PASS" if payload["profiling_integrity"] == "PASS" else "COMPLETE_FAIL"
        state_path.write_text(
            json.dumps({"status": status, "time": time.time()}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "PROFILE_COMPLETE", "status": status, "payload": payload}, ensure_ascii=False), flush=True)
        if payload["profiling_integrity"] != "PASS":
            raise SystemExit(2)
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            state_path.write_text(
                json.dumps({"status": "FAILED", "time": time.time(), "error": repr(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
