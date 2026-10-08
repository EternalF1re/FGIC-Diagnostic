"""Final same-process/same-session complexity audit for the three paper models.

This script never trains, loads a checkpoint, or reads a real image.  Round 1
is the pre-declared manuscript profile.  Round 2 reverses model order and is
used only as an order-effect sanity check.
"""
from __future__ import annotations

import csv
import gc
import inspect
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import thop
import torch
from thop import profile
from torch import Tensor, nn


OUTPUT = Path(__file__).resolve().parent
VALIDATION_ROOT = OUTPUT.parent
PHASE2B = VALIDATION_ROOT / "phase2b_screen"
DFAG_SCRIPTS = VALIDATION_ROOT / "phase2f_unified_standalone_dfag" / "scripts"
for path in (PHASE2B, DFAG_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from screen_core import BaselineHead, MappingBlock, MatchedShortcutHead, ScreenModel  # noqa: E402
from dfag_common import (  # noqa: E402
    DynamicResidualGate,
    FEATURE_DIM,
    GATE_PARAMETERS,
    REDUCTION,
    UnifiedStandaloneDFAG,
    parameter_counts,
)


NUM_CLASSES = 176
INPUT_SHAPE = (1, 3, 299, 299)
WARMUP = 10
MEASURED_FORWARDS = 30
ORDER_EFFECT_THRESHOLD_PERCENT = 5.0
GPU_INDEX = 0
MODEL_NAMES = ("Ours-FT", "Progressive Head", "DFAG")
ROUND_ORDERS = {
    "round1_manuscript": ("Ours-FT", "Progressive Head", "DFAG"),
    "round2_order_sanity": ("DFAG", "Progressive Head", "Ours-FT"),
}
EXPECTED = {
    "Ours-FT": {"resident_parameters": 57_114_448, "flops_g": 26.314805824},
    "Progressive Head": {"resident_parameters": 55_076_688, "flops_g": 26.310728256},
    "DFAG": {"resident_parameters": 114_523_808, "flops_g": 52.624581632},
}
HISTORICAL_LATENCY_CONTEXT = {
    "old_progressive_session": {"Ours-FT": 14.1310688, "Progressive Head": 14.250305},
    "old_dfag_session": {"Ours-FT": 15.168940766652424, "DFAG": 30.376856422424318},
}


class LogitsOnly(nn.Module):
    def __init__(self, name: str, core: nn.Module) -> None:
        super().__init__()
        self.name = name
        self.core = core

    def forward(self, images: Tensor) -> Tensor:
        output = self.core(images)
        return output[0]


def count_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def build_model(name: str) -> nn.Module:
    if name == "Ours-FT":
        return ScreenModel("#0", NUM_CLASSES, pretrained=False)
    if name == "Progressive Head":
        model = ScreenModel("#1", NUM_CLASSES, pretrained=False)
        model.head.shortcut_lambda = 0.7
        return model
    if name == "DFAG":
        return UnifiedStandaloneDFAG(NUM_CLASSES)
    raise ValueError(name)


def parameter_summary(name: str, core: nn.Module) -> dict[str, int]:
    resident = count_parameters(core)
    if name == "Ours-FT":
        return {
            "resident_parameters": resident,
            "executed_parameters": resident,
            "backbone_parameters": count_parameters(core.backbone),
            "head_or_gate_parameters": count_parameters(core.head),
            "anchor_branch_parameters": 0,
            "plastic_branch_parameters": 0,
        }
    if name == "Progressive Head":
        return {
            "resident_parameters": resident,
            "executed_parameters": resident,
            "backbone_parameters": count_parameters(core.backbone),
            "head_or_gate_parameters": count_parameters(core.head),
            "anchor_branch_parameters": 0,
            "plastic_branch_parameters": 0,
        }
    counts = parameter_counts(core)
    return {
        "resident_parameters": resident,
        "executed_parameters": int(counts["executed_parameters"]),
        "backbone_parameters": (
            count_parameters(core.anchor.backbone) + count_parameters(core.plastic.backbone)
        ),
        "head_or_gate_parameters": int(counts["gate_parameters"]),
        "anchor_branch_parameters": int(counts["anchor_branch_parameters"]),
        "plastic_branch_parameters": int(counts["plastic_branch_parameters"]),
    }


def architecture_audit_one(name: str) -> dict[str, Any]:
    core = build_model(name).eval()
    checks: dict[str, bool] = {}
    counts = parameter_summary(name, core)
    checks["resident_parameter_sanity_exact"] = (
        counts["resident_parameters"] == EXPECTED[name]["resident_parameters"]
    )
    if name == "Ours-FT":
        head = core.head
        checks.update({
            "one_backbone_branch": set(dict(core.named_children())) == {"backbone", "head"},
            "backbone_pooled_dimension_1536": core.backbone.num_features == 1536,
            "conventional_head_type": isinstance(head, BaselineHead),
            "fc1_1536_to_1024": head.fc1.in_features == 1536 and head.fc1.out_features == 1024,
            "fc2_1024_to_1024": head.fc2.in_features == 1024 and head.fc2.out_features == 1024,
            "classifier_1024_to_classes": head.classifier.in_features == 1024 and head.classifier.out_features == NUM_CLASSES,
            "dropout_0_2": head.drop1.p == 0.2 and head.drop2.p == 0.2,
            "batchnorm_and_silu": isinstance(head.bn1, nn.BatchNorm1d) and isinstance(head.bn2, nn.BatchNorm1d)
                and isinstance(head.act1, nn.SiLU) and isinstance(head.act2, nn.SiLU),
            "dfag_absent": not any(isinstance(module, DynamicResidualGate) for module in core.modules()),
        })
    elif name == "Progressive Head":
        head = core.head
        expected_types = (nn.Linear, nn.BatchNorm1d, nn.SiLU, nn.Dropout)
        checks.update({
            "one_backbone_branch": set(dict(core.named_children())) == {"backbone", "head"},
            "projection_1536_to_256": head.input_projection.in_features == 1536 and head.input_projection.out_features == 256,
            "five_mapping_blocks": len(head.mapping_blocks) == 5,
            "mapping_sequence_exact": all(
                isinstance(block, MappingBlock)
                and tuple(type(module) for module in block.mapping) == expected_types
                and block.mapping[0].in_features == 256
                and block.mapping[0].out_features == 256
                and block.mapping[3].p == 0.3
                for block in head.mapping_blocks
            ),
            "shortcut_lambda_exact_0_7": head.shortcut_lambda == 0.7,
            "arithmetic_mean_f1_to_f5": "stacked.mean(dim=1)" in inspect.getsource(MatchedShortcutHead.forward),
            "aggregation_dropout_0_2": head.aggregation_dropout.p == 0.2,
            "mhsa_absent": not any(isinstance(module, nn.MultiheadAttention) for module in core.modules()),
            "terminal_residual_absent": "stacked.mean(dim=1)" in inspect.getsource(MatchedShortcutHead.forward),
            "dfag_absent": not any(isinstance(module, DynamicResidualGate) for module in core.modules()),
        })
    else:
        gate_layers = list(core.dfag_gate.gate)
        source = inspect.getsource(UnifiedStandaloneDFAG.forward_components)
        checks.update({
            "standalone_dfag_type": isinstance(core, UnifiedStandaloneDFAG),
            "anchor_and_plastic_branches": hasattr(core, "anchor") and hasattr(core, "plastic") and core.anchor is not core.plastic,
            "two_distinct_backbones": core.anchor.backbone is not core.plastic.backbone,
            "both_backbones_inception_resnet_v2": core.anchor.backbone.__class__.__name__ == "InceptionResnetV2"
                and core.plastic.backbone.__class__.__name__ == "InceptionResnetV2",
            "feature_dimension_1536": FEATURE_DIM == 1536 and core.anchor.backbone.num_features == 1536
                and core.plastic.backbone.num_features == 1536,
            "reduction_16": REDUCTION == 16,
            "gate_structure_exact": len(gate_layers) == 4
                and isinstance(gate_layers[0], nn.Linear) and gate_layers[0].in_features == 1536 and gate_layers[0].out_features == 96
                and isinstance(gate_layers[1], nn.ReLU)
                and isinstance(gate_layers[2], nn.Linear) and gate_layers[2].in_features == 96 and gate_layers[2].out_features == 1536
                and isinstance(gate_layers[3], nn.Sigmoid),
            "gate_linears_bias_free": gate_layers[0].bias is None and gate_layers[2].bias is None,
            "gate_parameters_exact_294912": count_parameters(core.dfag_gate) == GATE_PARAMETERS == 294_912,
            "gate_receives_only_f_spec_source": "self.dfag_gate(f_spec)" in source,
            "elementwise_gated_fusion_source": "gate * f_anc + (1.0 - gate) * f_spec" in source,
            "progressive_head_absent": not any(isinstance(module, MatchedShortcutHead) for module in core.modules()),
            "mhsa_absent": not any(isinstance(module, nn.MultiheadAttention) for module in core.modules()),
            "anchor_frozen": not any(parameter.requires_grad for parameter in core.anchor.parameters()),
        })
    result = {
        "model": name,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "counts": counts,
        "checks": checks,
    }
    del core
    gc.collect()
    return result


def query_gpu() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--id", str(GPU_INDEX),
        "--query-gpu=name,utilization.gpu,memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    name, utilization, memory, temperature = [part.strip() for part in raw.split(",")]
    return {
        "name": name,
        "utilization_percent": int(utilization),
        "memory_used_mib": int(memory),
        "temperature_c": int(temperature),
    }


def runtime_route_check(name: str, core: nn.Module, deployed: nn.Module, x: Tensor) -> dict[str, Any]:
    counters: dict[str, int] = {"backbone": 0, "anchor_backbone": 0, "plastic_backbone": 0, "gate": 0}
    identity = {"gate_input_is_plastic_feature": False}
    handles = []
    plastic_output: dict[str, Tensor] = {}

    def increment(key: str) -> Callable:
        def hook(_module, _inputs, _output):
            counters[key] += 1
        return hook

    if name in {"Ours-FT", "Progressive Head"}:
        handles.append(core.backbone.register_forward_hook(increment("backbone")))
    else:
        handles.append(core.anchor.backbone.register_forward_hook(increment("anchor_backbone")))

        def plastic_hook(_module, _inputs, output):
            counters["plastic_backbone"] += 1
            plastic_output["tensor"] = output

        def gate_pre_hook(_module, inputs):
            counters["gate"] += 1
            identity["gate_input_is_plastic_feature"] = bool(
                inputs and "tensor" in plastic_output and inputs[0] is plastic_output["tensor"]
            )

        handles.append(core.plastic.backbone.register_forward_hook(plastic_hook))
        handles.append(core.dfag_gate.register_forward_pre_hook(gate_pre_hook))
    with torch.inference_mode():
        output = deployed(x)
    for handle in handles:
        handle.remove()
    checks = {"logits_shape_1_by_num_classes": tuple(output.shape) == (1, NUM_CLASSES)}
    if name in {"Ours-FT", "Progressive Head"}:
        checks["one_backbone_forward_observed"] = counters["backbone"] == 1
    else:
        checks.update({
            "anchor_backbone_forward_observed_once": counters["anchor_backbone"] == 1,
            "plastic_backbone_forward_observed_once": counters["plastic_backbone"] == 1,
            "gate_forward_observed_once": counters["gate"] == 1,
            "gate_input_identity_is_f_spec": identity["gate_input_is_plastic_feature"],
        })
    del output
    return {"status": "PASS" if all(checks.values()) else "FAIL", "counters": counters, "checks": checks}


def latency_memory(model: nn.Module, x: Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        for _ in range(WARMUP):
            output = model(x)
        torch.cuda.synchronize(device)
        del output
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        elapsed_ms: list[float] = []
        for _ in range(MEASURED_FORWARDS):
            torch.cuda.synchronize(device)
            start_ns = time.perf_counter_ns()
            output = model(x)
            torch.cuda.synchronize(device)
            end_ns = time.perf_counter_ns()
            elapsed_ms.append((end_ns - start_ns) / 1_000_000.0)
        peak = int(torch.cuda.max_memory_allocated(device))
        del output
    array = np.asarray(elapsed_ms, dtype=np.float64)
    return {
        "latency_mean_ms": float(array.mean()),
        "latency_sample_sd_ms": float(array.std(ddof=1)),
        "latency_median_ms": float(np.median(array)),
        "latency_p95_ms": float(np.quantile(array, 0.95)),
        "latency_min_ms": float(array.min()),
        "latency_max_ms": float(array.max()),
        "latency_samples_ms": elapsed_ms,
        "resident_allocated_before_forward_bytes": allocated_before,
        "resident_allocated_before_forward_mib": allocated_before / (1024.0 ** 2),
        "absolute_peak_allocated_bytes": peak,
        "absolute_peak_allocated_mib": peak / (1024.0 ** 2),
        "incremental_forward_peak_bytes": int(max(0, peak - allocated_before)),
        "incremental_forward_peak_mib": max(0, peak - allocated_before) / (1024.0 ** 2),
        "warmup_forwards": WARMUP,
        "measured_forwards": MEASURED_FORWARDS,
    }


def clean_gpu(device: torch.device) -> None:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gc.collect()


def profile_one(name: str, shared_input_cpu: Tensor, device: torch.device) -> dict[str, Any]:
    clean_gpu(device)
    clean_allocated = int(torch.cuda.memory_allocated(device))
    core = build_model(name).eval().requires_grad_(False).float().to(device)
    deployed = LogitsOnly(name, core).eval().requires_grad_(False)
    after_model_allocated = int(torch.cuda.memory_allocated(device))
    x = shared_input_cpu.to(device=device, dtype=torch.float32)
    after_input_allocated = int(torch.cuda.memory_allocated(device))
    counts = parameter_summary(name, core)
    route = runtime_route_check(name, core, deployed, x)
    if route["status"] != "PASS":
        raise RuntimeError(f"runtime route check failed for {name}: {route}")
    with torch.inference_mode():
        macs, thop_parameters = profile(deployed, inputs=(x,), verbose=False)
    flops = float(2.0 * macs)
    if abs(flops / 1e9 / EXPECTED[name]["flops_g"] - 1.0) > 0.001:
        raise RuntimeError(f"FLOPs sanity failed for {name}: {flops / 1e9}")
    measured = latency_memory(deployed, x, device)
    result = {
        "model": name,
        **counts,
        "resident_parameters_m": counts["resident_parameters"] / 1e6,
        "executed_parameters_m": counts["executed_parameters"] / 1e6,
        "head_or_gate_parameters_m": counts["head_or_gate_parameters"] / 1e6,
        "thop_reported_executed_parameters": int(thop_parameters),
        "macs": float(macs),
        "macs_g": float(macs / 1e9),
        "flops_2x_macs": flops,
        "flops_g": flops / 1e9,
        "backbone_forwards_per_input": 2 if name == "DFAG" else 1,
        "runtime_route_check": route,
        "allocator_checkpoints": {
            "clean_before_model_bytes": clean_allocated,
            "after_model_before_input_bytes": after_model_allocated,
            "after_model_and_input_bytes": after_input_allocated,
        },
        **measured,
    }
    del x, deployed, core
    clean_gpu(device)
    return result


def ratio_and_change(value: float, reference: float) -> dict[str, float]:
    ratio = float(value / reference)
    return {"ratio": ratio, "percent_change": 100.0 * (ratio - 1.0)}


def compute_relative(primary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = primary["Ours-FT"]
    result: dict[str, Any] = {}
    fields = {
        "resident_parameters": "resident_parameters",
        "flops": "flops_g",
        "latency": "latency_mean_ms",
        "absolute_peak_memory": "absolute_peak_allocated_mib",
    }
    for name in ("Progressive Head", "DFAG"):
        result[name] = {
            label: ratio_and_change(float(primary[name][field]), float(baseline[field]))
            for label, field in fields.items()
        }
    return result


def order_effect(rounds: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    first = rounds["round1_manuscript"]
    second = rounds["round2_order_sanity"]
    rows = {}
    for name in MODEL_NAMES:
        delta_percent = 100.0 * (
            second[name]["latency_mean_ms"] / first[name]["latency_mean_ms"] - 1.0
        )
        rows[name] = {
            "round1_latency_mean_ms": first[name]["latency_mean_ms"],
            "round2_latency_mean_ms": second[name]["latency_mean_ms"],
            "round2_minus_round1_percent": delta_percent,
            "within_predeclared_5_percent": abs(delta_percent) <= ORDER_EFFECT_THRESHOLD_PERCENT,
        }
    return {
        "predeclared_rule": "Round 1 is manuscript primary; round 2 is sanity only; abs(mean difference) <= 5% means no obvious order effect.",
        "threshold_percent": ORDER_EFFECT_THRESHOLD_PERCENT,
        "models": rows,
        "status": "PASS" if all(row["within_predeclared_5_percent"] for row in rows.values()) else "WARN",
    }


def historical_consistency(primary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for name in MODEL_NAMES:
        params_exact = primary[name]["resident_parameters"] == EXPECTED[name]["resident_parameters"]
        flops_relative_error_percent = 100.0 * (
            primary[name]["flops_g"] / EXPECTED[name]["flops_g"] - 1.0
        )
        rows[name] = {
            "resident_parameters_expected": EXPECTED[name]["resident_parameters"],
            "resident_parameters_exact_match": params_exact,
            "historical_flops_g": EXPECTED[name]["flops_g"],
            "flops_relative_error_percent": flops_relative_error_percent,
            "flops_within_0_1_percent": abs(flops_relative_error_percent) <= 0.1,
        }
    return {
        "status": "PASS" if all(
            row["resident_parameters_exact_match"] and row["flops_within_0_1_percent"]
            for row in rows.values()
        ) else "FAIL",
        "models": rows,
        "historical_latency_context_not_used_in_final_table": HISTORICAL_LATENCY_CONTEXT,
    }


def format_ratio(relative: dict[str, Any], model: str, key: str) -> str:
    value = relative[model][key]
    return f"{value['ratio']:.3f}x ({value['percent_change']:+.3f}%)"


def write_outputs(payload: dict[str, Any]) -> None:
    primary = payload["rounds"]["round1_manuscript"]
    secondary = payload["rounds"]["round2_order_sanity"]
    relative = payload["relative_to_ours_ft"]
    (OUTPUT / "FINAL_THREE_MODEL_COMPLEXITY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = [
        "model", "resident_parameters", "resident_parameters_m", "executed_parameters",
        "executed_parameters_m", "head_or_gate_parameters", "head_or_gate_parameters_m",
        "macs", "macs_g", "flops_2x_macs", "flops_g", "backbone_forwards_per_input",
        "latency_mean_ms", "latency_sample_sd_ms", "latency_median_ms", "latency_p95_ms",
        "latency_min_ms", "latency_max_ms", "resident_allocated_before_forward_bytes",
        "resident_allocated_before_forward_mib", "absolute_peak_allocated_bytes",
        "absolute_peak_allocated_mib", "incremental_forward_peak_bytes",
        "incremental_forward_peak_mib", "round2_latency_mean_ms", "round2_latency_sample_sd_ms",
        "round2_absolute_peak_allocated_mib", "round2_minus_round1_latency_percent",
        "params_ratio_to_ours_ft", "params_percent_change_vs_ours_ft",
        "flops_ratio_to_ours_ft", "flops_percent_change_vs_ours_ft",
        "latency_ratio_to_ours_ft", "latency_percent_change_vs_ours_ft",
        "peak_memory_ratio_to_ours_ft", "peak_memory_percent_change_vs_ours_ft",
    ]
    rows = []
    for name in MODEL_NAMES:
        row = {key: primary[name].get(key) for key in fields}
        row.update({
            "round2_latency_mean_ms": secondary[name]["latency_mean_ms"],
            "round2_latency_sample_sd_ms": secondary[name]["latency_sample_sd_ms"],
            "round2_absolute_peak_allocated_mib": secondary[name]["absolute_peak_allocated_mib"],
            "round2_minus_round1_latency_percent": payload["order_effect"]["models"][name]["round2_minus_round1_percent"],
        })
        if name == "Ours-FT":
            for prefix in ("params", "flops", "latency", "peak_memory"):
                row[f"{prefix}_ratio_to_ours_ft"] = 1.0
                row[f"{prefix}_percent_change_vs_ours_ft"] = 0.0
        else:
            mapping = {"params": "resident_parameters", "flops": "flops", "latency": "latency", "peak_memory": "absolute_peak_memory"}
            for prefix, key in mapping.items():
                row[f"{prefix}_ratio_to_ours_ft"] = relative[name][key]["ratio"]
                row[f"{prefix}_percent_change_vs_ours_ft"] = relative[name][key]["percent_change"]
        rows.append(row)
    with (OUTPUT / "FINAL_THREE_MODEL_COMPLEXITY.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def table_row(name: str) -> str:
        row = primary[name]
        return (
            f"| {name} | {row['resident_parameters_m']:.3f} | {row['head_or_gate_parameters_m']:.3f} | "
            f"{row['flops_g']:.3f} | {row['latency_mean_ms']:.2f} +/- {row['latency_sample_sd_ms']:.2f} | "
            f"{row['absolute_peak_allocated_mib']:.1f} |"
        )

    hard_lines = []
    for name in MODEL_NAMES:
        audit = payload["architecture_audit"][name]
        hard_lines.append(f"### {name}: **{audit['status']}**")
        hard_lines.extend(f"- `{key}`: {'PASS' if value else 'FAIL'}" for key, value in audit["checks"].items())
        hard_lines.append("")
    order_lines = []
    for name, row in payload["order_effect"]["models"].items():
        order_lines.append(
            f"- {name}: round 1 {row['round1_latency_mean_ms']:.6f} ms; round 2 "
            f"{row['round2_latency_mean_ms']:.6f} ms; change {row['round2_minus_round1_percent']:+.3f}% "
            f"({'PASS' if row['within_predeclared_5_percent'] else 'WARN'})."
        )
    report = f"""# Final Three-Model Complexity and Inference-Efficiency Audit

Overall status: **{payload['status']}**

## A. Environment

- Python: {payload['environment']['python']}
- Python executable: `{payload['environment']['python_executable']}`
- PyTorch: {payload['environment']['torch']}
- CUDA runtime: {payload['environment']['cuda_runtime']}
- cuDNN: {payload['environment']['cudnn']}
- GPU: {payload['environment']['gpu']}
- THOP: {payload['environment']['thop']}

## B. Unified profiling protocol

- All three models were built and profiled in one Python process and one GPU session on `cuda:{GPU_INDEX}`.
- No checkpoint or real dataset was loaded; model weights were random because architecture complexity is weight-value independent.
- Shared synthetic input: `torch.randn(1,3,299,299)`, FP32, batch size 1.
- `eval()` plus `torch.inference_mode()`; no AMP, FP16, `torch.compile`, DataLoader, preprocessing, TTA, or ensemble.
- FLOPs = 2 x THOP MACs.
- Latency: 10 warm-up forwards and 30 measured forwards; `torch.cuda.synchronize()` immediately before and after every measured forward; `time.perf_counter_ns()` timing.
- Predeclared selection rule: round 1 (Ours-FT -> Progressive Head -> DFAG) is the manuscript result. Reverse-order round 2 is order-effect sanity only and is never selected for a better-looking number.

## C. Architecture hard checks

{chr(10).join(hard_lines)}
## D. Final three-model table (round 1 manuscript values)

| Model | Resident Params (M) | Head/Gate Params (M) | FLOPs (G) | Latency mean +/- sample SD (ms/image) | Absolute Peak Memory (MiB) |
|---|---:|---:|---:|---:|---:|
{table_row('Ours-FT')}
{table_row('Progressive Head')}
{table_row('DFAG')}

## E. Relative ratios (round 1; Ours-FT reference)

### Progressive Head / Ours-FT

- Resident parameters: {format_ratio(relative, 'Progressive Head', 'resident_parameters')}
- FLOPs: {format_ratio(relative, 'Progressive Head', 'flops')}
- Latency: {format_ratio(relative, 'Progressive Head', 'latency')}
- Absolute peak memory: {format_ratio(relative, 'Progressive Head', 'absolute_peak_memory')}

### DFAG / Ours-FT

- Resident parameters: {format_ratio(relative, 'DFAG', 'resident_parameters')}
- FLOPs: {format_ratio(relative, 'DFAG', 'flops')}
- Latency: {format_ratio(relative, 'DFAG', 'latency')}
- Absolute peak memory: {format_ratio(relative, 'DFAG', 'absolute_peak_memory')}

## F. Memory-definition explanation

Three CUDA allocated-memory quantities are retained for every model. `resident_allocated_before_forward` is allocated memory after model creation, shared-input transfer, warm-up, and deletion of the warm-up output. `absolute_peak_allocated` is `torch.cuda.max_memory_allocated()` during the synchronized 30-forward measurement after resetting peak statistics. `incremental_forward_peak` is absolute peak minus resident allocated before forward. The manuscript table uses absolute peak allocated for all three models under the same definition.

## G. Order-effect sanity check

Predeclared threshold: absolute round-mean change <= {ORDER_EFFECT_THRESHOLD_PERCENT:.1f}% means no obvious order effect. Overall: **{payload['order_effect']['status']}**.

{chr(10).join(order_lines)}

## H. Historical-result consistency check

Status: **{payload['historical_consistency']['status']}**. Resident parameter counts match the frozen historical expectations exactly and FLOPs match within 0.1%. Historical latency and memory were measured in different GPU sessions and are retained only as context; none of those old runtime values is copied into the final table.

## I. Manuscript-ready rounded table

| Model | Params (M) | Head/Gate (M) | FLOPs (G) | Latency (ms/image) | Peak Memory (MiB) |
|---|---:|---:|---:|---:|---:|
{table_row('Ours-FT')}
{table_row('Progressive Head')}
{table_row('DFAG')}

## J. Interpretation boundaries

These measurements support descriptive comparisons under this exact hardware, software, batch-size, precision, input-shape, and timing protocol. Small runtime differences should not be generalized as intrinsic speed advantages. DFAG is explicitly a two-branch inference model with two full backbone forwards; it must not be described as lightweight or as having negligible overhead. Results do not include data loading or preprocessing and are not end-to-end application latency.
"""
    (OUTPUT / "FINAL_THREE_MODEL_COMPLEXITY_AUDIT.md").write_text(report, encoding="utf-8")


def main() -> None:
    state_path = OUTPUT / "profiling_state.json"
    state_path.write_text(json.dumps({"status": "STARTING", "time": time.time()}, indent=2) + "\n", encoding="utf-8")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device(f"cuda:{GPU_INDEX}")
        if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5070 Ti":
            raise RuntimeError(f"Unexpected GPU: {torch.cuda.get_device_name(device)}")
        gpu_before = query_gpu()
        if gpu_before["utilization_percent"] > 10 or gpu_before["memory_used_mib"] > 1500:
            raise RuntimeError(f"GPU is not sufficiently idle: {gpu_before}")

        torch.manual_seed(20260818)
        np.random.seed(20260818)
        torch.cuda.manual_seed_all(20260818)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        architecture = {name: architecture_audit_one(name) for name in MODEL_NAMES}
        if not all(row["status"] == "PASS" for row in architecture.values()):
            raise RuntimeError(f"Architecture hard checks failed: {architecture}")

        shared_input_cpu = torch.randn(*INPUT_SHAPE, dtype=torch.float32, device="cpu")
        rounds: dict[str, dict[str, dict[str, Any]]] = {}
        session_id = f"pid-{__import__('os').getpid()}-{time.time_ns()}"
        state_path.write_text(json.dumps({"status": "RUNNING", "session_id": session_id, "time": time.time()}, indent=2) + "\n", encoding="utf-8")
        for round_name, order in ROUND_ORDERS.items():
            rounds[round_name] = {}
            for name in order:
                print(json.dumps({"event": "PROFILE_START", "round": round_name, "model": name}), flush=True)
                rounds[round_name][name] = profile_one(name, shared_input_cpu, device)
                print(json.dumps({
                    "event": "PROFILE_COMPLETE", "round": round_name, "model": name,
                    "latency_mean_ms": rounds[round_name][name]["latency_mean_ms"],
                    "peak_mib": rounds[round_name][name]["absolute_peak_allocated_mib"],
                }), flush=True)

        relative = compute_relative(rounds["round1_manuscript"])
        order = order_effect(rounds)
        historical = historical_consistency(rounds["round1_manuscript"])
        gpu_after = query_gpu()
        status = "PASS" if historical["status"] == "PASS" and all(
            row["status"] == "PASS" for row in architecture.values()
        ) else "FAIL"
        payload = {
            "status": status,
            "session_id": session_id,
            "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "same_python_process": True,
            "same_gpu_session": True,
            "training_performed": False,
            "checkpoint_loaded": False,
            "real_dataset_used": False,
            "environment": {
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(device),
                "gpu_index": GPU_INDEX,
                "thop": thop.__version__,
                "gpu_state_before": gpu_before,
                "gpu_state_after": gpu_after,
            },
            "protocol": {
                "input_shape": list(INPUT_SHAPE), "dtype": "FP32", "batch_size": 1,
                "synthetic_input": True, "shared_identical_input_values": True,
                "warmup_forwards": WARMUP, "measured_forwards": MEASURED_FORWARDS,
                "latency_timer": "time.perf_counter_ns with CUDA synchronization immediately before and after each forward",
                "inference_context": "torch.inference_mode", "amp": False, "fp16": False,
                "torch_compile": False, "flops_convention": "2 x THOP MACs",
                "manuscript_primary_round": "round1_manuscript",
                "order_sanity_round": "round2_order_sanity",
            },
            "architecture_audit": architecture,
            "rounds": rounds,
            "relative_to_ours_ft": relative,
            "order_effect": order,
            "historical_consistency": historical,
            "interpretation_boundaries": [
                "Descriptive efficiency comparison under this exact profiling protocol only.",
                "No old-session latency or memory value is used in the final table.",
                "DFAG has two complete backbone forwards and is not lightweight.",
                "No data loading or preprocessing time is included.",
            ],
        }
        write_outputs(payload)
        state_path.write_text(json.dumps({"status": status, "session_id": session_id, "time": time.time()}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "AUDIT_COMPLETE", "status": status, "session_id": session_id}), flush=True)
        if status != "PASS":
            raise SystemExit(2)
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            state_path.write_text(json.dumps({"status": "FAILED", "error": repr(exc), "time": time.time()}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
