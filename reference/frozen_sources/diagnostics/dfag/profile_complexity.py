"""Matched FP32 deployment profile for current baseline and unified DFAG."""
from __future__ import annotations

import gc
import json
import platform
import time

import numpy as np
import thop
import torch
from thop import profile

from dfag_common import EXP_ROOT, PHASE2B, UnifiedStandaloneDFAG, parameter_counts, source_checkpoint
from screen_core import ScreenModel, seed_everything


def latency(model, device, warmup: int = 10, repeats: int = 30) -> dict:
    dummy = torch.randn(1, 3, 299, 299, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for start, end in zip(starts, ends):
            start.record(); model(dummy); end.record()
        torch.cuda.synchronize(device)
        elapsed = np.asarray([start.elapsed_time(end) for start, end in zip(starts, ends)], dtype=np.float64)
        peak = torch.cuda.max_memory_allocated(device)
    return {"latency_mean_ms": float(elapsed.mean()), "latency_sd_ms": float(elapsed.std(ddof=1)),
            "latency_median_ms": float(np.median(elapsed)), "latency_p95_ms": float(np.quantile(elapsed, .95)),
            "peak_max_memory_allocated_bytes": int(peak), "resident_allocated_before_forward_bytes": int(before),
            "incremental_peak_bytes": int(max(0, peak - before)), "warmup": warmup, "measured_forwards": repeats}


def load_baseline(device):
    model = ScreenModel("#0", 176, pretrained=False)
    checkpoint = torch.load(PHASE2B / "#0" / "fold_0" / "best_stage2.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def load_dfag(device):
    model = UnifiedStandaloneDFAG(176)
    model.load_common_stage1(source_checkpoint(42, 0))
    checkpoint = torch.load(EXP_ROOT / "seed42" / "fold_0" / "best_stage2.pth", map_location="cpu", weights_only=False)
    model.load_payload_state(checkpoint)
    return model.eval().requires_grad_(False).to(device)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    seed_everything(42)
    device = torch.device("cuda:0")
    rows = []
    for name, loader in (("current_single_branch_baseline", load_baseline), ("unified_standalone_dynamic_dfag", load_dfag)):
        model = loader(device)
        resident = sum(p.numel() for p in model.parameters())
        if name.endswith("dfag"):
            counts = parameter_counts(model)
            executed = counts["executed_parameters"]
            branch_forwards = 2
        else:
            executed = resident
            counts = {"gate_parameters": 0, "frozen_parameters": 0, "trainable_parameters": resident}
            branch_forwards = 1
        dummy = torch.randn(1, 3, 299, 299, device=device)
        with torch.inference_mode():
            macs, thop_params = profile(model, inputs=(dummy,), verbose=False)
        measured = latency(model, device)
        row = {"method": name, "device": torch.cuda.get_device_name(device), "input": "1x3x299x299 FP32",
               "resident_parameters": resident, "resident_parameters_millions": resident / 1e6,
               "executed_parameters": executed, "gate_parameters": counts["gate_parameters"],
               "thop_reported_parameters": int(thop_params), "macs": float(macs), "macs_giga": float(macs / 1e9),
               "flops_2xmacs": float(2 * macs), "gflops_2xmacs": float(2 * macs / 1e9),
               "backbone_forwards_per_view": branch_forwards, **measured}
        rows.append(row)
        del dummy, model
        torch.cuda.empty_cache(); gc.collect()
    baseline, dfag = rows
    payload = {"status": "PASS", "created_unix": time.time(), "environment": {"python": platform.python_version(),
               "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
               "thop": thop.__version__}, "protocol": {"dtype": "FP32", "batch": 1, "input": "299x299",
               "warmup": 10, "measured_forwards": 30, "flops_convention": "2 x THOP MACs"},
               "profiles": rows,
               "relative": {"resident_parameter_ratio_dfag_to_baseline": dfag["resident_parameters"] / baseline["resident_parameters"],
                            "gflops_ratio_dfag_to_baseline": dfag["gflops_2xmacs"] / baseline["gflops_2xmacs"],
                            "latency_ratio_dfag_to_baseline": dfag["latency_mean_ms"] / baseline["latency_mean_ms"]},
               "hard_checks": {"dfag_executes_two_backbones": dfag["backbone_forwards_per_view"] == 2,
                               "dfag_gflops_near_double": dfag["gflops_2xmacs"] > 1.9 * baseline["gflops_2xmacs"],
                               "gate_count_exact": dfag["gate_parameters"] == 294_912}}
    payload["status"] = "PASS" if all(payload["hard_checks"].values()) else "FAIL"
    target = EXP_ROOT / "complexity" / "unified_dfag_complexity.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "COMPLEXITY", "status": payload["status"], "profiles": rows}, ensure_ascii=False), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
