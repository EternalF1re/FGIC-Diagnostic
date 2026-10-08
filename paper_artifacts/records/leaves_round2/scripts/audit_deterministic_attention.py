"""GPU audit for bitwise-reproducible Round2A MHSA forward/backward."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import json
import warnings
from pathlib import Path

import torch

from round2a_common import MHSAShortcutHead, ROUND_ROOT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_pass(seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = MHSAShortcutHead(1536, 176, 0.7).to(device).train()
    inputs = torch.randn(8, 1536, device=device, requires_grad=True)
    logits, features = head(inputs)
    loss = logits.float().square().mean() + features.float().square().mean()
    loss.backward()
    return {
        "logits": logits.detach().cpu(),
        "features": features.detach().cpu(),
        "input_grad": inputs.grad.detach().cpu(),
        "mhsa_weight_grad": head.mhsa.in_proj_weight.grad.detach().cpu(),
    }


def main() -> None:
    output = ROUND_ROOT / "manifests" / "deterministic_attention_preflight.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the deterministic attention audit")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True, warn_only=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = one_pass(20260811, device)
        second = one_pass(20260811, device)
    comparisons = {
        name: {
            "bitwise_equal": bool(torch.equal(first[name], second[name])),
            "max_abs_difference": float((first[name] - second[name]).abs().max().item()),
        }
        for name in first
    }
    backend = {
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    warning_messages = [str(item.message) for item in caught]
    nondeterministic_warnings = [message for message in warning_messages if "non-determin" in message.lower()]
    passed = (
        not backend["flash_sdp_enabled"]
        and not backend["mem_efficient_sdp_enabled"]
        and backend["math_sdp_enabled"]
        and backend["deterministic_algorithms"]
        and not backend["deterministic_warn_only"]
        and all(item["bitwise_equal"] for item in comparisons.values())
        and not nondeterministic_warnings
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "backend": backend,
        "repeatability": comparisons,
        "warning_count": len(warning_messages),
        "warnings": warning_messages,
        "nondeterministic_warning_count": len(nondeterministic_warnings),
        "round2a_common_sha256": sha256(Path(__file__).with_name("round2a_common.py")),
        "audit_script_sha256": sha256(Path(__file__)),
        "training_performed": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "repeatability": comparisons, "backend": backend}))
    if not passed:
        raise RuntimeError("deterministic attention audit failed")


if __name__ == "__main__":
    main()
