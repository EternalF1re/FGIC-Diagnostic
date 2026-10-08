"""Read-only WSL/Linux CUDA and NCCL environment probe."""
from __future__ import annotations

import json
import platform

import torch


def version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "installed"))


def main() -> None:
    operations = []
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        tensor = torch.arange(16, dtype=torch.float32, device=device)
        result = (tensor * tensor).sum()
        torch.cuda.synchronize(index)
        value = float(result.cpu())
        operations.append({
            "device_index": index,
            "device_name": torch.cuda.get_device_name(index),
            "result": value,
            "pass": value == 1240.0,
        })
    try:
        nccl_version = torch.cuda.nccl.version()
    except Exception as error:
        nccl_version = {"error": repr(error)}
    dependencies = {}
    for name in ("timm", "torchvision", "sklearn", "PIL", "numpy"):
        try:
            dependencies[name] = version(name)
        except Exception as error:
            dependencies[name] = {"error": repr(error)}
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "distributed_available": torch.distributed.is_available(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "nccl_version": nccl_version,
        "minimal_cuda_operations": operations,
        "dependencies": dependencies,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
