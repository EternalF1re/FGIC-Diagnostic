"""Windows-safe two-process launcher for cal_ddp_preflight.py.

torchrun's elastic c10d rendezvous requests libuv even when the installed
Windows PyTorch was built without it. The ordinary en<LOCAL_PATH>
defaults to the non-libuv TCPStore on Windows, so this launcher uses PyTorch's
own spawn primitive without altering DDP, GPU affinity, SyncBN or batch sizes.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import torch.multiprocessing as mp


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def worker(local_rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["USE_LIBUV"] = "0"
    from cal_ddp_preflight import main
    code = main()
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    mp.spawn(worker, args=(free_port(),), nprocs=2, join=True)
