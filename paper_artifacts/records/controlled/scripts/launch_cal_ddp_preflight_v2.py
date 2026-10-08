"""Gloo-interface-pinned Windows launcher for the CAL topology preflight."""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import torch.multiprocessing as mp


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


MASTER_ADDRESS = "10.23.65.111"
GLOO_INTERFACE = "以太网"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind((MASTER_ADDRESS, 0))
        return int(handle.getsockname()[1])


def worker(local_rank: int, port: int) -> None:
    os.environ.update({
        "MASTER_ADDR": MASTER_ADDRESS,
        "MASTER_PORT": str(port),
        "WORLD_SIZE": "2",
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "USE_LIBUV": "0",
        "GLOO_SOCKET_IFNAME": GLOO_INTERFACE,
    })
    from cal_ddp_preflight import main
    code = main()
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    mp.spawn(worker, args=(free_port(),), nprocs=2, join=True)
