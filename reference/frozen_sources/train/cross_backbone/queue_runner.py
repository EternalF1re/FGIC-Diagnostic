"""Atomic shared-queue worker and queue initializer for smoke/formal jobs."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from cross_backbone_common import BACKBONES, FOLDS, METHODS, ROOT, output_dir


def db_path(run_type: str) -> Path:
    return ROOT / f"{run_type}_queue.sqlite"


def connect(run_type: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path(run_type), timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def initialize(run_type: str) -> None:
    path = db_path(run_type)
    if path.exists():
        with connect(run_type) as connection:
            count = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        print(json.dumps({"status": "EXISTS", "run_type": run_type, "jobs": count, "db": str(path)}))
        return
    folds = (0,) if run_type == "smoke" else FOLDS
    with connect(run_type) as connection:
        connection.execute("""CREATE TABLE jobs (
            id INTEGER PRIMARY KEY,
            run_type TEXT NOT NULL,
            backbone TEXT NOT NULL,
            method TEXT NOT NULL,
            fold INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            status TEXT NOT NULL,
            worker TEXT,
            device TEXT,
            started REAL,
            completed REAL,
            returncode INTEGER,
            log_path TEXT,
            error TEXT,
            UNIQUE(run_type, backbone, method, fold)
        )""")
        priority = {"ours_ft": 0, "progressive": 1, "dfag": 2}
        rows = [(run_type, backbone, method, fold, priority[method], "PENDING")
                for fold in folds for backbone in BACKBONES for method in METHODS]
        connection.executemany(
            "INSERT INTO jobs(run_type,backbone,method,fold,priority,status) VALUES(?,?,?,?,?,?)", rows
        )
    print(json.dumps({"status": "CREATED", "run_type": run_type, "jobs": len(rows), "db": str(path)}))


def dependency_complete(run_type: str, backbone: str, method: str, fold: int) -> bool:
    if method != "dfag":
        return True
    manifest = output_dir(run_type, backbone, "ours_ft", fold) / "run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "COMPLETE"
    except Exception:
        return False


def claim(run_type: str, worker: str, device: str):
    connection = connect(run_type)
    try:
        connection.execute("BEGIN IMMEDIATE")
        candidates = connection.execute(
            "SELECT * FROM jobs WHERE status='PENDING' ORDER BY priority,id"
        ).fetchall()
        chosen = next((row for row in candidates if dependency_complete(
            run_type, row["backbone"], row["method"], row["fold"]
        )), None)
        if chosen is None:
            connection.commit()
            return None
        changed = connection.execute(
            "UPDATE jobs SET status='RUNNING',worker=?,device=?,started=? WHERE id=? AND status='PENDING'",
            (worker, device, time.time(), chosen["id"]),
        ).rowcount
        connection.commit()
        return dict(chosen) if changed == 1 else None
    finally:
        connection.close()


def finish(run_type: str, job_id: int, returncode: int, log_path: Path, error: str | None) -> None:
    status = "COMPLETE" if returncode == 0 else "FAILED"
    with connect(run_type) as connection:
        connection.execute(
            "UPDATE jobs SET status=?,completed=?,returncode=?,log_path=?,error=? WHERE id=?",
            (status, time.time(), returncode, str(log_path.resolve()), error, job_id),
        )


def summary(run_type: str) -> dict:
    with connect(run_type) as connection:
        rows = connection.execute("SELECT status,COUNT(*) n FROM jobs GROUP BY status").fetchall()
        jobs = [dict(row) for row in connection.execute("SELECT * FROM jobs ORDER BY id").fetchall()]
    return {"run_type": run_type, "counts": {row["status"]: row["n"] for row in rows}, "jobs": jobs}


def work(run_type: str, device: str) -> None:
    worker = f"pid-{os.getpid()}-{device}"
    log_root = ROOT / "logs" / run_type
    log_root.mkdir(parents=True, exist_ok=True)
    while True:
        job = claim(run_type, worker, device)
        if job is None:
            state = summary(run_type)
            pending = state["counts"].get("PENDING", 0)
            running = state["counts"].get("RUNNING", 0)
            failed = state["counts"].get("FAILED", 0)
            if failed:
                print(json.dumps({"event": "QUEUE_STOP_FAILED", **state["counts"]}), flush=True)
                raise SystemExit(2)
            if pending == 0 and running == 0:
                print(json.dumps({"event": "QUEUE_COMPLETE", **state["counts"]}), flush=True)
                return
            time.sleep(10)
            continue
        tag = f"{job['backbone']}_{job['method']}_fold{job['fold']}"
        log_path = log_root / f"{tag}.log"
        command = [
            sys.executable, str(ROOT / "scripts" / "train_one_entry.py"),
            "--backbone", job["backbone"], "--method", job["method"],
            "--fold", str(job["fold"]), "--device", device,
            "--run-type", run_type,
        ]
        print(json.dumps({"event": "JOB_START", "worker": worker, "job": job, "command": command}), flush=True)
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=environment)
        error = None
        if completed.returncode:
            error = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        finish(run_type, job["id"], completed.returncode, log_path, error)
        print(json.dumps({"event": "JOB_END", "worker": worker, "job_id": job["id"],
                          "returncode": completed.returncode, "log": str(log_path)}), flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "work", "status"))
    parser.add_argument("--run-type", required=True, choices=("smoke", "formal"))
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"))
    args = parser.parse_args()
    if args.action == "init":
        initialize(args.run_type)
    elif args.action == "status":
        print(json.dumps(summary(args.run_type), indent=2))
    else:
        if args.device is None:
            raise ValueError("--device required for work")
        work(args.run_type, args.device)


if __name__ == "__main__":
    main()
