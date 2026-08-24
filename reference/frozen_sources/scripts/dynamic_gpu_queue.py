"""Auditable shared-queue scheduler for local multi-GPU experiments.

Tasks are never permanently assigned to a device.  Every idle GPU claims the
next pending task, so a faster GPU naturally executes more work.  The module
also provides fail-closed dispatch, append-only events, one log per task and an
atomically refreshed ledger suitable for later experiment audits.

Commands may contain the literal placeholders ``{device}``, ``{task_id}`` and
``{python}``.  They are replaced without invoking a shell.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TaskSpec:
    """One independently executable experiment task."""

    task_id: str
    command: tuple[str, ...]
    estimated_seconds: float = 1.0
    log_stem: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.command:
            raise ValueError(f"task {self.task_id!r} has an empty command")
        if self.estimated_seconds <= 0:
            raise ValueError(f"task {self.task_id!r} estimated_seconds must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            task_id=str(value["task_id"]),
            command=tuple(str(part) for part in value["command"]),
            estimated_seconds=float(value.get("estimated_seconds", 1.0)),
            log_stem=(str(value["log_stem"]) if value.get("log_stem") else None),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class SchedulerPaths:
    """Output paths owned by one scheduler invocation."""

    logs_dir: Path
    ledger_path: Path
    events_path: Path
    summary_path: Path

    @classmethod
    def under(cls, output_dir: Path | str) -> "SchedulerPaths":
        root = Path(output_dir)
        return cls(
            logs_dir=root / "logs",
            ledger_path=root / "run_ledger.csv",
            events_path=root / "orchestrator_events.jsonl",
            summary_path=root / "orchestrator_summary.json",
        )


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "task"


def _render_command(task: TaskSpec, device: str) -> list[str]:
    replacements = {
        "{device}": device,
        "{task_id}": task.task_id,
        "{python}": sys.executable,
    }
    rendered: list[str] = []
    for part in task.command:
        for token, replacement in replacements.items():
            part = part.replace(token, replacement)
        rendered.append(part)
    return rendered


class DynamicGpuQueue:
    """Run tasks from one shared queue on a set of GPU workers."""

    _LEDGER_FIELDS = (
        "task_id",
        "status",
        "device",
        "pid",
        "queue_rank",
        "estimated_seconds",
        "elapsed_seconds",
        "exit_code",
        "started_unix",
        "finished_unix",
        "log_path",
        "command_json",
        "metadata_json",
        "error",
    )

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        devices: Sequence[str],
        paths: SchedulerPaths,
        *,
        cwd: Path | str | None = None,
        poll_seconds: float = 10.0,
        longest_first: bool = True,
        completed_task_ids: Iterable[str] = (),
    ) -> None:
        if not tasks:
            raise ValueError("at least one task is required")
        if not devices:
            raise ValueError("at least one device is required")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if len(set(devices)) != len(devices):
            raise ValueError("devices must be unique")

        ids = [task.task_id for task in tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task_id values must be unique")

        completed = set(completed_task_ids)
        unknown_completed = completed.difference(ids)
        if unknown_completed:
            raise ValueError(f"unknown completed task ids: {sorted(unknown_completed)}")

        indexed = list(enumerate(tasks))
        if longest_first:
            indexed.sort(key=lambda item: (-item[1].estimated_seconds, item[0]))

        self.tasks = {task.task_id: task for task in tasks}
        self.task_order = ids
        self.queue = [task.task_id for _, task in indexed if task.task_id not in completed]
        self.devices = tuple(devices)
        self.paths = paths
        self.cwd = Path(cwd).resolve() if cwd is not None else None
        self.poll_seconds = poll_seconds
        self.longest_first = longest_first
        self.running: dict[str, dict[str, Any]] = {}
        self.records: dict[str, dict[str, Any]] = {}
        for task in tasks:
            self.records[task.task_id] = {
                "status": "COMPLETE" if task.task_id in completed else "QUEUED",
                "device": "",
                "pid": "",
                "elapsed_seconds": "",
                "exit_code": "",
                "started_unix": "",
                "finished_unix": "",
                "log_path": "",
                "error": "",
            }

    def _prepare_outputs(self) -> None:
        for path in (
            self.paths.ledger_path,
            self.paths.events_path,
            self.paths.summary_path,
        ):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite scheduler artifact: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)

    def _event(self, handle: Any, event: str, **payload: Any) -> None:
        row = {"event": event, "time": time.time(), **payload}
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _write_ledger(self) -> None:
        queue_rank = {task_id: rank for rank, task_id in enumerate(self.queue, start=1)}
        rows: list[dict[str, Any]] = []
        for task_id in self.task_order:
            task = self.tasks[task_id]
            record = self.records[task_id]
            rows.append(
                {
                    "task_id": task_id,
                    "status": record["status"],
                    "device": record["device"],
                    "pid": record["pid"],
                    "queue_rank": queue_rank.get(task_id, ""),
                    "estimated_seconds": task.estimated_seconds,
                    "elapsed_seconds": record["elapsed_seconds"],
                    "exit_code": record["exit_code"],
                    "started_unix": record["started_unix"],
                    "finished_unix": record["finished_unix"],
                    "log_path": record["log_path"],
                    "command_json": json.dumps(task.command, ensure_ascii=False),
                    "metadata_json": json.dumps(task.metadata, ensure_ascii=False, default=str),
                    "error": record["error"],
                }
            )

        target = self.paths.ledger_path
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._LEDGER_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)

    def _dispatch(self, task_id: str, device: str, event_handle: Any) -> bool:
        task = self.tasks[task_id]
        log_path = self.paths.logs_dir / f"{_safe_stem(task.log_stem or task_id)}.log"
        command = _render_command(task, device)
        record = self.records[task_id]
        started = time.time()
        log_handle = None
        try:
            log_handle = log_path.open("x", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                command,
                cwd=str(self.cwd) if self.cwd is not None else None,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
            )
        except Exception as exc:
            if log_handle is not None:
                log_handle.close()
            record.update(
                status="FAILED",
                device=device,
                started_unix=started,
                finished_unix=time.time(),
                log_path=str(log_path),
                exit_code="DISPATCH_ERROR",
                error=f"{type(exc).__name__}: {exc}",
            )
            self._event(
                event_handle,
                "DISPATCH_ERROR",
                task_id=task_id,
                device=device,
                error=record["error"],
            )
            return False

        record.update(
            status="RUNNING",
            device=device,
            pid=process.pid,
            started_unix=started,
            log_path=str(log_path),
        )
        self.running[task_id] = {
            "process": process,
            "device": device,
            "handle": log_handle,
            "started_unix": started,
        }
        self._event(
            event_handle,
            "START",
            task_id=task_id,
            device=device,
            pid=process.pid,
            command=command,
            log=str(log_path),
        )
        return True

    def _finish_ready(self, event_handle: Any) -> bool:
        failed = False
        for task_id, job in list(self.running.items()):
            code = job["process"].poll()
            if code is None:
                continue
            finished = time.time()
            job["handle"].close()
            del self.running[task_id]
            record = self.records[task_id]
            record.update(
                status="COMPLETE" if code == 0 else "FAILED",
                exit_code=code,
                elapsed_seconds=finished - job["started_unix"],
                finished_unix=finished,
            )
            self._event(
                event_handle,
                "EXIT",
                task_id=task_id,
                device=job["device"],
                pid=record["pid"],
                exit_code=code,
                elapsed_seconds=record["elapsed_seconds"],
                log=record["log_path"],
            )
            failed = failed or code != 0
        return failed

    def _summary(self, halted: bool, started_unix: float) -> dict[str, Any]:
        counts: dict[str, int] = {}
        device_completed: dict[str, int] = {device: 0 for device in self.devices}
        device_seconds: dict[str, float] = {device: 0.0 for device in self.devices}
        for record in self.records.values():
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            if record["status"] == "COMPLETE" and record["device"] in device_completed:
                device_completed[record["device"]] += 1
                if record["elapsed_seconds"] != "":
                    device_seconds[record["device"]] += float(record["elapsed_seconds"])
        finished = time.time()
        return {
            "status": "FAILED_HALTED" if halted else "ALL_COMPLETE",
            "task_count": len(self.tasks),
            "counts": counts,
            "devices": list(self.devices),
            "completed_by_device": device_completed,
            "busy_seconds_by_device": device_seconds,
            "longest_first": self.longest_first,
            "started_unix": started_unix,
            "finished_unix": finished,
            "wall_seconds": finished - started_unix,
            "fail_closed": True,
        }

    def run(self) -> dict[str, Any]:
        """Execute the queue and return the persisted final summary."""

        self._prepare_outputs()
        started_unix = time.time()
        halted = False
        with self.paths.events_path.open("x", encoding="utf-8", buffering=1) as events:
            self._event(
                events,
                "SCHEDULER_START",
                devices=self.devices,
                task_count=len(self.tasks),
                pending_count=len(self.queue),
                longest_first=self.longest_first,
            )
            self._write_ledger()
            while self.queue or self.running:
                used = {job["device"] for job in self.running.values()}
                if not halted:
                    for device in self.devices:
                        if not self.queue or device in used:
                            continue
                        task_id = self.queue.pop(0)
                        if not self._dispatch(task_id, device, events):
                            halted = True
                            break
                        used.add(device)
                    self._write_ledger()

                if self.running:
                    time.sleep(self.poll_seconds)
                    if self._finish_ready(events):
                        halted = True
                        self._event(events, "HALT_NEW_DISPATCH", reason="task_failure")
                    self._write_ledger()

                if halted and not self.running:
                    for task_id in self.queue:
                        self.records[task_id]["status"] = "HALTED_AFTER_FAILURE"
                    self.queue.clear()
                    self._write_ledger()
                    break

            summary = self._summary(halted, started_unix)
            self._event(events, "SCHEDULER_EXIT", **summary)

        self.paths.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if halted:
            raise RuntimeError("a task failed; new dispatch was halted")
        return summary


def load_task_file(path: Path | str) -> list[TaskSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload["tasks"] if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("task JSON must be a list or an object containing a 'tasks' list")
    return [TaskSpec.from_mapping(value) for value in values]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=("cuda:0", "cuda:1"))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--fifo", action="store_true", help="preserve task-file order")
    parser.add_argument("--completed-task-id", action="append", default=[])
    args = parser.parse_args(argv)
    scheduler = DynamicGpuQueue(
        load_task_file(args.tasks_json),
        args.devices,
        SchedulerPaths.under(args.output_dir),
        cwd=args.work_dir,
        poll_seconds=args.poll_seconds,
        longest_first=not args.fifo,
        completed_task_ids=args.completed_task_id,
    )
    summary = scheduler.run()
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
