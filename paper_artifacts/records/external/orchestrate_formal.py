"""Execute exactly the frozen 40-row ledger with a two-GPU shared queue."""

from __future__ import annotations

import csv
import json
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import pandas as pd

import formal_common as common


LEDGER_FIELDS = [
    "job_id", "dataset", "method", "lambda", "training_seed", "split_random_state", "fold",
    "formal_status", "protocol_config_sha256", "start_time", "end_time", "elapsed_seconds",
    "device", "output_dir", "stage1_checkpoint_sha256", "stage2_checkpoint_sha256",
    "prediction_artifact_paths", "metrics_path", "metrics_sha256", "stdout_path", "stderr_path",
    "stdout_status", "stderr_status", "returncode", "technical_failure",
]


class Coordinator:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.started = time.time()
        self.status_path = common.ROOT / "formal_orchestrator_status.json"

    def write_ledger(self) -> None:
        with common.RUNTIME_LEDGER_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def status_counts(self) -> dict:
        return {status: sum(row["formal_status"] == status for row in self.rows) for status in ("PENDING", "RUNNING", "COMPLETE", "FAILED")}

    def write_status(self, state: str = "RUNNING", extra: dict | None = None) -> None:
        record = {
            "status": state, "started_unix": self.started, "updated_unix": time.time(),
            "elapsed_seconds": time.time() - self.started, "expected_jobs": 40,
            "counts": self.status_counts(), "config_sha256": common.EXPECTED_CONFIG_SHA256,
            "shared_dynamic_queue": True, "devices": ["cuda:0", "cuda:1"],
            "result_independent_scheduling": True,
        }
        if extra:
            record.update(extra)
        self.status_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def update(self, job_id: str, **values) -> None:
        with self.lock:
            row = next(row for row in self.rows if row["job_id"] == job_id)
            row.update({key: "" if value is None else value for key, value in values.items()})
            self.write_ledger()
            self.write_status()


def iso_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def prelaunch() -> tuple[list[dict], dict]:
    config = common.load_config()
    if common.RUNTIME_LEDGER_PATH.exists() or common.RUNS_ROOT.exists():
        raise RuntimeError("formal runtime outputs already exist; refusing ambiguous relaunch")
    source = pd.read_csv(common.SOURCE_LEDGER_PATH, keep_default_na=False)
    if len(source) != 40 or set(source["formal_status"]) != {"PENDING"}:
        raise ValueError("source ledger must contain exactly 40 PENDING rows")
    if source.duplicated(subset=["dataset", "method", "lambda", "fold"]).any():
        raise ValueError("duplicate source ledger jobs")
    if set(source["protocol_config_sha256"]) != {common.EXPECTED_CONFIG_SHA256}:
        raise ValueError("source ledger config hash mismatch")
    if set(source["training_seed"]) != {42} or set(source["split_random_state"]) != {42}:
        raise ValueError("source ledger seed mismatch")
    manifest = json.loads((common.ROOT / "final_external_protocol_manifest.json").read_text(encoding="utf-8"))
    if manifest["FINAL_40_JOB_RUN_READY"] != "YES" or manifest["NEW_CONFIG_SHA256"] != common.EXPECTED_CONFIG_SHA256:
        raise ValueError("final prelaunch manifest is not ready")
    transforms = common.corrected_transform.serialized_transforms()
    transform_checks = {
        "cub_rotation_absent": not transforms["cub"]["random_rotation_present"],
        "cars_rotation_absent": not transforms["cars"]["random_rotation_present"],
        "flowers_rotation_present": transforms["flowers"]["random_rotation_present"],
        "vertical_flip_absent": all(not row["vertical_flip_present"] for row in transforms.values()),
        "affine_degrees_zero": all(row["random_affine_degrees"] == [0.0, 0.0] for row in transforms.values()),
    }
    dataset_counts = {}
    for dataset in ("CUB-200-2011", "Stanford Cars", "Oxford Flowers-102"):
        development, test = common.records_for_dataset(dataset, config)
        dataset_counts[dataset] = {"development": len(development), "official_test": len(test)}
    architecture_checks = {}
    for method, shortcut_lambda in (("Ours-FT", ""), ("Progressive Head", "0.1"), ("Progressive Head", "0.7"), ("Progressive Head", "1.0")):
        model = common.build_model(method, shortcut_lambda, 200, pretrained=False)
        types = [type(module).__name__ for module in model.modules()]
        key = f"{method}:{shortcut_lambda}"
        architecture_checks[key] = {
            "mhsa_absent": "MultiheadAttention" not in types,
            "terminal_residual_absent": not any("terminal" in name.lower() for name, _ in model.named_modules()),
            "lambda_exact": method == "Ours-FT" or model.head.shortcut_lambda == float(shortcut_lambda),
            "five_mapping_blocks": method == "Ours-FT" or len(model.head.mapping_blocks) == 5,
        }
    checks = {
        "config_sha_match": common.sha256_file(common.CONFIG_PATH) == common.EXPECTED_CONFIG_SHA256,
        "source_ledger_sha": common.sha256_file(common.SOURCE_LEDGER_PATH),
        "source_ledger_rows": len(source) == 40,
        "source_ledger_unique": not source.duplicated(subset=["dataset", "method", "lambda", "fold"]).any(),
        "source_ledger_pending": int((source["formal_status"] == "PENDING").sum()) == 40,
        "stage1_smoothing": config["training"]["stage1"]["label_smoothing"] == 0.05,
        "stage2_smoothing": config["training"]["stage2"]["label_smoothing"] == 0.03,
        "batch_size": config["training"]["batch_size"] == 32,
        "transforms": transform_checks,
        "architecture": architecture_checks,
        "dataset_counts": dataset_counts,
    }
    if not all(transform_checks.values()) or not all(all(item.values()) for item in architecture_checks.values()):
        raise ValueError("transform or architecture hard check failed")
    rows = []
    for row in source.to_dict("records"):
        shortcut_lambda = str(row["lambda"])
        identifier = common.job_id(row["dataset"], row["method"], shortcut_lambda, int(row["fold"]))
        runtime = {field: "" for field in LEDGER_FIELDS}
        runtime.update({
            "job_id": identifier, "dataset": row["dataset"], "method": row["method"], "lambda": shortcut_lambda,
            "training_seed": 42, "split_random_state": 42, "fold": int(row["fold"]),
            "formal_status": "PENDING", "protocol_config_sha256": common.EXPECTED_CONFIG_SHA256,
            "output_dir": str(common.output_dir(identifier).resolve()),
        })
        rows.append(runtime)
    preflight = {
        "status": "PASS", "generated_at": iso_now(), "instruction_sha256": common.INSTRUCTION_SHA256,
        "formal_jobs_started": 0, "checks": checks,
        "source_hashes": {
            "config": common.sha256_file(common.CONFIG_PATH), "ledger": common.sha256_file(common.SOURCE_LEDGER_PATH),
            "formal_common": common.sha256_file(Path(common.__file__)),
            "runner": common.sha256_file(common.ROOT / "formal_train_one.py"),
            "orchestrator": common.sha256_file(Path(__file__)),
            "aggregator": common.sha256_file(common.ROOT / "aggregate_formal.py"),
        },
    }
    (common.ROOT / "formal_prelaunch_audit.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, preflight


def worker(device: str, tasks: queue.Queue, coordinator: Coordinator) -> None:
    while not coordinator.stop.is_set():
        try:
            row = tasks.get_nowait()
        except queue.Empty:
            return
        job = row["job_id"]
        started = time.time()
        stdout_path = common.LOGS_ROOT / f"{job}.stdout.log"
        stderr_path = common.LOGS_ROOT / f"{job}.stderr.log"
        coordinator.update(job, formal_status="RUNNING", start_time=iso_now(), device=device,
                           stdout_path=str(stdout_path.resolve()), stderr_path=str(stderr_path.resolve()))
        command = [
            sys.executable, str(common.ROOT / "formal_train_one.py"), "--job-id", job,
            "--dataset", row["dataset"], "--method", row["method"], "--lambda", row["lambda"],
            "--fold", str(row["fold"]), "--device", device,
        ]
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                process = subprocess.run(command, cwd=str(common.ROOT), stdout=stdout, stderr=stderr, check=False)
            elapsed = time.time() - started
            stdout_status = "EMPTY" if stdout_path.stat().st_size == 0 else "NONEMPTY"
            stderr_status = "EMPTY" if stderr_path.stat().st_size == 0 else "NONEMPTY"
            manifest_path = common.output_dir(job) / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            required = (
                process.returncode == 0 and manifest.get("status") == "COMPLETE"
                and manifest.get("config_sha256") == common.EXPECTED_CONFIG_SHA256
                and Path(manifest.get("best_stage1_checkpoint", "missing")).is_file()
                and Path(manifest.get("best_stage2_checkpoint", "missing")).is_file()
                and Path(manifest.get("metrics_path", "missing")).is_file()
            )
            if not required:
                failure = manifest.get("error", f"returncode={process.returncode}; missing/incomplete outputs")
                coordinator.update(job, formal_status="FAILED", end_time=iso_now(), elapsed_seconds=elapsed,
                                   returncode=process.returncode, stdout_status=stdout_status, stderr_status=stderr_status,
                                   technical_failure=failure)
                coordinator.stop.set()
            else:
                prediction_paths = [path for group in manifest["prediction_artifacts"].values() for path in group.values()]
                coordinator.update(
                    job, formal_status="COMPLETE", end_time=iso_now(), elapsed_seconds=elapsed,
                    returncode=process.returncode, stdout_status=stdout_status, stderr_status=stderr_status,
                    stage1_checkpoint_sha256=manifest["best_stage1_sha256"],
                    stage2_checkpoint_sha256=manifest["best_stage2_sha256"],
                    prediction_artifact_paths=json.dumps(prediction_paths, ensure_ascii=False),
                    metrics_path=manifest["metrics_path"], metrics_sha256=manifest["metrics_sha256"], technical_failure="",
                )
        except Exception as exc:
            coordinator.update(job, formal_status="FAILED", end_time=iso_now(), elapsed_seconds=time.time() - started,
                               returncode=-1, technical_failure=repr(exc), stderr_status="ORCHESTRATOR_EXCEPTION")
            coordinator.stop.set()
        finally:
            tasks.task_done()


def main() -> None:
    rows, _ = prelaunch()
    common.RUNS_ROOT.mkdir(parents=True, exist_ok=False)
    common.LOGS_ROOT.mkdir(parents=True, exist_ok=False)
    coordinator = Coordinator(rows)
    coordinator.write_ledger()
    coordinator.write_status("RUNNING")
    task_queue: queue.Queue = queue.Queue()
    priority = {"Stanford Cars": 0, "CUB-200-2011": 1, "Oxford Flowers-102": 2}
    for row in sorted(rows, key=lambda item: (priority[item["dataset"]], item["method"], item["lambda"], int(item["fold"]))):
        task_queue.put(row)
    threads = [threading.Thread(target=worker, args=(f"cuda:{index}", task_queue, coordinator), daemon=False) for index in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    counts = coordinator.status_counts()
    if coordinator.stop.is_set() or counts["FAILED"] or counts["COMPLETE"] != 40:
        coordinator.write_status("STOPPED_ON_TECHNICAL_FAILURE", {"aggregation_launched": False})
        raise SystemExit(2)
    aggregate_stdout = common.LOGS_ROOT / "aggregation.stdout.log"
    aggregate_stderr = common.LOGS_ROOT / "aggregation.stderr.log"
    with aggregate_stdout.open("w", encoding="utf-8") as stdout, aggregate_stderr.open("w", encoding="utf-8") as stderr:
        result = subprocess.run([sys.executable, str(common.ROOT / "aggregate_formal.py")], cwd=str(common.ROOT), stdout=stdout, stderr=stderr, check=False)
    if result.returncode:
        coordinator.write_status("AGGREGATION_FAILED", {"aggregation_returncode": result.returncode})
        raise SystemExit(result.returncode)
    coordinator.write_status("COMPLETE", {"aggregation_returncode": 0, "stopped_after_aggregation": True})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (common.ROOT / "formal_orchestrator_exception.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
