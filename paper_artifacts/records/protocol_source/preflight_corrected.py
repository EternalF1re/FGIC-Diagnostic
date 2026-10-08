"""Corrected hard-check preflight and exact 40-row formal ledger generator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import corrected_transform
import protocol_core


corrected_transform.install()
import preflight_external_protocol as base  # noqa: E402


def add(checks: list, name: str, passed: bool, detail) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def build_ledger(root: Path, config: dict) -> tuple[Path, list]:
    config_hash = protocol_core.sha256_file(protocol_core.CONFIG_PATH)
    specifications = [
        ("CUB-200-2011", "Ours-FT", "", range(5)),
        ("CUB-200-2011", "Progressive Head", "0.7", range(5)),
        ("CUB-200-2011", "Progressive Head", "0.1", range(5)),
        ("CUB-200-2011", "Progressive Head", "1.0", range(5)),
        ("Stanford Cars", "Ours-FT", "", range(5)),
        ("Stanford Cars", "Progressive Head", "0.7", range(5)),
        ("Oxford Flowers-102", "Ours-FT", "", range(5)),
        ("Oxford Flowers-102", "Progressive Head", "0.7", range(5)),
    ]
    rows = []
    for dataset, method, shortcut_lambda, folds in specifications:
        for fold in folds:
            rows.append({
                "dataset": dataset,
                "method": method,
                "lambda": shortcut_lambda,
                "training_seed": 42,
                "split_random_state": 42,
                "fold": fold,
                "formal_status": "PENDING",
                "protocol_config_sha256": config_hash,
            })
    path = root / "formal_job_ledger.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def main() -> None:
    base.main()
    root = protocol_core.ROOT
    preflight_path = root / "manifests" / "preflight.json"
    record = json.loads(preflight_path.read_text(encoding="utf-8"))
    transforms = corrected_transform.serialized_transforms()
    checks = record["checks"]
    add(checks, "cub_random_rotation_absent", not transforms["cub"]["random_rotation_present"], transforms["cub"])
    add(checks, "cars_random_rotation_absent", not transforms["cars"]["random_rotation_present"], transforms["cars"])
    add(checks, "flowers_random_rotation_present", transforms["flowers"]["random_rotation_present"], transforms["flowers"])
    for key in ("cub", "cars", "flowers"):
        add(checks, f"{key}_random_affine_degrees_exactly_zero",
            transforms[key]["random_affine_degrees"] == [0.0, 0.0], transforms[key]["random_affine_degrees"])
        add(checks, f"{key}_random_affine_shear_exactly_minus5_plus5",
            transforms[key]["random_affine_shear"] == [-5.0, 5.0], transforms[key]["random_affine_shear"])
    config = protocol_core.load_config()
    ledger_path, ledger = build_ledger(root, config)
    lambdas = sorted({row["lambda"] for row in ledger if row["dataset"] == "CUB-200-2011" and row["method"] == "Progressive Head"})
    add(checks, "formal_ledger_row_count_exactly_40", len(ledger) == 40, len(ledger))
    add(checks, "formal_ledger_all_pending", all(row["formal_status"] == "PENDING" for row in ledger), None)
    add(checks, "formal_ledger_seed42_only", {row["training_seed"] for row in ledger} == {42}, None)
    add(checks, "formal_ledger_split42_only", {row["split_random_state"] for row in ledger} == {42}, None)
    add(checks, "formal_ledger_cub_lambdas_exact", lambdas == ["0.1", "0.7", "1.0"], lambdas)
    add(checks, "formal_ledger_no_extra_jobs", len({(row["dataset"], row["method"], row["lambda"], row["fold"]) for row in ledger}) == 40, None)
    record.update({
        "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
        "corrected_transform_hard_checks": True,
        "transforms": transforms,
        "formal_job_ledger": {
            "path": str(ledger_path.resolve()),
            "sha256": protocol_core.sha256_file(ledger_path),
            "row_count": len(ledger),
            "all_pending": all(row["formal_status"] == "PENDING" for row in ledger),
        },
        "formal_jobs_launched": 0,
        "previous_smoke_reused": False,
    })
    preflight_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    failed = [row["name"] for row in checks if not row["pass"]]
    print(json.dumps({"status": record["status"], "checks": len(checks), "failed": failed,
                      "ledger_rows": len(ledger), "ledger_sha256": record["formal_job_ledger"]["sha256"]}, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

