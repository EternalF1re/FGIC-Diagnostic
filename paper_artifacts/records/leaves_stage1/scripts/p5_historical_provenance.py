"""Phase 2C P5: recover only evidence preserved for the historical motivation."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "p5_historical_motivation_provenance"
LOG = REPO / "实验进程" / "未使用微残差且将x放入注意力中计算的注意力分数分配.txt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = LOG.read_text(encoding="utf-8")
    blocks = re.findall(r"\[模型\s+(\d+)\s+\|\s+Epoch\s+(\d+)[^\]]*\]:\s*(.*?)(?=-{10,}|\Z)", text, re.S)
    records = []
    for model, epoch, block in blocks:
        values = dict((name, float(value)) for name, value in re.findall(r"(x0|f[1-5])\s*:\s*([0-9.]+)", block))
        if list(values) != ["x0", "f1", "f2", "f3", "f4", "f5"]:
            raise RuntimeError("INTEGRITY_FAILURE: malformed archived six-source record")
        records.append({"model_index": int(model), "epoch": int(epoch), **values, "rounded_sum": sum(values.values())})
    if len(records) != 5:
        raise RuntimeError("INTEGRITY_FAILURE: expected five archived model-index records")
    matrix = np.asarray([[r[k] for k in ("x0", "f1", "f2", "f3", "f4", "f5")] for r in records])
    means = matrix.mean(axis=0)
    expected = np.asarray([0.27656, 0.16156, 0.10922, 0.08326, 0.08240, 0.28696])
    if not np.allclose(means, expected, atol=5e-7):
        raise RuntimeError(f"INTEGRITY_FAILURE: archived means {means}")
    with (OUT / "figure1b_archived_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    inventory = [
        {"evidence_id":"P5-E1","classification":"HISTORICAL_ONLY","artifact":str(LOG.relative_to(REPO)),"fact":"five rounded model-index records; Epoch 150; sources x0,f1..f5","numeric_result":", ".join(f"{k}={v:.5f}" for k,v in zip(("x0","f1","f2","f3","f4","f5"),means)),"limitations":"model index is not proven to equal fold; checkpoint, sample count, dataset, seed, query/head aggregation absent"},
        {"evidence_id":"P5-E2","classification":"PROVENANCE_INCOMPLETE","artifact":"figure1-gen.py:34-57,95-129","fact":"current Figure1 generator parses E1 and labels it exploratory six-source attention","numeric_result":"mean of five rounded records","limitations":"plot code does not restore missing run metadata"},
        {"evidence_id":"P5-E3","classification":"PROVENANCE_INCOMPLETE","artifact":"figure1-gen.py.bak_20260808:97","fact":"backup hard-codes the same six means","numeric_result":", ".join(f"{v:.5f}" for v in means),"limitations":"hard-coded derivative, not an independent experiment record"},
        {"evidence_id":"P5-E4","classification":"SUPPORTED_CURRENTLY","artifact":"phase2_controlled_validation/phase2b_screen/configs/controlled_screen.json:architectures.#0","fact":"current #0 head is 1536-1024-1024-176","numeric_result":"two hidden transformations, not a five-stage progressive head","limitations":"cannot support a claim about five intermediate representations"},
        {"evidence_id":"P5-E5","classification":"SUPPORTED_CURRENTLY","artifact":"phase2_controlled_validation/phase2c_inference_diagnostic/p3_mslfa_representation/","fact":"new controlled #1/#2 post-shortcut representation diagnostic","numeric_result":"see P3 cosine/CKA/adjacent-change CSVs","limitations":"new Phase2C evidence; no MHSA; not provenance for the old Figure1(b) experiment"},
    ]
    with (OUT / "provenance_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0])); writer.writeheader(); writer.writerows(inventory)
    recovery = {
        "status": "EXPLORATORY / PROVENANCE PARTIAL", "classification": "PROVENANCE_INCOMPLETE",
        "source_log": str(LOG), "source_log_sha256": sha256(LOG), "source_log_mtime": LOG.stat().st_mtime,
        "recovered": {"record_count": 5, "model_indices": [0,1,2,3,4], "epoch": 150,
                      "source_labels": ["x0","f1","f2","f3","f4","f5"], "rounded_mean_weights": means.tolist()},
        "not_recovered": ["exact fold identity", "checkpoint path/hash", "dataset assertion", "training seed",
                          "sample count", "sample population", "attention head aggregation", "query aggregation",
                          "code version that generated this six-source log"],
        "safe_use": "May be described only as an exploratory historical five-record attention summary with partial provenance.",
        "unsafe_use": "Must not be presented as a current #0 result, a representation-similarity measurement, or a fully reproducible quantitative experiment.",
        "training_performed": False,
    }
    (OUT / "figure1b_recovery.json").write_text(json.dumps(recovery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# P5 historical MS-LFA motivation provenance

Overall status: **PROVENANCE_INCOMPLETE**. Figure1(b) remains **EXPLORATORY / PROVENANCE PARTIAL**.

## Directly recovered

- The archived text file contains five records labelled model 0..4, all at Epoch 150, with six rounded sources x0,f1..f5.
- Their arithmetic means are x0={means[0]:.5f}, f1={means[1]:.5f}, f2={means[2]:.5f}, f3={means[3]:.5f}, f4={means[4]:.5f}, f5={means[5]:.5f}.
- `figure1-gen.py` parses those five records and explicitly labels the panel exploratory.
- The current Phase2B #0 is a 1536->1024->1024->176 head, not a five-stage progressive head.

## Not recovered

Exact fold identity, checkpoint, dataset assertion, seed, sample count/population, query aggregation, attention-head aggregation, and the exact generating code version are absent. A nearby current helper averages attention weights over two dimensions, but it handles five sources and cannot be used to infer the missing six-source aggregation rule.

## Scientific use boundary

The old observation cannot legally support a current quantitative claim that the present #0 has redundant intermediate representations. It can only be mentioned, with explicit caveats, as a historical exploratory attention-allocation observation. The new P3 controlled representation measurements are separate current evidence, not recovered provenance for Figure1(b).
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print("P5 COMPLETE")


if __name__ == "__main__":
    main()
