"""Add the direct lambda=0.9 versus lambda=0.7 comparison omitted upstream."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from round1_common import ROUND_ROOT
from summarize_round1 import SEED, paired, verify, write_csv


def load_oof(run_id: str, stage: int) -> dict[str, np.ndarray]:
    path = ROUND_ROOT / "oof" / f"{run_id}_stage{stage}_oof.npz"
    with np.load(path) as data:
        result = {
            "sample_ids": data["sample_id"],
            "folds": data["fold"],
            "labels": data["label"],
            "predictions": data["prediction"],
            "logits": data["logits"],
        }
    verify(result, f"{run_id} stage{stage} supplemental")
    return result


def main() -> None:
    pooled_path = ROUND_ROOT / "statistics" / "lambda_0_9_vs_0_7_paired.csv"
    fold_path = ROUND_ROOT / "statistics" / "lambda_0_9_vs_0_7_paired_per_fold.csv"
    for path in (pooled_path, fold_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")

    pooled_rows = []
    fold_rows = []
    for stage in (1, 2):
        a = load_oof("lambda_0_9_seed42", stage)
        b = load_oof("lambda_0_7_seed42", stage)
        if not np.array_equal(a["sample_ids"], b["sample_ids"]):
            raise RuntimeError("lambda sample-id mismatch")
        if not np.array_equal(a["labels"], b["labels"]):
            raise RuntimeError("lambda label mismatch")
        pooled_rows.append(
            {
                "stage": stage,
                "a": "lambda_0_9",
                "b": "lambda_0_7",
                "comparison": "lambda_0_9_vs_lambda_0_7",
                **paired(a["labels"], a["predictions"], b["predictions"]),
            }
        )
        for fold in range(5):
            mask = a["folds"] == fold
            result = paired(
                a["labels"][mask],
                a["predictions"][mask],
                b["predictions"][mask],
                SEED + fold + 1,
            )
            delta = result["delta_b_minus_a_pp"]
            fold_rows.append(
                {
                    "stage": stage,
                    "a": "lambda_0_9",
                    "b": "lambda_0_7",
                    "fold": fold,
                    "direction": "positive" if delta > 0 else "negative" if delta < 0 else "equal",
                    **result,
                }
            )
    write_csv(pooled_path, pooled_rows)
    write_csv(fold_path, fold_rows)
    print(
        json.dumps(
            {
                "status": "LAMBDA_0_7_VS_0_9_COMPLETE",
                "pooled_rows": len(pooled_rows),
                "per_fold_rows": len(fold_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
