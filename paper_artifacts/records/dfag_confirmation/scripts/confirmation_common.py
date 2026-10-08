"""Reference, interpolation, OOF, and paired-statistics helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from dfag_common import EXPECTED_N, EXP_ROOT, FOLDS, SEEDS, source_dir


BOOTSTRAP_REPLICATES = 100_000


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def reference_fold(seed: int, fold: int, stage: int) -> dict[str, np.ndarray]:
    if seed not in SEEDS or fold not in FOLDS or stage not in (1, 2):
        raise ValueError((seed, fold, stage))
    root = source_dir(seed, fold)
    result = {
        "sample_id": np.load(root / f"stage{stage}_validation_sample_ids.npy").astype(np.int64),
        "label": np.load(root / f"stage{stage}_validation_labels.npy").astype(np.int64),
        "prediction": np.load(root / f"stage{stage}_validation_predictions.npy").astype(np.int64),
        "logits": np.load(root / f"stage{stage}_validation_logits.npy").astype(np.float32),
    }
    result["fold"] = np.full(len(result["sample_id"]), fold, dtype=np.int8)
    return result


def validate_oof(data: dict[str, np.ndarray], label: str) -> None:
    ids = data["sample_id"]
    if len(ids) != EXPECTED_N or len(np.unique(ids)) != EXPECTED_N:
        raise RuntimeError(f"{label}: OOF coverage")
    if not np.array_equal(ids, np.arange(EXPECTED_N)):
        raise RuntimeError(f"{label}: OOF order")
    if not np.isfinite(data["logits"]).all():
        raise RuntimeError(f"{label}: non-finite logits")
    if not np.array_equal(data["logits"].argmax(axis=1), data["prediction"]):
        raise RuntimeError(f"{label}: prediction/logit mismatch")


def assemble_reference(seed: int, stage: int) -> dict[str, np.ndarray]:
    parts = [reference_fold(seed, fold, stage) for fold in FOLDS]
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(merged["sample_id"])
    merged = {key: value[order] for key, value in merged.items()}
    validate_oof(merged, f"seed{seed}/stage{stage}")
    return merged


def make_interpolated_state(
    expected: dict[str, torch.Tensor],
    stage1: dict[str, torch.Tensor],
    stage2: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], dict]:
    if set(stage1) != set(stage2) or set(stage1) != set(expected):
        raise RuntimeError("checkpoint-state keys mismatch")
    if alpha == 0.0:
        return {key: value.clone() for key, value in stage2.items()}, {"endpoint_direct_full_state": "Stage2"}
    if alpha == 1.0:
        return {key: value.clone() for key, value in stage1.items()}, {"endpoint_direct_full_state": "Stage1"}
    output: dict[str, torch.Tensor] = {}
    floating = nonfloating = 0
    for key in expected:
        left, right = stage1[key], stage2[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(key)
        if torch.is_floating_point(left):
            output[key] = left.mul(alpha).add(right, alpha=1.0 - alpha)
            floating += 1
        else:
            output[key] = right.clone()
            nonfloating += 1
    return output, {
        "formula": "alpha*Stage1+(1-alpha)*Stage2",
        "alpha": alpha,
        "floating_state_tensors_interpolated": floating,
        "intermediate_nonfloating_buffer_rule": "copy Stage2",
        "nonfloating_state_tensors_copied": nonfloating,
    }


def comparator_task_dir(seed: int, fold: int) -> Path:
    return EXP_ROOT / "comparators" / "tasks" / f"seed{seed}" / f"fold_{fold}"


def paired(
    seed: int,
    comparison: str,
    labels: np.ndarray,
    dfag_prediction: np.ndarray,
    reference_prediction: np.ndarray,
    bootstrap_seed: int,
) -> dict:
    if not (len(labels) == len(dfag_prediction) == len(reference_prediction) == EXPECTED_N):
        raise ValueError("paired OOF length mismatch")
    dfag_correct = dfag_prediction == labels
    reference_correct = reference_prediction == labels
    n10 = int(np.sum(dfag_correct & ~reference_correct))
    n01 = int(np.sum(~dfag_correct & reference_correct))
    rng = np.random.default_rng(bootstrap_seed)
    # This is the exact Phase2F bootstrap implementation: multinomial counts
    # for DFAG-only-correct, reference-only-correct, and tied correctness.
    draws = rng.multinomial(
        EXPECTED_N,
        [n10 / EXPECTED_N, n01 / EXPECTED_N, 1.0 - (n10 + n01) / EXPECTED_N],
        size=BOOTSTRAP_REPLICATES,
    )
    bootstrap = (draws[:, 0] - draws[:, 1]) / EXPECTED_N * 100.0
    dfag_metrics = metrics(labels, dfag_prediction)
    reference_metrics = metrics(labels, reference_prediction)
    dfag_error, reference_error = ~dfag_correct, ~reference_correct
    both_wrong = int(np.sum(dfag_error & reference_error))
    error_union = int(np.sum(dfag_error | reference_error))
    return {
        "seed": seed,
        "comparison": comparison,
        "n": EXPECTED_N,
        "dfag_accuracy": dfag_metrics["accuracy"],
        "reference_accuracy": reference_metrics["accuracy"],
        "delta_accuracy_pp": 100.0 * (dfag_metrics["accuracy"] - reference_metrics["accuracy"]),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": bootstrap_seed,
        "ci95_low_pp": float(np.quantile(bootstrap, 0.025)),
        "ci95_high_pp": float(np.quantile(bootstrap, 0.975)),
        "n10_dfag_correct_reference_wrong": n10,
        "n01_dfag_wrong_reference_correct": n01,
        "mcnemar_exact_two_sided_p": float(
            binomtest(min(n10, n01), n10 + n01, 0.5, alternative="two-sided").pvalue
        ) if n10 + n01 else 1.0,
        "dfag_macro_f1": dfag_metrics["macro_f1"],
        "reference_macro_f1": reference_metrics["macro_f1"],
        "delta_macro_f1": dfag_metrics["macro_f1"] - reference_metrics["macro_f1"],
        "dfag_balanced_accuracy": dfag_metrics["balanced_accuracy"],
        "reference_balanced_accuracy": reference_metrics["balanced_accuracy"],
        "delta_balanced_accuracy": dfag_metrics["balanced_accuracy"] - reference_metrics["balanced_accuracy"],
        "prediction_disagreement_count": int(np.sum(dfag_prediction != reference_prediction)),
        "prediction_disagreement_rate": float(np.mean(dfag_prediction != reference_prediction)),
        "both_wrong_count": both_wrong,
        "error_union_count": error_union,
        "error_set_jaccard": both_wrong / error_union if error_union else math.nan,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
