"""Phase 2C P2: recompute #0 versus #2 statistics from existing Stage2 OOF.

Inference-only/provenance-only: this script never imports torch and never trains.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


VALIDATION_ROOT = Path(__file__).resolve().parents[2]
SCREEN = VALIDATION_ROOT / "phase2b_screen"
OUT = Path(__file__).resolve().parents[1] / "p2_direct_paired_stats"
EXPECTED = {"#0": 0.977006483953577, "#2": 0.9760802048711382}
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260807


def load_variant(name: str) -> dict[str, np.ndarray]:
    base = SCREEN / name
    result = {
        "ids": np.load(base / "oof_sample_ids.npy"),
        "labels": np.load(base / "oof_labels.npy"),
        "predictions": np.load(base / "oof_predictions.npy"),
    }
    rows = list(csv.DictReader((base / "oof_sample_outputs.csv").open(encoding="utf-8")))
    result["folds"] = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    result["csv_ids"] = np.asarray([int(row["dataset_index"]) for row in rows], dtype=np.int64)
    return result


def paired_stats(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, seed: int) -> dict[str, object]:
    ca = pred_a == labels
    cb = pred_b == labels
    n = len(labels)
    n11 = int(np.sum(ca & cb))
    n10 = int(np.sum(ca & ~cb))
    n01 = int(np.sum(~ca & cb))
    n00 = int(np.sum(~ca & ~cb))
    delta_obs = float((cb.mean() - ca.mean()) * 100.0)
    # A paired bootstrap of the per-sample correctness difference. Multinomial
    # counts are exactly equivalent to resampling n paired rows with replacement.
    categories = np.bincount((ca.astype(np.int8) * 2 + cb.astype(np.int8)), minlength=4)
    probs = categories / n
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, probs, size=BOOTSTRAP_REPLICATES)
    # category 1 = A wrong/B correct; category 2 = A correct/B wrong
    boot_delta_pp = (draws[:, 1] - draws[:, 2]) * (100.0 / n)
    discordant = n10 + n01
    p_value = float(binomtest(min(n10, n01), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "n": n,
        "accuracy_a": float(ca.mean()),
        "accuracy_b": float(cb.mean()),
        "delta_b_minus_a_pp": delta_obs,
        "bootstrap_ci95_low_pp": float(np.quantile(boot_delta_pp, 0.025)),
        "bootstrap_ci95_high_pp": float(np.quantile(boot_delta_pp, 0.975)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
        "mcnemar_exact_two_sided_p": p_value,
        "prediction_changed_n": int(np.sum(pred_a != pred_b)),
        "n11_both_correct": n11,
        "n10_a_correct_b_wrong": n10,
        "n01_a_wrong_b_correct": n01,
        "n00_both_wrong": n00,
        "correctness_discordant_n": discordant,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = load_variant("#0")
    b = load_variant("#2")
    integrity: list[str] = []
    for key in ("ids", "labels", "folds"):
        if not np.array_equal(a[key], b[key]):
            raise RuntimeError(f"PREDICTION_PROVENANCE_MISMATCH: #0/#2 {key} differ")
    if not np.array_equal(a["ids"], a["csv_ids"]) or not np.array_equal(b["ids"], b["csv_ids"]):
        raise RuntimeError("PREDICTION_PROVENANCE_MISMATCH: NPY and CSV sample IDs differ")
    if len(a["ids"]) != 18_353 or len(np.unique(a["ids"])) != 18_353:
        raise RuntimeError("PREDICTION_PROVENANCE_MISMATCH: expected 18,353 unique OOF IDs")
    if set(np.unique(a["folds"]).tolist()) != {0, 1, 2, 3, 4}:
        raise RuntimeError("PREDICTION_PROVENANCE_MISMATCH: fold IDs are not 0..4")
    observed = {name: float(accuracy_score(a["labels"], data["predictions"])) for name, data in (("#0", a), ("#2", b))}
    for name, expected in EXPECTED.items():
        if observed[name] != expected:
            raise RuntimeError(f"PREDICTION_PROVENANCE_MISMATCH: {name} accuracy {observed[name]!r} != {expected!r}")
        integrity.append(f"{name} pooled accuracy exactly reproduced: {observed[name]:.15f}")

    pooled = paired_stats(a["labels"], a["predictions"], b["predictions"], BOOTSTRAP_SEED)
    pooled.update({
        "comparison": "#0_vs_#2",
        "a": "#0 Original baseline",
        "b": "#2 Scaling-only lambda=0.1",
        "macro_f1_a": float(f1_score(a["labels"], a["predictions"], average="macro")),
        "macro_f1_b": float(f1_score(a["labels"], b["predictions"], average="macro")),
        "macro_f1_delta_b_minus_a": float(f1_score(a["labels"], b["predictions"], average="macro") - f1_score(a["labels"], a["predictions"], average="macro")),
        "balanced_accuracy_a": float(balanced_accuracy_score(a["labels"], a["predictions"])),
        "balanced_accuracy_b": float(balanced_accuracy_score(a["labels"], b["predictions"])),
        "balanced_accuracy_delta_b_minus_a": float(balanced_accuracy_score(a["labels"], b["predictions"]) - balanced_accuracy_score(a["labels"], a["predictions"])),
    })
    write_csv(OUT / "#0_vs_#2_paired.csv", [pooled])

    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        mask = a["folds"] == fold
        row = paired_stats(a["labels"][mask], a["predictions"][mask], b["predictions"][mask], BOOTSTRAP_SEED + fold + 1)
        delta = float(row["delta_b_minus_a_pp"])
        row.update({"fold": fold, "direction": "+" if delta > 0 else "-" if delta < 0 else "="})
        fold_rows.append(row)
    write_csv(OUT / "#0_vs_#2_per_fold.csv", fold_rows)

    verification = [
        "# P2 verification",
        "",
        "Status: PASS",
        "",
        *[f"- {line}" for line in integrity],
        "- 18,353 unique samples; no duplicates or missing IDs.",
        "- #0/#2 sample IDs, labels, and fold assignments are exactly aligned.",
        "- Statistics were recomputed from existing Stage2 OOF predictions; no inference or training was run.",
        f"- Paired bootstrap: {BOOTSTRAP_REPLICATES:,} replicates, seed {BOOTSTRAP_SEED}.",
        "- McNemar: exact two-sided binomial test on correctness-discordant pairs.",
        "",
        "## Directly measured pooled result",
        "",
        f"- Delta (#2 - #0): {pooled['delta_b_minus_a_pp']:.6f} pp.",
        f"- 95% paired-bootstrap CI: [{pooled['bootstrap_ci95_low_pp']:.6f}, {pooled['bootstrap_ci95_high_pp']:.6f}] pp.",
        f"- Exact McNemar p: {pooled['mcnemar_exact_two_sided_p']:.12g}.",
    ]
    (OUT / "verification.md").write_text("\n".join(verification) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "phase": "Phase2C P2",
        "training_performed": False,
        "source_paths": [str(SCREEN / "#0"), str(SCREEN / "#2")],
        "expected_accuracies": EXPECTED,
        "sample_count": 18_353,
        "integrity": "PASS",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("P2 COMPLETE")


if __name__ == "__main__":
    main()
