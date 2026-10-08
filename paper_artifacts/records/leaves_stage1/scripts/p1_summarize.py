"""Aggregate P1 only after all 15 Stage1 reproduction gates pass."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


VALIDATION_ROOT = Path(__file__).resolve().parents[2]
SCREEN = VALIDATION_ROOT / "phase2b_screen"
OUT = Path(__file__).resolve().parents[1] / "p1_stage1_only_oof"
FOLD_OUT = OUT / "fold_outputs"
VARIANTS = ("#0", "#1", "#2")
REPS = 100_000
SEED = 20260807


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def stats(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int) -> dict[str, object]:
    ca, cb = a == y, b == y
    n11, n10 = int(np.sum(ca & cb)), int(np.sum(ca & ~cb))
    n01, n00 = int(np.sum(~ca & cb)), int(np.sum(~ca & ~cb))
    n = len(y); rng = np.random.default_rng(seed)
    categories = np.bincount(ca.astype(np.int8) * 2 + cb.astype(np.int8), minlength=4)
    draws = rng.multinomial(n, categories / n, size=REPS)
    delta = (draws[:, 1] - draws[:, 2]) * 100.0 / n
    disc = n10 + n01
    return {
        "n": n, "stage1_accuracy": float(ca.mean()), "stage2_accuracy": float(cb.mean()),
        "delta_stage2_minus_stage1_pp": float((cb.mean() - ca.mean()) * 100),
        "bootstrap_ci95_low_pp": float(np.quantile(delta, 0.025)),
        "bootstrap_ci95_high_pp": float(np.quantile(delta, 0.975)),
        "bootstrap_replicates": REPS, "bootstrap_seed": seed,
        "mcnemar_exact_two_sided_p": float(binomtest(min(n10, n01), disc, 0.5).pvalue) if disc else 1.0,
        "prediction_changed_n": int(np.sum(a != b)), "n11_both_correct": n11,
        "n10_stage1_correct_stage2_wrong": n10, "n01_stage1_wrong_stage2_correct": n01,
        "n00_both_wrong": n00, "correctness_discordant_n": disc,
    }


def main() -> None:
    reproduction_rows: list[dict[str, object]] = []
    variant_data: dict[str, dict[str, np.ndarray]] = {}
    metadata_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        parts = []
        for fold in range(5):
            base = FOLD_OUT / variant.replace("#", "variant_") / f"fold_{fold}"
            manifest = json.loads((base / "inference_manifest.json").read_text(encoding="utf-8"))
            reproduction_rows.append({k: manifest[k] for k in (
                "variant", "fold", "status", "n", "expected_correct_count", "reproduced_correct_count",
                "expected_accuracy", "reproduced_accuracy", "checkpoint_path", "checkpoint_sha256",
                "deterministic_repeat_exact", "deterministic_repeat_max_abs_diff", "elapsed_seconds")})
            if manifest["status"] != "PASS":
                write_csv(OUT / "stage1_checkpoint_reproduction.csv", reproduction_rows)
                raise RuntimeError("STAGE1_CHECKPOINT_REPRODUCTION_FAILED: refusing OOF aggregation")
            with np.load(base / "stage1_validation.npz") as data:
                parts.append({name: data[name] for name in data.files})
        combined = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
        order = np.argsort(combined["sample_id"])
        combined = {name: values[order] for name, values in combined.items()}
        ids = combined["sample_id"]
        if len(ids) != 18_353 or not np.array_equal(ids, np.arange(18_353)):
            raise RuntimeError(f"INTEGRITY_FAILURE: {variant} OOF IDs incomplete/duplicated")
        if not np.isfinite(combined["logits"]).all() or not np.array_equal(combined["logits"].argmax(1), combined["prediction"]):
            raise RuntimeError(f"INTEGRITY_FAILURE: {variant} logits")
        out_name = OUT / f"stage1_oof_predictions_{variant.replace('#', 'variant_')}.npz"
        if out_name.exists():
            raise RuntimeError(f"refusing to overwrite {out_name}")
        np.savez_compressed(out_name, **combined)
        variant_data[variant] = combined
        for scope, fold in [("pooled_oof", None), *[("fold", i) for i in range(5)]]:
            mask = np.ones(len(ids), dtype=bool) if fold is None else combined["fold"] == fold
            y, p = combined["label"][mask], combined["prediction"][mask]
            metric_rows.append({
                "variant": variant, "scope": scope, "fold": "ALL" if fold is None else fold,
                "n": int(mask.sum()), "accuracy": float(accuracy_score(y, p)),
                "macro_f1": float(f1_score(y, p, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            })
        metadata_rows.extend({
            "variant": variant, "sample_id": int(combined["sample_id"][i]), "fold": int(combined["fold"][i]),
            "label": int(combined["label"][i]), "prediction": int(combined["prediction"][i]),
            "image_name": str(combined["image_name"][i]),
        } for i in range(len(ids)))

    write_csv(OUT / "stage1_checkpoint_reproduction.csv", reproduction_rows)
    write_csv(OUT / "stage1_oof_metrics.csv", metric_rows)
    write_csv(OUT / "stage1_oof_metadata.csv", metadata_rows)

    pooled_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        d = variant_data[variant]
        stage2_ids = np.load(SCREEN / variant / "oof_sample_ids.npy")
        stage2_y = np.load(SCREEN / variant / "oof_labels.npy")
        stage2_p = np.load(SCREEN / variant / "oof_predictions.npy")
        if not np.array_equal(d["sample_id"], stage2_ids) or not np.array_equal(d["label"], stage2_y):
            raise RuntimeError(f"INTEGRITY_FAILURE: {variant} Stage1/Stage2 OOF alignment")
        row = stats(d["label"], d["prediction"], stage2_p, SEED)
        row.update({
            "variant": variant,
            "stage1_macro_f1": float(f1_score(d["label"], d["prediction"], average="macro")),
            "stage2_macro_f1": float(f1_score(d["label"], stage2_p, average="macro")),
            "macro_f1_delta_stage2_minus_stage1": float(f1_score(d["label"], stage2_p, average="macro") - f1_score(d["label"], d["prediction"], average="macro")),
            "stage1_balanced_accuracy": float(balanced_accuracy_score(d["label"], d["prediction"])),
            "stage2_balanced_accuracy": float(balanced_accuracy_score(d["label"], stage2_p)),
            "balanced_accuracy_delta_stage2_minus_stage1": float(balanced_accuracy_score(d["label"], stage2_p) - balanced_accuracy_score(d["label"], d["prediction"])),
        })
        pooled_rows.append(row)
        for fold in range(5):
            mask = d["fold"] == fold
            fr = stats(d["label"][mask], d["prediction"][mask], stage2_p[mask], SEED + fold + 1)
            delta = float(fr["delta_stage2_minus_stage1_pp"])
            fr.update({"variant": variant, "fold": fold, "direction": "+" if delta > 0 else "-" if delta < 0 else "="})
            fold_rows.append(fr)
    write_csv(OUT / "stage1_vs_stage2_paired.csv", pooled_rows)
    write_csv(OUT / "stage1_vs_stage2_per_fold.csv", fold_rows)
    readme = [
        "# P1 Stage1-only pooled OOF", "", "Status: COMPLETE; all 15 checkpoint reproduction gates passed.", "",
        "- Deterministic original-view inference only; model.eval() and torch.inference_mode().",
        "- 18,353 unique samples per variant; exact Stage1/Stage2 ID and label alignment.",
        "- FP32 logits; argmax equals saved prediction; first-batch repeated inference is bitwise identical.",
        "- Paired bootstrap uses 100,000 replicates and seed 20260807; McNemar is exact two-sided.",
        "- Fold directions describe five models from one controlled training seed, not independent-seed significance.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print("P1 AGGREGATION COMPLETE")


if __name__ == "__main__":
    main()
