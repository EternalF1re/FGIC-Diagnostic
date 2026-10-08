"""Phase 2C P3: cosine, centered linear CKA, and adjacent-change summaries."""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "p3_mslfa_representation"
FOLD_FEATURES = OUT / "fold_features"
VARIANTS = ("#1", "#2")
PAIRS = list(combinations(range(5), 2))


def summary(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise RuntimeError("INTEGRITY_FAILURE: NaN/Inf diagnostic values")
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1)), "median": float(q[2]),
            "p05": float(q[0]), "p25": float(q[1]), "p75": float(q[3]), "p95": float(q[4])}


def cosine_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    denom = np.linalg.norm(x64, axis=1) * np.linalg.norm(y64, axis=1)
    if np.any(denom == 0):
        raise RuntimeError("INTEGRITY_FAILURE: zero-norm representation")
    return np.sum(x64 * y64, axis=1) / denom


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA = ||Xc^T Yc||_F^2 / sqrt(||Xc^T Xc||_F^2 ||Yc^T Yc||_F^2)."""
    xc = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    yc = y.astype(np.float64) - y.astype(np.float64).mean(axis=0, keepdims=True)
    cross = xc.T @ yc; xx = xc.T @ xc; yy = yc.T @ yc
    numerator = float(np.sum(cross * cross))
    denominator = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("INTEGRITY_FAILURE: degenerate centered Gram in CKA")
    value = numerator / denominator
    if not np.isfinite(value):
        raise RuntimeError("INTEGRITY_FAILURE: non-finite CKA")
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def load(variant: str, fold: int) -> dict[str, np.ndarray]:
    path = FOLD_FEATURES / variant.replace("#", "variant_") / f"fold_{fold}" / "post_shortcut_stages.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    cosine_pairwise: list[dict[str, object]] = []
    cosine_fold: list[dict[str, object]] = []
    cka_foldwise: list[dict[str, object]] = []
    adjacent: list[dict[str, object]] = []
    cosine_matrices = {v: [] for v in VARIANTS}; cka_matrices = {v: [] for v in VARIANTS}
    fold_delta_means: list[float] = []
    manifest_folds: list[dict[str, object]] = []

    for fold in range(5):
        data = {variant: load(variant, fold) for variant in VARIANTS}
        if not np.array_equal(data["#1"]["sample_id"], data["#2"]["sample_id"]) or not np.array_equal(data["#1"]["label"], data["#2"]["label"]):
            raise RuntimeError(f"INTEGRITY_FAILURE: #1/#2 alignment fold {fold}")
        if data["#1"]["f1"].shape[1] != 256 or data["#2"]["f1"].shape[1] != 256:
            raise RuntimeError("INTEGRITY_FAILURE: feature dimension is not 256")
        pair_cosines: dict[str, dict[tuple[int, int], np.ndarray]] = {v: {} for v in VARIANTS}
        fold_cka: dict[str, dict[tuple[int, int], float]] = {v: {} for v in VARIANTS}
        for variant in VARIANTS:
            cm = np.eye(5); km = np.eye(5)
            for i, j in PAIRS:
                c = cosine_rows(data[variant][f"f{i+1}"], data[variant][f"f{j+1}"])
                k = linear_cka(data[variant][f"f{i+1}"], data[variant][f"f{j+1}"])
                pair_cosines[variant][(i, j)] = c; fold_cka[variant][(i, j)] = k
                cm[i, j] = cm[j, i] = c.mean(); km[i, j] = km[j, i] = k
                row: dict[str, object] = {"variant": variant, "fold": fold, "pair": f"f{i+1}-f{j+1}"}
                row.update(summary(c)); cosine_pairwise.append(row)
            cosine_matrices[variant].append(cm); cka_matrices[variant].append(km)
            offdiag = np.mean(np.stack(list(pair_cosines[variant].values()), axis=1), axis=1)
            row = {"variant": variant, "fold": fold, "metric": "sample_mean_offdiagonal_cosine"}
            row.update(summary(offdiag)); cosine_fold.append(row)
            for i in range(1, 5):
                prev, curr = data[variant][f"f{i}"].astype(np.float64), data[variant][f"f{i+1}"].astype(np.float64)
                raw = np.linalg.norm(curr - prev, axis=1)
                normalized = raw / (np.linalg.norm(prev, axis=1) + 1e-12)
                cos_dist = 1.0 - cosine_rows(curr, prev)
                for metric, values in (("raw_change", raw), ("normalized_change", normalized), ("adjacent_cosine_distance", cos_dist)):
                    ar: dict[str, object] = {"variant": variant, "fold": fold, "transition": f"f{i}-f{i+1}", "metric": metric}
                    ar.update(summary(values)); adjacent.append(ar)
        for i, j in PAIRS:
            c1, c2 = pair_cosines["#1"][(i, j)], pair_cosines["#2"][(i, j)]
            delta = c2 - c1
            row = {"variant": "#2-minus-#1", "fold": fold, "metric": f"paired_cosine_delta_f{i+1}-f{j+1}"}
            row.update(summary(delta)); cosine_fold.append(row)
            cka_foldwise.append({"fold": fold, "pair": f"f{i+1}-f{j+1}", "cka_#1": fold_cka["#1"][(i, j)],
                                 "cka_#2": fold_cka["#2"][(i, j)], "cka_delta_#2_minus_#1": fold_cka["#2"][(i, j)] - fold_cka["#1"][(i, j)]})
        all_delta = np.mean(np.stack([pair_cosines["#2"][p] - pair_cosines["#1"][p] for p in PAIRS], axis=1), axis=1)
        fold_delta_means.append(float(all_delta.mean()))
        row = {"variant": "#2-minus-#1", "fold": fold, "metric": "paired_sample_mean_offdiagonal_cosine_delta"}
        row.update(summary(all_delta)); cosine_fold.append(row)
        manifest_folds.append({"fold": fold, "n": len(data["#1"]["sample_id"]),
                               "sample_ids_aligned": True, "labels_aligned": True})

    cka_summary: list[dict[str, object]] = []
    for i, j in PAIRS:
        pair = f"f{i+1}-f{j+1}"; rows = [r for r in cka_foldwise if r["pair"] == pair]
        for metric in ("cka_#1", "cka_#2", "cka_delta_#2_minus_#1"):
            values = np.asarray([r[metric] for r in rows], dtype=np.float64)
            cka_summary.append({"pair": pair, "metric": metric, "folds": 5,
                                "mean": float(values.mean()), "std_across_folds": float(values.std(ddof=1)),
                                "min": float(values.min()), "max": float(values.max())})
    write_csv(OUT / "cosine_pairwise.csv", cosine_pairwise)
    write_csv(OUT / "cosine_fold_summary.csv", cosine_fold)
    write_csv(OUT / "cka_foldwise.csv", cka_foldwise)
    write_csv(OUT / "cka_summary.csv", cka_summary)
    write_csv(OUT / "adjacent_change.csv", adjacent)

    mean_cos = {v: np.mean(cosine_matrices[v], axis=0) for v in VARIANTS}
    mean_cka = {v: np.mean(cka_matrices[v], axis=0) for v in VARIANTS}
    for name, matrices, cmap, vrange in (("cosine", mean_cos, "viridis", (-1, 1)), ("cka", mean_cka, "magma", (0, 1))):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, variant in zip(axes, VARIANTS):
            im = ax.imshow(matrices[variant], cmap=cmap, vmin=vrange[0], vmax=vrange[1])
            ax.set_xticks(range(5), [f"f{i}" for i in range(1, 6)]); ax.set_yticks(range(5), [f"f{i}" for i in range(1, 6)])
            ax.set_title(f"{variant} fold-mean {name.upper()}")
        fig.colorbar(im, ax=axes, shrink=0.8); fig.subplots_adjust(wspace=0.25, right=0.88)
        fig.savefig(OUT / f"diagnostic_{name}_matrix.png", dpi=220); fig.savefig(OUT / f"diagnostic_{name}_matrix.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    transitions = ["f1-f2", "f2-f3", "f3-f4", "f4-f5"]
    for variant, color in (("#1", "#31688e"), ("#2", "#35b779")):
        means, errors = [], []
        for t in transitions:
            vals = [r["mean"] for r in adjacent if r["variant"] == variant and r["transition"] == t and r["metric"] == "normalized_change"]
            means.append(np.mean(vals)); errors.append(np.std(vals, ddof=1))
        ax.errorbar(transitions, means, yerr=errors, marker="o", capsize=3, label=variant, color=color)
    ax.set(ylabel="Mean normalized adjacent change", title="Post-shortcut adjacent representation change"); ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "diagnostic_adjacent_normalized_change.png", dpi=220); fig.savefig(OUT / "diagnostic_adjacent_normalized_change.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4)); ax.axhline(0, color="black", lw=0.8)
    ax.bar(np.arange(5), fold_delta_means, color=["#35b779" if x >= 0 else "#d95f02" for x in fold_delta_means])
    ax.set(xticks=np.arange(5), xlabel="Fold", ylabel="cosine #2 - #1", title="Fold-wise mean off-diagonal cosine delta"); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(OUT / "diagnostic_foldwise_cosine_delta.png", dpi=220); fig.savefig(OUT / "diagnostic_foldwise_cosine_delta.pdf"); plt.close(fig)

    overall_cos = {v: float(np.mean([r["mean"] for r in cosine_fold if r["variant"] == v and r["metric"] == "sample_mean_offdiagonal_cosine"])) for v in VARIANTS}
    cka_overall = {v: float(np.mean([r[f"cka_{v}"] for r in cka_foldwise])) for v in VARIANTS}
    feature_manifest = {
        "phase": "Phase2C P3", "training_performed": False, "integrity": "PASS",
        "scope": "Phase2B controlled #1/#2 heads (no MHSA; not the full historical MS-LFA head)",
        "feature_definition": "post-shortcut outputs f1..f5 exactly used in forward; [N_fold,256] each",
        "formula_centered_linear_cka": "||Xc^T Yc||_F^2 / sqrt(||Xc^T Xc||_F^2 ||Yc^T Yc||_F^2)",
        "cka_accumulation": "float64", "folds_are_independent_models_not_training_seeds": True,
        "folds": manifest_folds,
    }
    (OUT / "feature_manifest.json").write_text(json.dumps(feature_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# P3 controlled #1 versus #2 representation diagnostic

Status: COMPLETE. This is a diagnostic of the Phase2B controlled mean-aggregation heads. They have no MHSA and are not the full historical MS-LFA head.

## Directly measured

- Each f1..f5 is the true post-shortcut tensor with shape [N_fold, 256].
- Mean sample-wise off-diagonal cosine (fold means): #1 {overall_cos['#1']:.8f}; #2 {overall_cos['#2']:.8f}; #2-#1 {overall_cos['#2']-overall_cos['#1']:.8f}.
- Mean of ten fold-wise centered-linear CKA pairs: #1 {cka_overall['#1']:.8f}; #2 {cka_overall['#2']:.8f}; #2-#1 {cka_overall['#2']-cka_overall['#1']:.8f}.

## Interpretation boundary

The fixed lambda=0.1 setting is associated with the measured inter-stage similarity changes under this controlled screen. Lower CKA/cosine is not equivalent to better representations, and the five folds are not independent training-seed significance tests.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("P3 ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
