"""Analyze true post-shortcut f1..f5 for all Round1 lambdas and stages."""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from round1_common import ROUND_ROOT


OUT = ROUND_ROOT / "representation"
FEATURES = OUT / "fold_features"
LAMBDAS = (("lambda_0_1", 0.1), ("lambda_0_7", 0.7), ("lambda_0_9", 0.9), ("lambda_1_0", 1.0))
PAIRS = tuple(combinations(range(5), 2))
TRANSITIONS = tuple((i, i + 1) for i in range(4))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def describe(values: np.ndarray) -> dict[str, object]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise RuntimeError("empty or non-finite diagnostic values")
    q = np.quantile(x, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "median": float(q[2]), "p05": float(q[0]), "p25": float(q[1]),
            "p75": float(q[3]), "p95": float(q[4]), "min": float(x.min()), "max": float(x.max())}


def cosine_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    denom = np.linalg.norm(x64, axis=1) * np.linalg.norm(y64, axis=1)
    if np.any(denom <= 0):
        raise RuntimeError("zero-norm representation")
    out = np.sum(x64 * y64, axis=1) / denom
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite cosine")
    return out


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    xc, yc = x64 - x64.mean(0, keepdims=True), y64 - y64.mean(0, keepdims=True)
    xy, xx, yy = xc.T @ yc, xc.T @ xc, yc.T @ yc
    denom = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    if not np.isfinite(denom) or denom <= 0:
        raise RuntimeError("degenerate centered-linear CKA")
    value = float(np.sum(xy * xy) / denom)
    if not np.isfinite(value):
        raise RuntimeError("non-finite CKA")
    return value


def load(key: str, stage: int, fold: int) -> dict[str, np.ndarray]:
    path = FEATURES / key / f"stage{stage}" / f"fold_{fold}" / "stages.npz"
    with np.load(path) as z:
        data = {name: z[name] for name in z.files}
    if len(data["sample_id"]) == 0 or any(data[f"f{i}"].shape != (len(data["sample_id"]), 256) for i in range(1, 6)):
        raise RuntimeError(f"shape failure: {path}")
    return data


def main() -> None:
    cosine_fold: list[dict[str, object]] = []
    cosine_pooled: list[dict[str, object]] = []
    cka_fold: list[dict[str, object]] = []
    cka_summary: list[dict[str, object]] = []
    adjacent: list[dict[str, object]] = []
    norms: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    matrices: dict[tuple[str, int, str], list[np.ndarray]] = {}

    for key, lam in LAMBDAS:
        for stage in (1, 2):
            pooled_pair: dict[tuple[int, int], list[np.ndarray]] = {p: [] for p in PAIRS}
            pooled_offdiag: list[np.ndarray] = []
            ids_all: list[np.ndarray] = []
            for fold in range(5):
                data = load(key, stage, fold)
                ids_all.append(data["sample_id"])
                fold_pair: dict[tuple[int, int], np.ndarray] = {}
                cos_matrix, cka_matrix = np.eye(5), np.eye(5)
                for i, j in PAIRS:
                    cos = cosine_rows(data[f"f{i+1}"], data[f"f{j+1}"])
                    cka = linear_cka(data[f"f{i+1}"], data[f"f{j+1}"])
                    fold_pair[(i, j)] = cos; pooled_pair[(i, j)].append(cos)
                    cos_matrix[i, j] = cos_matrix[j, i] = float(cos.mean())
                    cka_matrix[i, j] = cka_matrix[j, i] = cka
                    cosine_fold.append({"lambda_key": key, "lambda": lam, "stage": stage, "fold": fold,
                                        "pair": f"f{i+1}-f{j+1}", **describe(cos)})
                    cka_fold.append({"lambda_key": key, "lambda": lam, "stage": stage, "fold": fold,
                                     "pair": f"f{i+1}-f{j+1}", "centered_linear_cka": cka})
                offdiag = np.stack([fold_pair[p] for p in PAIRS], axis=1).mean(1)
                pooled_offdiag.append(offdiag)
                cosine_fold.append({"lambda_key": key, "lambda": lam, "stage": stage, "fold": fold,
                                    "pair": "mean_offdiagonal_per_sample", **describe(offdiag)})
                matrices.setdefault((key, stage, "cosine"), []).append(cos_matrix)
                matrices.setdefault((key, stage, "cka"), []).append(cka_matrix)
                for i, j in TRANSITIONS:
                    prev, cur = data[f"f{i+1}"].astype(np.float64), data[f"f{j+1}"].astype(np.float64)
                    raw = np.linalg.norm(cur - prev, axis=1)
                    normalized = raw / np.maximum(np.linalg.norm(prev, axis=1), 1e-12)
                    cosdist = 1.0 - cosine_rows(prev, cur)
                    for metric, values in (("raw_change", raw), ("normalized_change", normalized), ("cosine_distance", cosdist)):
                        adjacent.append({"lambda_key": key, "lambda": lam, "stage": stage, "fold": fold,
                                         "transition": f"f{i+1}-f{j+1}", "metric": metric, **describe(values)})
                for i in range(5):
                    norms.append({"lambda_key": key, "lambda": lam, "stage": stage, "fold": fold,
                                  "layer": f"f{i+1}", **describe(np.linalg.norm(data[f"f{i+1}"].astype(np.float64), axis=1))})
            ids = np.concatenate(ids_all)
            coverage.append({"lambda_key": key, "lambda": lam, "stage": stage, "n": len(ids),
                             "unique_n": len(np.unique(ids)), "covers_0_to_18352": bool(set(ids.tolist()) == set(range(18353)))})
            for i, j in PAIRS:
                cosine_pooled.append({"lambda_key": key, "lambda": lam, "stage": stage,
                                      "pair": f"f{i+1}-f{j+1}", **describe(np.concatenate(pooled_pair[(i, j)]))})
            cosine_pooled.append({"lambda_key": key, "lambda": lam, "stage": stage,
                                  "pair": "mean_offdiagonal_per_sample", **describe(np.concatenate(pooled_offdiag))})

    for key, lam in LAMBDAS:
        for stage in (1, 2):
            for pair in tuple(f"f{i+1}-f{j+1}" for i, j in PAIRS):
                values = np.asarray([r["centered_linear_cka"] for r in cka_fold
                                     if r["lambda_key"] == key and r["stage"] == stage and r["pair"] == pair])
                cka_summary.append({"lambda_key": key, "lambda": lam, "stage": stage, "pair": pair,
                                    "folds": len(values), "mean": float(values.mean()), "std_across_folds": float(values.std(ddof=1)),
                                    "min": float(values.min()), "max": float(values.max())})
            values = np.asarray([r["centered_linear_cka"] for r in cka_fold if r["lambda_key"] == key and r["stage"] == stage])
            cka_summary.append({"lambda_key": key, "lambda": lam, "stage": stage, "pair": "mean_over_10_pairs_and_5_folds",
                                "folds": 5, "mean": float(values.mean()), "std_across_folds": float(np.asarray([
                                    np.mean([r["centered_linear_cka"] for r in cka_fold if r["lambda_key"] == key and r["stage"] == stage and r["fold"] == f])
                                    for f in range(5)]).std(ddof=1)), "min": float(values.min()), "max": float(values.max())})

    analytic = [{"lambda": lam, "lambda_power_5": float(lam ** 5),
                 "interpretation": "direct repeated-shortcut coefficient after five mappings (analytic; not an empirical contribution estimate)"}
                for _, lam in LAMBDAS]
    write_csv(OUT / "cosine_foldwise.csv", cosine_fold)
    write_csv(OUT / "cosine_pooled_summary.csv", cosine_pooled)
    write_csv(OUT / "cka_foldwise.csv", cka_fold)
    write_csv(OUT / "cka_summary.csv", cka_summary)
    write_csv(OUT / "adjacent_change.csv", adjacent)
    write_csv(OUT / "feature_norms.csv", norms)
    write_csv(OUT / "lambda_power5_analytic.csv", analytic)
    write_csv(OUT / "coverage.csv", coverage)

    # Compact lambda-response plots; points are descriptive fold means, not independent-seed inference.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for stage, marker in ((1, "o"), (2, "s")):
        means, errors = [], []
        for _, lam in LAMBDAS:
            vals = [r["mean"] for r in cosine_fold if r["lambda"] == lam and r["stage"] == stage and r["pair"] == "mean_offdiagonal_per_sample"]
            means.append(float(np.mean(vals))); errors.append(float(np.std(vals, ddof=1)))
        axes[0].errorbar([x[1] for x in LAMBDAS], means, yerr=errors, marker=marker, capsize=3, label=f"Stage {stage}")
        means, errors = [], []
        for _, lam in LAMBDAS:
            vals = [r["mean"] for r in cka_summary if r["lambda"] == lam and r["stage"] == stage and r["pair"] == "mean_over_10_pairs_and_5_folds"]
            foldvals = [np.mean([r["centered_linear_cka"] for r in cka_fold if r["lambda"] == lam and r["stage"] == stage and r["fold"] == f]) for f in range(5)]
            means.append(vals[0]); errors.append(float(np.std(foldvals, ddof=1)))
        axes[1].errorbar([x[1] for x in LAMBDAS], means, yerr=errors, marker=marker, capsize=3, label=f"Stage {stage}")
    axes[0].set(xlabel="Shortcut lambda", ylabel="Mean off-diagonal cosine", title="Inter-stage cosine")
    axes[1].set(xlabel="Shortcut lambda", ylabel="Mean centered-linear CKA", title="Inter-stage CKA")
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "lambda_response_similarity.png", dpi=220); fig.savefig(OUT / "lambda_response_similarity.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = plt.cm.viridis(np.linspace(.1, .9, len(LAMBDAS)))
    for ax, stage in zip(axes, (1, 2)):
        for (key, lam), color in zip(LAMBDAS, colors):
            means, errors = [], []
            for transition in ("f1-f2", "f2-f3", "f3-f4", "f4-f5"):
                vals = [r["mean"] for r in adjacent if r["lambda_key"] == key and r["stage"] == stage and r["transition"] == transition and r["metric"] == "normalized_change"]
                means.append(float(np.mean(vals))); errors.append(float(np.std(vals, ddof=1)))
            ax.errorbar(range(4), means, yerr=errors, marker="o", capsize=2, color=color, label=f"lambda={lam:g}")
        ax.set(xticks=range(4), xticklabels=("f1-f2", "f2-f3", "f3-f4", "f4-f5"), title=f"Stage {stage}", xlabel="Transition")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel("Normalized adjacent change")
    fig.tight_layout(); fig.savefig(OUT / "adjacent_change_by_lambda.png", dpi=220); fig.savefig(OUT / "adjacent_change_by_lambda.pdf"); plt.close(fig)

    # Eight-panel fold-mean matrices for each metric.
    for metric, cmap, limits in (("cosine", "viridis", (-1, 1)), ("cka", "magma", (0, 1))):
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for row, stage in enumerate((1, 2)):
            for col, (key, lam) in enumerate(LAMBDAS):
                matrix = np.mean(matrices[(key, stage, metric)], axis=0)
                im = axes[row, col].imshow(matrix, cmap=cmap, vmin=limits[0], vmax=limits[1])
                axes[row, col].set_xticks(range(5), [f"f{i}" for i in range(1, 6)])
                axes[row, col].set_yticks(range(5), [f"f{i}" for i in range(1, 6)])
                axes[row, col].set_title(f"Stage {stage}, lambda={lam:g}")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.75); fig.subplots_adjust(wspace=.25, hspace=.3, right=.9)
        fig.savefig(OUT / f"fold_mean_{metric}_matrices.png", dpi=220); fig.savefig(OUT / f"fold_mean_{metric}_matrices.pdf"); plt.close(fig)

    overview = []
    for key, lam in LAMBDAS:
        for stage in (1, 2):
            cos = next(r["mean"] for r in cosine_pooled if r["lambda_key"] == key and r["stage"] == stage and r["pair"] == "mean_offdiagonal_per_sample")
            cka = next(r["mean"] for r in cka_summary if r["lambda_key"] == key and r["stage"] == stage and r["pair"] == "mean_over_10_pairs_and_5_folds")
            overview.append((lam, stage, cos, cka))
    lines = ["# Round1 representation diagnostic", "", "Status: COMPLETE.", "",
             "Each f1..f5 array is the true post-shortcut tensor used by the frozen mean-aggregation head, shape [N_fold, 256]. No MHSA and no terminal residual are present.", "",
             "| lambda | stage | pooled mean off-diagonal cosine | mean centered-linear CKA |", "|---:|---:|---:|---:|"]
    lines += [f"| {lam:g} | {stage} | {cos:.8f} | {cka:.8f} |" for lam, stage, cos, cka in overview]
    lines += ["", "Centered-linear CKA is computed separately within each fold/model in float64, then summarized across folds. Cosine is sample-wise and pooled only after fold-local extraction.",
              "", "The lambda^5 table is an analytic coefficient illustration, not a measured causal contribution. Lower cosine/CKA denotes changed geometry, not better representations. The five folds are not five independent training seeds."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"status": "COMPLETE", "training_performed": False, "lambda_keys": [x[0] for x in LAMBDAS],
                "stages": [1, 2], "folds": 5, "feature_dimension": 256, "feature_files": 40,
                "cosine_definition": "sample-wise cosine for each of 10 f1..f5 pairs",
                "cka_definition": "centered linear CKA computed fold-wise in float64",
                "coverage": coverage, "scope_boundary": "descriptive mechanism diagnostic; folds are not independent seeds"}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPRESENTATION_ANALYSIS_COMPLETE", "feature_files": 40, "coverage_rows": len(coverage)}))


if __name__ == "__main__":
    main()
