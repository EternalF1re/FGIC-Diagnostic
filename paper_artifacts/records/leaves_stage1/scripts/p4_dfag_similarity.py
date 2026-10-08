"""Phase 2C P4: summarize existing standalone-DFAG branch features."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VALIDATION_ROOT = Path(__file__).resolve().parents[2]
SOURCE = VALIDATION_ROOT / "inference" / "dfag"
OUT = Path(__file__).resolve().parents[1] / "p4_dfag_anchor_similarity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dist_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("INTEGRITY_FAILURE: NaN/Inf in feature diagnostic")
    q = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=1)),
        "median": float(q[3]), "min": float(values.min()), "max": float(values.max()),
        "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
        "p75": float(q[4]), "p95": float(q[5]), "p99": float(q[6]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    fold_summary: list[dict[str, object]] = []
    norm_summary: list[dict[str, object]] = []
    provenance: dict[str, object] = {
        "phase": "Phase2C P4", "training_performed": False,
        "artifact_type": "existing Phase2A standalone DFAG original-view features",
        "feature_definition": "anchor_features.npy and specific_features.npy immediately before gate fusion",
        "D_g": 1536, "folds": [],
    }

    for fold in range(5):
        base = SOURCE / f"fold_{fold}" / "original" / "dynamic"
        manifest_path = base / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != "dfag" or manifest.get("tta_mode") != "original" or manifest.get("fusion_mode") != "dynamic":
            raise RuntimeError(f"INTEGRITY_FAILURE: wrong DFAG source semantics in fold {fold}")
        if manifest.get("training_performed") is not False:
            raise RuntimeError(f"INTEGRITY_FAILURE: source is not checkpoint-only in fold {fold}")
        anc = np.load(base / "anchor_features.npy")
        spec = np.load(base / "specific_features.npy")
        ids = np.load(base / "dataset_indices.npy")
        labels = np.load(base / "labels.npy")
        gates = np.load(base / "gate_vectors.npy")
        if anc.shape != spec.shape or anc.shape[1:] != (1536,):
            raise RuntimeError(f"INTEGRITY_FAILURE: fold {fold} feature shapes {anc.shape}/{spec.shape}")
        if not (len(ids) == len(labels) == len(anc) == len(gates)):
            raise RuntimeError(f"INTEGRITY_FAILURE: fold {fold} array lengths differ")
        if not all(np.isfinite(x).all() for x in (anc, spec, gates)):
            raise RuntimeError(f"INTEGRITY_FAILURE: fold {fold} contains NaN/Inf")
        anc64, spec64 = anc.astype(np.float64), spec.astype(np.float64)
        anc_norm = np.linalg.norm(anc64, axis=1)
        spec_norm = np.linalg.norm(spec64, axis=1)
        if np.any(anc_norm == 0) or np.any(spec_norm == 0):
            raise RuntimeError(f"INTEGRITY_FAILURE: fold {fold} has zero-norm branch features")
        cosine = np.sum(anc64 * spec64, axis=1) / (anc_norm * spec_norm)
        distance = 1.0 - cosine
        for i in range(len(ids)):
            all_rows.append({
                "sample_id": int(ids[i]), "fold": fold, "label": int(labels[i]),
                "cosine_similarity": float(cosine[i]), "cosine_distance": float(distance[i]),
                "anchor_norm": float(anc_norm[i]), "plastic_norm": float(spec_norm[i]),
                "gate_mean": float(np.mean(gates[i])), "gate_std_across_dimensions": float(np.std(gates[i])),
            })
        for metric, values in (("cosine_similarity", cosine), ("cosine_distance", distance)):
            row: dict[str, object] = {"scope": "fold", "fold": fold, "metric": metric}
            row.update(dist_stats(values))
            fold_summary.append(row)
        for metric, values in (("anchor_norm", anc_norm), ("plastic_norm", spec_norm), ("gate_mean", gates.mean(axis=1))):
            row = {"scope": "fold", "fold": fold, "metric": metric}
            row.update(dist_stats(values))
            norm_summary.append(row)
        provenance["folds"].append({
            "fold": fold, "manifest": str(manifest_path), "n": len(ids),
            "checkpoint_filename": manifest["checkpoint"]["filename"],
            "checkpoint_recorded_sha256": manifest["checkpoint"]["sha256"],
            "anchor_checkpoint_filename": manifest["anchor_checkpoint"]["filename"],
            "anchor_checkpoint_recorded_sha256": manifest["anchor_checkpoint"]["sha256"],
            "feature_sha256": {name: sha256(base / name) for name in ("anchor_features.npy", "specific_features.npy", "dataset_indices.npy", "labels.npy", "gate_vectors.npy")},
        })

    ids_all = np.asarray([row["sample_id"] for row in all_rows], dtype=np.int64)
    if len(ids_all) != 18_353 or len(np.unique(ids_all)) != 18_353 or set(ids_all.tolist()) != set(range(18_353)):
        raise RuntimeError("INTEGRITY_FAILURE: DFAG OOF does not contain exactly 18,353 unique IDs 0..18352")
    cosine_all = np.asarray([row["cosine_similarity"] for row in all_rows])
    distance_all = 1.0 - cosine_all
    for metric, values in (("cosine_similarity", cosine_all), ("cosine_distance", distance_all)):
        row = {"scope": "pooled_oof", "fold": "ALL", "metric": metric}
        row.update(dist_stats(values))
        fold_summary.append(row)
    for metric in ("anchor_norm", "plastic_norm", "gate_mean"):
        values = np.asarray([row[metric] for row in all_rows])
        summary = {"scope": "pooled_oof", "fold": "ALL", "metric": metric}
        summary.update(dist_stats(values))
        norm_summary.append(summary)

    write_csv(OUT / "anchor_plastic_similarity.csv", all_rows)
    write_csv(OUT / "fold_summary.csv", fold_summary)
    write_csv(OUT / "feature_norm_summary.csv", norm_summary)
    provenance.update({"sample_count": 18_353, "unique_sample_count": 18_353, "integrity": "PASS"})
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    order = np.argsort(cosine_all)
    ecdf = np.arange(1, len(cosine_all) + 1) / len(cosine_all)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(cosine_all, bins=80, color="#31688e", alpha=0.85)
    axes[0].set(xlabel="cos(f_anc, f_spec)", ylabel="Samples", title="Standalone DFAG: cosine similarity")
    axes[1].plot(cosine_all[order], ecdf, color="#35b779", lw=1.5)
    axes[1].set(xlabel="cos(f_anc, f_spec)", ylabel="ECDF", title="OOF empirical CDF")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "similarity_distribution.png", dpi=220)
    fig.savefig(OUT / "similarity_distribution.pdf")
    plt.close(fig)

    pooled = next(row for row in fold_summary if row["scope"] == "pooled_oof" and row["metric"] == "cosine_similarity")
    gate = next(row for row in norm_summary if row["scope"] == "pooled_oof" and row["metric"] == "gate_mean")
    readme = f"""# P4 standalone DFAG anchor-plastic similarity

Status: COMPLETE (existing-artifact recomputation; no inference or training).

## Directly measured

- Source: Phase2A standalone DFAG, original view, dynamic fusion, D_g=1536.
- OOF coverage: 18,353 unique samples across five held-out folds.
- Cosine similarity: mean {pooled['mean']:.8f}, std {pooled['std']:.8f}, median {pooled['median']:.8f}, p05 {pooled['p05']:.8f}, p95 {pooled['p95']:.8f}.
- Sample-wise gate mean: mean {gate['mean']:.8f}, std {gate['std']:.8f}.

## Interpretation boundary

High similarity, if observed, is only consistent with limited routing pressure. It does not prove the cause of weak gate dynamics. Feature norms and gate summaries are descriptive only.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("P4 COMPLETE")


if __name__ == "__main__":
    main()
