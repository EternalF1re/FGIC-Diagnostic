"""CPU-only verification of published historical predictions and statistics.

Reads real published CSVs; never trains a model or rewrites historical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or ":" in relative or "\\" in relative:
        raise ValueError(f"Unsafe artifact path: {relative}")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact escapes root: {relative}") from exc
    return target


def check_hash(root, path, expected):
    actual = sha256(safe_path(root, path))
    if actual.lower() != expected.lower():
        raise ValueError(f"Hash mismatch: {path}")


def check_config_references(index, manifest):
    sources = manifest.set_index("path")
    checked = 0
    for row in index.to_dict("records"):
        if not row["public_config"]:
            if row["config_source_sha256"]:
                raise ValueError("Original configuration hash has no public reference")
            continue  # Explicitly documented historical provenance gap.
        original = sources.loc[row["public_config"], "source_sha256"]
        if not original or original != row["config_source_sha256"]:
            raise ValueError("Original configuration hash does not match referenced source")
        checked += 1
    return checked


def check_assertion_sources(root, assertions):
    tables = {}
    for row in assertions.to_dict("records"):
        name = row["source_table"]
        if name not in tables:
            tables[name] = pd.read_csv(safe_path(root, name))
        value = float(tables[name].iloc[int(row["source_row"])][row["source_column"]])
        if not np.isclose(value, float(row["value"]), atol=1e-12, rtol=1e-12):
            raise ValueError("Archived assertion disagrees with its source table")


def read_oof(path, expected_n, num_classes):
    frame = pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False)
    required = ["sample_id", "fold", "y_true", "y_pred"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing OOF columns: {path}")
    if len(frame) != expected_n or frame.sample_id.duplicated().any() or frame.sample_id.eq("").any():
        raise ValueError(f"Incomplete or duplicate OOF: {path}")
    for column in required[1:]:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Noninteger {column}: {path}")
        frame[column] = values.astype(np.int64)
    if set(frame.fold) != set(range(5)):
        raise ValueError(f"Not a complete five-fold OOF: {path}")
    for column in ["y_true", "y_pred"]:
        if not frame[column].between(0, num_classes - 1).all():
            raise ValueError(f"Invalid class: {path}")
    return frame


def metrics(frame):
    return {
        "accuracy": float(accuracy_score(frame.y_true, frame.y_pred)),
        "macro_f1": float(f1_score(frame.y_true, frame.y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(frame.y_true, frame.y_pred)),
    }


def align_pair(reference, candidate):
    # Index bootstrap depends on row order. Preserve the archived reference
    # order while explicitly aligning the candidate by sample_id.
    a = reference.reset_index(drop=True)
    if reference.sample_id.duplicated().any() or candidate.sample_id.duplicated().any():
        raise ValueError("Duplicate paired sample IDs")
    if set(reference.sample_id) != set(candidate.sample_id):
        raise ValueError("Paired sample IDs differ")
    b = candidate.set_index("sample_id").loc[a.sample_id].reset_index()
    if not a[["sample_id", "fold", "y_true"]].equals(b[["sample_id", "fold", "y_true"]]):
        raise ValueError("Paired sample IDs, labels or folds differ")
    return a, b


def paired(reference, candidate, implementation, seed, replicates=100_000, bootstrap=True):
    a, b = align_pair(reference, candidate)
    ca = (a.y_pred == a.y_true).to_numpy()
    cb = (b.y_pred == b.y_true).to_numpy()
    n = len(ca)
    n10 = int(np.sum(ca & ~cb))
    n01 = int(np.sum(~ca & cb))
    result = {
        "delta_accuracy_pp": float(100 * (cb.mean() - ca.mean())),
        "mcnemar_p": float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0,
        "n10": n10,
        "n01": n01,
    }
    if not bootstrap:
        return result
    rng = np.random.default_rng(seed)
    if implementation in ["delta3", "delta3_external"]:
        difference = cb.astype(np.int8) - ca.astype(np.int8)
        counts = np.array([(difference == -1).sum(), (difference == 0).sum(), (difference == 1).sum()])
        draws = rng.multinomial(n, counts / n, size=replicates)
        if implementation == "delta3_external":
            samples = (draws[:, 2] - draws[:, 0]) * (100.0 / n)
        else:
            samples = 100.0 * (draws[:, 2] - draws[:, 0]) / n
    elif implementation == "joint4":
        counts = np.bincount(ca.astype(np.int8) * 2 + cb.astype(np.int8), minlength=4)
        draws = rng.multinomial(n, counts / n, size=replicates)
        samples = (draws[:, 1] - draws[:, 2]) * 100 / n
    elif implementation == "joint4_candidate_first":
        counts = np.bincount(cb.astype(np.int8) * 2 + ca.astype(np.int8), minlength=4)
        draws = rng.multinomial(n, counts / n, size=replicates)
        samples = (draws[:, 2] - draws[:, 1]) * 100 / n
    elif implementation == "candidate_gain_loss":
        draws = rng.multinomial(n, [n01 / n, n10 / n, 1 - (n01 + n10) / n], size=replicates)
        samples = (draws[:, 0] - draws[:, 1]) / n * 100.0
    elif implementation == "sample_indices":
        difference = cb.astype(np.float64) - ca.astype(np.float64)
        samples = np.empty(replicates, dtype=np.float64)
        # Subdivide the historical 1000-row batches to bound memory, preserving
        # exactly the same PCG64 integer stream and sample-index resampling.
        for start in range(0, replicates, 128):
            count = min(128, replicates - start)
            indices = rng.integers(0, n, size=(count, n))
            samples[start : start + count] = difference[indices].mean(axis=1)
        samples *= 100
    else:
        raise ValueError(f"Unknown historical bootstrap implementation: {implementation}")
    result["ci95_low_pp"], result["ci95_high_pp"] = map(float, np.percentile(samples, [2.5, 97.5]))
    return result


def verify(root, bootstrap=True):
    root = Path(root)
    manifest = pd.read_csv(root / "MANIFEST.csv", keep_default_na=False)
    if manifest.path.duplicated().any():
        raise ValueError("Duplicate manifest paths")
    for row in manifest.to_dict("records"):
        check_hash(root, row["path"], row["sha256"])
    actual = {f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file() and f != root / "MANIFEST.csv"}
    if actual != set(manifest.path):
        raise ValueError("Manifest does not enumerate exactly the published artifacts")
    index = pd.read_csv(root / "OOF_INDEX.csv", keep_default_na=False)
    if index.artifact_id.duplicated().any():
        raise ValueError("Duplicate OOF artifact IDs")
    configuration_references = check_config_references(index, manifest)
    public_config_assertions = 0
    config_list = root.parent / "docs" / "CONFIG_SHA256.csv"
    if config_list.is_file():
        for row in pd.read_csv(config_list).to_dict("records"):
            check_hash(root.parent, row["config_path"], row["sha256"])
            public_config_assertions += 1
    frames, rows = {}, []
    pools = {}
    for record in index.to_dict("records"):
        frame = read_oof(safe_path(root, record["path"]), int(record["n"]), int(record["num_classes"]))
        frames[record["artifact_id"]] = frame
        # Every experiment in a dataset must have the same frozen sample, label
        # and fold pool; do not confuse a count check with sample coverage.
        pool = frame[["sample_id", "fold", "y_true"]].sort_values("sample_id").reset_index(drop=True)
        if record["pool_id"] in pools and not pools[record["pool_id"]].equals(pool):
            raise ValueError(f"Pool coverage/fold mismatch: {record['artifact_id']}")
        pools[record["pool_id"]] = pool
        rows.append({"artifact_id": record["artifact_id"], **metrics(frame)})
    split_records = pd.read_csv(root / "SPLIT_INDEX.csv", keep_default_na=False)
    for record in split_records.to_dict("records"):
        split = pd.read_csv(safe_path(root, record["path"]), dtype={"sample_id": str})
        if (
            len(split) != int(record["n"])
            or split.dataset_index.duplicated().any()
            or split.sample_id.duplicated().any()
        ):
            raise ValueError("Incomplete or overlapping archived split")
        if set(split.partition) != {"train", "heldout"}:
            raise ValueError("Unknown archived partition")
        heldout = split[split.partition == "heldout"]
        pool = pools[record["pool_id"]]
        expected = pool[pool.fold == int(record["fold"])]
        if set(heldout.sample_id) != set(expected.sample_id):
            raise ValueError("OOF IDs not in the corresponding archived heldout partition")
        if len(heldout) != int(record["heldout_n"]):
            raise ValueError("Archived heldout count mismatch")
        if (split.partition == "train").sum() != int(record["training_n"]):
            raise ValueError("Archived training count mismatch")
    recomputed = pd.DataFrame(rows).set_index("artifact_id")
    differences = []
    expectations = pd.read_csv(root / "HISTORICAL_METRICS.csv", keep_default_na=False)
    check_assertion_sources(root, expectations)
    for row in expectations.to_dict("records"):
        actual_value = recomputed.loc[row["artifact_id"], row["metric"]]
        difference = float(actual_value - float(row["value"]))
        differences.append(
            {
                "kind": "metric",
                "artifact_id": row["artifact_id"],
                "metric": row["metric"],
                "archived": float(row["value"]),
                "recomputed": float(actual_value),
                "difference": difference,
            }
        )
    recipes = pd.read_csv(root / "PAIRED_RECIPES.csv", keep_default_na=False)
    for recipe in recipes.to_dict("records"):
        values = paired(
            frames[recipe["reference"]],
            frames[recipe["candidate"]],
            recipe["implementation"],
            int(recipe["bootstrap_seed"]),
            int(recipe["bootstrap_replicates"]),
            bootstrap,
        )
        for key, actual_value in values.items():
            difference = float(actual_value - float(recipe[key]))
            differences.append(
                {
                    "kind": "paired",
                    "artifact_id": recipe["comparison_id"],
                    "metric": key,
                    "archived": float(recipe[key]),
                    "recomputed": float(actual_value),
                    "difference": difference,
                }
            )
    bad = [r for r in differences if not np.isclose(r["archived"], r["recomputed"], rtol=1e-12, atol=1e-12)]
    # Fold SD is never relabeled as independent-seed SD.
    seeds = pd.read_csv(root / "SEED_SUMMARY_EXPECTATIONS.csv", keep_default_na=False)
    for row in seeds.to_dict("records"):
        ids = row["artifact_ids"].split("|")
        metadata = index.set_index("artifact_id").loc[ids]
        if metadata.seed.nunique() != len(ids) or metadata.pool_id.nunique() != 1:
            raise ValueError("Seed summary requires distinct training seeds on one frozen pool")
        values = recomputed.loc[ids, row["metric"]].to_numpy()
        for statistic in ["mean", "sample_sd"]:
            actual_value = float(values.mean() if statistic == "mean" else values.std(ddof=1))
            delta = actual_value - float(row[statistic])
            item = {
                "kind": "seed_summary",
                "artifact_id": row["summary_id"],
                "metric": statistic,
                "archived": float(row[statistic]),
                "recomputed": actual_value,
                "difference": delta,
            }
            differences.append(item)
            if not np.isclose(actual_value, float(row[statistic]), atol=1e-12, rtol=1e-12):
                bad.append(item)
    return {
        "status": "FAIL" if bad else "PASS",
        "files_verified": len(manifest),
        "oof_sets": len(index),
        "oof_prediction_rows": int(index.n.sum()),
        "historical_metric_assertions": len(expectations),
        "paired_comparisons": len(recipes),
        "independent_seed_summary_assertions": len(seeds) * 2,
        "archived_split_checks": len(split_records),
        "configuration_source_references": configuration_references,
        "configuration_reference_gaps": int(index.public_config.eq("").sum()),
        "portable_config_hash_assertions": public_config_assertions,
        "bootstrap_recomputed": bool(bootstrap),
        "differences_outside_tolerance": len(bad),
        "differences": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("paper_artifacts"))
    parser.add_argument(
        "--skip-bootstrap", action="store_true", help="Fast integrity/metrics/McNemar only; does not validate CIs"
    )
    parser.add_argument("--output", type=Path, help="Write fresh audit outside the immutable artifact directory")
    args = parser.parse_args()
    if args.output:
        try:
            args.output.resolve().relative_to(args.artifact_root.resolve())
        except ValueError:
            pass
        else:
            parser.error("--output must be outside artifact-root")
    result = verify(args.artifact_root, not args.skip_bootstrap)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "differences"}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
