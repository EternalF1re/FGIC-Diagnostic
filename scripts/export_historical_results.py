"""Read-only exporter for the historical FGIC experiment workspace.

Requires actual local experiment outputs. It never fabricates missing results.
Private absolute paths are recorded only in --internal-audit, outside this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {
    "external": "final_external_protocol",
    "cross_backbone": "final_cross_backbone_evaluation",
    "controlled": "final_controlled_reimplementation",
    "leaves_round1": "phase2d_round1",
    "leaves_round2": "phase2d_round2a",
    "leaves_screen": "phase2b_screen",
    "leaves_stage1": "phase2c_inference_diagnostic",
    "dfag": "phase2f_unified_standalone_dfag",
    "dfag_confirmation": "phase2f_seed45_46_confirmation",
    "state_interpolation": "phase2d_checkpoint_state_interpolation",
    "complexity": "final_three_model_complexity",
    "progressive_complexity": "final_progressive_head_complexity",
    "protocol_source": "corrected_external_protocol_smoke/scripts",
}
POOL = {"leaves": (18353, 176), "cub": (5994, 200), "cars": (8144, 196), "flowers": (2040, 102)}
DATASETS = {
    "CUB-200-2011": "cub",
    "Stanford Cars": "cars",
    "Oxford Flowers-102": "flowers",
    "Classify Leaves": "leaves",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sanitize(text, source_root, identities=()):
    for form in [str(source_root), str(source_root).replace("\\", "/"), str(source_root).replace("\\", "\\\\")]:
        text = text.replace(form, "<EXPERIMENT_ROOT>")
    text = re.sub(r"[A-Za-z]:[\\/][^\r\n\t\x22\x27,;<>|]*", "<LOCAL_PATH>", text)
    text = re.sub(r"/(?:mnt/[a-z]|home|Users)/[^\s\x22\x27,;<>|]*", "<LOCAL_PATH>", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<EMAIL_REDACTED>", text)
    for word in ["Administrator", *identities]:
        text = text.replace(word, "<IDENTITY_REDACTED>")
    return text


class Exporter:
    def __init__(self, source_root, out, audit, identities=()):
        self.source = source_root.resolve()
        self.base = self.source / "phase2_controlled_validation"
        self.out = out.resolve()
        self.audit = audit.resolve()
        self.identities = identities
        if self.out.exists():
            raise ValueError("Refusing to overwrite an existing artifact directory")
        try:
            self.audit.relative_to(self.out.parent)
        except ValueError:
            pass
        else:
            raise ValueError("Internal audit must be outside the public repository")
        self.out.mkdir(parents=True)
        self.files, self.private, self.index, self.metrics, self.recipes, self.seeds = [], [], [], [], [], []
        self.split_index = []
        self.frames, self.copied = {}, {}
        self.gaps = []

    def path(self, group, relative=""):
        return self.base / GROUPS[group] / relative

    def record(self, dest, source=None, role="", group="", transform="generated_from_actual_artifacts"):
        self.files.append(
            {
                "path": dest.relative_to(self.out).as_posix(),
                "sha256": digest(dest),
                "bytes": dest.stat().st_size,
                "role": role,
                "experiment_group": group,
                "source_id": source.relative_to(self.source).as_posix() if source else "",
                "source_sha256": digest(source) if source else "",
                "transformation": transform,
            }
        )
        if source:
            self.private.append(
                {
                    "public_path": dest.relative_to(self.out).as_posix(),
                    "original_path": str(source),
                    "original_sha256": digest(source),
                }
            )

    def write_csv(self, relative, frame, source=None, role="", group=""):
        dest = self.out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, index=False, lineterminator="\n")
        self.record(dest, source, role, group)
        return dest.relative_to(self.out).as_posix()

    def copy_text(self, group, source):
        if source in self.copied:
            return self.copied[source]
        relative = source.relative_to(self.base / GROUPS[group])
        dest = self.out / "records" / group / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            original = source.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            encoding = "utf-16" if source.read_bytes().startswith((b"\xff\xfe", b"\xfe\xff")) else "gb18030"
            original = source.read_text(encoding=encoding)
            self.gaps.append(f"Legacy {encoding} text transcoded to UTF-8: {group}/{relative.as_posix()}")
        text = sanitize(original, self.source, self.identities)
        # Validate structured copies after path redaction.
        if source.suffix == ".json":
            json.loads(text)
        dest.write_text(text, encoding="utf-8")
        self.record(
            dest,
            source,
            "historical_record",
            group,
            "path_and_identity_redaction_utf8" if text != original else "utf8_copy",
        )
        self.copied[source] = dest.relative_to(self.out).as_posix()
        return self.copied[source]

    def load(self, source):
        with np.load(source, allow_pickle=False) as archive:
            keys = archive.files
            aliases = {
                "sample_id": ["sample_id", "sample_ids"],
                "fold": ["fold", "folds", "fold_ids"],
                "y_true": ["y_true", "label", "labels"],
                "y_pred": ["y_pred", "prediction", "predictions"],
            }
            result = {name: archive[next(k for k in options if k in keys)].copy() for name, options in aliases.items()}
            if "logits" in keys:
                logits = archive["logits"]
                if not np.isfinite(logits).all() or not np.array_equal(logits.argmax(axis=1), result["y_pred"]):
                    raise ValueError(f"Invalid original logits/predictions: {source.name}")
        return pd.DataFrame(result)

    def add(
        self,
        group,
        name,
        frame,
        source,
        dataset="leaves",
        seed=42,
        method="",
        backbone="inception_resnet_v2",
        stage="",
        config=None,
        sources=None,
    ):
        artifact_id = f"{group}/{name}"
        n, classes = POOL[dataset]
        frame = frame[["sample_id", "fold", "y_true", "y_pred"]].copy()
        frame["sample_id"] = frame.sample_id.astype(str)
        if len(frame) != n or frame.sample_id.duplicated().any() or set(frame.fold) != set(range(5)):
            raise ValueError(f"Incomplete historical OOF: {artifact_id}")
        if not frame.y_true.between(0, classes - 1).all() or not frame.y_pred.between(0, classes - 1).all():
            raise ValueError(f"Invalid historical classes: {artifact_id}")
        frame = frame.reset_index(drop=True)
        rel = self.write_csv(f"oof/{group}/{name}.csv", frame, source, "complete_five_fold_oof", group)
        config_hash = digest(config) if config and config.exists() else ""
        config_path = self.copy_text(self.source_group(config), config) if config and config.exists() else ""
        self.index.append(
            {
                "artifact_id": artifact_id,
                "path": rel,
                "dataset": dataset,
                "pool_id": dataset,
                "n": n,
                "num_classes": classes,
                "seed": seed,
                "method": method or name,
                "backbone": backbone,
                "checkpoint_stage": stage
                or (re.search(r"stage[12]", name)[0] if re.search(r"stage[12]", name) else name),
                "config_source_sha256": config_hash,
                "public_config": config_path,
                "formal_status": "formal_or_completed_frozen_diagnostic",
                "evaluation": "original_view_no_tta_no_fold_ensemble",
                "source_sha256": digest(source),
                "source_verification": "archived_output; numeric_summary_crosscheck_when_available",
            }
        )
        self.frames[artifact_id] = frame
        for dep in sources or []:
            self.private.append({"public_path": rel, "original_path": str(dep), "original_sha256": digest(dep)})
        return artifact_id

    def metric_expectation(self, artifact_id, row, source, row_index, aliases=None):
        aliases = aliases or {"accuracy": "accuracy", "macro_f1": "macro_f1", "balanced_accuracy": "balanced_accuracy"}
        public = self.copy_text(self.source_group(source), source)
        for metric, column in aliases.items():
            if column in row and pd.notna(row[column]):
                self.metrics.append(
                    {
                        "artifact_id": artifact_id,
                        "metric": metric,
                        "value": float(row[column]),
                        "source_table": public,
                        "source_row": row_index,
                        "source_column": column,
                    }
                )

    def source_group(self, source):
        for group, folder in GROUPS.items():
            try:
                source.relative_to(self.base / folder)
                return group
            except ValueError:
                continue
        raise ValueError("Unknown source group")

    def recipe(self, source, i, reference, candidate, implementation, row, delta, p, n10, n01, seed):
        if reference not in self.frames or candidate not in self.frames:
            self.gaps.append(f"Pair unavailable: {reference} vs {candidate} from {source.name}")
            return
        self.recipes.append(
            {
                "comparison_id": f"{self.source_group(source)}/{source.stem}/{i}",
                "reference": reference,
                "candidate": candidate,
                "implementation": implementation,
                "bootstrap_seed": int(seed),
                "bootstrap_replicates": int(row.get("bootstrap_replicates", 100000)),
                "delta_accuracy_pp": row[delta],
                "ci95_low_pp": row["ci95_low_pp"],
                "ci95_high_pp": row["ci95_high_pp"],
                "mcnemar_p": row[p],
                "n10": row[n10],
                "n01": row[n01],
                "historical_source": self.copy_text(self.source_group(source), source),
                "source_row": i,
                "difference_definition": "candidate_minus_reference_percentage_points",
            }
        )

    def export_primary(self):
        # Config hashes are for exact original bytes; redacted copies have their own manifest hash.
        for group, folder, config_name in [
            ("external", "final_oof_predictions", "final_external_protocol_config.json"),
            ("cross_backbone", "final_cross_backbone_oof_predictions", "final_cross_backbone_protocol_config.json"),
            ("controlled", "formal_run/pooled_oof", ""),
            ("leaves_round1", "oof", ""),
            ("leaves_round2", "oof", ""),
            ("dfag", "oof", ""),
            ("dfag_confirmation", "oof", ""),
            ("state_interpolation", "oof", ""),
        ]:
            for source in sorted(self.path(group, folder).glob("*.npz")):
                name = source.stem
                dataset = name.split("__")[0] if group in ["external", "controlled"] else "leaves"
                seed_match = re.search(r"seed(\d+)", name)
                seed = int(seed_match[1]) if seed_match else 42
                backbone = (
                    ("convnext_tiny" if name.startswith("convnext_tiny") else "resnet50")
                    if group == "cross_backbone"
                    else ("resnet50" if group == "controlled" else "inception_resnet_v2")
                )
                config = self.path(group, config_name) if config_name else None
                if group == "controlled":
                    data, method = name.split("__")
                    config = self.path(group, f"configs/{method}_{data}.json")
                elif group in ["dfag", "dfag_confirmation"]:
                    config = self.path(group, f"configs/seed{seed}.json")
                elif group in ["leaves_round1", "leaves_round2"]:
                    config = self.path(group, f"configs/{name.split('_stage')[0]}.json")
                elif group == "state_interpolation":
                    config = (
                        self.path("leaves_screen", "configs/controlled_screen.json")
                        if seed == 42
                        else self.path("leaves_round1", f"configs/baseline_seed{seed}.json")
                    )
                self.add(group, name, self.load(source), source, dataset, seed, backbone=backbone, config=config)
        # Earlier frozen screen supplies Leaves baseline42 and lambda .1/1.0.
        for variant, name in [("#0", "baseline"), ("#1", "lambda_1_0"), ("#2", "lambda_0_1")]:
            for stage in [1, 2]:
                if stage == 1:
                    source = self.path(
                        "leaves_stage1", f"p1_stage1_only_oof/stage1_oof_predictions_variant_{variant[1:]}.npz"
                    )
                    frame = self.load(source)
                    dependencies = []
                else:
                    folder = self.path("leaves_screen", variant)
                    source = folder / "oof_predictions.npy"
                    ids = np.load(folder / "oof_sample_ids.npy")
                    folds = np.full(len(ids), -1)
                    id_to_position = {int(v): i for i, v in enumerate(ids)}
                    dependencies = [source, folder / "oof_sample_ids.npy", folder / "oof_labels.npy"]
                    for fold in range(5):
                        held = folder / f"fold_{fold}/validation_sample_ids.npy"
                        dependencies.append(held)
                        for sample in np.load(held):
                            pos = id_to_position[int(sample)]
                            if folds[pos] != -1:
                                raise ValueError("Overlapping heldout folds")
                            folds[pos] = fold
                    frame = pd.DataFrame(
                        {
                            "sample_id": ids,
                            "fold": folds,
                            "y_true": np.load(folder / "oof_labels.npy"),
                            "y_pred": np.load(source),
                        }
                    )
                self.add(
                    "leaves_screen",
                    f"{name}_seed42_stage{stage}",
                    frame,
                    source,
                    stage=f"stage{stage}",
                    sources=dependencies,
                    config=self.path("leaves_screen", "configs/controlled_screen.json"),
                )
        # Baseline seeds45/46 are exported from real per-fold arrays, not reconstructed from aggregate tables.
        for seed in [45, 46]:
            folder = self.path("leaves_round2", f"baseline_seed{seed}")
            for stage in [1, 2]:
                pieces, dependencies = [], []
                for fold in range(5):
                    files = [
                        folder / f"fold_{fold}/stage{stage}_validation_{kind}.npy"
                        for kind in ["sample_ids", "labels", "predictions"]
                    ]
                    arrays = [np.load(f, allow_pickle=False) for f in files]
                    pieces.append(
                        pd.DataFrame({"sample_id": arrays[0], "y_true": arrays[1], "y_pred": arrays[2], "fold": fold})
                    )
                    dependencies.extend(files)
                self.add(
                    "leaves_round2",
                    f"baseline_seed{seed}_stage{stage}",
                    pd.concat(pieces),
                    dependencies[0],
                    seed=seed,
                    stage=f"stage{stage}",
                    sources=dependencies,
                    config=self.path("leaves_round2", f"configs/baseline_seed{seed}.json"),
                )
        # Counterfactual predictions were recorded at inference; retain their distinct semantics.
        for group, seeds in [("dfag", [42, 43, 44]), ("dfag_confirmation", [45, 46])]:
            for seed in seeds:
                pieces = []
                deps = []
                for fold in range(5):
                    source = self.path(group, f"seed{seed}/fold_{fold}/heldout_dfag_artifacts.npz")
                    deps.append(source)
                    with np.load(source, allow_pickle=False) as z:
                        pieces.append(
                            {
                                k: z[k].copy()
                                for k in z.files
                                if k in ["sample_id", "label"] or k.startswith("prediction_")
                            }
                        )
                for mode in [
                    "dynamic",
                    "mean_vector",
                    "mean_scalar",
                    "constant_0_5",
                    "anchor_forced",
                    "plastic_forced",
                ]:
                    frame = pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "sample_id": x["sample_id"],
                                    "y_true": x["label"],
                                    "y_pred": x[f"prediction_{mode}"],
                                    "fold": fold,
                                }
                            )
                            for fold, x in enumerate(pieces)
                        ]
                    )
                    self.add(
                        group,
                        f"seed{seed}_counterfactual_{mode}",
                        frame,
                        deps[0],
                        seed=seed,
                        stage="counterfactual",
                        config=self.path(group, f"configs/seed{seed}.json"),
                        sources=deps,
                    )
        for seed in [45, 46]:
            source = self.path("dfag_confirmation", f"comparators/seed{seed}_alpha_0_5.npz")
            self.add(
                "dfag_confirmation",
                f"seed{seed}_alpha05_comparator",
                self.load(source),
                source,
                seed=seed,
                stage="alpha0.5",
            )

    def baseline(self, seed, stage):
        group = "leaves_screen" if seed == 42 else ("leaves_round1" if seed in [43, 44] else "leaves_round2")
        name = (
            f"baseline_seed{seed}_stage{stage}"
            if seed == 42
            else f"baseline_seed{seed}_stage{stage}" + ("_oof" if seed in [43, 44] else "")
        )
        return f"{group}/{name}"

    def leaves_variant(self, name, stage):
        if name == "baseline":
            return self.baseline(42, stage)
        group = "leaves_screen" if name in ["lambda_0_1", "lambda_1_0"] else "leaves_round1"
        return f"{group}/{name}_seed42_stage{stage}" + ("_oof" if group == "leaves_round1" else "")

    def statistics(self):
        f = self.path("external", "final_external_oof_summary.csv")
        for i, r in pd.read_csv(f, keep_default_na=False).iterrows():
            data = DATASETS[r.dataset]
            method = (
                "ours_ft" if r.method == "Ours-FT" else f"progressive_lambda_{float(r['lambda']):.1f}".replace(".", "_")
            )
            self.metric_expectation(f"external/{data}__{method}__stage{r.stage}", r, f, i)
        f = self.path("cross_backbone", "FINAL_CROSS_BACKBONE_OOF_SUMMARY.csv")
        for i, r in pd.read_csv(f).iterrows():
            self.metric_expectation(f"cross_backbone/{r.backbone}_{r.outcome}", r, f, i)
        f = self.path("controlled", "formal_run/final_pooled_oof_metrics.csv")
        for i, r in pd.read_csv(f).iterrows():
            self.metric_expectation(f"controlled/{r.dataset}__{r.method}", r, f, i)
        for group in ["leaves_round1", "leaves_round2"]:
            f = self.path(
                group, "oof/round1_oof_metrics.csv" if group == "leaves_round1" else "statistics/mhsa_oof_metrics.csv"
            )
            if f.exists():
                for i, r in pd.read_csv(f).iterrows():
                    if r.get("scope", "pooled") == "pooled":
                        rid = r.get("run_id", r.get("configuration", ""))
                        aid = f"{group}/{rid}_stage{int(r.stage)}_oof"
                        if aid in self.frames:
                            self.metric_expectation(aid, r, f, i)
        f = self.path("leaves_round2", "statistics/baseline_five_seed_per_seed.csv")
        df = pd.read_csv(f)
        for i, r in df.iterrows():
            seed = int(r.get("training_seed", r.get("seed")))
            for stage in [1, 2]:
                self.metric_expectation(
                    self.baseline(seed, stage),
                    r,
                    f,
                    i,
                    {m: f"stage{stage}_{m}" for m in ["accuracy", "macro_f1", "balanced_accuracy"]},
                )
        # Seed SD expectations come from the archived five-seed report, not a fresh expected value.
        fsum = self.path("leaves_round2", "statistics/baseline_five_seed_summary.csv")
        for i, r in pd.read_csv(fsum).iterrows():
            if "stage" in r and "metric" in r:
                stage = int(r.stage)
                self.seeds.append(
                    {
                        "summary_id": f"baseline_five_seed_stage{stage}_{r.metric}",
                        "artifact_ids": "|".join(self.baseline(s, stage) for s in [42, 43, 44, 45, 46]),
                        "metric": r.metric,
                        "mean": r["mean"],
                        "sample_sd": r.sample_sd,
                        "historical_source": self.copy_text("leaves_round2", fsum),
                        "source_row": i,
                    }
                )
                continue
            for stage in [1, 2]:
                for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
                    key = f"stage{stage}_{metric}"
                    sd = next((k for k in [key + "_std", key + "_sample_sd"] if k in r), None)
                    if key + "_mean" in r and sd:
                        self.seeds.append(
                            {
                                "summary_id": f"baseline_five_seed_stage{stage}_{metric}",
                                "artifact_ids": "|".join(self.baseline(s, stage) for s in [42, 43, 44, 45, 46]),
                                "metric": metric,
                                "mean": r[key + "_mean"],
                                "sample_sd": r[sd],
                                "historical_source": self.copy_text("leaves_round2", fsum),
                                "source_row": i,
                            }
                        )
        for group, relative in [
            ("dfag", "metrics/dfag_per_seed.csv"),
            ("dfag_confirmation", "metrics/dfag_seed45_46_per_seed.csv"),
        ]:
            f = self.path(group, relative)
            for i, r in pd.read_csv(f).iterrows():
                self.metric_expectation(f"{group}/seed{int(r.seed)}_dfag_oof", r, f, i)
        f = self.path("dfag_confirmation", "dfag_five_seed_summary.csv")
        for i, r in pd.read_csv(f).iterrows():
            self.seeds.append(
                {
                    "summary_id": f"dfag_five_seed_{r.metric}",
                    "artifact_ids": "|".join(
                        f"{('dfag' if s < 45 else 'dfag_confirmation')}/seed{s}_dfag_oof" for s in [42, 43, 44, 45, 46]
                    ),
                    "metric": r.metric,
                    "mean": r.mean_across_seeds,
                    "sample_sd": r.sample_sd_across_seeds,
                    "historical_source": self.copy_text("dfag_confirmation", f),
                    "source_row": i,
                }
            )
        f = self.path("external", "FINAL_EXTERNAL_PAIRED_STATISTICS.csv")
        for i, r in pd.read_csv(f, keep_default_na=False).iterrows():
            dataset = DATASETS[r.dataset]

            def aid(method, lam, dataset=dataset, stage=r.stage):
                key = "ours_ft" if method == "Ours-FT" else f"progressive_lambda_{float(lam):.1f}".replace(".", "_")
                return f"external/{dataset}__{key}__stage{stage}"

            self.recipe(
                f,
                i,
                aid(r.method_a, r.lambda_a),
                aid(r.method_b, r.lambda_b),
                "delta3_external",
                r,
                "delta_accuracy_pp",
                "mcnemar_p",
                "n10",
                "n01",
                20260818,
            )
        f = self.path("cross_backbone", "FINAL_CROSS_BACKBONE_PAIRED_STATISTICS.csv")
        mapping = {
            "Ours-FT Stage1": "ours_stage1",
            "Ours-FT Stage2": "ours_stage2",
            "Progressive Stage1": "progressive_stage1",
            "Progressive Stage2": "progressive_stage2",
            "DFAG": "dfag",
            "Ours-FT alpha=0.5": "ours_alpha0.5",
            "Ours-FT alpha0.5": "ours_alpha0.5",
        }
        for i, r in pd.read_csv(f).iterrows():
            if r.reference not in mapping or r.candidate not in mapping:
                self.gaps.append(f"Unmapped cross-backbone pair: {r.reference} / {r.candidate}")
                continue
            self.recipe(
                f,
                i,
                f"cross_backbone/{r.backbone}_{mapping[r.reference]}",
                f"cross_backbone/{r.backbone}_{mapping[r.candidate]}",
                "delta3",
                r,
                "delta_accuracy_pp",
                "mcnemar_exact_two_sided_p",
                "n10_reference_correct_candidate_wrong",
                "n01_reference_wrong_candidate_correct",
                r.bootstrap_seed,
            )
        f = self.path("controlled", "formal_run/final_paired_statistics.csv")
        for i, r in pd.read_csv(f).iterrows():
            self.recipe(
                f,
                i,
                f"controlled/{r.dataset}__{r.reference}",
                f"controlled/{r.dataset}__{r.candidate}",
                "sample_indices",
                r,
                "delta_accuracy_pp",
                "mcnemar_exact_two_sided_p",
                "mcnemar_b",
                "mcnemar_c",
                42,
            )
        f = self.path("leaves_round1", "statistics/lambda_direct_paired.csv")
        for i, r in pd.read_csv(f).iterrows():
            self.recipe(
                f,
                i,
                self.leaves_variant(r.a, int(r.stage)),
                self.leaves_variant(r.b, int(r.stage)),
                "joint4",
                r,
                "delta_b_minus_a_pp",
                "mcnemar_exact_two_sided_p",
                "n10",
                "n01",
                r.bootstrap_seed,
            )
        for group, seeds in [("dfag", [42, 43, 44]), ("dfag_confirmation", [45, 46])]:
            for seed in seeds:
                for stage in [1, 2]:
                    if group == "dfag":
                        f = self.path(group, f"statistics/dfag_vs_stage{stage}.csv")
                        selected = pd.read_csv(f)
                        selected = selected[selected.seed == seed]
                    else:
                        f = self.path(group, "statistics/dfag_seed45_46_paired.csv")
                        selected = pd.read_csv(f)
                        selected = selected[(selected.seed == seed) & selected.comparison.str.contains(f"stage{stage}")]
                    for i, r in selected.iterrows():
                        self.recipe(
                            f,
                            i,
                            self.baseline(seed, stage),
                            f"{group}/seed{seed}_dfag_oof",
                            "candidate_gain_loss",
                            r,
                            "delta_accuracy_pp",
                            "mcnemar_exact_two_sided_p",
                            "n01_dfag_wrong_reference_correct",
                            "n10_dfag_correct_reference_wrong",
                            r.bootstrap_seed,
                        )
            f = self.path(
                group, "statistics/dfag_vs_alpha_0_5.csv" if group == "dfag" else "statistics/dfag_seed45_46_paired.csv"
            )
            selected = pd.read_csv(f)
            if group == "dfag_confirmation":
                selected = selected[selected.comparison.str.contains("alpha")]
            for i, r in selected.iterrows():
                seed = int(r.seed)
                ref = (
                    f"state_interpolation/seed{seed}_alpha_0_5"
                    if seed < 45
                    else f"{group}/seed{seed}_alpha05_comparator"
                )
                self.recipe(
                    f,
                    i,
                    ref,
                    f"{group}/seed{seed}_dfag_oof",
                    "candidate_gain_loss",
                    r,
                    "delta_accuracy_pp",
                    "mcnemar_exact_two_sided_p",
                    "n01_dfag_wrong_reference_correct",
                    "n10_dfag_correct_reference_wrong",
                    r.bootstrap_seed,
                )
        for group, relative in [
            ("dfag", "counterfactual/gate_counterfactual_per_seed.csv"),
            ("dfag_confirmation", "counterfactual/gate_counterfactual_seed45_46_per_seed.csv"),
        ]:
            f = self.path(group, relative)
            if not f.exists():
                alternatives = list(self.path(group, "counterfactual").glob("*per_seed*.csv"))
                if len(alternatives) != 1:
                    self.gaps.append(f"Counterfactual pair schema unavailable: {group}")
                    continue
                f = alternatives[0]
            for i, r in pd.read_csv(f).iterrows():
                seed = int(r.seed)
                mode = r.comparison.replace("dynamic_vs_", "")
                candidate = f"{group}/seed{seed}_counterfactual_dynamic"
                reference = f"{group}/seed{seed}_counterfactual_{mode}"
                self.recipe(
                    f,
                    i,
                    reference,
                    candidate,
                    "candidate_gain_loss",
                    r,
                    "delta_accuracy_pp",
                    "mcnemar_exact_two_sided_p",
                    "n01_dfag_wrong_reference_correct",
                    "n10_dfag_correct_reference_wrong",
                    r.bootstrap_seed,
                )
                self.metric_expectation(
                    candidate, r, f, i, {m: "dfag_" + m for m in ["accuracy", "macro_f1", "balanced_accuracy"]}
                )
                self.metric_expectation(
                    reference, r, f, i, {m: "reference_" + m for m in ["accuracy", "macro_f1", "balanced_accuracy"]}
                )
        f = self.path("state_interpolation", "curves/checkpoint_interpolation_curve_per_seed.csv")
        for i, r in pd.read_csv(f).iterrows():
            aid = f"state_interpolation/seed{int(r.training_seed)}_alpha_{r.alpha:.1f}".replace(".", "_")
            self.metric_expectation(aid, r, f, i)
        f = self.path("state_interpolation", "curves/checkpoint_interpolation_curve_multiseed_summary.csv")
        for i, r in pd.read_csv(f).iterrows():
            for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
                self.seeds.append(
                    {
                        "summary_id": f"alpha{r.alpha}_{metric}",
                        "artifact_ids": "|".join(
                            f"state_interpolation/seed{s}_alpha_{r.alpha:.1f}".replace(".", "_") for s in [42, 43, 44]
                        ),
                        "metric": metric,
                        "mean": r[metric + "_mean"],
                        "sample_sd": r[metric + "_sample_sd"],
                        "historical_source": self.copy_text("state_interpolation", f),
                        "source_row": i,
                    }
                )
        for stage in [1, 2]:
            f = self.path("state_interpolation", f"statistics/alpha_0_5_vs_stage{stage}.csv")
            for i, r in pd.read_csv(f).iterrows():
                seed = int(r.training_seed)
                self.recipe(
                    f,
                    i,
                    self.baseline(seed, stage),
                    f"state_interpolation/seed{seed}_alpha_0_5",
                    "joint4_candidate_first",
                    r,
                    "accuracy_delta_alpha_minus_reference_pp",
                    "mcnemar_exact_two_sided_p",
                    "n01_alpha_wrong_reference_correct",
                    "n10_alpha_correct_reference_wrong",
                    r.bootstrap_seed,
                )

    def split_evidence(self):
        for group in ["external", "cross_backbone", "leaves_round1", "leaves_round2", "leaves_screen"]:
            folder = self.path(group)
            for source in sorted(folder.rglob("split_indices.npz")):
                relative = source.relative_to(folder).as_posix()
                if any(x in relative.lower() for x in ["smoke", "failed", "aborted"]):
                    continue
                fold_match = re.search(r"fold_(\d)", relative)
                if not fold_match:
                    continue
                fold = int(fold_match[1])
                with np.load(source, allow_pickle=False) as z:
                    train = z["train_indices"].astype(np.int64)
                    heldout = z["validation_indices"].astype(np.int64)
                    if len(np.intersect1d(train, heldout)) or len(np.unique(np.r_[train, heldout])) != len(train) + len(
                        heldout
                    ):
                        raise ValueError("Archived training/heldout split overlap")
                    data = {
                        "dataset_index": np.r_[train, heldout],
                        "partition": ["train"] * len(train) + ["heldout"] * len(heldout),
                    }
                    if "train_sample_ids" in z.files:
                        data["sample_id"] = np.r_[z["train_sample_ids"], z["validation_sample_ids"]].astype(str)
                    else:
                        data["sample_id"] = np.r_[train, heldout].astype(str)
                dataset = "leaves"
                if group == "external":
                    dataset = re.search(r"(cars|cub|flowers)__", relative)[1]
                n, _ = POOL[dataset]
                if len(train) + len(heldout) != n:
                    raise ValueError("Archived split development-pool size mismatch")
                dest = self.write_csv(
                    f"splits/{group}/{relative.replace('.npz', '.csv')}",
                    pd.DataFrame(data),
                    source,
                    "historical_train_heldout_assignment",
                    group,
                )
                self.split_index.append(
                    {
                        "path": dest,
                        "dataset": dataset,
                        "pool_id": dataset,
                        "fold": fold,
                        "n": n,
                        "training_n": len(train),
                        "heldout_n": len(heldout),
                    }
                )
        self.write_csv("SPLIT_INDEX.csv", pd.DataFrame(self.split_index), role="historical_split_evidence")

    def source_audit(self):
        hashes = {}
        for group in GROUPS:
            for source in self.path(group).rglob("*"):
                if (
                    source.is_file()
                    and source.suffix in [".py", ".json", ".sh"]
                    and "wsl_python_deps" not in str(source)
                ):
                    hashes.setdefault(digest(source), []).append(source)
        entries = []
        keys = [
            "config_sha256",
            "runner_sha256",
            "common_sha256",
            "runtime_sha256",
            "protocol_core_sha256",
            "round1_runner_sha256",
            "phase2b_train_one_sha256",
            "phase2b_screen_core_sha256",
        ]
        for group in GROUPS:
            for source in self.path(group).rglob("run_manifest.json"):
                if any(x in str(source).lower() for x in ["smoke", "failed", "aborted"]):
                    continue
                manifest = json.loads(source.read_text(encoding="utf-8-sig"))
                for key in keys:
                    recorded = manifest.get(key, manifest.get("job", {}).get(key, ""))
                    if not recorded:
                        continue
                    matches = hashes.get(recorded, [])
                    entries.append(
                        {
                            "run_manifest": self.copy_text(group, source),
                            "field": key,
                            "recorded_sha256": recorded,
                            "status": "MATCHED_ORIGINAL_FILE_BYTES" if matches else "ORIGINAL_FILE_NOT_LOCATED_BY_HASH",
                            "matching_public_copy": self.copy_text(self.source_group(matches[0]), matches[0])
                            if matches
                            else "",
                        }
                    )
        self.write_csv("SOURCE_HASH_AUDIT.csv", pd.DataFrame(entries), role="recorded_source_hash_crosscheck")
        missing = sum(x["status"] == "ORIGINAL_FILE_NOT_LOCATED_BY_HASH" for x in entries)
        if missing:
            self.gaps.append(
                f"{missing} recorded code/config hash entries have no byte-identical source file in the selected historical groups; see SOURCE_HASH_AUDIT.csv"
            )

    def records(self):
        extensions = {".json", ".csv", ".md", ".log", ".jsonl", ".py", ".sh", ".txt"}
        for group in GROUPS:
            folder = self.path(group)
            for source in sorted(folder.rglob("*")):
                rel = source.relative_to(folder).as_posix()
                if not source.is_file() or source.suffix not in extensions:
                    continue
                lower = rel.lower()
                if any(
                    x in lower
                    for x in [
                        "smoke",
                        "failed_attempt",
                        "aborted_attempt",
                        "preflight/",
                        "wsl_python_deps",
                        "gpt56_review",
                        "__pycache__",
                        "immutability",
                        "snapshot_before",
                        "oof_sample_outputs",
                    ]
                ):
                    continue
                if source.stat().st_size > 10 * 1024 * 1024:
                    self.gaps.append(f"Large ancillary text omitted: {group}/{rel}")
                    continue
                self.copy_text(group, source)

    def finish(self):
        self.write_csv("OOF_INDEX.csv", pd.DataFrame(self.index), role="oof_provenance_index")
        self.write_csv("HISTORICAL_METRICS.csv", pd.DataFrame(self.metrics), role="archived_metric_assertions")
        self.write_csv("PAIRED_RECIPES.csv", pd.DataFrame(self.recipes), role="historical_statistical_recipes")
        self.write_csv(
            "SEED_SUMMARY_EXPECTATIONS.csv",
            pd.DataFrame(
                self.seeds,
                columns=[
                    "summary_id",
                    "artifact_ids",
                    "metric",
                    "mean",
                    "sample_sd",
                    "historical_source",
                    "source_row",
                ],
            ),
            role="independent_seed_expectations",
        )
        gap = self.out / "EXPORT_GAPS.json"
        gap.write_text(json.dumps(self.gaps, indent=2) + "\n", encoding="utf-8")
        self.record(gap, role="export_gaps")
        pd.DataFrame(self.files).to_csv(self.out / "MANIFEST.csv", index=False, lineterminator="\n")
        self.audit.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.private).to_csv(self.audit / "INTERNAL_SOURCE_PATHS.csv", index=False)
        # Sources remain byte-identical; this is verified independently at the end.
        for row in self.private:
            if digest(row["original_path"]) != row["original_sha256"]:
                raise ValueError("Original source changed during export")
        print(
            json.dumps(
                {
                    "oof_sets": len(self.index),
                    "published_files": len(self.files),
                    "historical_metric_assertions": len(self.metrics),
                    "paired_comparisons": len(self.recipes),
                    "gaps": self.gaps,
                },
                indent=2,
            )
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("paper_artifacts"))
    p.add_argument("--internal-audit", type=Path, required=True)
    p.add_argument("--redact-identity", action="append", default=[], help="Private strings to redact; repeat as needed")
    args = p.parse_args()
    exporter = Exporter(args.source_root, args.output, args.internal_audit, args.redact_identity)
    exporter.export_primary()
    exporter.statistics()
    exporter.split_evidence()
    exporter.source_audit()
    exporter.records()
    exporter.finish()


if __name__ == "__main__":
    main()
