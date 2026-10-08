# Historical results provenance

This release publishes predictions and records from completed local experiments. It does not train models, reselect checkpoints, tune hyperparameters, or replace recorded statistics.

## Public records

- `paper_artifacts/MANIFEST.csv`: every public artifact, public SHA-256, original source identifier and original SHA-256, transformation, and experiment group.
- `OOF_INDEX.csv`: dataset, backbone, training seed, stage, original configuration hash, public configuration copy and development-pool size for each OOF set.
- `SOURCE_HASH_AUDIT.csv`: historical recorded code/config hashes checked against the original source bytes. This is distinct from the hash of a sanitized public copy.
- `records/`: configuration snapshots, final metrics and statistics, run manifests, job ledgers, training histories, historical statistical source files, diagnostics and logs.
- `SPLIT_INDEX.csv` and `splits/`: actual saved training/held-out assignments for experiments where split archives were retained.
- `HISTORICAL_METRICS.csv`, `PAIRED_RECIPES.csv`, `SEED_SUMMARY_EXPECTATIONS.csv`: assertions copied from archived tables; these values are separate from verifier calculations.

Original absolute source paths are available in the owner's internal audit, and intentionally absent from the public release. Source identifiers are relative to the historical experiment workspace. Public text copies remove local paths and personal identifiers and transcode legacy logs to UTF-8. Their hashes can differ from original hashes. The local exporter checked original bytes again after export.

## Experiment contexts

| Group | Dataset / backbone | Seeds | Source context |
|---|---|---|---|
| external | CUB, Cars, Flowers / Inception-ResNet-v2 | 42 | Newly frozen unified 40-job cross-dataset validation; includes CUB shortcut diagnostics |
| cross_backbone | Leaves / ResNet-50 and ConvNeXt-Tiny | 42 | 30-job frozen cross-architecture run |
| controlled | CUB, Cars / ResNet-50 | 42 | 60-job L2-SP, MC-Loss, CAL, Ours-FT, Progressive and DFAG comparison |
| leaves_screen | Leaves / Inception-ResNet-v2 | 42 | Completed controlled screen; baseline and lambda 0.1/1.0, both stages |
| leaves_round1 | Leaves / Inception-ResNet-v2 | 42, 43, 44 | Lambda 0.7/0.9 and additional baseline seeds |
| leaves_round2 | Leaves / Inception-ResNet-v2 | 42, 45, 46 | MHSA diagnostics and additional baseline seeds; MHSA variants are not the final no-attention head |
| dfag / dfag_confirmation | Leaves / Inception-ResNet-v2 | 42–46 | Standalone DFAG and frozen seed confirmation; six counterfactual modes |
| state_interpolation | Leaves / Inception-ResNet-v2 | 42–44 | Complete checkpoint-state interpolation at eleven alpha values; seed45/46 alpha0.5 lives in dfag_confirmation |

The controlled screen is included as a historical diagnostic, not represented as the final external protocol. Do not pool different method/backbone/seed contexts as repeated estimates.

## Frozen configuration identity

External monolithic config: `7f667c125983ca82ec36214b727ee8fbe3215f21562abd260f4018559f87e3e5`.

Cross-backbone monolithic config: `1a075d4241358e4681f2e7c70523a85bf286d97193fb1104b18bd713c5006fee`.

For all other configurations use the per-record original hashes. Existing public cross-dataset/cross-backbone configs are portable projections; do not treat their hashes as hashes of the original monolithic files. The controlled per-method configs are also available under `configs/controlled/`. The verifier checks their documented hashes.

Checkpoint selection is documented in the copied per-job manifests and configs: the best held-out original-view accuracy, with the exact recorded stage and tie rule. Stage2 starts from selected Stage1 where specified; DFAG reuses its same-fold anchor. OOF here means held-out training folds; those validation folds also support checkpoint selection. This release does not claim a separately untouched nested selection set.

## Statistical implementation

Every recipe uses 100,000 resamples, a 95% percentile CI, and exact two-sided McNemar. All differences in the normalized recipe are candidate minus reference, in percentage points. The copied historical CSVs preserve their original sign conventions and column names.

| Recipe | Historical sampling implementation |
|---|---|
| external | PCG64, seed 20260818, three multinomial categories negative/tie/positive |
| cross_backbone | PCG64, seed 20260818, same categories with its recorded arithmetic order |
| Leaves shortcut | PCG64, seed 20260807, four joint correctness categories |
| DFAG and counterfactuals | PCG64, recorded per-comparison seed, candidate gain/loss/tie categories |
| checkpoint-state interpolation | PCG64, recorded seed, four joint categories with candidate first |
| controlled | PCG64, seed 42, explicit paired sample-index draws |

Multinomial category sampling has the same sampling distribution as sample-index bootstrap for accuracy differences, but can produce a different finite Monte Carlo realization. Therefore implementations are not silently interchanged. The verifier preserves archived reference row order for index resampling, and aligns other predictions by sample_id. It bounds memory by subdividing index batches while preserving the integer RNG stream.

Fold variability is descriptive. Independent-seed summaries use sample SD with ddof=1 and explicitly list distinct training seeds. No official-test predictions or fold ensembles are used as the published OOF.
