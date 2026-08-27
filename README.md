# FGIC-Diagnostic

Reproducible code for diagnostic experiments and controlled comparisons in fine-grained image classification.

This repository is deliberately not presented as a leaderboard or a SOTA-claim package. Values quoted from original papers and values produced by our controlled reimplementations follow different protocols and must remain clearly separated.

## What is included

- Ours-FT with the final conventional classification head;
- the final five-block Progressive Head with fixed 0.7 residual shortcuts and mean aggregation;
- DFAG with a frozen same-fold Stage-1 anchor, a plastic branch and a feature-wise gate;
- controlled L2-SP, Mutual-Channel Loss and CAL implementations;
- five-fold pooled-OOF evaluation and paired statistics;
- cross-dataset configurations for CUB-200-2011, Stanford Cars and Oxford Flowers-102;
- frozen cross-backbone and DFAG diagnostic sources for result provenance.

## Installation

Python 3.9 or newer is required. The versions used by the paper are recorded in `requirements-lock.txt`; the portable lower bounds are in `pyproject.toml`.

```bash
git clone <REPOSITORY_URL> fgic-diagnostic
cd fgic-diagnostic
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Dataset manifests

Datasets are not redistributed. Each development manifest is a CSV with exactly these required columns:

```text
path,label,sample_id
images/001.jpg,0,cub:001
```

Paths may be absolute or relative to the manifest. Labels are zero-based; sample IDs must be unique. Copy `configs/datasets.example.json`, edit only the manifest paths, then validate:

```bash
cp configs/datasets.example.json configs/datasets.local.json
fgic validate-data --experiment-config configs/controlled/ours_ft_cub.json \
  --dataset-config configs/datasets.local.json
```

For class-named ImageFolder layouts, a manifest can be generated with:

```bash
fgic make-imagefolder-manifest --image-root /data/cub/images-by-class \
  --dataset cub --output data/manifests/cub_development.csv
```

See `DATASETS.md` for the exact development-pool boundary used by each benchmark.

## Run one fold

```bash
fgic train --experiment-config configs/controlled/ours_ft_cub.json \
  --dataset-config configs/datasets.local.json --fold 0 --device cuda:0 \
  --output outputs/cub/ours_ft/fold_0
```

DFAG must receive the selected Ours-FT Stage-1 checkpoint from the same dataset and fold:

```bash
fgic train --experiment-config configs/controlled/dfag_cub.json \
  --dataset-config configs/datasets.local.json --fold 0 --device cuda:0 \
  --stage1-checkpoint outputs/cub/ours_ft/fold_0/best_stage1.pth \
  --output outputs/cub/dfag/fold_0
```

CAL preserves the formal two-GPU NCCL/SyncBN topology:

```bash
torchrun --standalone --nproc_per_node=2 -m fgic_diagnostic.cal_ddp \
  --experiment-config configs/controlled/cal_cub.json \
  --dataset-config configs/datasets.local.json --fold 0 \
  --output outputs/cub/cal/fold_0
```

The `--smoke` option shortens optimization only to test mechanics. Smoke outputs are not paper results.

## Reproduce experiment groups

Set `DATASET_CONFIG` and optionally `OUTPUT_ROOT`/`DEVICE`, then use:

```bash
bash reproduce/main_results.sh
bash reproduce/cross_dataset.sh
bash reproduce/cross_backbone.sh
bash reproduce/controlled_comparison.sh
```

The scripts are intentionally fail-closed: output directories are never silently reused. Independent large runs may instead be placed into the shared dynamic GPU queue described in `docs/DYNAMIC_GPU_SCHEDULING.md`.

### Paper section mapping

| Paper section | Experiment scope | Reproduction entry point | Public reproduction configurations |
|---|---|---|---|
| Section 4.3 | Inception-ResNet-v2 cross-dataset evaluation on CUB-200-2011, Stanford Cars and Oxford Flowers-102 | `reproduce/cross_dataset.sh` | `configs/cross_dataset/` |
| Section 4.7 | Cross-architecture evaluation on Classify Leaves with ResNet-50 and ConvNeXt-Tiny | `reproduce/cross_backbone.sh` | `configs/cross_backbone/` |
| Section 4.8 | Common-ResNet-50 controlled reimplementation on CUB-200-2011 and Stanford Cars | `reproduce/main_results.sh` and `reproduce/controlled_comparison.sh` | `configs/controlled/` |

The experiment cells above are independently frozen protocol contexts. Their effect sizes must not be treated as directly comparable repeated estimates.

### Configuration identity

SHA-256 values for every public experiment configuration are listed in `docs/CONFIG_SHA256.csv`. Each new `fgic train` run also records the exact `config_sha256` in its `run_manifest.json`.

The formal Section 4.3 run used the monolithic config SHA-256 `7f667c125983ca82ec36214b727ee8fbe3215f21562abd260f4018559f87e3e5`. The formal Section 4.7 run used the monolithic config SHA-256 `1a075d4241358e4681f2e7c70523a85bf286d97193fb1104b18bd713c5006fee`. Their public per-method JSON files are path-free projections of those protocols, so their file hashes are intentionally different from the monolithic formal hashes. The Section 4.8 files under `configs/controlled/` are byte-identical copies of the corresponding frozen formal per-method configs.

A public config hash can be verified independently with:

```bash
sha256sum configs/controlled/ours_ft_cub.json
```

On PowerShell, use `Get-FileHash configs\controlled\ours_ft_cub.json -Algorithm SHA256`.

### Five-fold split reproduction

All public runners construct folds with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the ordered development manifest. The implementation is `fold_indices` in `src/fgic_diagnostic/data.py`. Reproducing a fold therefore requires the same ordered `path,label,sample_id` manifest and the same zero-based fold ID (`0` through `4`). Each development sample appears in exactly one held-out fold, and final metrics are recomputed from pooled OOF predictions rather than averaged across fold-level metrics.

## OOF aggregation and paired statistics

```bash
fgic aggregate --runs outputs/cub/ours_ft --output outputs/cub/ours_ft/summary
fgic paired --first outputs/cub/progressive/summary/pooled_oof_predictions.npz \
  --second outputs/cub/ours_ft/summary/pooled_oof_predictions.npz \
  --output outputs/cub/progressive_vs_ours_ft.json
```

The paired command uses 100,000 paired bootstrap samples, a 95% percentile interval and an exact two-sided McNemar test.

Cross-backbone Ours-FT also exposes the paper's complete-state interpolation diagnostic through `fgic interpolate`; floating tensors use `alpha*Stage1 + (1-alpha)*Stage2`, non-floating tensors copy Stage 2, and no BN refresh is applied.

## Repository layout

- `src/fgic_diagnostic`: maintained public package and command-line tools;
- `configs`: frozen controlled configs and portable cross-dataset configs;
- `reproduce`: experiment-group launchers;
- `tests`: fast protocol and architecture tests;
- `docs`: frozen protocol and method adaptation records;
- `reference/frozen_sources`: immutable source snapshot used to produce the audited results.

The reference snapshot is retained for forensic traceability. It contains original experiment-root assumptions and is not the public CLI; use `fgic` for new runs. See `SOURCE_MAP.md`.

## Results and checkpoints

Large checkpoints, datasets, raw OOF arrays and machine-local logs are intentionally excluded. Published numeric tables should be accompanied by their final audit documents and configuration hashes. The repository does not fabricate downloadable weights that were not approved for release.

## License and attribution

The repository is released under the MIT License. Controlled baseline implementations are adaptations of MIT-licensed upstream repositories; see `THIRD_PARTY_NOTICES.md`. Datasets and pretrained weights retain their own licenses.

## Availability

During double-anonymized review, the code, frozen configuration files and protocol documentation are provided through the anonymous artifact link accompanying the manuscript. A permanent public repository will be disclosed after the review process. Historical formal job ledgers are not claimed as part of this review artifact; newly reproduced runs emit their own `run_manifest.json` and training history.
