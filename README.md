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
git clone <repository-url>
cd FGIC-Diagnostic
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
