#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_CONFIG:?Set DATASET_CONFIG to your datasets JSON}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/controlled_comparison}"
DEVICE="${DEVICE:-cuda:0}"

for dataset in cub cars; do
  for fold in 0 1 2 3 4; do
    for method in l2_sp mc_loss; do
      fgic train --experiment-config "configs/controlled/${method}_${dataset}.json" \
        --dataset-config "$DATASET_CONFIG" --fold "$fold" --device "$DEVICE" \
        --output "$OUTPUT_ROOT/${dataset}/${method}/fold_${fold}"
    done
    torchrun --standalone --nproc_per_node=2 -m fgic_diagnostic.cal_ddp \
      --experiment-config "configs/controlled/cal_${dataset}.json" \
      --dataset-config "$DATASET_CONFIG" --fold "$fold" \
      --output "$OUTPUT_ROOT/${dataset}/cal/fold_${fold}"
  done
  for method in l2_sp mc_loss cal; do
    fgic aggregate --runs "$OUTPUT_ROOT/${dataset}/${method}" \
      --output "$OUTPUT_ROOT/${dataset}/${method}/summary"
  done
done

