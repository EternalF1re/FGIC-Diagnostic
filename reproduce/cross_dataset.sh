#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_CONFIG:?Set DATASET_CONFIG to your datasets JSON}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cross_dataset}"
DEVICE="${DEVICE:-cuda:0}"

for dataset in cub cars flowers; do
  for method in ours_ft progressive; do
    for fold in 0 1 2 3 4; do
      fgic train --experiment-config "configs/cross_dataset/${method}_${dataset}.json" \
        --dataset-config "$DATASET_CONFIG" --fold "$fold" --device "$DEVICE" \
        --output "$OUTPUT_ROOT/${dataset}/${method}/fold_${fold}"
    done
    fgic aggregate --runs "$OUTPUT_ROOT/${dataset}/${method}" \
      --output "$OUTPUT_ROOT/${dataset}/${method}/summary"
  done
done

