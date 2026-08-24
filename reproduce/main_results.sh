#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_CONFIG:?Set DATASET_CONFIG to your datasets JSON}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/main_results}"
DEVICE="${DEVICE:-cuda:0}"

for dataset in cub cars; do
  for fold in 0 1 2 3 4; do
    for method in ours_ft progressive; do
      fgic train --experiment-config "configs/controlled/${method}_${dataset}.json" \
        --dataset-config "$DATASET_CONFIG" --fold "$fold" --device "$DEVICE" \
        --output "$OUTPUT_ROOT/${dataset}/${method}/fold_${fold}"
    done
    fgic train --experiment-config "configs/controlled/dfag_${dataset}.json" \
      --dataset-config "$DATASET_CONFIG" --fold "$fold" --device "$DEVICE" \
      --stage1-checkpoint "$OUTPUT_ROOT/${dataset}/ours_ft/fold_${fold}/best_stage1.pth" \
      --output "$OUTPUT_ROOT/${dataset}/dfag/fold_${fold}"
  done
  for method in ours_ft progressive dfag; do
    fgic aggregate --runs "$OUTPUT_ROOT/${dataset}/${method}" \
      --output "$OUTPUT_ROOT/${dataset}/${method}/summary"
  done
done

