#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_CONFIG:?Set DATASET_CONFIG to your datasets JSON}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cross_backbone}"
DEVICE="${DEVICE:-cuda:0}"

for backbone in resnet50 convnext_tiny; do
  for fold in 0 1 2 3 4; do
    for method in ours_ft progressive; do
      fgic train --experiment-config "configs/cross_backbone/${method}_${backbone}.json" \
        --dataset-config "$DATASET_CONFIG" --fold "$fold" --device "$DEVICE" \
        --output "$OUTPUT_ROOT/${backbone}/${method}/fold_${fold}"
    done
    fgic interpolate \
      --stage1 "$OUTPUT_ROOT/${backbone}/ours_ft/fold_${fold}/best_stage1.pth" \
      --stage2 "$OUTPUT_ROOT/${backbone}/ours_ft/fold_${fold}/best_stage2.pth" \
      --alpha 0.5 --output "$OUTPUT_ROOT/${backbone}/interpolated_checkpoints/fold_${fold}.pth"
    fgic evaluate --experiment-config "configs/cross_backbone/ours_ft_${backbone}.json" \
      --dataset-config "$DATASET_CONFIG" --fold "$fold" --device "$DEVICE" \
      --checkpoint "$OUTPUT_ROOT/${backbone}/interpolated_checkpoints/fold_${fold}.pth" \
      --output "$OUTPUT_ROOT/${backbone}/alpha0_5/fold_${fold}"
    fgic train --experiment-config "configs/cross_backbone/dfag_${backbone}.json" \
      --dataset-config "$DATASET_CONFIG" --fold "$fold" --device "$DEVICE" \
      --stage1-checkpoint "$OUTPUT_ROOT/${backbone}/ours_ft/fold_${fold}/best_stage1.pth" \
      --output "$OUTPUT_ROOT/${backbone}/dfag/fold_${fold}"
  done
  for method in ours_ft progressive dfag; do
    fgic aggregate --runs "$OUTPUT_ROOT/${backbone}/${method}" \
      --output "$OUTPUT_ROOT/${backbone}/${method}/summary"
  done
  fgic aggregate --runs "$OUTPUT_ROOT/${backbone}/alpha0_5" \
    --output "$OUTPUT_ROOT/${backbone}/alpha0_5/summary"
done
