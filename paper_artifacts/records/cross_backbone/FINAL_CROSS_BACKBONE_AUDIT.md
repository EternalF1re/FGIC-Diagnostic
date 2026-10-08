# Final Cross-Backbone Evaluation Audit

Status: **COMPLETE**

## A. Protocol provenance

The run used the frozen Classify Leaves controlled protocol and standalone DFAG semantics. No old cross-backbone result or checkpoint was reused. Config SHA256: `1a075d4241358e4681f2e7c70523a85bf286d97193fb1104b18bd713c5006fee`.

## B. Backbone dimensions

| Backbone | Actual D | H | D/H | Gate parameters |
|---|---:|---:|---:|---:|
| ResNet-50 | 2048 | 256 | 8.0 | 524,288 |
| ConvNeXt-Tiny | 768 | 256 | 3.0 | 73,728 |

H=256 is fixed across backbones, so the projection compression ratio differs across architectures and is not independently controlled.

## C. Job completion

- Preflight: PASS
- Six technical smoke configurations: PASS
- Formal jobs: 30/30 COMPLETE
- Failed formal jobs: 0

## D. OOF coverage

Each of 12 final pooled artifacts contains all 18,353 held-out samples exactly once, with aligned labels and fold identities. Original-view predictions only; no TTA or fold ensemble.

## E-H. Ours-FT, Progressive Head, and standalone DFAG results

| Backbone | Outcome | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---|---:|---:|---:|
| resnet50 | ours_stage1 | 97.41% | 0.9769 | 0.9774 |
| resnet50 | ours_stage2 | 97.60% | 0.9788 | 0.9790 |
| resnet50 | ours_alpha0.5 | 97.47% | 0.9775 | 0.9780 |
| resnet50 | progressive_stage1 | 97.15% | 0.9736 | 0.9740 |
| resnet50 | progressive_stage2 | 97.22% | 0.9749 | 0.9754 |
| resnet50 | dfag | 97.55% | 0.9781 | 0.9785 |
| convnext_tiny | ours_stage1 | 97.99% | 0.9828 | 0.9830 |
| convnext_tiny | ours_stage2 | 97.80% | 0.9808 | 0.9813 |
| convnext_tiny | ours_alpha0.5 | 97.93% | 0.9824 | 0.9827 |
| convnext_tiny | progressive_stage1 | 97.94% | 0.9819 | 0.9820 |
| convnext_tiny | progressive_stage2 | 97.74% | 0.9802 | 0.9804 |
| convnext_tiny | dfag | 98.03% | 0.9832 | 0.9833 |

## G/I. Paired OOF statistics

| Backbone | Comparison | Reference Acc. | Candidate Acc. | Delta (pp) | 95% CI | McNemar | Fold direction | Disagreement |
|---|---|---:|---:|---:|---:|---:|---|---:|
| resnet50 | progressive_minus_ours_stage1 | 97.41% | 97.15% | -0.26 | [-0.45, -0.08] | p=0.006 | negative 5/5, positive 0/5, tie 0/5 | 1.78% |
| resnet50 | progressive_minus_ours_stage2 | 97.60% | 97.22% | -0.38 | [-0.54, -0.21] | p<0.001 | negative 5/5, positive 0/5, tie 0/5 | 1.41% |
| resnet50 | dfag_minus_stage1 | 97.41% | 97.55% | +0.14 | [0.02, 0.26] | p=0.026 | negative 0/5, positive 5/5, tie 0/5 | 0.74% |
| resnet50 | dfag_minus_stage2 | 97.60% | 97.55% | -0.04 | [-0.16, 0.08] | p=0.536 | negative 3/5, positive 1/5, tie 1/5 | 0.77% |
| resnet50 | dfag_minus_alpha0.5 | 97.47% | 97.55% | +0.09 | [-0.02, 0.20] | p=0.152 | negative 0/5, positive 5/5, tie 0/5 | 0.64% |
| convnext_tiny | progressive_minus_ours_stage1 | 97.99% | 97.94% | -0.05 | [-0.20, 0.10] | p=0.571 | negative 3/5, positive 1/5, tie 1/5 | 1.16% |
| convnext_tiny | progressive_minus_ours_stage2 | 97.80% | 97.74% | -0.06 | [-0.22, 0.10] | p=0.507 | negative 3/5, positive 2/5, tie 0/5 | 1.34% |
| convnext_tiny | dfag_minus_stage1 | 97.99% | 98.03% | +0.04 | [-0.05, 0.13] | p=0.494 | negative 2/5, positive 3/5, tie 0/5 | 0.42% |
| convnext_tiny | dfag_minus_stage2 | 97.80% | 98.03% | +0.23 | [0.08, 0.38] | p=0.003 | negative 0/5, positive 5/5, tie 0/5 | 1.10% |
| convnext_tiny | dfag_minus_alpha0.5 | 97.93% | 98.03% | +0.10 | [-0.02, 0.21] | p=0.111 | negative 1/5, positive 3/5, tie 1/5 | 0.64% |

## J. DFAG gate and Stage-1 provenance

- ResNet-50 gate: 524,288 parameters (2048 -> 128 -> 2048; bias-free).
- ConvNeXt-Tiny gate: 73,728 parameters (768 -> 48 -> 768; bias-free).
- All ten DFAG folds reused the exact same-run, same-backbone, same-fold Ours-FT selected Stage-1 SHA as common anchor/plastic initialization; Stage 1 was not retrained inside DFAG.

## K-L. Fold direction and prediction disagreement

Fold-direction counts and prediction disagreement are reported in the paired table. Folds are descriptive strata and were not treated as independent significance units.

## Backbone-dependent direction changes

- `dfag_minus_alpha0.5`: NO; ResNet-50 +0.087 pp, ConvNeXt-Tiny +0.098 pp.
- `dfag_minus_stage1`: NO; ResNet-50 +0.142 pp, ConvNeXt-Tiny +0.038 pp.
- `dfag_minus_stage2`: YES; ResNet-50 -0.044 pp, ConvNeXt-Tiny +0.229 pp.
- `progressive_minus_ours_stage1`: NO; ResNet-50 -0.262 pp, ConvNeXt-Tiny -0.049 pp.
- `progressive_minus_ours_stage2`: NO; ResNet-50 -0.376 pp, ConvNeXt-Tiny -0.060 pp.

## M. Protocol limitations

These are supporting results under the frozen cross-backbone protocol. They do not prove architecture generalization, backbone independence, universal benefit/harm, or a causal effect of compression ratio. H is fixed at 256, so D/H is confounded with backbone identity. Result directions are reported without result-dependent retuning.
