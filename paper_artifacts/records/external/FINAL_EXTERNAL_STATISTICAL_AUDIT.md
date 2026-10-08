# Final External Statistical Audit

## A. Integrity / provenance

- `PROVENANCE_PASS = YES`
- `CONFIG_SHA256 = 7f667c125983ca82ec36214b727ee8fbe3215f21562abd260f4018559f87e3e5`
- `FORMAL_JOBS_COMPLETE = 40/40`
- `FORMAL_JOBS_FAILED = 0`
- `OOF_COVERAGE_PASS = YES`
- `OOF_ARTIFACT_COUNT = 16/16`
- `OOF_ARTIFACT_HASH_MATCH = 16/16`
- `ALL_PAIRED_SAMPLE_ALIGNMENT_PASS = YES`
- Every paired comparison was explicitly aligned by `sample_id`; N, ground-truth labels, and fold assignments matched exactly.

## B. Protocol summary

This analysis belongs to `FINAL_NEWLY_FROZEN_UNIFIED_CROSS_DATASET_VALIDATION_PROTOCOL`. The statistical unit is the complete pooled five-fold OOF sample. No fold-mean significance test, official-test primary analysis, TTA, cross-fold ensemble, checkpoint reselection, model training, or OOF modification was performed.

Paired bootstrap used 100,000 resamples, seed 20260818, and a percentile 95% CI. The difference is always method B minus method A in percentage points. `n10` means method A is correct and method B is wrong; `n01` means method A is wrong and method B is correct. McNemar p-values are exact and two-sided.

## C. Primary six paired comparisons

| Dataset | Stage | Comparison (B - A) | A Acc. | B Acc. | Delta (pp) | 95% CI (pp) | n10 | n01 | Exact McNemar | Fold direction |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| CUB-200-2011 | 1 | Progressive Head lambda=0.7 - Ours-FT | 80.78% | 79.15% | -1.63 | [-2.54, -0.75] | 423 | 325 | p<0.001 | negative 5/5, positive 0/5, tie 0/5 |
| CUB-200-2011 | 2 | Progressive Head lambda=0.7 - Ours-FT | 80.56% | 78.26% | -2.30 | [-3.24, -1.38] | 470 | 332 | p<0.001 | negative 5/5, positive 0/5, tie 0/5 |
| Stanford Cars | 1 | Progressive Head lambda=0.7 - Ours-FT | 91.42% | 90.72% | -0.70 | [-1.24, -0.16] | 283 | 226 | p=0.013 | negative 4/5, positive 1/5, tie 0/5 |
| Stanford Cars | 2 | Progressive Head lambda=0.7 - Ours-FT | 91.26% | 90.19% | -1.07 | [-1.63, -0.50] | 316 | 229 | p<0.001 | negative 5/5, positive 0/5, tie 0/5 |
| Oxford Flowers-102 | 1 | Progressive Head lambda=0.7 - Ours-FT | 97.84% | 97.06% | -0.78 | [-1.37, -0.20] | 26 | 10 | p=0.011 | negative 4/5, positive 0/5, tie 1/5 |
| Oxford Flowers-102 | 2 | Progressive Head lambda=0.7 - Ours-FT | 97.45% | 96.76% | -0.69 | [-1.42, 0.00] | 35 | 21 | p=0.081 | negative 4/5, positive 1/5, tie 0/5 |

## D. CUB lambda versus Ours-FT

| Dataset | Stage | Comparison (B - A) | A Acc. | B Acc. | Delta (pp) | 95% CI (pp) | n10 | n01 | Exact McNemar | Fold direction |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| CUB-200-2011 | 1 | Progressive Head lambda=0.1 - Ours-FT | 80.78% | 79.88% | -0.90 | [-1.80, 0.00] | 405 | 351 | p=0.054 | negative 4/5, positive 1/5, tie 0/5 |
| CUB-200-2011 | 1 | Progressive Head lambda=0.7 - Ours-FT | 80.78% | 79.15% | -1.63 | [-2.54, -0.75] | 423 | 325 | p<0.001 | negative 5/5, positive 0/5, tie 0/5 |
| CUB-200-2011 | 1 | Progressive Head lambda=1.0 - Ours-FT | 80.78% | 79.73% | -1.05 | [-1.94, -0.17] | 398 | 335 | p=0.022 | negative 5/5, positive 0/5, tie 0/5 |
| CUB-200-2011 | 2 | Progressive Head lambda=0.1 - Ours-FT | 80.56% | 79.36% | -1.20 | [-2.14, -0.28] | 438 | 366 | p=0.012 | negative 5/5, positive 0/5, tie 0/5 |
| CUB-200-2011 | 2 | Progressive Head lambda=0.7 - Ours-FT | 80.56% | 78.26% | -2.30 | [-3.24, -1.38] | 470 | 332 | p<0.001 | negative 5/5, positive 0/5, tie 0/5 |
| CUB-200-2011 | 2 | Progressive Head lambda=1.0 - Ours-FT | 80.56% | 79.43% | -1.13 | [-2.05, -0.22] | 426 | 358 | p=0.017 | negative 4/5, positive 1/5, tie 0/5 |

## E. CUB lambda-to-lambda diagnostics

| Dataset | Stage | Comparison (B - A) | A Acc. | B Acc. | Delta (pp) | 95% CI (pp) | n10 | n01 | Exact McNemar | Fold direction |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| CUB-200-2011 | 1 | Progressive Head lambda=0.7 - Progressive Head lambda=0.1 | 79.88% | 79.15% | -0.73 | [-1.63, 0.17] | 396 | 352 | p=0.116 | negative 4/5, positive 1/5, tie 0/5 |
| CUB-200-2011 | 1 | Progressive Head lambda=1.0 - Progressive Head lambda=0.1 | 79.88% | 79.73% | -0.15 | [-1.05, 0.75] | 378 | 369 | p=0.770 | negative 2/5, positive 2/5, tie 1/5 |
| CUB-200-2011 | 1 | Progressive Head lambda=1.0 - Progressive Head lambda=0.7 | 79.15% | 79.73% | 0.58 | [-0.32, 1.48] | 357 | 392 | p=0.214 | negative 1/5, positive 4/5, tie 0/5 |
| CUB-200-2011 | 2 | Progressive Head lambda=0.7 - Progressive Head lambda=0.1 | 79.36% | 78.26% | -1.10 | [-2.07, -0.15] | 463 | 397 | p=0.027 | negative 5/5, positive 0/5, tie 0/5 |
| CUB-200-2011 | 2 | Progressive Head lambda=1.0 - Progressive Head lambda=0.1 | 79.36% | 79.43% | 0.07 | [-0.88, 1.02] | 417 | 421 | p=0.917 | negative 2/5, positive 3/5, tie 0/5 |
| CUB-200-2011 | 2 | Progressive Head lambda=1.0 - Progressive Head lambda=0.7 | 78.26% | 79.43% | 1.17 | [0.25, 2.10] | 362 | 432 | p=0.014 | negative 0/5, positive 4/5, tie 1/5 |

These lambda-to-lambda results are diagnostic sensitivity checks only; they do not establish universal lambda superiority.

## F. Representation-similarity summary

The values below are unweighted arithmetic means of five independently computed fold-level diagnostics. Feature matrices from different fold models were not concatenated for global CKA.

| Stage | Lambda | OOF Accuracy | Mean off-diagonal cosine | Centered-linear CKA |
|---:|---:|---:|---:|---:|
| 1 | 0.1 | 79.88% | 0.262 | 0.639 |
| 1 | 0.7 | 79.15% | 0.693 | 0.883 |
| 1 | 1.0 | 79.73% | 0.858 | 0.933 |
| 2 | 0.1 | 79.36% | 0.226 | 0.557 |
| 2 | 0.7 | 78.26% | 0.672 | 0.868 |
| 2 | 1.0 | 79.43% | 0.841 | 0.912 |

- Stage 1, lambda 1.0 - lambda 0.1: accuracy -0.15 pp; cosine 0.596; CKA 0.295.
- Stage 2, lambda 1.0 - lambda 0.1: accuracy 0.07 pp; cosine 0.615; CKA 0.355.

These results describe systematic representation similarity changes associated with the shortcut coefficient. Cosine or CKA magnitude is not interpreted as representation quality and no causal accuracy claim is made.

## G. Fold-direction consistency

Primary directions are reported in the final column of Section C. Counts use the sign of each fold's unrounded paired accuracy difference; folds are descriptive strata, not independent samples for significance testing.

## H. Interpretation boundaries

1. This is a new unified cross-dataset controlled validation, not an exact reproduction of the historical main experiments.
2. These numbers must not directly replace the historical CUB, Cars, or Flowers main-table values unless the entire table is explicitly switched to the new protocol.
3. The primary comparison supports only the paired OOF performance difference between Progressive Head lambda=0.7 and Ours-FT under this frozen protocol. It does not show that Progressive Head intrinsically harms performance or that shortcut scaling universally reduces accuracy.
4. CUB representation diagnostics describe similarity structure changes associated with the shortcut coefficient; they do not measure representation quality.

## I. Manuscript-ready rounded preview

| Dataset | Stage | Comparison (B - A) | A Acc. | B Acc. | Delta (pp) | 95% CI (pp) | n10 | n01 | Exact McNemar | Fold direction |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| CUB-200-2011 | 1 | Progressive Head lambda=0.7 - Ours-FT | 80.78% | 79.15% | -1.63 | [-2.54, -0.75] | 423 | 325 | p<0.001 | negative 5/5, positive 0/5, tie 0/5 |
| CUB-200-2011 | 2 | Progressive Head lambda=0.7 - Ours-FT | 80.56% | 78.26% | -2.30 | [-3.24, -1.38] | 470 | 332 | p<0.001 | negative 5/5, positive 0/5, tie 0/5 |
| Stanford Cars | 1 | Progressive Head lambda=0.7 - Ours-FT | 91.42% | 90.72% | -0.70 | [-1.24, -0.16] | 283 | 226 | p=0.013 | negative 4/5, positive 1/5, tie 0/5 |
| Stanford Cars | 2 | Progressive Head lambda=0.7 - Ours-FT | 91.26% | 90.19% | -1.07 | [-1.63, -0.50] | 316 | 229 | p<0.001 | negative 5/5, positive 0/5, tie 0/5 |
| Oxford Flowers-102 | 1 | Progressive Head lambda=0.7 - Ours-FT | 97.84% | 97.06% | -0.78 | [-1.37, -0.20] | 26 | 10 | p=0.011 | negative 4/5, positive 0/5, tie 1/5 |
| Oxford Flowers-102 | 2 | Progressive Head lambda=0.7 - Ours-FT | 97.45% | 96.76% | -0.69 | [-1.42, 0.00] | 35 | 21 | p=0.081 | negative 4/5, positive 1/5, tie 0/5 |

Machine-readable CSV and JSON retain full precision. Accuracy is shown to two decimals, delta and CI to two decimals, p-values to three decimals or `p<0.001`, and cosine/CKA to three decimals only in this Markdown preview.
