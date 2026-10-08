# DFAG Five-Seed Final Audit

**FINAL CLASSIFICATION: FAIL**

## Frozen A/B/C/D criteria

- PASS — `A_stage1_5_of_5_positive`
- PASS — `B_alpha_0_5_5_of_5_positive`
- FAIL — `C_at_least_one_primary_has_3_of_5_strictly_positive_ci`
- PASS — `D_other_primary_has_no_negative_direction_or_fully_negative_ci`

## Five-seed DFAG performance

| Seed | Accuracy | Macro-F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 42 | 0.978369 | 0.980663 | 0.980490 |
| 43 | 0.979622 | 0.982043 | 0.982080 |
| 44 | 0.978968 | 0.981486 | 0.981525 |
| 45 | 0.978587 | 0.981278 | 0.981362 |
| 46 | 0.978968 | 0.981317 | 0.981317 |

## Interpretation boundary

The inference-only counterfactual analysis does not isolate an accuracy advantage attributable specifically to sample-dependent gate variation in the trained DFAG models.

These counterfactuals are not independently trained fixed-gating baselines and therefore do not establish that dynamic gating was unnecessary during training or that the observed DFAG performance originates solely from dual-branch fusion.

Gate variation and anchor/plastic cosine/L2 are descriptive only; they do not establish semantic restoration, useful routing, redundancy, overfitting prevention, or a causal mechanism.

No pooled 91,765-sample significance test was performed. The independent unit is the training seed.

STOP
