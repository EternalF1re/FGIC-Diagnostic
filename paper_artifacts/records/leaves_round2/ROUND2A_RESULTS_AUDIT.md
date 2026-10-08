# Round2A Results Audit

Postflight: **PASS**.

All 25/25 formal training jobs and all 30/30 checkpoint-only OOF diagnostic jobs completed. No diagnostic job trained or modified a model.

## 1. MHSA ON versus OFF

| lambda | OFF Accuracy (%) | ON Accuracy (%) | Delta ON-OFF (pp) | 95% paired bootstrap CI | Exact McNemar p |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 97.6080 | 97.5699 | -0.0381 | [-0.2125, 0.1362] | 0.717563 |
| 0.7 | 97.6734 | 97.5426 | -0.1308 | [-0.3051, 0.0436] | 0.158346 |
| 1.0 | 97.5699 | 97.5971 | +0.0272 | [-0.1580, 0.2125] | 0.815896 |

All three intervals include zero. The observed results do not establish a stable MHSA accuracy benefit at any tested lambda.

## 2. Difference-in-differences interactions

| Contrast | Role | Point estimate (pp) | 95% paired bootstrap CI |
|---|---|---:|---:|
| I_0.1_minus_1.0 | primary | -0.0654 | [-0.3215, 0.1853] |
| I_0.1_minus_0.7 | secondary | +0.0926 | [-0.1471, 0.3324] |
| I_0.7_minus_1.0 | secondary | -0.1580 | [-0.4087, 0.0926] |

The primary and both secondary interaction intervals include zero. The experiment does not support the claim that the MHSA effect depends on shortcut lambda.

## 3. Five-seed baseline confirmation

| Seed | Stage1 Accuracy (%) | Stage2 Accuracy (%) | Stage2-Stage1 (pp) |
|---:|---:|---:|---:|
| 42 | 97.8096 | 97.7006 | -0.1090 |
| 43 | 97.9295 | 97.7061 | -0.2234 |
| 44 | 97.8042 | 97.7115 | -0.0926 |
| 45 | 97.7933 | 97.7224 | -0.0708 |
| 46 | 97.8314 | 97.6353 | -0.1962 |

Stage1 mean +/- sample SD is 97.8336 +/- 0.0554%; Stage2 is 97.6952 +/- 0.0345%.
Stage2 pairwise prediction disagreement remains 1.4984% to 1.8144%.
Therefore only `performance-level convergence` is supported; same basin, same solution, and same prediction function are not supported.

## 4. Representation diagnostics

True post-shortcut f1..f5 are summarized below. Cosine is pooled sample-wise; centered-linear CKA is computed fold-wise in float64 and then summarized.

| lambda | MHSA | Stage | Mean off-diagonal cosine | Mean centered-linear CKA |
|---:|---|---:|---:|---:|
| 0.1 | OFF | 1 | 0.16073483 | 0.57777664 |
| 0.1 | OFF | 2 | 0.11932485 | 0.55837749 |
| 0.1 | ON | 1 | 0.28075678 | 0.68717040 |
| 0.1 | ON | 2 | 0.28439669 | 0.68031581 |
| 0.7 | OFF | 1 | 0.65052828 | 0.85591669 |
| 0.7 | OFF | 2 | 0.63544866 | 0.83996394 |
| 0.7 | ON | 1 | 0.79412041 | 0.89793754 |
| 0.7 | ON | 2 | 0.85187463 | 0.92544367 |
| 1.0 | OFF | 1 | 0.82310906 | 0.91882862 |
| 1.0 | OFF | 2 | 0.82436200 | 0.89872488 |
| 1.0 | ON | 1 | 0.96873724 | 0.97638858 |
| 1.0 | ON | 2 | 0.98080239 | 0.98426196 |

These are descriptive geometry measurements. Lower similarity or a Stage1/Stage2 change is not evidence of better/worse representation quality, restoration, collapse, or a causal MHSA mechanism.

## 5. Attention diagnostics

Normalized entropy and maximum attention share are first computed for every sample x head x query distribution, then aggregated.

| lambda | Stage | Metric | Atomic units | Mean | SD | Median |
|---:|---:|---|---:|---:|---:|---:|
| 0.1 | 1 | normalized_entropy | 367060 | 0.003145 | 0.032084 | 0.000000 |
| 0.1 | 1 | max_attention_share | 367060 | 0.997956 | 0.024554 | 1.000000 |
| 0.1 | 2 | normalized_entropy | 367060 | 0.001958 | 0.024752 | 0.000000 |
| 0.1 | 2 | max_attention_share | 367060 | 0.998722 | 0.019297 | 1.000000 |
| 0.7 | 1 | normalized_entropy | 367060 | 0.153283 | 0.174303 | 0.089090 |
| 0.7 | 1 | max_attention_share | 367060 | 0.915656 | 0.129657 | 0.968446 |
| 0.7 | 2 | normalized_entropy | 367060 | 0.275721 | 0.173676 | 0.243404 |
| 0.7 | 2 | max_attention_share | 367060 | 0.837267 | 0.139425 | 0.879519 |
| 1.0 | 1 | normalized_entropy | 367060 | 0.764118 | 0.254804 | 0.859258 |
| 1.0 | 1 | max_attention_share | 367060 | 0.484382 | 0.212711 | 0.431057 |
| 1.0 | 2 | normalized_entropy | 367060 | 0.796540 | 0.205973 | 0.858268 |
| 1.0 | 2 | max_attention_share | 367060 | 0.466199 | 0.178320 | 0.432937 |

Attention concentration is descriptive and is not automatically an explanation for classification performance.

## 6. Supported conclusion for manuscript review

Within the frozen lambda x MHSA design, MHSA produced small and directionally inconsistent OOF changes, while all paired effect and interaction intervals included zero. The current evidence therefore does not support retaining MHSA on the basis of a demonstrated accuracy gain or lambda-dependent benefit. Any architectural decision to remove it should additionally consider complexity and the descriptive diagnostics, not claim proof of harm.

## 7. Principal review files

- `statistics/mhsa_on_vs_off_lambda_0_1.csv`
- `statistics/mhsa_on_vs_off_lambda_0_7.csv`
- `statistics/mhsa_on_vs_off_lambda_1_0.csv`
- `statistics/mhsa_interaction_difference_in_differences.csv`
- `statistics/baseline_five_seed_per_seed.csv`
- `statistics/baseline_five_seed_summary.csv`
- `statistics/baseline_five_seed_prediction_audit.csv`
- `representation/representation_overview.csv`
- `representation/cosine_pooled_summary.csv`
- `representation/cka_summary.csv`
- `attention/attention_overview.csv`
- `attention/attention_by_head_query.csv`
- `attention/position_allocation_summary.csv`
- `manifests/postflight_audit.json`

## 8. Stop boundary

No DFAG or result-dependent follow-up variant was launched by this finalization. Stop for human/GPT-5.6 review.
