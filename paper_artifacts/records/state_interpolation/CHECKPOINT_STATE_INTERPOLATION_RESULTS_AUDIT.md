# Checkpoint-State Interpolation Results Audit

## Integrity
PASS. All 30 endpoints reproduced original predictions exactly before intermediate inference. All 165 models used eval-only complete checkpoint-state interpolation with no BN refresh, optimizer, scheduler, backward, gradient, TTA, or source modification.

## Endpoint reproduction
Alpha 0 directly loaded original Stage2 full state; alpha 1 directly loaded original Stage1 full state. All sample IDs, labels, predictions, Accuracy, Macro-F1 and Balanced Accuracy were exact for every seed/fold. Logit reproduction details are in `manifests/endpoint_fidelity_gate.json`.

## Alpha 0.5 primary results
| Seed | Accuracy (%) | vs Stage1 delta pp [95% CI] | vs Stage2 delta pp [95% CI] |
|---:|---:|---:|---:|
| 42 | 97.8042 | -0.0054 [-0.1417, 0.1308] | 0.1035 [-0.0272, 0.2343] |
| 43 | 97.8369 | -0.0926 [-0.2071, 0.0218] | 0.1308 [0.0163, 0.2452] |
| 44 | 97.8096 | 0.0054 [-0.1253, 0.1362] | 0.0981 [-0.0054, 0.2016] |

Across seeds: alpha=.5 vs Stage1 -0.0309 +/- 0.0538 pp; vs Stage2 0.1108 +/- 0.0175 pp. Exact McNemar results are in the primary CSVs.

## Full descriptive curve
| alpha | Primary | Accuracy mean +/- sample SD (%) |
|---:|:---:|---:|
| 0.0 | No | 97.7061 +/- 0.0054 |
| 0.1 | No | 97.7642 +/- 0.0206 |
| 0.2 | No | 97.7896 +/- 0.0274 |
| 0.3 | No | 97.8024 +/- 0.0083 |
| 0.4 | No | 97.7987 +/- 0.0094 |
| 0.5 | Yes | 97.8169 +/- 0.0175 |
| 0.6 | No | 97.8169 +/- 0.0269 |
| 0.7 | No | 97.8278 +/- 0.0280 |
| 0.8 | No | 97.8223 +/- 0.0457 |
| 0.9 | No | 97.8296 +/- 0.0494 |
| 1.0 | No | 97.8478 +/- 0.0708 |

No alpha was selected; alpha=.6 and all non-primary points are descriptive only.

## Prediction overlap
Per-seed alpha=.5 disagreement and error Jaccard against each endpoint are in `statistics/prediction_overlap.csv`.

## Conclusions and limits
The report supports only measured endpoint-faithful checkpoint-state interpolation performance and prediction overlap. It does not support semantic, representation, collapse, basin, attractor, optimal-alpha, or DFAG-necessity claims. Branch count remains one and Stage1-only remains the simple comparator unless the pre-registered alpha=.5 evidence demonstrates otherwise.

Round2A and the old reset/recalibration experiment were unchanged. STOP: no DFAG task was launched.
