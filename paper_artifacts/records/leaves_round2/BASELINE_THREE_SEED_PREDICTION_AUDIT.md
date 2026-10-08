# Baseline three-seed prediction audit

Status: PASS — Round2A training may proceed.

Prediction disagreement and correctness discordants are reported as distinct quantities.

| Stage | Seed pair | Prediction disagreement | Disagreement % | n10 | n01 | Both wrong | Error Jaccard |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 42 vs 43 | 253 | 1.3785% | 107 | 129 | 273 | 0.536346 |
| 1 | 42 vs 44 | 297 | 1.6183% | 141 | 140 | 262 | 0.482505 |
| 1 | 43 vs 44 | 230 | 1.2532% | 120 | 97 | 283 | 0.566000 |
| 2 | 42 vs 43 | 314 | 1.7109% | 143 | 144 | 278 | 0.492035 |
| 2 | 42 vs 44 | 307 | 1.6728% | 143 | 145 | 277 | 0.490265 |
| 2 | 43 vs 44 | 295 | 1.6074% | 134 | 135 | 286 | 0.515315 |

If Stage2 performance variance contracts while disagreement remains material, this supports only the phrase `performance-level convergence`, not solution convergence.
