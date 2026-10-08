# Final External Validation Audit

- `FORMAL_JOBS_EXPECTED = 40`
- `FORMAL_JOBS_COMPLETE = 40`
- `FORMAL_JOBS_FAILED = 0`
- `CONFIG_SHA_MATCH = YES`
- `OOF_COVERAGE_PASS = YES`
- `CUB_DIAGNOSTICS_COMPLETE = YES`
- `FINAL_EXTERNAL_VALIDATION_READY_FOR_ANALYSIS = YES`

Primary metrics below are recomputed on each complete pooled five-fold OOF prediction set, not arithmetic means of fold metrics. Official-test metrics remain secondary.

| Dataset | Method | Lambda | Stage | N | OOF Accuracy | Macro-F1 | Balanced Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| CUB-200-2011 | Ours-FT | - | 1 | 5994 | 0.807808 | 0.807684 | 0.807736 |
| CUB-200-2011 | Ours-FT | - | 2 | 5994 | 0.805639 | 0.805354 | 0.805534 |
| CUB-200-2011 | Progressive Head | 0.7 | 1 | 5994 | 0.791458 | 0.791492 | 0.791356 |
| CUB-200-2011 | Progressive Head | 0.7 | 2 | 5994 | 0.782616 | 0.783552 | 0.782552 |
| CUB-200-2011 | Progressive Head | 0.1 | 1 | 5994 | 0.798799 | 0.797692 | 0.798684 |
| CUB-200-2011 | Progressive Head | 0.1 | 2 | 5994 | 0.793627 | 0.792462 | 0.793552 |
| CUB-200-2011 | Progressive Head | 1.0 | 1 | 5994 | 0.797297 | 0.797026 | 0.797201 |
| CUB-200-2011 | Progressive Head | 1.0 | 2 | 5994 | 0.794294 | 0.793917 | 0.794184 |
| Stanford Cars | Ours-FT | - | 1 | 8144 | 0.914170 | 0.912030 | 0.912775 |
| Stanford Cars | Ours-FT | - | 2 | 8144 | 0.912574 | 0.910479 | 0.911215 |
| Stanford Cars | Progressive Head | 0.7 | 1 | 8144 | 0.907171 | 0.905447 | 0.905725 |
| Stanford Cars | Progressive Head | 0.7 | 2 | 8144 | 0.901891 | 0.900422 | 0.900803 |
| Oxford Flowers-102 | Ours-FT | - | 1 | 2040 | 0.978431 | 0.978263 | 0.978431 |
| Oxford Flowers-102 | Ours-FT | - | 2 | 2040 | 0.974510 | 0.974183 | 0.974510 |
| Oxford Flowers-102 | Progressive Head | 0.7 | 1 | 2040 | 0.970588 | 0.970081 | 0.970588 |
| Oxford Flowers-102 | Progressive Head | 0.7 | 2 | 2040 | 0.967647 | 0.967395 | 0.967647 |

Prediction-level OOF artifacts are preserved under `final_oof_predictions/`. Paired OOF accuracy differences are stored in `final_external_validation_results.json`; no bootstrap, McNemar, or significance interpretation was invented automatically.
