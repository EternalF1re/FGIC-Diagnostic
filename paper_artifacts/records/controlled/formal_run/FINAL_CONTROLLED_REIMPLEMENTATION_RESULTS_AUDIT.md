# Final Controlled Reimplementation Results

`FORMAL_RUN_COMPLETE`

All 60 jobs completed with complete five-fold pooled OOF coverage. Every job used the same one-pass Cars filename index; runtime and protocol-core SHA-256 identities were checked before aggregation. Its frozen assignments, folds, class map and path semantics were verified by `scripts/verify_cars_index_optimization.py`. See `final_pooled_oof_metrics.csv`, `final_fold_descriptive_metrics.csv`, and `final_paired_statistics.csv`. Paired statistics use 100,000 paired bootstrap resamples, 95% percentile intervals and exact two-sided McNemar tests on pooled OOF samples; folds are descriptive strata, not significance units.
