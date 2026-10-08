# Paper-to-artifact mapping

This is an experiment/section mapping, not a claim that the latest submitted manuscript's table numbers were checked. The available manuscript files were older May/June drafts. The owner must check current main-text/SI table numbers before claiming exact submitted-table coverage.

| Main-text/SI result topic | Public artifacts | Independent numeric verification |
|---|---|---|
| Leaves baseline and multiseed results | `oof/leaves_screen/baseline_seed42_*.csv`, `oof/leaves_round1/baseline_seed4[34]_*.csv`, `oof/leaves_round2/baseline_seed4[56]_*.csv`; `records/leaves_round2/statistics/baseline_five_seed_*.csv` | Pooled three metrics; five-seed mean and sample SD |
| Progressive shortcut coefficient, Leaves | `oof/leaves_screen/lambda_*.csv`, `oof/leaves_round1/lambda_*.csv`; `records/leaves_round1/statistics/lambda_direct_paired.csv` | Paired delta, CI, exact McNemar; representation summaries copied separately |
| Section 4.3 cross-dataset validation | `oof/external/*.csv`; `records/external/final_external_oof_summary.csv`, `FINAL_EXTERNAL_PAIRED_STATISTICS.csv` | Pooled three metrics and 18 archived paired rows |
| CUB lambda diagnostic | `oof/external/cub__progressive_lambda_*.csv`; `records/external/FINAL_CUB_LAMBDA_DIAGNOSTIC_SUMMARY.csv`, `final_cub_lambda_representation_diagnostics.csv` | OOF and paired rows; cosine/CKA available as historical summaries |
| Section 4.7 cross-architecture experiments | `oof/cross_backbone/*.csv`; `records/cross_backbone/FINAL_CROSS_BACKBONE_*.csv` | Pooled three metrics and paired rows; training seed42, not five independent seeds |
| DFAG multiseed and counterfactual diagnostics | `oof/dfag/*.csv`, `oof/dfag_confirmation/*.csv`; their `records/` metrics, statistics, gate, representation, counterfactual, and decision files | Five-seed metrics, paired diagnostics and all six inference modes |
| Complete checkpoint-state interpolation | `oof/state_interpolation/*.csv`; `records/state_interpolation/curves/`, `statistics/` | Eleven-alpha three-seed metrics/mean/sample SD; alpha0.5 paired rows |
| Section 4.8 controlled comparison | `oof/controlled/*.csv`; `records/controlled/formal_run/final_pooled_oof_metrics.csv`, `final_paired_statistics.csv` | Twelve complete OOF sets, three metrics, ten paired comparisons |
| Complexity table | `records/complexity/`, `records/progressive_complexity/` | Recorded parameters/FLOPs/latency/memory; no new hardware profiling in this release |
| MHSA/attention SI diagnostics | `records/leaves_round2/attention/`, `representation/`; `oof/leaves_round2/mhsa_*.csv` | Historical summaries; distinct attention variants remain labeled |

Quoted numbers from original baseline papers are not controlled experimental outputs. The existing third-party attribution and the original papers remain the sources for those rows.

Verification asserts unrounded historical output values at tolerance 1e-12. The original numerical tables are retained even if a fresh calculation disagrees; every discrepancy is reported in the verification JSON. A PASS against these tables does not prove that a subsequently edited manuscript reproduces them faithfully.
