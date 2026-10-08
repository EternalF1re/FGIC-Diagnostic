# DFAG Seed45/46 Results Audit

**Postflight integrity: PASS**

| Seed | Accuracy | Macro-F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 45 | 0.978587 | 0.981278 | 0.981362 |
| 46 | 0.978968 | 0.981317 | 0.981317 |

## Postflight gates

- PASS — `ten_of_ten_complete`
- PASS — `zero_failed_folds`
- PASS — `oof_18353_per_new_seed`
- PASS — `oof_sample_ids_labels_and_folds_exact`
- PASS — `anchor_unchanged`
- PASS — `anchor_bn_unchanged`
- PASS — `plastic_stage2_protocol_matched`
- PASS — `gate_architecture_exact`
- PASS — `seed45_46_exact`
- PASS — `counterfactual_inference_only`
- PASS — `historical_seed42_44_artifacts_unchanged`
- PASS — `no_pooled_five_seed_significance_test`
- PASS — `statistics_independent_unit_seed`
- PASS — `no_fixed_g_training`
- PASS — `no_anchor_sweep`
- PASS — `no_gating_source_ablation`
- PASS — `no_joint_or_progressive_training`
- PASS — `no_seed47_plus`
- PASS — `scheduler_contains_only_ten_authorized_tasks`
- PASS — `gate_training_sanity_pass`
- PASS — `final_classification_not_borderline`

Only seed45/46 unified standalone dynamic DFAG was trained. No later experiment was launched.

STOP
