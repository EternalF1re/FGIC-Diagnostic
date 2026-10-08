# Unified Standalone Dynamic DFAG Results Audit

**Integrity: PASS**

**Pre-declared classification: BORDERLINE** — Type B1: 3/3 positive against both primary references, but paired-CI evidence does not reach PASS.

## Hard gates

- PASS — `15_of_15_jobs_complete`
- PASS — `no_failed_fold`
- PASS — `oof_18353_exact_per_seed`
- PASS — `sample_ids_and_labels_exact`
- PASS — `source_stage1_mapping_exact`
- PASS — `anchor_unchanged`
- PASS — `anchor_bn_unchanged`
- PASS — `plastic_stage2_protocol_matched`
- PASS — `gate_architecture_exact`
- PASS — `feature_dim_1536_and_r16_bias_false`
- PASS — `fspec_only_conditioning_and_formula_exact`
- PASS — `no_ssph_or_joint_model`
- PASS — `no_fixed_g_training`
- PASS — `no_anchor_sweep`
- PASS — `no_gating_source_ablation`
- PASS — `old_artifacts_unchanged`
- PASS — `gate_tensors_saved`
- PASS — `counterfactuals_inference_only`
- PASS — `complexity_includes_both_backbones`
- PASS — `statistics_computed_per_seed`
- PASS — `no_55059_pooled_significance_test`
- PASS — `no_result_dependent_training_followup`
- PASS — `gate_training_sanity_pass`

## Per-seed DFAG performance

| Seed | Accuracy | Macro-F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 42 | 0.978369 | 0.980663 | 0.980490 |
| 43 | 0.979622 | 0.982043 | 0.982080 |
| 44 | 0.978968 | 0.981486 | 0.981525 |

## Interpretation boundary

The experiment supports reporting measured OOF performance, paired deltas, gate variability, branch discrepancy, counterfactual replacement effects, and deployment cost. It does not directly establish semantic restoration, useful routing solely from gate variance, necessity, overfitting prevention, redundancy from cosine, or a causal mechanism.

STOP

No seed45/46 automatically launched.
No fixed-g training launched.
No anchor sweep launched.
No gating-source ablation launched.
No SSPH+DFAG joint launched.
