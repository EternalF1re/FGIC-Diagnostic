# Unified Standalone Dynamic DFAG Preflight

**Status: PASS**

Historical `DFAG.py`, the frozen unified evaluator, and the controlled baseline agree on the 1536-D standalone gate and conventional classifier.

## Classifier and BN-refresh resolution

No conflict was found. The 50-batch pre-optimization refresh updates plastic-backbone BN and both BN layers in the plastic conventional classifier. The anchor remains in eval mode, its parameters/buffers do not change, and the gate contains no BN.

## Hard gates

- PASS — `01_15_selected_stage1_sources_available`
- PASS — `02_source_sha256_exact`
- PASS — `03_stage1_comparator_oof_available`
- PASS — `04_current_stage2_comparator_oof_available`
- PASS — `05_alpha05_oof_available`
- PASS — `06_oof_18353_exact`
- PASS — `07_labels_exact`
- PASS — `08_fold_mapping_exact`
- PASS — `09_seed_mapping_exact`
- PASS — `10_representation_dimension_1536`
- PASS — `11_gate_dimensions_exact`
- PASS — `12_reduction_16`
- PASS — `13_gate_bias_false`
- PASS — `14_gate_input_fspec_only`
- PASS — `15_fusion_formula_exact`
- PASS — `16_common_initialization_exact`
- PASS — `17_anchor_frozen_eval`
- PASS — `18_plastic_current_bn_refresh_retained`
- PASS — `19_stage2_protocol_otherwise_exact`
- PASS — `20_conventional_classifier_exact`
- PASS — `21_no_ssph`
- PASS — `22_no_mhsa`
- PASS — `23_no_terminal`
- PASS — `24_output_directory_isolated`
- PASS — `25_protected_snapshot_saved`
- PASS — `26_no_fixed_g_training_scheduled`
- PASS — `27_no_anchor_or_gating_source_experiment`
- PASS — `28_no_joint_model_scheduled`
- PASS — `gate_parameter_count_exact`
- PASS — `classifier_input_1536`
- PASS — `state_schema_identical_all_sources`
- PASS — `protected_sources_unchanged_during_preflight`
- PASS — `disk_free_at_least_12GB`

## Frozen architecture

- backbone representation: 1536
- fusion representation: 1536
- classifier input: 1536
- gate: 1536 -> 96 -> 1536, ReLU/Sigmoid, no bias, 294,912 parameters
- gate input: f_spec only
- fusion: g*f_anc + (1-g)*f_spec

Exactly 15 Stage2 jobs are authorized only when this report is PASS. No existing comparator is retrained.
