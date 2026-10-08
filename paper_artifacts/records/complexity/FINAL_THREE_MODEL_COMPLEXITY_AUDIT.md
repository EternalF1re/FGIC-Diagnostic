# Final Three-Model Complexity and Inference-Efficiency Audit

Overall status: **PASS**

## A. Environment

- Python: 3.9.23
- Python executable: `<LOCAL_PATH>
- PyTorch: 2.8.0+cu129
- CUDA runtime: 12.9
- cuDNN: 91002
- GPU: NVIDIA GeForce RTX 5070 Ti
- THOP: 0.1.1

## B. Unified profiling protocol

- All three models were built and profiled in one Python process and one GPU session on `cuda:0`.
- No checkpoint or real dataset was loaded; model weights were random because architecture complexity is weight-value independent.
- Shared synthetic input: `torch.randn(1,3,299,299)`, FP32, batch size 1.
- `eval()` plus `torch.inference_mode()`; no AMP, FP16, `torch.compile`, DataLoader, preprocessing, TTA, or ensemble.
- FLOPs = 2 x THOP MACs.
- Latency: 10 warm-up forwards and 30 measured forwards; `torch.cuda.synchronize()` immediately before and after every measured forward; `time.perf_counter_ns()` timing.
- Predeclared selection rule: round 1 (Ours-FT -> Progressive Head -> DFAG) is the manuscript result. Reverse-order round 2 is order-effect sanity only and is never selected for a better-looking number.

## C. Architecture hard checks

### Ours-FT: **PASS**
- `resident_parameter_sanity_exact`: PASS
- `one_backbone_branch`: PASS
- `backbone_pooled_dimension_1536`: PASS
- `conventional_head_type`: PASS
- `fc1_1536_to_1024`: PASS
- `fc2_1024_to_1024`: PASS
- `classifier_1024_to_classes`: PASS
- `dropout_0_2`: PASS
- `batchnorm_and_silu`: PASS
- `dfag_absent`: PASS

### Progressive Head: **PASS**
- `resident_parameter_sanity_exact`: PASS
- `one_backbone_branch`: PASS
- `projection_1536_to_256`: PASS
- `five_mapping_blocks`: PASS
- `mapping_sequence_exact`: PASS
- `shortcut_lambda_exact_0_7`: PASS
- `arithmetic_mean_f1_to_f5`: PASS
- `aggregation_dropout_0_2`: PASS
- `mhsa_absent`: PASS
- `terminal_residual_absent`: PASS
- `dfag_absent`: PASS

### DFAG: **PASS**
- `resident_parameter_sanity_exact`: PASS
- `standalone_dfag_type`: PASS
- `anchor_and_plastic_branches`: PASS
- `two_distinct_backbones`: PASS
- `both_backbones_inception_resnet_v2`: PASS
- `feature_dimension_1536`: PASS
- `reduction_16`: PASS
- `gate_structure_exact`: PASS
- `gate_linears_bias_free`: PASS
- `gate_parameters_exact_294912`: PASS
- `gate_receives_only_f_spec_source`: PASS
- `elementwise_gated_fusion_source`: PASS
- `progressive_head_absent`: PASS
- `mhsa_absent`: PASS
- `anchor_frozen`: PASS

## D. Final three-model table (round 1 manuscript values)

| Model | Resident Params (M) | Head/Gate Params (M) | FLOPs (G) | Latency mean +/- sample SD (ms/image) | Absolute Peak Memory (MiB) |
|---|---:|---:|---:|---:|---:|
| Ours-FT | 57.114 | 2.808 | 26.315 | 15.15 +/- 0.90 | 246.2 |
| Progressive Head | 55.077 | 0.770 | 26.311 | 14.96 +/- 0.53 | 239.1 |
| DFAG | 114.524 | 0.295 | 52.625 | 29.98 +/- 1.21 | 468.6 |

## E. Relative ratios (round 1; Ours-FT reference)

### Progressive Head / Ours-FT

- Resident parameters: 0.964x (-3.568%)
- FLOPs: 1.000x (-0.015%)
- Latency: 0.988x (-1.201%)
- Absolute peak memory: 0.971x (-2.889%)

### DFAG / Ours-FT

- Resident parameters: 2.005x (+100.516%)
- FLOPs: 2.000x (+99.981%)
- Latency: 1.980x (+97.965%)
- Absolute peak memory: 1.904x (+90.356%)

## F. Memory-definition explanation

Three CUDA allocated-memory quantities are retained for every model. `resident_allocated_before_forward` is allocated memory after model creation, shared-input transfer, warm-up, and deletion of the warm-up output. `absolute_peak_allocated` is `torch.cuda.max_memory_allocated()` during the synchronized 30-forward measurement after resetting peak statistics. `incremental_forward_peak` is absolute peak minus resident allocated before forward. The manuscript table uses absolute peak allocated for all three models under the same definition.

## G. Order-effect sanity check

Predeclared threshold: absolute round-mean change <= 5.0% means no obvious order effect. Overall: **WARN**.

- Ours-FT: round 1 15.146040 ms; round 2 14.961440 ms; change -1.219% (PASS).
- Progressive Head: round 1 14.964103 ms; round 2 15.783440 ms; change +5.475% (WARN).
- DFAG: round 1 29.983787 ms; round 2 30.337973 ms; change +1.181% (PASS).

## H. Historical-result consistency check

Status: **PASS**. Resident parameter counts match the frozen historical expectations exactly and FLOPs match within 0.1%. Historical latency and memory were measured in different GPU sessions and are retained only as context; none of those old runtime values is copied into the final table.

## I. Manuscript-ready rounded table

| Model | Params (M) | Head/Gate (M) | FLOPs (G) | Latency (ms/image) | Peak Memory (MiB) |
|---|---:|---:|---:|---:|---:|
| Ours-FT | 57.114 | 2.808 | 26.315 | 15.15 +/- 0.90 | 246.2 |
| Progressive Head | 55.077 | 0.770 | 26.311 | 14.96 +/- 0.53 | 239.1 |
| DFAG | 114.524 | 0.295 | 52.625 | 29.98 +/- 1.21 | 468.6 |

## J. Interpretation boundaries

These measurements support descriptive comparisons under this exact hardware, software, batch-size, precision, input-shape, and timing protocol. Small runtime differences should not be generalized as intrinsic speed advantages. DFAG is explicitly a two-branch inference model with two full backbone forwards; it must not be described as lightweight or as having negligible overhead. Results do not include data loading or preprocessing and are not end-to-end application latency.
