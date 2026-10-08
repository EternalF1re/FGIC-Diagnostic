# Final Cross-Backbone Supplementary Audit

Status: **COMPLETE**

No training, optimizer update, BatchNorm recalibration, checkpoint modification, or formal-output overwrite was performed.

## 1. Ours-FT Stage 2 minus Stage 1 paired pooled-OOF statistics

The exact formal paired protocol was reused: N=18,353, 100,000 paired bootstrap resamples with seed 20260818, 95% percentile CI, and exact two-sided McNemar. Folds are descriptive strata only.

| Backbone | Stage 1 Acc. | Stage 2 Acc. | Delta pp | 95% CI pp | McNemar | Fold direction | Disagreement |
|---|---:|---:|---:|---:|---:|---|---:|
| resnet50 | 97.4119% | 97.5971% | +0.1853 | [0.0598, 0.3160] | p=0.006 | negative 1/5, positive 4/5, tie 0/5 | 161 (0.8772%) |
| convnext_tiny | 97.9894% | 97.7987% | -0.1907 | [-0.3433, -0.0381] | p=0.017 | negative 5/5, positive 0/5, tie 0/5 | 218 (1.1878%) |

## 2. Alpha interpolation implementation semantics

- Every floating state tensor, including parameters, BatchNorm running_mean, and BatchNorm running_var, uses `alpha*Stage1 + (1-alpha)*Stage2`.
- Every non-floating state tensor is copied from Stage 2.
- There is no endpoint-specific branch for alpha=0 or alpha=1.
- In these checkpoints the only non-floating state tensors are BatchNorm `num_batches_tracked` counters.

## 3. Per-fold endpoint fidelity

| Backbone | Fold | alpha=0 pred | alpha=0 logits | alpha=0 full state=Stage2 | alpha=1 pred | alpha=1 logits | alpha=1 full state=Stage1 | Inference endpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet50 | 0 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| resnet50 | 1 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| resnet50 | 2 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| resnet50 | 3 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| resnet50 | 4 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| convnext_tiny | 0 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| convnext_tiny | 1 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| convnext_tiny | 2 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| convnext_tiny | 3 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| convnext_tiny | 4 | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |

## 4. Endpoint conclusion

- alpha=0: strict checkpoint-state identity with Stage 2 passes, and predictions/logits are bitwise exact in all 10 backbone-fold cases.
- alpha=1: all floating state, including BN running statistics, is exactly Stage 1; predictions/logits are bitwise exact in all 10 cases. Strict full-state identity fails because Stage-2 `num_batches_tracked` counters are retained.
- Retaining Stage-2 `num_batches_tracked` does not alter these eval-mode endpoint logits because the counters are not used by BatchNorm in eval mode.

## 5. Accurate alpha=0.5 description

The current alpha=0.5 artifact is a complete floating-state checkpoint interpolation: all floating parameters and floating buffers are averaged as `0.5*Stage1 + 0.5*Stage2`, while non-floating BatchNorm `num_batches_tracked` buffers are copied from Stage 2. It is not a literal interpolation of every checkpoint state tensor.
