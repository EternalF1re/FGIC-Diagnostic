# Final External Protocol Manifest

## Final pre-launch status

- `STAGE1_LABEL_SMOOTHING = 0.05`
- `STAGE2_LABEL_SMOOTHING = 0.03`
- `EXTERNAL_BATCH_SIZE = 32`
- `NEW_CONFIG_SHA256 = 7f667c125983ca82ec36214b727ee8fbe3215f21562abd260f4018559f87e3e5`
- `FORMAL_LEDGER_ROWS = 40`
- `FORMAL_LEDGER_PENDING = 40`
- `FORMAL_JOBS_STARTED = 0`
- `EPOCH0_SEMANTICS_CONFIRMED = YES`
- `LIGHTWEIGHT_SANITY_PASS = YES`
- `FINAL_40_JOB_RUN_READY = YES`

Formal training remains stopped pending manual approval.

## Superseded protocol identity

- Previous config SHA-256: `a4a975bf1636e91e27621bd666d91e8bab3b5f960e7d964bbf50fd50f1e990fc`
- Reason: Stage1 label smoothing corrected from manuscript-recorded 0.03 to audited controlled-protocol value 0.05 before formal training.
- Formal results under the previous config: **0**
- Formal results under this final config: **0**

This correction comes from protocol provenance audit, not smoke accuracy or result-dependent tuning.

## Frozen protocol delta

The only scientific protocol correction is Stage1 label smoothing `0.03 -> 0.05`. Stage2 remains `0.03`. It applies identically to all 40 CUB/Cars/Flowers jobs and all methods/lambdas.

External batch size remains 32. Classify Leaves controlled analyses use 64; this is an explicit protocol difference, not an error. Every configuration within an external dataset uses the same batch size.

All other data pools, folds, transforms, optimization, augmentation, BN recalibration, checkpoint selection, evaluation, architectures, lambdas, and per-fold CKA/cosine aggregation semantics remain frozen.

## Lightweight sanity check

Both CUB Ours-FT and Progressive lambda=0.7 constructed the corrected legal batch-size-32 Stage1 path, used `CrossEntropyLoss(label_smoothing=0.05)`, produced finite loss, and completed one isolated backward pass with finite gradients. No optimizer step, checkpoint, accuracy, result, or ledger-status change was produced. This was a technical configuration check, not an experiment.

## Stage2 epoch-0 invariant

Confirmed directly from `<EXPERIMENT_ROOT>\phase2_controlled_validation\corrected_external_protocol_smoke\scripts\train_cub_smoke.py` (`a2d851317dc69ee0f6e70b5b08ca3053dc6f79a9d67e6e8b551d9e1fb9cf1f63`): the runner completes the training batch loop, loss backward, and optimizer step before validation and checkpoint selection. Therefore zero-based Stage2 epoch 0 is validation after the first completed gradient-based Stage2 epoch. Behavior was not changed.

## Formal ledger and reporting

`formal_job_ledger.csv` contains exactly 40 unique PENDING jobs with the new config hash. Started jobs: 0.

Future Leaves-vs-CUB shortcut comparison uses only common lambda points `0.1, 0.7, 1.0`. Leaves `0.9` is an additional Leaves-only descriptive point; the four-point Leaves curve and three-point CUB curve are not fully matched.
