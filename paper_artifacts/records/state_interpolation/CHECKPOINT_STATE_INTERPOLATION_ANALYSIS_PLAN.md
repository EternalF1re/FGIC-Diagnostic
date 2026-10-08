# Endpoint-faithful Checkpoint-State Interpolation Plan

Frozen before inference on 2026-08-12. Sources are formal unified-protocol baseline seeds 42/43/44, five folds each. The fixed state path is `state_alpha = alpha*Stage1 + (1-alpha)*Stage2` for every compatible floating-point state tensor, including trainable parameters and floating buffers. For intermediate alphas, non-floating buffers are copied from Stage2. Alpha 0 and 1 bypass generic interpolation and directly load the pristine full Stage2 and Stage1 state respectively.

Alpha 0.5 is the sole pre-registered primary comparator. The sensitivity grid is fixed at 0.0 through 1.0 in increments of 0.1 and is descriptive only; no selection, refinement, or optimal-alpha claim is allowed.

No BN reset, refresh, calibration loader, train-mode forward, TTA, optimizer, scheduler, backward, gradient update, or learned module is authorized. Every model is loaded strictly, placed in eval mode, and evaluated on deterministic original-view held-out samples. Endpoint predictions must exactly reproduce the original per-fold Stage2/Stage1 reference predictions before any intermediate alpha for that fold is accepted.

Primary statistics are per-seed alpha 0.5 versus original Stage1 and Stage2: accuracy delta, 100,000 paired bootstrap percentile CI (seed 20260807), exact two-sided McNemar, correctness discordants, Macro-F1 and Balanced-Accuracy deltas. Seeds, not folds or pooled cross-seed samples, are independent replicates.

