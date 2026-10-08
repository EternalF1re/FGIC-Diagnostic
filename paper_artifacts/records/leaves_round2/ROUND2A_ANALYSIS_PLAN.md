# Phase2D Round2A frozen analysis plan

Instruction SHA-256: `FC4D9CD4DA231A1117581AD0EE06F40AFA521B83A4FA770E704648DE731D4307`

This plan is frozen before any Round2A training starts. Round2A is formal
controlled validation, not exploratory model selection.

## Primary scientific question

Does the effect of MHSA depend on the similarity level of progressively
transformed representations induced by shortcut coefficient lambda?

## Factorial design

- Factor A: lambda in {0.1, 0.7, 1.0}.
- Factor B: MHSA OFF versus ON.
- Training seed: 42.
- OFF arms are reused from verified Phase2B/Round1 artifacts.
- Only the three ON arms are newly trained, with five fixed folds each.
- Progressive mapping: five 256-D post-shortcut stages.
- MHSA: four heads, attention dropout 0.1, pre-LN attention and residual
  fusion `LN(S + MHSA(LN(S)))`, followed by mean pooling.
- f0 is excluded from the token sequence and no terminal f0 residual is used.

## Outcomes and inference

Primary outcome: Stage2 pooled held-out OOF accuracy over exactly 18,353
samples. Secondary outcomes: Macro-F1 and Balanced Accuracy.

For each lambda:

`Delta_MHSA(lambda) = Accuracy(MHSA ON) - Accuracy(MHSA OFF)`.

Primary interaction:

`I_0.1_vs_1.0 = Delta_MHSA(0.1) - Delta_MHSA(1.0)`.

Secondary interactions:

- `Delta_MHSA(0.1) - Delta_MHSA(0.7)`;
- `Delta_MHSA(0.7) - Delta_MHSA(1.0)`.

All effects and interaction contrasts use sample-level paired bootstrap on the
same OOF indices, 100,000 repetitions, seed 20260807, with percentile 95% CI.
Pairwise ON/OFF comparisons additionally report exact two-sided McNemar p,
n10, n01, changed-prediction count, Macro-F1 delta and Balanced-Accuracy delta.
The primary interaction is inferred from its direct difference-in-differences
bootstrap CI, not from visual comparison of three McNemar p-values.

## Frozen baseline prediction audit

Before training, baseline seeds 42, 43 and 44 are checked at Stage1 and Stage2.
For pairs 42/43, 42/44 and 43/44, report separately:

- prediction disagreement count and percentage;
- A-correct/B-wrong n10 and A-wrong/B-correct n01;
- both-wrong, A-only-wrong, B-only-wrong and error-set Jaccard.

OOF sample order, labels, folds, one-time coverage, prediction provenance and
checkpoint SHA-256 must pass. Checkpoint hashes must be distinct across seeds.
Training is blocked if any Stage2 pair has fewer than 10 prediction changes or
less than 0.1% disagreement, or if any provenance/integrity check fails.

If accuracy variance contracts while predictions remain materially different,
the only allowed phrase is `performance-level convergence`; solution-level
convergence, attractor, same basin and same solution are prohibited.

## Baseline five-seed confirmation

Baseline seeds 45 and 46 are fixed before MHSA results are known. Each uses the
same architecture, five folds and frozen protocol as seeds 42/43/44. The seed
is the independent repetition unit; folds are OOF construction, not 25
independent replicates. Report per-seed Stage1/Stage2 metrics and across-seed
mean, sample SD, min, max and range, plus all ten seed-pair prediction audits.

## Representation and attention diagnostics

For each ON arm, Stage1 and Stage2 held-out OOF forward passes save true
post-shortcut f1..f5 and compute Round1-compatible cosine, centered-linear CKA,
adjacent normalized change and per-layer feature norms.

Attention weights retain sample x head x query x key for five tokens. For each
sample/head/query distribution, compute normalized entropy and maximum
attention share before any averaging. Raw reproducible statistics and
fold-wise/lambda-stage summaries are required. Lower similarity, norm changes
or concentrated attention are descriptive measurements, not automatic
evidence of better representations or a mechanism.

## Protocol lock and stop gate

Dataset, folds, augmentation, class weights, optimizer, scheduler, two-stage
training, BN adaptation, AMP, gradient clipping and evaluation are byte-level
or field-level matched to Phase2B/Round1 except for declared seed/MHSA factors.

Forbidden in Round2A: lambda 0.5/0.8, terminal residual, DFAG, earlier-anchor
DFAG, learnable lambda, other head counts/dropout/depth, extra MS-LFA seeds or
any result-dependent additions. After the specified postflight audit and
`ROUND2A_RESULTS_AUDIT.md` are produced, STOP for human review.
