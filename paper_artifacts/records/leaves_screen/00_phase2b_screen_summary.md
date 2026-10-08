# Phase 2B-S1 controlled screen

## Completion and protocol

All and only #0/#1/#2 completed five folds with `split_random_state=42` and `training_seed=42` independently recorded for every fold. The primary series is deterministic original-view pooled held-out OOF. This is a single-training-seed screen, not independent-seed confirmation.

| variant | OOF accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| #0 Original baseline | 97.7006% | 97.9467% | 97.9518% |
| #1 Deep/narrow λ=1.0 | 97.5699% | 97.8143% | 97.8542% |
| #2 Scaling-only λ=0.1 | 97.6080% | 97.8794% | 97.8841% |

## Q1 — #0 vs #1: deep/narrow head architecture

pooled Δ=-0.1308 pp, paired 95% CI [-0.3160, +0.0490] pp; fold directions +/−/= 1/4/0; changed predictions 322, correctness discordants 292; Macro-F1 Δ=-0.1323 pp, balanced-accuracy Δ=-0.0976 pp.

Interpretation must combine the effect, interval, fold direction, changed samples, Macro-F1 and balanced accuracy above; no fixed 0.1 pp or p<0.05 gate was imposed.

## Q2 — #1 vs #2: shortcut scaling only

pooled Δ=+0.0381 pp, paired 95% CI [-0.1471, +0.2234] pp; fold directions +/−/= 4/1/0; changed predictions 337, correctness discordants 301; Macro-F1 Δ=+0.0651 pp, balanced-accuracy Δ=+0.0299 pp.

#1 and #2 have the same tensor graph, module/parameter counts, mean aggregation, optimizer path and data protocol; the declared forward difference is only fixed shortcut λ=1.0 versus 0.1.

## Mandatory stop gate

**STOPPED_AFTER_#0_#1_#2.** No #3/#4/#5 and no new DFAG training was started. Human review is required before any continuation to MHSA, scaling+MHSA, or terminal identity experiments.

Detailed per-fold metrics are in `oof_metrics.csv`; paired effect, bootstrap CI, exact McNemar, changed/discordant counts and fold directions are in `paired_comparisons.csv`.
