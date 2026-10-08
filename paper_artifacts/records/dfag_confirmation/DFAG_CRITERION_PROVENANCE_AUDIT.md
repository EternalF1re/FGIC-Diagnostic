# DFAG Criterion Provenance Audit

Generated: 2026-08-17 15:07 +08:00  
Mode: read-only provenance audit; no training, inference, checkpoint mutation, criterion recomputation, or new significance test.  
Instruction SHA256: `4c0bf2508a17887eb537d9a9352388c311bc933a7f2e5f99814920da0e477c0b`

## Executive result

The apparent seed-count conflict is resolved. Appendix D / Task A contains the original endpoint-faithful alpha=0.5 comparators for seeds 42--44. The later seed45/46 confirmation run generated two additional endpoint-faithful alpha=0.5 OOF comparators, each with complete fold 0--4 coverage and 18,353 aligned OOF samples. `DFAG_FIVE_SEED_FINAL_AUDIT.md` used the combined seeds 42--46 table, not the three-seed Appendix D table alone.

Therefore Criterion B is scientifically supported as a true 5/5 positive point-delta criterion. No seed-count assumption, duplicated source, missing-seed default, stale-manifest substitution, reporting-label error, or criterion computation error was found.

## A. Endpoint-faithful alpha=0.5 comparators

| Seed | Exact seed-level OOF artifact | SHA256 | Folds | OOF n | alpha=0.5 acc. | DFAG acc. | DFAG-alpha0.5 (pp) | Existing paired-bootstrap 95% CI (pp) | Exact two-sided McNemar p | n10 / n01 | Disagreement |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 45 | `comparators/seed45_alpha_0_5.npz` | `2d012caed8481ef12c78f506d513ab0e61a522ff11893dbc956ad8532e02bfdf` | 0,1,2,3,4 | 18,353 | 0.9780417370457146 | 0.9785866070942081 | +0.05448700484934621 | [-0.07083310630414646, 0.18525581648776768] | 0.45648223496512585 | 78 / 68 | 151 (0.008227537732250859) |
| 46 | `comparators/seed46_alpha_0_5.npz` | `fbaa86e7ac1b062a8736cc93d30e42943a39070632582a7dcdc5547e314935aa` | 0,1,2,3,4 | 18,353 | 0.9776058410069198 | 0.9789680161281534 | +0.13621751212335997 | [0.0, 0.2724350242467172] | 0.056655996833771365 | 92 / 67 | 169 (0.00920830381953904) |

The seed-level comparator OOF files were written on 2026-08-16 12:17:22 +08:00. The five-seed comparison tables and final decision were generated on 2026-08-16 21:42:08 +08:00.

Protocol provenance is consistent with Task A:

- `inference_only = true`; `model_eval = true`.
- `optimizer = false`; `scheduler = false`; `backward = false`; all gradients remained `None`.
- `bn_refresh = false`; no BN reset or BN recalibration route is present.
- No TTA: the comparator uses the deterministic evaluation transform and a non-shuffled validation loader.
- alpha=0 directly copies the complete Stage2 state; alpha=1 directly copies the complete Stage1 state.
- alpha=0.5 applies `0.5*Stage1 + 0.5*Stage2` to floating state tensors and copies Stage2 non-floating buffers.
- All ten seed45/46 fold manifests are `COMPLETE` and pass endpoint fidelity against the existing Stage1/Stage2 fold predictions.

The fold-level artifact paths and hashes are recorded in `manifests/alpha05_comparator_ledger.csv`; its SHA256 is `6b3f67232e1232e4be6abfb08cc49658f6e6c8dd660f3d11d5006ded6340f1d0`.

## B. DFAG versus Stage2 five-seed paired evidence

Prediction-level evidence exists for all five seeds. For seeds 45/46, DFAG OOF artifacts are `oof/seed45_dfag_oof.npz` (SHA256 `4c58221fac1f75d4c44678d5056f54655fc57da6d2ba78cecfc30b9f6ac54b2d`) and `oof/seed46_dfag_oof.npz` (SHA256 `e28bda66a5865bf132e018844e185700c913f1301a457f9de27fdbfc074e86ba`). Stage2 predictions are assembled from `phase2d_round2a/baseline_seed{45,46}/fold_{0..4}/stage2_validation_{sample_ids,labels,predictions}.npy`. Finalization checks exact sample-ID, label, and fold alignment before paired statistics.

| Seed | OOF n | DFAG acc. | Stage2 acc. | Delta (pp) | Existing paired-bootstrap 95% CI (pp) | Exact two-sided McNemar p | n10 / n01 | Disagreement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 18,353 | 0.9783686590748106 | 0.9770064839535770 | +0.1362175121233599 | [-0.032692202909606, 0.3051272271563232] | existing | existing | existing |
| 43 | 18,353 | 0.9796218601863456 | 0.9770609709584264 | +0.2560889227919194 | [0.1035253092137525, 0.4086525363700757] | existing | existing | existing |
| 44 | 18,353 | 0.9789680161281534 | 0.9771154579632758 | +0.1852558164877615 | [0.0272435024246717, 0.3432681305508636] | existing | existing | existing |
| 45 | 18,353 | 0.9785866070942081 | 0.9772244319729745 | +0.13621751212335997 | [-0.02179480193973737, 0.2942298261864545] | 0.10781097222614858 | 124 / 99 | 241 (0.013131368168691766) |
| 46 | 18,353 | 0.9789680161281534 | 0.9763526398953849 | +0.2615376232768529 | [0.09807660872881818, 0.4304473383098131] | 0.0024467848432771946 | 145 / 97 | 256 (0.013948673241431918) |

Arithmetic-only five-seed summary: mean delta = `0.195063477360651` pp; sample SD = `0.0615727263619774` pp. No new bootstrap or McNemar calculation was run in this audit.

## C. Frozen criterion provenance

| Criterion | Reference comparator | Required seeds | Actually available | Actual source files | Result | Provenance status |
|---|---|---|---|---|---|---|
| A: Stage1 5/5 positive | Stage1 | 42--46 | 42--46 | old seeds: `phase2f_unified_standalone_dfag/statistics/dfag_vs_stage1.csv`; new seeds: `statistics/dfag_seed45_46_paired.csv`; combined: `dfag_vs_stage1_five_seed.csv` | PASS, 5 positive deltas | Complete |
| B: alpha=0.5 5/5 positive | endpoint-faithful alpha=0.5 | 42--46 | 42--46 | old seeds: `phase2f_unified_standalone_dfag/statistics/dfag_vs_alpha_0_5.csv`; new seeds: the two comparator OOF files above plus `statistics/dfag_seed45_46_paired.csv`; combined: `dfag_vs_alpha_0_5_five_seed.csv` | PASS, 5 positive deltas | Complete; true 5/5 |
| C: at least one primary has >=3/5 strictly positive CI lower bounds | Stage1 and alpha=0.5 | 42--46 | 42--46 | the two combined five-seed CSVs | FAIL; Stage1=0/5, alpha=0.5=0/5 with lower bound strictly >0 | Complete |
| D: other primary has no negative direction or fully negative CI | Stage1 and alpha=0.5 because C has no supporting primary | 42--46 | 42--46 | the two combined five-seed CSVs | PASS; zero negative deltas and zero CIs fully below zero | Complete |

Criterion computation is in `scripts/finalize_confirmation.py`: it requires exact seed coverage 42--46, computes B from `positive_seed_count == 5`, and refuses a coverage mismatch. Existing 42--44 rows are loaded from the old Phase2F statistics; seed45/46 rows are appended and sorted before the frozen decision is evaluated. The paired-statistics implementation is in `scripts/confirmation_common.py`.

## Required terminal fields

`ALPHA05_AVAILABLE_SEEDS = [42, 43, 44, 45, 46]`  
`ALPHA05_SEED45_EXISTS = YES`  
`ALPHA05_SEED46_EXISTS = YES`  
`CRITERION_B_5_OF_5_SUPPORTED = YES`  
`DFAG_STAGE2_ACCURACY_DELTA_SEEDS = 5`  
`DFAG_STAGE2_PAIRED_STATS_AVAILABLE_SEEDS = [42, 43, 44, 45, 46]`  
`DFAG_STAGE2_DELTA_MEAN_PP = 0.195063477360651`  
`DFAG_STAGE2_DELTA_SAMPLE_SD_PP = 0.0615727263619774`  
`FROZEN_FINAL_CLASSIFICATION_CHANGES = NO`

The frozen final classification remains `FAIL` because Criterion C fails. No old artifact was modified.

STOP
