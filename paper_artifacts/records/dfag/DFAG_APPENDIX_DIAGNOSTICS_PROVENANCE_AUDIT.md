# DFAG Appendix Diagnostics Provenance Audit

Audit date: 2026-08-17  
Instruction SHA256: `77c31c130bed025463246511fcd524732012c1d5fa0136c5e7e0c766f6e70af0`  
Scope: final controlled standalone DFAG runs, training seeds 42, 43, and 44 only. This was a read-only provenance audit. No model was trained or rerun and no existing result was modified.

## 1. Gate SD definition

The reported gate SD values `0.0150 / 0.0039 / 0.0049` are the rounded values of `sample_mean_sd`:

- seed 42: `0.015047849376286006`
- seed 43: `0.003862978362472668`
- seed 44: `0.004916804396509503`

For each seed, the five held-out folds are first concatenated, sorted by `sample_id`, and checked for exactly 18,353 unique OOF samples. Let the resulting gate tensor be `G` with shape `[18353, 1536]`. For sample `i`, the scalar gate summary is

`m_i = mean_c G[i,c]`, over all 1,536 gate channels.

The reported SD is then the sample standard deviation of the 18,353 values `m_i`, using `ddof=1`. It is not the SD of all 28,190,208 atomic gate entries, not the mean channel-wise SD, and not an average of five fold-level SDs.

Code provenance:

- `scripts/finalize_dfag.py`, lines 73--87: concatenate folds, sort by sample ID, and validate OOF coverage.
- `scripts/finalize_dfag.py`, lines 149--167: `sample_mean = gate.mean(axis=1)` followed by the common `describe()` function.
- `scripts/finalize_dfag.py`, lines 30--35: `values.std(ddof=1)` and NumPy quantiles.

The distinction matters: the atomic gate SDs are `0.0866474388 / 0.0335529003 / 0.0531736312`, whereas the manuscript values are the across-sample SDs of the per-sample 1,536-channel means.

## 2. Gate distribution on the formal OOF samples

The following statistics all describe the same per-sample scalar distribution `{m_i}` used by the reported gate SD. `IQR = p75 - p25`.

| Seed | OOF n | Mean | Sample SD | Median | IQR | Min | Max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 18,353 | 0.5185675949568093 | 0.015047849376286006 | 0.5100218438310549 | 0.030019155751991433 | 0.5013503559360591 | 0.5579730310419109 |
| 43 | 18,353 | 0.5065473585556084 | 0.003862978362472668 | 0.5053025821107440 | 0.006165964606528518 | 0.5008813057793304 | 0.5163032331814369 |
| 44 | 18,353 | 0.5099781755492846 | 0.004916804396509503 | 0.5113185798982158 | 0.009633896232116967 | 0.5027873411891051 | 0.5228388878555658 |

`gate mean approximately 0.519 / 0.507 / 0.510` = **VERIFIED**. These are the rounded formal OOF per-sample gate means for seeds 42/43/44, respectively. The mean of the per-sample means equals the mean of all atomic gate entries for each rectangular OOF tensor.

Primary provenance:

- `gate/gate_summary_per_seed.csv`, SHA256 `2c057e7ec40ea9a71a9bff16bd6de7a54357ae2d71aa865dad85337e21aa6b85`
- `gate/gate_sample_summary.csv`, SHA256 `2b499d1ea91f607188d404148d32490ce0462fbd5600a5e385a2cfeb18097249`
- `gate/seed42_gate_oof.npz`, SHA256 `c705b33068491212d671bf870f088741e58e67c6d171ae5a9db5d7ff4a3cf28c`
- `gate/seed43_gate_oof.npz`, SHA256 `a685d11fa75e9bbd56a092c7572053b63aae05268e24dffbbe5432a8cf0a12f8`
- `gate/seed44_gate_oof.npz`, SHA256 `015a7f706fdf56cf74b574c428beaa043f050efe8e6f93f16307a1a2aff78982`

## 3. Sample/channel/interaction variance decomposition

**NOT AVAILABLE.**

No current final-controlled-run file, script output, manifest field, or audit result contains a sample-effect, channel-effect, and sample-by-channel interaction variance decomposition. The existing files provide:

- one `mean_gate` and one RMS diagnostic per sample in `gate_sample_summary.csv`;
- one mean and across-sample SD per channel in `gate_channel_summary.csv`;
- the complete raw OOF gate matrix for each seed in `seed{42,43,44}_gate_oof.npz`.

These marginal summaries and raw matrices are not an already-computed variance decomposition: they contain no defined sums-of-squares convention, no three component estimates, and no percentage attribution. Therefore the earlier Phase2A values `1.20% / 19.51% / 79.30%` are not reused or reported as current controlled-run results.

Additional marginal-summary provenance:

- `gate/gate_channel_summary.csv`, SHA256 `4c466b078e2478d77f1199e83398b9ef436ad90c8c5303d8a7b9dd4461078659`

## 4. Anchor--plastic L2 distance

For every formal OOF sample `i`, the existing finalizer computes

`d_i = ||f_anc[i] - f_spec[i]||_2`

over the complete anchor and plastic representation vectors. It computes one L2 distance per sample first, then applies the same OOF-level `describe()` function across all 18,353 distances for each seed. The SD is a sample SD with `ddof=1`. The five folds are not summarized separately or averaged; they are concatenated and sample-ID sorted before calculation.

| Seed | OOF n | Mean L2 | Sample SD | Median L2 |
|---:|---:|---:|---:|---:|
| 42 | 18,353 | 3.1711424779589628 | 0.9307571165802503 | 2.995575501401104 |
| 43 | 18,353 | 4.2334565981423120 | 1.1153099770509325 | 4.292613151683982 |
| 44 | 18,353 | 3.7238977740584070 | 1.4264350226255114 | 3.3879866515490185 |

Provenance:

- `scripts/finalize_dfag.py`, lines 174--178: sample-wise cosine and L2 definitions.
- `representation/anchor_plastic_similarity_summary.csv`, SHA256 `67edefe74798722cd9862297d04b0969d27954b481480194ae1604913a47f9af`
- `scripts/finalize_dfag.py`, SHA256 `9ecf1c6e9be4cb7248f1ca3d5dd63713d63c59a1f0b5487d617af14b910aea89`
- `manifests/postflight_audit.json`, SHA256 `a1f8059678040cfc098fea8a5871432d29212f0c7924d0e41d7013251077c879`

## Readiness

`GATE_DISTRIBUTION_READY = YES`  
`VARIANCE_DECOMPOSITION_READY = NO`  
`ANCHOR_PLASTIC_L2_READY = YES`

STOP
