# P2 verification

Status: PASS

- #0 pooled accuracy exactly reproduced: 0.977006483953577
- #2 pooled accuracy exactly reproduced: 0.976080204871138
- 18,353 unique samples; no duplicates or missing IDs.
- #0/#2 sample IDs, labels, and fold assignments are exactly aligned.
- Statistics were recomputed from existing Stage2 OOF predictions; no inference or training was run.
- Paired bootstrap: 100,000 replicates, seed 20260807.
- McNemar: exact two-sided binomial test on correctness-discordant pairs.

## Directly measured pooled result

- Delta (#2 - #0): -0.092628 pp.
- 95% paired-bootstrap CI: [-0.277884, 0.092628] pp.
- Exact McNemar p: 0.353212040006.
