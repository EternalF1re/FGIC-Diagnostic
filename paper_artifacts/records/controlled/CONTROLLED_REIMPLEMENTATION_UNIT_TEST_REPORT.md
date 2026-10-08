# Controlled Reimplementation Unit Test Report

Overall: `PASS`; result SHA `ef850187356bdf9c73d2a21d436fcd98a73b619768fbee1028526d7557b190bf`.

| Suite | Status | Key facts |
|---|---|---|
| Common | PASS | CUB/Cars IDs, fold membership, no overlap, class maps, deterministic 299 eval, pretrained SHA/state, pooled OOF semantics |
| L2-SP | PASS | immutable w0; 53 inherited weights; SP value/gradient zero at w0; classifier L2 separate; WD0 |
| MC-Loss | PASS | exact totals/boundaries; CWA counts/reproducibility; golden L_dis=5.349156380; golden L_div=0.097818211; sign and inference tests |
| CAL | PASS | 32 maps, 65,536-D BAP, train/eval fake attention, causal logits, train-only center, second forward, no eval augmentation leakage |
| Ours/Progressive/DFAG | PASS | fixed .7/5 blocks, no obsolete modules, same-fold provenance, frozen anchor/trainable plastic, 2048-D gate |

Machine-readable details: `preflight/unit_test_results.json`.
