# Round1 representation diagnostic

Status: COMPLETE.

Each f1..f5 array is the true post-shortcut tensor used by the frozen mean-aggregation head, shape [N_fold, 256]. No MHSA and no terminal residual are present.

| lambda | stage | pooled mean off-diagonal cosine | mean centered-linear CKA |
|---:|---:|---:|---:|
| 0.1 | 1 | 0.16073483 | 0.57777664 |
| 0.1 | 2 | 0.11932485 | 0.55837749 |
| 0.7 | 1 | 0.65052828 | 0.85591669 |
| 0.7 | 2 | 0.63544866 | 0.83996394 |
| 0.9 | 1 | 0.79600966 | 0.91262004 |
| 0.9 | 2 | 0.78777635 | 0.88614578 |
| 1 | 1 | 0.82310906 | 0.91882862 |
| 1 | 2 | 0.82436200 | 0.89872488 |

Centered-linear CKA is computed separately within each fold/model in float64, then summarized across folds. Cosine is sample-wise and pooled only after fold-local extraction.

The lambda^5 table is an analytic coefficient illustration, not a measured causal contribution. Lower cosine/CKA denotes changed geometry, not better representations. The five folds are not five independent training seeds.
