# P3 controlled #1 versus #2 representation diagnostic

Status: COMPLETE. This is a diagnostic of the Phase2B controlled mean-aggregation heads. They have no MHSA and are not the full historical MS-LFA head.

## Directly measured

- Each f1..f5 is the true post-shortcut tensor with shape [N_fold, 256].
- Mean sample-wise off-diagonal cosine (fold means): #1 0.82436226; #2 0.11932517; #2-#1 -0.70503709.
- Mean of ten fold-wise centered-linear CKA pairs: #1 0.89872488; #2 0.55837749; #2-#1 -0.34034739.

## Interpretation boundary

The fixed lambda=0.1 setting is associated with the measured inter-stage similarity changes under this controlled screen. Lower CKA/cosine is not equivalent to better representations, and the five folds are not independent training-seed significance tests.
