# P5 historical MS-LFA motivation provenance

Overall status: **PROVENANCE_INCOMPLETE**. Figure1(b) remains **EXPLORATORY / PROVENANCE PARTIAL**.

## Directly recovered

- The archived text file contains five records labelled model 0..4, all at Epoch 150, with six rounded sources x0,f1..f5.
- Their arithmetic means are x0=0.27656, f1=0.16156, f2=0.10922, f3=0.08326, f4=0.08240, f5=0.28696.
- `figure1-gen.py` parses those five records and explicitly labels the panel exploratory.
- The current Phase2B #0 is a 1536->1024->1024->176 head, not a five-stage progressive head.

## Not recovered

Exact fold identity, checkpoint, dataset assertion, seed, sample count/population, query aggregation, attention-head aggregation, and the exact generating code version are absent. A nearby current helper averages attention weights over two dimensions, but it handles five sources and cannot be used to infer the missing six-source aggregation rule.

## Scientific use boundary

The old observation cannot legally support a current quantitative claim that the present #0 has redundant intermediate representations. It can only be mentioned, with explicit caveats, as a historical exploratory attention-allocation observation. The new P3 controlled representation measurements are separate current evidence, not recovered provenance for Figure1(b).
