# Cars Index Optimization Audit

## Decision

`PASS` — the repeated WSL DrvFS wildcard lookup was replaced by a one-pass,
per-split filename index. This is a storage lookup optimization only; the
formal sample order, labels, folds, class map, sample IDs, relative image names,
and resolved image paths are unchanged.

## Interrupted run

The first formal attempt was stopped with user authorization before changing
executed source code. Its archived output was permanently deleted with user
authorization on 2026-08-20 and is not part of the replacement result set.

## Equivalence result

The WSL pytorch environment ran `scripts/verify_cars_index_optimization.py`:

- optimized Cars initialization: 1.0690658278763294 seconds;
- development records: 8,144;
- assignments: exact match;
- all five fold definitions: exact match;
- class map: exact match;
- path semantics: exact match;
- fold manifest SHA-256: `e2a18afd04c54105215b97cd02a2687b4ead9ac87c6dbe77a58b9508e02507df`;
- class map SHA-256: `2cf8d881310afbf88fb3785191403a54ca1d4fff90d187fb68391286db646dcc`.

## Executed code identities

- `common/runtime.py`: `f76e5747cb066907e24549f98709b4fcace26d777eed08e69d66af2bd03ee0e5`;
- `scripts/formal_single_job.py`: `e6588fe0334b6bcc60cd0070b4dae300a703e9ab3a4c921b0a96b0966a333a0a`;
- `scripts/formal_cal_job.py`: `c630e7742f2354f4ac4f3ae9386b08ce0bf2028dc6d2b4879eda36591e284841`;
- `scripts/formal_cal_job_v2.py`: `df657acdbc8f6465bb9980336914e2ede4d2207f89b3ba60dcd09214c97f9f64`;
- frozen `protocol_core.py`: `a4ec8e36e4850f6710d4b9854b315e77ff3e084ccd2bb28c3cac80c5906e6f42`;
- finalizer: `5732c880c5b58b442ef0c90046f51f5f230a763a125a4d773f513e3a7b134043`;
- verification script: `ae911e63779a88ffb3343b70789198a656c222e6c0dcc71bf04db4400e6929fc`.

Every replacement job manifest records the runtime, protocol-core, and runner
identities. Final aggregation fails closed if any job mixes loader versions.
