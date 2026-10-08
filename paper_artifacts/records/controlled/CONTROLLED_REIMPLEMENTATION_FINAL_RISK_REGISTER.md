# Controlled Reimplementation Final Risk Register

| ID | Severity | State | Evidence / control |
|---|---|---|---|
| R1 | Former blocker | CLOSED | CUB and Cars each passed NCCL 2-rank, rank-to-GPU binding, per-rank8/global16, 54 SyncBN modules, complete CAL training pathway, backward/optimizer and checkpoint roundtrip under WSL2. |
| R2 | Protocol-change risk | CONTROLLED | Single GPU, ordinary BN, DataParallel, smaller topology and different batch size remain forbidden; none was used. |
| R3 | Medium | CONTROLLED | MC pretrained augmentation remains a predeclared source-backed candidate for an originally NOT_SPECIFIED setting; no OOF selection. |
| R4 | Medium | CONTROLLED | L2-SP geometry/common-normalization adaptations remain manifest-locked. |
| R5 | Medium | CONTROLLED | Ours/Progressive/DFAG workspace semantics remain content-addressed by SHA because the workspace is not a Git worktree. |
| R6 | High | CONTROLLED | DFAG same-fold Stage-1 dependency and frozen-anchor assertions remain mandatory. |
| R7 | High | CONTROLLED | All smoke/preflight checkpoints were technical and transient; no accuracy was computed or interpreted. |
