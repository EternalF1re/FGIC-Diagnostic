# Controlled Reimplementation Frozen Protocol

Status: `READY_FOR_FORMAL_RUN`; `training_authorized=false`; `formal_jobs_started=0`.

## Common evaluation lock

- CUB development 5,994; Cars development 8,144. Official test is excluded from training, tuning, checkpoint selection, and OOF.
- SKF5, shuffle=true, random_state=42. Training seed is exactly 42 in every fold, never seed+fold.
- Input 299x299. Common pretrained artifact `773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb` and loaded backbone state `45179af731d5ea5470ba5906fadf0b55b884338ca64d2b7f53f46096665ec6c6`.
- Deterministic original-view held-out inference; strict max accuracy, exact tie keeps earlier checkpoint.
- Complete pooled OOF with each development sample exactly once. Metrics: Accuracy, Macro-F1, Balanced Accuracy. No TTA or cross-fold ensemble.

## Method locks

- L2-SP: `CE + .1/2||w_inherited-w0||^2 + .01/2||w_classifier||^2`; SGD .01, momentum .9, batch64, 9000 optimizer iterations, x.1 at 6000, WD0, Xavier, no smoothing.
- MC-Loss: exact CUB/Cars contiguous Table-II groups covering 2048; `CE+.005*(L_dis-10L_div)`; SGD, batch32, momentum.9, backbone/FC LR 1e-4/1e-2, 300 epochs, steps150/225, WD5e-4, no smoothing.
- CAL: 32-map full method, causal primary, 160 epochs, SGD1e-3 momentum.9 WD1e-5, fractional schedule, beta.05, crop/drop and second forward; target topology 2x8 DDP+SyncBN.
- Ours/Progressive: current two-stage CUB/Cars controlled training semantics, batch32; Progressive is fixed lambda .7 five-block final architecture.
- DFAG: no independent Stage-1; exact same-dataset/fold selected Ours checkpoint initializes anchor/plastic; anchor frozen; feature-wise gate; Stage2 current semantics.

Augmentation is locked separately in `CONTROLLED_REIMPLEMENTATION_AUGMENTATION_MANIFEST.md`. Per-job exact JSON configs and SHA values are under `configs/` and in the 60-job plan.

## Final execution-readiness closure

CAL CUB and Cars passed the frozen 2x8 NCCL/DDP/SyncBN preflight under WSL2. This closes only the environment blocker; no protocol field or formal job was changed. `training_authorized=false`; `formal_jobs_started=0`.
