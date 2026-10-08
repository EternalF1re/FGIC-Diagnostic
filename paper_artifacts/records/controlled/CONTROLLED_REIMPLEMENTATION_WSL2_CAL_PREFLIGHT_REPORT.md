# Controlled Reimplementation Preflight Report

## Final outcome

`PASS / READY_FOR_FORMAL_RUN`

The historical Windows-only blocker is closed by the final WSL2 NCCL execution. No formal training was started.

## WSL2/Linux environment

- Distribution: Ubuntu 22.04.1 LTS under WSL2
- Python: 3.10.12
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- timm: 1.0.21
- CUDA runtime: 12.8
- NCCL: 2.28.9
- GPUs: 2 x NVIDIA GeForce RTX 5070 Ti
- `torch.distributed.is_available()`: true
- `torch.distributed.is_nccl_available()`: true
- Independent minimal CUDA tensor operation: GPU0 PASS, GPU1 PASS

## Identity gate

All identity checks PASS: CAL upstream commit and implementation SHA, CUB/Cars config SHA, pretrained artifact SHA, loaded initial backbone state SHA, CUB/Cars fold and class-map SHA, augmentation identity, and evaluation protocol identifier. Evidence: `preflight/wsl2_final/cal_protocol_identity.json`.

## Frozen CAL topology results

| Dataset | Ranks | Mapping | Batch | SyncBN | Forward / p-p_cf / center / crop-drop second forward | Backward / optimizer | Checkpoint roundtrip | Accuracy |
|---|---:|---|---|---|---|---|---|---|
| CUB | 2 | rank0-GPU0, rank1-GPU1 | 8/rank, global16 | PASS (54 modules) | PASS | PASS | PASS | not computed |
| Cars | 2 | rank0-GPU0, rank1-GPU1 | 8/rank, global16 | PASS (54 modules) | PASS | PASS | PASS | not computed |

Both jobs used NCCL and completed without deadlock, device mismatch, NaN, or Inf. Transient technical checkpoints were deleted after verified save/load. `training_authorized=false`; `formal_jobs_started=0`.
