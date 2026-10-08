# Final Progressive Head Complexity Audit

Architecture audit: **PASS**

- Backbone: Inception-ResNet-v2; pooled dimension 1536
- Projection: 1536 -> 256; H=256; N=5; shortcut lambda=0.7
- Mapping block: Linear -> BatchNorm1d -> SiLU -> Dropout(p=0.3)
- Aggregation: mean(f1,...,f5) -> Dropout(p=0.2) -> Linear(256,176)
- MHSA absent; terminal residual absent; DFAG absent; branch count=1
- Backbone parameters: 54,306,464
- Progressive head parameters: 770,224
- Total resident parameters: 55,076,688

## Ours-FT sanity

- Resident Params: 57,114,448 (57.114448 M)
- Head Params: 2,807,984 (2.807984 M)
- FLOPs: 26.314805824 G
- Latency: 14.131069 +/- 0.895673 ms
- Peak Mem: 247.878418 MiB
- Sanity status: **PASS**

## Progressive Head

- Resident Params: 55,076,688 (55.076688 M)
- Head Params: 770,224 (0.770224 M)
- FLOPs: 26.310728256 G
- Latency mean +/- sample SD: 14.250305 +/- 0.350744 ms
- Peak Mem: 239.268066 MiB

## Relative to Ours-FT (Progressive minus Ours-FT)

- Total Params: -2,037,760 (-3.568%)
- Head Params: -2,037,760 (-72.570%)
- FLOPs: -0.004078 G (-0.015%)
- Latency: +0.119236 ms (+0.844%)
- Peak Mem: -8.610352 MiB (-3.474%)

## Profiling integrity

Status: **PASS**

- Hardware: NVIDIA GeForce RTX 5070 Ti
- Input: 1 x 3 x 299 x 299, FP32, eval mode, torch.no_grad
- Latency: 10 warm-up and 30 measured forwards, synchronized before and after every forward
- FLOPs: THOP 0.1.1, reported as 2 x MACs to match the historical table
- Memory: absolute peak allocated memory from torch.cuda.max_memory_allocated
- No training, optimizer, backward, BN refresh, checkpoint modification, DFAG, MHSA, terminal residual, TTA, or ensemble was run.
