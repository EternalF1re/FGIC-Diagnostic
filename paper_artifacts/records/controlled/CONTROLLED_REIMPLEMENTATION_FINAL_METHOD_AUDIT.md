# Controlled Reimplementation Final Method Audit

审计日期：2026-08-20。定位：controlled reimplementation under a common evaluation protocol，不是原论文数字 reproduction。

## 总判定

`READY_FOR_FORMAL_RUN`

所有 source/config/init/fold/class-map/augmentation/evaluation identity 检查、unit tests、12/12 single-GPU technical smoke，以及 CUB/Cars CAL 的 WSL2 NCCL 2×8 DDP/SyncBN final preflight 均已通过。

`CAL_DDP_SYNCBN_ENVIRONMENT = CLOSED`。

冻结协议没有修改：CAL 仍为 32 maps、完整 BAP/counterfactual/feature-center/crop-drop second forward/loss，primary logits 仍为 `p-p_cf`；rank0→GPU0，rank1→GPU1，每 rank batch8，global batch16，54 个 BatchNorm 模块成功转换为 SyncBatchNorm。

其他方法状态不变：L2-SP、MC-Loss、Ours-FT、Progressive Head、DFAG 均为 `READY_NOT_AUTHORIZED`；DFAG 仍依赖 same-dataset/same-fold Ours Stage-1 checkpoint。

本轮没有计算或报告 accuracy，没有访问 official-test 参与训练或选择，没有启动任何正式 fold job。

- `training_authorized=false`
- `formal_jobs_started=0`
- `formal_job_count=60`
