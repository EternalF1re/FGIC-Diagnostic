# Phase2D Round1 实验结果审计与论文整理

## 1. 完整性结论

- 正式任务：20/20 COMPLETE，0 RUNNING，0 QUEUED，0 FAILED。
- 数据划分：5-fold StratifiedKFold，OOF 样本总数 18,353；每个 OOF 样本恰好出现一次。
- 训练记录：每个任务包含 Stage 1 的 150 epoch 和 Stage 2 的 60 epoch。
- 产物：20 份 manifest、20 份 Stage 1 checkpoint、20 份 Stage 2 checkpoint、8 份新 OOF。
- 训练前审计：44/44 PASS。
- 训练后审计：366/366 PASS，包括 40 个 checkpoint SHA-256、历史行数、验证数组形状、预测与 logits 一致性、OOF 覆盖和统计表行数。
- 正式统计：100,000 次 paired bootstrap；McNemar exact two-sided test；固定 seed 20260807。

结论：本轮结果完整、可复核，可以用于论文分析。

## 2. 新增实验的 pooled OOF 指标

以下数值均为百分比（%）。

| Run | Stage | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|
| Baseline seed43 | 1 | 97.9295 | 98.2001 | 98.2254 |
| Baseline seed43 | 2 | 97.7061 | 97.9860 | 98.0035 |
| Baseline seed44 | 1 | 97.8042 | 98.0907 | 98.1068 |
| Baseline seed44 | 2 | 97.7115 | 97.9981 | 98.0075 |
| lambda=0.7 seed42 | 1 | 97.7878 | 98.0648 | 98.0605 |
| lambda=0.7 seed42 | 2 | 97.6734 | 97.9750 | 97.9781 |
| lambda=0.9 seed42 | 1 | 97.7606 | 97.9995 | 98.0122 |
| lambda=0.9 seed42 | 2 | 97.6353 | 97.9054 | 97.9174 |

## 3. Stage 1 与 Stage 2 的直接比较

Delta 定义为 `Stage 2 - Stage 1`，单位为百分点（pp）。

| Run | Delta (pp) | 95% CI (pp) | McNemar p | 判断 |
|---|---:|---:|---:|---|
| Baseline seed43 | -0.2234 | [-0.3760, -0.0708] | 0.0047 | Stage 2 显著下降 |
| Baseline seed44 | -0.0926 | [-0.2506, 0.0654] | 0.2839 | 无显著差异 |
| lambda=0.7 seed42 | -0.1144 | [-0.2670, 0.0381] | 0.1623 | 无显著差异 |
| lambda=0.9 seed42 | -0.1253 | [-0.2779, 0.0272] | 0.1242 | 无显著差异 |

四个新增 run 的 Stage 2 accuracy 均低于 Stage 1。只有 baseline seed43 的置信区间不跨 0 且 p<0.05；其余三项不能认定存在显著变化。因此，本轮证据不支持“Stage 2 提升准确率”的表述。

## 4. Baseline 三 seed 稳定性

| Training seed | Stage 1 Accuracy (%) | Stage 2 Accuracy (%) | Delta (pp) | 95% CI (pp) | p |
|---:|---:|---:|---:|---:|---:|
| 42 | 97.8096 | 97.7006 | -0.1090 | [-0.2888, 0.0708] | 0.2578 |
| 43 | 97.9295 | 97.7061 | -0.2234 | [-0.3760, -0.0708] | 0.0047 |
| 44 | 97.8042 | 97.7115 | -0.0926 | [-0.2506, 0.0654] | 0.2839 |

- Stage 1：97.8478% ± 0.0708 pp（mean ± sample SD）。
- Stage 2：97.7061% ± 0.0054 pp。
- Stage 2 - Stage 1：-0.1417 ± 0.0713 pp。

Stage 2 在这三个 seed 上表现出更小的数值波动，但平均 accuracy、Macro-F1 和 Balanced Accuracy 均略低。论文中可以报告“观察到更低的跨 seed 方差”，不应据此声称 Stage 2 提升性能。

## 5. lambda=0.7 与 lambda=0.9

Delta 定义为 `lambda=0.7 - lambda=0.9`。

| Stage | Accuracy Delta (pp) | 95% CI (pp) | McNemar p | Macro-F1 Delta (pp) | Balanced Acc. Delta (pp) |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.0272 | [-0.1362, 0.1907] | 0.7959 | +0.0654 | +0.0483 |
| 2 | +0.0381 | [-0.1199, 0.1962] | 0.6892 | +0.0696 | +0.0607 |

lambda=0.7 在两个 stage 的三项指标上均数值略高于 lambda=0.9，但差异很小，accuracy 的置信区间均跨 0。因此可以把 lambda=0.7 作为数值上更优的候选，不能写成统计显著优于 lambda=0.9。

lambda=0.7 与既有 seed42 对照的 accuracy 比较如下：

| Stage | Reference | Delta: lambda=0.7 - reference (pp) | 95% CI (pp) | p |
|---:|---|---:|---:|---:|
| 1 | lambda=1.0 | +0.0708 | [-0.0926, 0.2290] | 0.4258 |
| 1 | lambda=0.1 | -0.0163 | [-0.1798, 0.1471] | 0.8962 |
| 1 | Baseline | -0.0218 | [-0.1908, 0.1526] | 0.8513 |
| 2 | lambda=1.0 | +0.1035 | [-0.0817, 0.2888] | 0.2963 |
| 2 | lambda=0.1 | +0.0654 | [-0.0926, 0.2234] | 0.4604 |
| 2 | Baseline | -0.0272 | [-0.2071, 0.1526] | 0.8121 |

所有比较均未达到显著水平。Stage 2 中 lambda=0.7 是已测试 progressive-mapping shortcut 强度里数值最好的候选，但没有证据证明它是统计意义上的最优值。

## 6. 参数量与性能权衡

| Architecture | Total Parameters | Head Parameters |
|---|---:|---:|
| Baseline | 57,114,448 | 2,807,984 |
| Deep-narrow progressive mapping | 55,076,688 | 770,224 |

- 总参数减少 2,037,760（约 3.57%）。
- Head 参数减少 2,037,760（约 72.57%）。
- lambda=0.7 相对 seed42 baseline：Stage 2 accuracy -0.0272 pp，Macro-F1 +0.0283 pp，Balanced Accuracy +0.0262 pp；差异均很小，其中 Macro-F1/Balanced Accuracy 这里只作描述性比较。

因此，当前证据更适合支持“在显著压缩分类头参数的同时保持与 baseline 相当的 OOF 性能”，不支持“显著超越 baseline”。

## 7. 论文表述边界

可以使用：

- lambda=0.7 在新增候选中数值优于 lambda=0.9，并在 Stage 2 获得最好的 progressive-mapping accuracy。
- lambda=0.7 与 baseline 的 OOF accuracy 差异很小且统计不显著，同时显著减少分类头参数。
- 三 seed 结果显示 Stage 2 的跨 seed 数值波动更小，但平均指标略低。

不应使用：

- lambda=0.7 显著优于 lambda=0.9、lambda=1.0、lambda=0.1 或 baseline。
- Stage 2 提升了分类性能。
- lambda=0.7 已被证明为全局最优 shortcut 系数。

## 8. 可复核文件

- `oof/round1_oof_metrics.csv`：全部新增 run 的 pooled/per-fold 指标。
- `statistics/stage1_vs_stage2_paired.csv`：Stage 1/2 pooled 配对检验。
- `statistics/baseline_multiseed_per_seed.csv`：baseline 三 seed 结果。
- `statistics/baseline_multiseed_summary.csv`：baseline mean ± SD。
- `statistics/lambda_direct_paired.csv`：lambda 候选与既有对照的直接比较。
- `statistics/lambda_0_9_vs_0_7_paired.csv`：lambda=0.7 与 0.9 的补充直接比较。
- `manifests/postflight_audit.json`：366 项训练后完整性审计及结果文件 SHA-256。
