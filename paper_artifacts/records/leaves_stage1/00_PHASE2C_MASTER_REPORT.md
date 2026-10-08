# Phase 2C - Inference-only / Provenance Diagnostic master report

## 1. EXECUTIVE SUMMARY

本阶段未执行任何训练、反向传播或参数更新；未运行 #3/#4/#5，未改论文或 Fig.1-Fig.4。P7 完整性检查全部通过。

直接测量显示：(1) Phase2A Ours-FT 97.5372% 与 Phase2B #0 97.7006% 虽使用相同数据与折分，但 36/59 个历史训练字段未恢复且 checkpoint 选择机制不同，不能直接归因；(2) Stage2 相对 Stage1-only 在三变体 pooled OOF 均下降，其中仅 #2 的 paired CI 不跨 0；(3) lambda=0.1 极大改变了层间表征相似性，但没有带来可确认的 accuracy 提升；(4) DFAG anchor/plastic 表征高度相似。

## 2. P0 - Cross-phase comparability

结论：**C. NOT DIRECTLY COMPARABLE**。Phase2A=0.975371873808097，Phase2B=0.977006483953577，差值=0.163461 pp。字段：SAME=19，DIFFERENT=4，UNRESOLVED=36。关键障碍是 Phase2A 训练 seed/loader、优化器、LR、增强、class weight、BN adaptation、AMP/GradScaler 等未被 checkpoint 序列化，且候选 checkpoint 与选择规则不同。these protocol differences prevent direct attribution.

## 3. P1 - Does Stage2 reduce OOF performance?

| Variant | Stage1-only Acc. | Stage2 Acc. | Delta pp | 95% paired CI pp | McNemar p | Fold direction |
|---|---:|---:|---:|---:|---:|---|
| #0 | 97.8096% | 97.7006% | -0.108974 | [-0.288781, 0.070833] | 0.257834 | +:1 / -:4 / =:0 |
| #1 | 97.7170% | 97.5699% | -0.147115 | [-0.321473, 0.027244] | 0.106008 | +:1 / -:4 / =:0 |
| #2 | 97.8042% | 97.6080% | -0.196153 | [-0.365063, -0.027244] | 0.024849 | +:1 / -:4 / =:0 |

直接测量：pooled OOF 中三者均为下降；#0/#1 的 CI 跨 0，#2 的 CI 不跨 0。该结果仍来自单一训练 seed=42 的五折模型，不能包装为独立多 seed 确认。

## 4. P2 - #0 versus #2

#0=97.7006%，#2=97.6080%，delta (#2-#0)=-0.092628 pp，95% CI=[-0.277884, 0.092628]，exact McNemar p=0.353212；prediction_changed=321，n10=157，n01=140。结论：没有可确认的 accuracy 提升。

## 5. P3 - Does lambda=0.1 change representation behavior?

Accuracy effect：#2-#1=+0.038141 pp，95% CI=[-0.147115, 0.223397]，p=0.729531，未确认性能提升。

Representation effect：十个 pair 的 sample-wise off-diagonal cosine（五折均值）#1=0.82436226，#2=0.11932517，差=-0.70503709；centered-linear CKA 平均 #1=0.89872488，#2=0.55837749，差=-0.34034739。

Adjacent normalized change（五折均值）：

| Transition | #1 | #2 |
|---|---:|---:|
| f1-f2 | 0.258137 | 0.778284 |
| f2-f3 | 0.292988 | 0.975128 |
| f3-f4 | 0.361302 | 1.015933 |
| f4-f5 | 0.581740 | 1.212326 |

直接测量支持：lambda=0.1 与更低层间相似度和更大相邻变化相关。推断边界：lower cosine/CKA 不等于 better representation。

## 6. P4 - Are anchor and plastic representations too similar?

Standalone DFAG（D_g=1536）pooled cosine：mean=0.95341117，std=0.02625865，median=0.96014639，p05=0.90227135，p95=0.97925165，min=0.70495366，max=0.99068356。gate sample mean：mean=0.50385104，std=0.00338717。高相似度与 gate 缺少强 sample-dependent routing pressure 的假设一致，但不证明 gate 弱动态的原因。

## 7. P5 - Can the historical MS-LFA motivation be recovered?

结论：**PROVENANCE_INCOMPLETE**；Figure1(b) 维持 **EXPLORATORY / PROVENANCE PARTIAL**。已恢复五条 model-index、Epoch-150、x0/f1...f5 的四位小数记录及均值；未恢复 exact fold、checkpoint、sample count/population、seed、query/head aggregation。它不能作为当前 #0 表征冗余的完整可复现实验证据。

## 8. SCIENTIFIC DECISION TABLE

| Question | Evidence for | Evidence against / unresolved | Current evidence state |
|---|---|---|---|
| A. Deep/narrow 是否值得继续 | #1 的 P3 层间相似度很高，说明机制问题真实可测 | #1-#0=-0.130769 pp，CI 跨 0；历史 motivation provenance 不完整 | 性能证据不支持优势；机制证据支持继续分析，但是否训练由人工决定 |
| B. Shortcut scaling 是否改变 representation | cosine 差 -0.7050，CKA 差 -0.3403，adjacent change 全部增大 | #2-#1 accuracy CI 跨 0 | 表征改变：有；性能改善：未确认 |
| C. 是否启动 MHSA #3/#4 | 表征冗余为机制探索提供理由 | 当前没有 MHSA 的 controlled performance evidence；单 seed | UNRESOLVED，不能自动启动 |
| D. Stage2 decline 是否 pooled OOF 复现 | 三变体点估计均下降；#2 CI 不跨 0 | #0/#1 CI 跨 0；非多 seed | pooled 复现，强度依变体而异 |
| E. gate 弱动态是否可能与 branch similarity 有关 | mean cos=0.9534，与假设一致 | 观察性诊断，不能证明因果 | CONSISTENT WITH，非因果确认 |
| F. Phase2A DFAG vs Phase2B baseline 可否跨 Phase effect subtraction | 数据/折分一致 | P0 有 4 DIFFERENT、36 UNRESOLVED | 不允许 |

## 9. RECOMMENDED NEXT HUMAN DECISION

1. 不要用 Phase2A 与 Phase2B 的点估计做模块 effect subtraction；如需跨 Phase 结论，先由人工决定是否建立完全统一且多 seed 的新协议。
2. 先人工审阅 #2 的 Stage2 decline，决定后续是否研究 Stage2 checkpoint/early-stopping/协议，而不是自动扩展训练。
3. 若未来决定运行 MHSA #3/#4，应预注册问题与评价标准，并保持 #1/#2 相同协议；Phase2C 本身不执行。
4. Figure1(b) 只能保留 exploratory/provenance-partial 限定，或由人工决定删除其定量动机表述。

## Integrity and scope

P7 status: PASS; checks=37; failures=0。15/15 Stage1 reproduction gates、10/10 P3 feature manifests、18,353/18,353 OOF coverage、NaN/Inf、ID/label/fold alignment、checkpoint hash、strict architecture load、logits argmax、deterministic repeat 均已核验。
