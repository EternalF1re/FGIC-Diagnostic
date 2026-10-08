"""Phase 2C P6/P7 final integrity cross-check and master report generation."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parents[1]
VALIDATION = Path(__file__).resolve().parents[2]
SCRIPTS = BASE / "scripts"
EXPECTED_INSTRUCTION_HASH = "E1EC0AD1AE186BF8E6AEE0599F628F4A203A45F4270DD3D84C504ED5077DF337"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    checks: list[dict[str, object]] = []
    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            failures.append(f"{name}: {detail}")

    instruction = next((BASE / "manifests").glob("instruction_*.txt"))
    check("instruction_sha256", sha256(instruction).upper() == EXPECTED_INSTRUCTION_HASH, str(instruction))

    # Static safety: no executable training/update calls in any newly created script.
    forbidden_hits: list[str] = []
    for script in sorted(SCRIPTS.glob("*.py")):
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "backward": forbidden_hits.append(f"{script.name}:{node.lineno}:backward")
                if node.func.attr == "step" and isinstance(node.func.value, ast.Name) and node.func.value.id in {"optimizer", "scaler"}:
                    forbidden_hits.append(f"{script.name}:{node.lineno}:{node.func.value.id}.step")
    check("no_training_or_parameter_updates", not forbidden_hits, "; ".join(forbidden_hits) or "no backward/optimizer.step/scaler.step calls")

    # P0
    p0 = json.loads((BASE / "p0_phase2a_vs_phase2b" / "provenance_sources.json").read_text(encoding="utf-8"))
    p0_table = rows(BASE / "p0_phase2a_vs_phase2b" / "protocol_comparison.csv")
    check("p0_59_fields", len(p0_table) == 59 and [int(r["index"]) for r in p0_table] == list(range(1,60)), f"rows={len(p0_table)}")
    check("p0_oof_alignment", p0["sample_count"] == 18353 and p0["sample_ids_labels_aligned"] and p0["exact_fold_membership_aligned"], "Phase2A/Phase2B IDs, labels, folds")

    # P1: all fold gates and pooled arrays.
    p1_repro = rows(BASE / "p1_stage1_only_oof" / "stage1_checkpoint_reproduction.csv")
    check("p1_15_reproduction_gates", len(p1_repro) == 15 and all(r["status"] == "PASS" for r in p1_repro), f"pass={sum(r['status']=='PASS' for r in p1_repro)}/15")
    p1_labels_ref = None
    for variant in ("#0", "#1", "#2"):
        path = BASE / "p1_stage1_only_oof" / f"stage1_oof_predictions_{variant.replace('#','variant_')}.npz"
        with np.load(path) as z:
            ids, labels, logits, pred, folds = z["sample_id"], z["label"], z["logits"], z["prediction"], z["fold"]
            ok = (len(ids)==18353 and np.array_equal(ids,np.arange(18353)) and len(np.unique(ids))==18353
                  and np.isfinite(logits).all() and np.array_equal(logits.argmax(1),pred)
                  and set(np.unique(folds).tolist())=={0,1,2,3,4})
            check(f"p1_{variant}_oof_integrity", ok, f"n={len(ids)}, logits={logits.shape}")
            check(f"p1_{variant}_stage2_alignment", np.array_equal(ids,np.load(VALIDATION/"phase2b_screen"/variant/"oof_sample_ids.npy")) and np.array_equal(labels,np.load(VALIDATION/"phase2b_screen"/variant/"oof_labels.npy")), "exact IDs/labels")
            if p1_labels_ref is None: p1_labels_ref = labels.copy()
            else: check(f"p1_{variant}_cross_variant_labels", np.array_equal(p1_labels_ref,labels), "exact")
    check("p1_deterministic_spotchecks", all(r["deterministic_repeat_exact"] == "True" and float(r["deterministic_repeat_max_abs_diff"]) == 0 for r in p1_repro), "15/15 bitwise repeated first batch")

    # P2
    p2 = rows(BASE / "p2_direct_paired_stats" / "#0_vs_#2_paired.csv")[0]
    check("p2_expected_accuracies", float(p2["accuracy_a"])==0.977006483953577 and float(p2["accuracy_b"])==0.9760802048711382, "exact pooled values")

    # P3: 10 manifests and extracted arrays.
    p3_manifests = list((BASE / "p3_mslfa_representation" / "fold_features").glob("variant_*/fold_*/feature_manifest.json"))
    check("p3_10_feature_manifests", len(p3_manifests)==10, f"count={len(p3_manifests)}")
    check("p3_source_prediction_and_repeat", all((lambda m: m["status"]=="PASS" and m["source_predictions_exact"] and m["forward_equivalence_exact"] and m["deterministic_repeat_exact"] and m["source_logits_max_abs_diff"]<=1e-6)(json.loads(p.read_text(encoding="utf-8"))) for p in p3_manifests), "all folds")
    for variant in ("#1", "#2"):
        seen = []
        for fold in range(5):
            path = BASE/"p3_mslfa_representation"/"fold_features"/variant.replace("#","variant_")/f"fold_{fold}"/"post_shortcut_stages.npz"
            with np.load(path) as z:
                features = [z[f"f{i}"] for i in range(1,6)]
                ok = all(x.shape==(len(z["sample_id"]),256) and np.isfinite(x).all() for x in features)
                check(f"p3_{variant}_fold{fold}_shape_finite", ok, f"n={len(z['sample_id'])}, D=256")
                seen.append(z["sample_id"])
        all_ids=np.concatenate(seen)
        check(f"p3_{variant}_oof_coverage", len(all_ids)==18353 and len(np.unique(all_ids))==18353 and set(all_ids.tolist())==set(range(18353)), "18,353 unique IDs")
    for csv_name in ("cosine_pairwise.csv","cosine_fold_summary.csv","cka_foldwise.csv","cka_summary.csv","adjacent_change.csv"):
        content = rows(BASE/"p3_mslfa_representation"/csv_name)
        numeric_bad = any(str(value).lower() in {"nan","inf","-inf"} for r in content for value in r.values())
        check(f"p3_{csv_name}_finite", not numeric_bad and len(content)>0, f"rows={len(content)}")

    # P4/P5
    p4 = json.loads((BASE/"p4_dfag_anchor_similarity"/"provenance.json").read_text(encoding="utf-8"))
    check("p4_oof_integrity", p4["integrity"]=="PASS" and p4["sample_count"]==18353 and p4["unique_sample_count"]==18353 and p4["D_g"]==1536, "standalone DFAG D_g=1536")
    p5 = json.loads((BASE/"p5_historical_motivation_provenance"/"figure1b_recovery.json").read_text(encoding="utf-8"))
    check("p5_recovery_classification", p5["classification"]=="PROVENANCE_INCOMPLETE" and p5["recovered"]["record_count"]==5, p5["status"])

    # Required diagnostic plots.
    plots = list((BASE/"p3_mslfa_representation").glob("diagnostic_*.png"))+list((BASE/"p3_mslfa_representation").glob("diagnostic_*.pdf"))+list((BASE/"p4_dfag_anchor_similarity").glob("similarity_distribution.*"))
    check("diagnostic_plots_nonempty", len(plots)>=10 and all(p.stat().st_size>0 for p in plots), f"files={len(plots)}")

    status = "PASS" if not failures else "FAIL"
    integrity = {"phase":"Phase2C P7","status":status,"training_performed":False,"checks":checks,"failures":failures}
    (BASE/"manifests"/"integrity_report.json").write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (BASE/"INTEGRITY_FAILURES.md").write_text("# Integrity failures\n\nStatus: "+("NONE" if not failures else "FAIL\n\n"+"\n".join(f"- {x}" for x in failures))+"\n",encoding="utf-8")
    (BASE/"OPEN_QUESTIONS.md").write_text("""# Open questions requiring human review

- Phase2A Ours-FT historical training configuration is not fully serialized; 36/59 P0 fields remain unresolved.
- Figure1(b) lacks exact checkpoint, sample count/population, fold identity, and query/head aggregation provenance.
- The five controlled folds use one training seed (42), so no finding is an independent multi-seed confirmation.
- Whether to run MHSA variants #3/#4/#5 remains a human decision; Phase2C did not start them.
- P4 similarity is consistent with limited gate routing pressure but does not establish its cause.
""",encoding="utf-8")
    (BASE/"manifests"/"instruction_baseline.json").write_text(json.dumps({
        "source_path":r"<LOCAL_PATH>",
        "frozen_copy":str(instruction),"bytes":instruction.stat().st_size,"sha256":sha256(instruction).upper(),
        "phase":"Phase 2C Inference-only / Provenance Diagnostic","training_allowed":False,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if failures:
        raise RuntimeError("P7 INTEGRITY FAILURE; see INTEGRITY_FAILURES.md")

    # Read measured results for the master report.
    p1_metrics = {r["variant"]:r for r in rows(BASE/"p1_stage1_only_oof"/"stage1_oof_metrics.csv") if r["scope"]=="pooled_oof"}
    p1_pairs = {r["variant"]:r for r in rows(BASE/"p1_stage1_only_oof"/"stage1_vs_stage2_paired.csv")}
    p1_folds = rows(BASE/"p1_stage1_only_oof"/"stage1_vs_stage2_per_fold.csv")
    screen_pairs = rows(VALIDATION/"phase2b_screen"/"paired_comparisons.csv")
    pair01 = next(r for r in screen_pairs if r["comparison"]=="#0_vs_#1" and r["scope"]=="POOLED_OOF")
    pair12 = next(r for r in screen_pairs if r["comparison"]=="#1_vs_#2" and r["scope"]=="POOLED_OOF")
    cos_rows = rows(BASE/"p3_mslfa_representation"/"cosine_fold_summary.csv")
    cos_mean = {v:float(np.mean([float(r["mean"]) for r in cos_rows if r["variant"]==v and r["metric"]=="sample_mean_offdiagonal_cosine"])) for v in ("#1","#2")}
    cka_rows = rows(BASE/"p3_mslfa_representation"/"cka_foldwise.csv")
    cka_mean = {v:float(np.mean([float(r[f"cka_{v}"]) for r in cka_rows])) for v in ("#1","#2")}
    adj_rows = rows(BASE/"p3_mslfa_representation"/"adjacent_change.csv")
    transitions=("f1-f2","f2-f3","f3-f4","f4-f5")
    adj_mean={v:{t:float(np.mean([float(r["mean"]) for r in adj_rows if r["variant"]==v and r["transition"]==t and r["metric"]=="normalized_change"])) for t in transitions} for v in ("#1","#2")}
    p4_rows=rows(BASE/"p4_dfag_anchor_similarity"/"fold_summary.csv")
    p4_cos=next(r for r in p4_rows if r["scope"]=="pooled_oof" and r["metric"]=="cosine_similarity")
    gate_rows=rows(BASE/"p4_dfag_anchor_similarity"/"feature_norm_summary.csv")
    gate=next(r for r in gate_rows if r["scope"]=="pooled_oof" and r["metric"]=="gate_mean")
    def directions(v: str) -> str:
        ds=[r["direction"] for r in p1_folds if r["variant"]==v]
        return f"+:{ds.count('+')} / -:{ds.count('-')} / =:{ds.count('=')}"
    lines=[
        "# Phase 2C - Inference-only / Provenance Diagnostic master report","",
        "## 1. EXECUTIVE SUMMARY","",
        "本阶段未执行任何训练、反向传播或参数更新；未运行 #3/#4/#5，未改论文或 Fig.1-Fig.4。P7 完整性检查全部通过。", "",
        f"直接测量显示：(1) Phase2A Ours-FT {p0['phase2a_accuracy']*100:.4f}% 与 Phase2B #0 {p0['phase2b_accuracy']*100:.4f}% 虽使用相同数据与折分，但 36/59 个历史训练字段未恢复且 checkpoint 选择机制不同，不能直接归因；(2) Stage2 相对 Stage1-only 在三变体 pooled OOF 均下降，其中仅 #2 的 paired CI 不跨 0；(3) lambda=0.1 极大改变了层间表征相似性，但没有带来可确认的 accuracy 提升；(4) DFAG anchor/plastic 表征高度相似。", "",
        "## 2. P0 - Cross-phase comparability","",
        f"结论：**C. NOT DIRECTLY COMPARABLE**。Phase2A={p0['phase2a_accuracy']:.15f}，Phase2B={p0['phase2b_accuracy']:.15f}，差值={p0['difference_phase2b_minus_phase2a_pp']:.6f} pp。字段：SAME={p0['status_counts']['SAME']}，DIFFERENT={p0['status_counts']['DIFFERENT']}，UNRESOLVED={p0['status_counts']['UNRESOLVED']}。关键障碍是 Phase2A 训练 seed/loader、优化器、LR、增强、class weight、BN adaptation、AMP/GradScaler 等未被 checkpoint 序列化，且候选 checkpoint 与选择规则不同。these protocol differences prevent direct attribution.", "",
        "## 3. P1 - Does Stage2 reduce OOF performance?","",
        "| Variant | Stage1-only Acc. | Stage2 Acc. | Delta pp | 95% paired CI pp | McNemar p | Fold direction |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for v in ("#0","#1","#2"):
        r=p1_pairs[v]; lines.append(f"| {v} | {float(r['stage1_accuracy'])*100:.4f}% | {float(r['stage2_accuracy'])*100:.4f}% | {float(r['delta_stage2_minus_stage1_pp']):+.6f} | [{float(r['bootstrap_ci95_low_pp']):.6f}, {float(r['bootstrap_ci95_high_pp']):.6f}] | {float(r['mcnemar_exact_two_sided_p']):.6g} | {directions(v)} |")
    lines += ["", "直接测量：pooled OOF 中三者均为下降；#0/#1 的 CI 跨 0，#2 的 CI 不跨 0。该结果仍来自单一训练 seed=42 的五折模型，不能包装为独立多 seed 确认。", "",
              "## 4. P2 - #0 versus #2","",
              f"#0={float(p2['accuracy_a'])*100:.4f}%，#2={float(p2['accuracy_b'])*100:.4f}%，delta (#2-#0)={float(p2['delta_b_minus_a_pp']):+.6f} pp，95% CI=[{float(p2['bootstrap_ci95_low_pp']):.6f}, {float(p2['bootstrap_ci95_high_pp']):.6f}]，exact McNemar p={float(p2['mcnemar_exact_two_sided_p']):.6g}；prediction_changed={p2['prediction_changed_n']}，n10={p2['n10_a_correct_b_wrong']}，n01={p2['n01_a_wrong_b_correct']}。结论：没有可确认的 accuracy 提升。", "",
              "## 5. P3 - Does lambda=0.1 change representation behavior?","",
              f"Accuracy effect：#2-#1={float(pair12['delta_b_minus_a_pp']):+.6f} pp，95% CI=[{float(pair12['paired_bootstrap_95ci_low_pp']):.6f}, {float(pair12['paired_bootstrap_95ci_high_pp']):.6f}]，p={float(pair12['mcnemar_exact_p']):.6g}，未确认性能提升。", "",
              f"Representation effect：十个 pair 的 sample-wise off-diagonal cosine（五折均值）#1={cos_mean['#1']:.8f}，#2={cos_mean['#2']:.8f}，差={cos_mean['#2']-cos_mean['#1']:.8f}；centered-linear CKA 平均 #1={cka_mean['#1']:.8f}，#2={cka_mean['#2']:.8f}，差={cka_mean['#2']-cka_mean['#1']:.8f}。", "",
              "Adjacent normalized change（五折均值）：", "",
              "| Transition | #1 | #2 |", "|---|---:|---:|"]
    for t in transitions: lines.append(f"| {t} | {adj_mean['#1'][t]:.6f} | {adj_mean['#2'][t]:.6f} |")
    lines += ["", "直接测量支持：lambda=0.1 与更低层间相似度和更大相邻变化相关。推断边界：lower cosine/CKA 不等于 better representation。", "",
              "## 6. P4 - Are anchor and plastic representations too similar?","",
              f"Standalone DFAG（D_g=1536）pooled cosine：mean={float(p4_cos['mean']):.8f}，std={float(p4_cos['std']):.8f}，median={float(p4_cos['median']):.8f}，p05={float(p4_cos['p05']):.8f}，p95={float(p4_cos['p95']):.8f}，min={float(p4_cos['min']):.8f}，max={float(p4_cos['max']):.8f}。gate sample mean：mean={float(gate['mean']):.8f}，std={float(gate['std']):.8f}。高相似度与 gate 缺少强 sample-dependent routing pressure 的假设一致，但不证明 gate 弱动态的原因。", "",
              "## 7. P5 - Can the historical MS-LFA motivation be recovered?","",
              "结论：**PROVENANCE_INCOMPLETE**；Figure1(b) 维持 **EXPLORATORY / PROVENANCE PARTIAL**。已恢复五条 model-index、Epoch-150、x0/f1...f5 的四位小数记录及均值；未恢复 exact fold、checkpoint、sample count/population、seed、query/head aggregation。它不能作为当前 #0 表征冗余的完整可复现实验证据。", "",
              "## 8. SCIENTIFIC DECISION TABLE","",
              "| Question | Evidence for | Evidence against / unresolved | Current evidence state |", "|---|---|---|---|",
              f"| A. Deep/narrow 是否值得继续 | #1 的 P3 层间相似度很高，说明机制问题真实可测 | #1-#0={float(pair01['delta_b_minus_a_pp']):+.6f} pp，CI 跨 0；历史 motivation provenance 不完整 | 性能证据不支持优势；机制证据支持继续分析，但是否训练由人工决定 |",
              f"| B. Shortcut scaling 是否改变 representation | cosine 差 {cos_mean['#2']-cos_mean['#1']:.4f}，CKA 差 {cka_mean['#2']-cka_mean['#1']:.4f}，adjacent change 全部增大 | #2-#1 accuracy CI 跨 0 | 表征改变：有；性能改善：未确认 |",
              "| C. 是否启动 MHSA #3/#4 | 表征冗余为机制探索提供理由 | 当前没有 MHSA 的 controlled performance evidence；单 seed | UNRESOLVED，不能自动启动 |",
              "| D. Stage2 decline 是否 pooled OOF 复现 | 三变体点估计均下降；#2 CI 不跨 0 | #0/#1 CI 跨 0；非多 seed | pooled 复现，强度依变体而异 |",
              f"| E. gate 弱动态是否可能与 branch similarity 有关 | mean cos={float(p4_cos['mean']):.4f}，与假设一致 | 观察性诊断，不能证明因果 | CONSISTENT WITH，非因果确认 |",
              "| F. Phase2A DFAG vs Phase2B baseline 可否跨 Phase effect subtraction | 数据/折分一致 | P0 有 4 DIFFERENT、36 UNRESOLVED | 不允许 |", "",
              "## 9. RECOMMENDED NEXT HUMAN DECISION","",
              "1. 不要用 Phase2A 与 Phase2B 的点估计做模块 effect subtraction；如需跨 Phase 结论，先由人工决定是否建立完全统一且多 seed 的新协议。",
              "2. 先人工审阅 #2 的 Stage2 decline，决定后续是否研究 Stage2 checkpoint/early-stopping/协议，而不是自动扩展训练。",
              "3. 若未来决定运行 MHSA #3/#4，应预注册问题与评价标准，并保持 #1/#2 相同协议；Phase2C 本身不执行。",
              "4. Figure1(b) 只能保留 exploratory/provenance-partial 限定，或由人工决定删除其定量动机表述。", "",
              "## Integrity and scope", "",
              f"P7 status: {status}; checks={len(checks)}; failures=0。15/15 Stage1 reproduction gates、10/10 P3 feature manifests、18,353/18,353 OOF coverage、NaN/Inf、ID/label/fold alignment、checkpoint hash、strict architecture load、logits argmax、deterministic repeat 均已核验。",]
    (BASE/"00_PHASE2C_MASTER_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    # Inventory after all main outputs exist (self-inventory intentionally excluded).
    inventory=[]
    for path in sorted(p for p in BASE.rglob("*") if p.is_file() and p.name!="artifact_inventory.csv"):
        inventory.append({"relative_path":str(path.relative_to(BASE)),"bytes":path.stat().st_size,"sha256":sha256(path)})
    with (BASE/"manifests"/"artifact_inventory.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(inventory[0])); writer.writeheader(); writer.writerows(inventory)
    print(f"P6/P7 COMPLETE: {len(checks)} checks, {len(inventory)} inventoried files")


if __name__ == "__main__":
    main()
