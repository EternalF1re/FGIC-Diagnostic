"""Phase 2C P0: evidence-constrained 59-field cross-phase protocol audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


VALIDATION = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "p0_phase2a_vs_phase2b"
A_RUN = "phase2_controlled_validation/inference/ours-ft/fold_*/original/dynamic/run_manifest.json"
A_EVAL = "phase2_controlled_validation/unified_evaluator.py:179-206,306-309,455-458,537-541,576,795"
A_CKPT = "temp_weights_classifyleaves/Ours-FT/temp_weights_Classifyleaves + Ours-FT/weights_stage2_*"
A_LIMIT = "audit_phase01/03_batch_size_audit.csv; current script snapshots are not linked to the selected checkpoints"
B_CFG = "phase2_controlled_validation/phase2b_screen/configs/controlled_screen.json"
B_CODE = "phase2_controlled_validation/phase2b_screen/screen_core.py"
B_RUN = "phase2_controlled_validation/phase2b_screen/#0/fold_*/run_manifest.json"


def row(i: int, field: str, status: str, av: str, bv: str, source_a: str, source_b: str, note: str = "") -> dict[str, object]:
    return {"index": i, "field": field, "status": status, "phase2a_value": av, "phase2b_value": bv,
            "phase2a_source": source_a, "phase2b_source": source_b, "evidence_note": note}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Direct artifact check for the two exact OOF metrics and fold membership.
    a_parts = []
    fold_assignment_exact = True
    for fold in range(5):
        a_dir = VALIDATION / "inference" / "ours-ft" / f"fold_{fold}" / "original" / "dynamic"
        ids = np.load(a_dir / "dataset_indices.npy").astype(np.int64)
        labels = np.load(a_dir / "labels.npy").astype(np.int64)
        logits = np.load(a_dir / "logits.npy")
        a_parts.append((ids, labels, logits.argmax(1).astype(np.int64), fold))
        b_dir = VALIDATION / "phase2b_screen" / "#0" / f"fold_{fold}"
        fold_assignment_exact &= set(ids.tolist()) == set(np.load(b_dir / "validation_sample_ids.npy").astype(np.int64).tolist())
    ids = np.concatenate([p[0] for p in a_parts]); labels = np.concatenate([p[1] for p in a_parts]); preds = np.concatenate([p[2] for p in a_parts])
    order = np.argsort(ids); ids, labels, preds = ids[order], labels[order], preds[order]
    if not np.array_equal(ids, np.arange(18_353)) or len(np.unique(ids)) != 18_353:
        raise RuntimeError("INTEGRITY_FAILURE: Phase2A Ours-FT OOF coverage")
    a_accuracy = float(np.mean(preds == labels))
    b_ids = np.load(VALIDATION / "phase2b_screen" / "#0" / "oof_sample_ids.npy")
    b_labels = np.load(VALIDATION / "phase2b_screen" / "#0" / "oof_labels.npy")
    b_preds = np.load(VALIDATION / "phase2b_screen" / "#0" / "oof_predictions.npy")
    b_accuracy = float(np.mean(b_preds == b_labels))
    if not np.array_equal(ids, b_ids) or not np.array_equal(labels, b_labels) or not fold_assignment_exact:
        raise RuntimeError("INTEGRITY_FAILURE: Phase2A/Phase2B sample, label, or fold alignment")
    if round(a_accuracy * 100, 4) != 97.5372 or b_accuracy != 0.977006483953577:
        raise RuntimeError(f"INTEGRITY_FAILURE: baseline metrics {a_accuracy}/{b_accuracy}")

    U = "UNRESOLVED; checkpoint artifacts do not serialize this field and available script snapshots are not proven run sources"
    rows = [
        row(1,"dataset root","SAME","<LOCAL_PATH>","<LOCAL_PATH>",A_RUN,B_CFG+":dataset.root"),
        row(2,"train.csv","SAME",".../classify-leaves/train.csv",".../classify-leaves/train.csv",A_RUN,B_CFG+":dataset.train_csv"),
        row(3,"label mapping","SAME","first appearance order in train.csv","first appearance order in train.csv",A_RUN,B_CFG+":dataset.label_mapping"),
        row(4,"num classes","SAME","176","176",A_RUN,B_CFG+":dataset.num_classes"),
        row(5,"input resolution","SAME","299x299","299x299",A_RUN,B_CFG+":dataset.input_size"),
        row(6,"split method","SAME","StratifiedKFold","StratifiedKFold",A_RUN,B_CFG+":split.type"),
        row(7,"number of folds","SAME","5","5",A_RUN,B_CFG+":split.n_splits"),
        row(8,"shuffle","SAME","True","True",A_RUN,B_CFG+":split.shuffle"),
        row(9,"split_random_state","SAME","42","42",A_RUN,B_CFG+":split.split_random_state"),
        row(10,"exact fold assignments","SAME","artifact-verified exact held-out membership","artifact-verified exact held-out membership",A_RUN+":dataset_indices.npy",B_RUN+":validation_sample_ids.npy","Compared all five fold ID sets"),
        row(11,"training seed","UNRESOLVED",U,"42",A_LIMIT,B_CFG+":seed.training_seed"),
        row(12,"seed is +fold","UNRESOLVED",U,"False; same seed every fold",A_LIMIT,B_CFG+":seed.same_training_seed_every_fold/seed_plus_fold"),
        row(13,"DataLoader seed","UNRESOLVED",U,"controlled generator from seed 42",A_LIMIT,B_CFG+":seed.controlled_sources"),
        row(14,"worker seed","UNRESOLVED",U,"worker_init_fn seeded",A_LIMIT,B_CFG+":seed.controlled_sources"),
        row(15,"model backbone","SAME","timm inception_resnet_v2; GAP 1536-D","timm inception_resnet_v2; GAP 1536-D",A_EVAL,B_CFG+":architectures.#0.backbone"),
        row(16,"pretrained source","UNRESOLVED","checkpoint proves architecture, not initialization source","timm ImageNet pretrained",A_CKPT,B_CFG+":common_training_protocol.pretrained_backbone"),
        row(17,"baseline head architecture","SAME","1536-1024-BN-SiLU-Drop0.2-1024-BN-SiLU-Drop0.2-176","same",A_EVAL,B_CFG+":architectures.#0.head"),
        row(18,"feature extraction location","DIFFERENT","Phase2A Ours-FT run persists logits but no head feature; evaluator internal specific is 1536-D backbone GAP","saved 1024-D second head-block output after Dropout",A_RUN+":outputs; "+A_EVAL,B_CFG+":architectures.#0.feature_extraction_location"),
        row(19,"batch size","UNRESOLVED","training batch unresolved; checkpoint-only inference used 32","training 64",A_RUN+":runtime.batch_size; "+A_LIMIT,B_CFG+":common_training_protocol.batch_size","Inference batch is not evidence of historical training batch"),
        row(20,"num_workers","UNRESOLVED","training workers unresolved; checkpoint-only inference used 4","training 4",A_RUN+":runtime.num_workers; "+A_LIMIT,B_CFG+":common_training_protocol.num_workers"),
        row(21,"Stage1 epochs","UNRESOLVED",U,"150",A_LIMIT,B_CFG+":common_training_protocol.stage1_epochs"),
        row(22,"Stage1 optimizer","UNRESOLVED",U,"AdamW",A_LIMIT,B_CFG+":common_training_protocol.optimizer"),
        row(23,"backbone LR","UNRESOLVED",U,"1e-4",A_LIMIT,B_CFG+":common_training_protocol.stage1_backbone_lr"),
        row(24,"head LR","UNRESOLVED",U,"1e-3",A_LIMIT,B_CFG+":common_training_protocol.stage1_head_lr"),
        row(25,"weight decay","UNRESOLVED",U,"1e-3",A_LIMIT,B_CFG+":common_training_protocol.stage1_weight_decay"),
        row(26,"scheduler","UNRESOLVED",U,"linear warmup then cosine T_max=147 eta_min=1e-5",A_LIMIT,B_CFG+":common_training_protocol.stage1_scheduler"),
        row(27,"warmup","UNRESOLVED",U,"3 epochs",A_LIMIT,B_CFG+":common_training_protocol.stage1_warmup_epochs"),
        row(28,"label smoothing","UNRESOLVED",U,"0.05",A_LIMIT,B_CFG+":common_training_protocol.stage1_label_smoothing"),
        row(29,"Mixup","UNRESOLVED",U,"p=0.4 alpha=0.4",A_LIMIT,B_CFG+":common_training_protocol.stage1_batch_augmentation"),
        row(30,"CutMix","UNRESOLVED",U,"p=0.5 alpha=1.0",A_LIMIT,B_CFG+":common_training_protocol.stage1_batch_augmentation"),
        row(31,"augmentation attenuation schedule","UNRESOLVED",U,"alpha linearly decays after 70% epochs",A_LIMIT,B_CFG+":common_training_protocol.stage1_batch_augmentation"),
        row(32,"spatial augmentation","UNRESOLVED",U,"Resize, H/V flip, rotation, ColorJitter, affine, normalize",A_LIMIT,B_CFG+":common_training_protocol.spatial_augmentation_both_stages"),
        row(33,"class weighting","UNRESOLVED",U,"inverse fold-train frequency, clipped/normalized, minority x1.1",A_LIMIT,B_CFG+":common_training_protocol.loss_class_weight"),
        row(34,"class weight based on full train.csv","UNRESOLVED",U,"False",A_LIMIT,B_CFG+":common_training_protocol.loss_class_weight"),
        row(35,"class weight based on fold training subset","UNRESOLVED",U,"True",A_LIMIT,B_CFG+":common_training_protocol.loss_class_weight"),
        row(36,"Stage1 checkpoint candidate mechanism","DIFFERENT","multiple candidate-index Stage2 artifacts (_0/_1/_2); exact Stage1 links not serialized","single best Stage1 checkpoint feeds one Stage2 run",A_CKPT,B_RUN+":best_stage1_checkpoint"),
        row(37,"ever used top-3 candidate","DIFFERENT","three candidate indices are preserved per fold; historical script linkage remains partial","No",A_CKPT,B_CFG+":common_training_protocol.checkpoint_selection"),
        row(38,"current is single-best checkpoint","SAME","one highest-metric Stage2 checkpoint per fold used for Phase2A OOF","one best Stage2 checkpoint per fold used for Phase2B OOF",A_RUN+":checkpoint_selection_rule",B_RUN+":best_stage2_checkpoint"),
        row(39,"exact checkpoint selection rule","DIFFERENT","highest stored/filename val_acc among existing artifacts; lexical filename tie-break","max deterministic held-out accuracy; exact tie selects earlier epoch",A_RUN+":checkpoint_selection_rule",B_CFG+":common_training_protocol.checkpoint_selection"),
        row(40,"Stage2 epochs","UNRESOLVED",U,"60",A_LIMIT,B_CFG+":common_training_protocol.stage2_epochs"),
        row(41,"Stage2 optimizer","UNRESOLVED",U,"AdamW",A_LIMIT,B_CFG+":common_training_protocol.optimizer"),
        row(42,"Stage2 LR","UNRESOLVED",U,"5e-5 all parameters",A_LIMIT,B_CFG+":common_training_protocol.stage2_all_parameter_lr"),
        row(43,"Stage2 weight decay","UNRESOLVED",U,"5e-4",A_LIMIT,B_CFG+":common_training_protocol.stage2_weight_decay"),
        row(44,"Stage2 scheduler","UNRESOLVED",U,"CosineAnnealingLR T_max=60 eta_min=1e-6",A_LIMIT,B_CFG+":common_training_protocol.stage2_scheduler"),
        row(45,"Stage2 label smoothing","UNRESOLVED",U,"0.03",A_LIMIT,B_CFG+":common_training_protocol.stage2_label_smoothing"),
        row(46,"Stage2 augmentation","UNRESOLVED",U,"spatial augmentation; no batch Mixup/CutMix",A_LIMIT,B_CFG+":common_training_protocol.stage2_batch_augmentation"),
        row(47,"BN adaptation","UNRESOLVED",U,"Yes",A_LIMIT,B_CFG+":common_training_protocol.bn_adaptation"),
        row(48,"50-batch no-grad train-mode BN update","UNRESOLVED",U,"50 batches",A_LIMIT,B_CFG+":common_training_protocol.bn_adaptation_batches_before_stage2"),
        row(49,"BN adaptation before/after optimizer creation","UNRESOLVED",U,"before Stage2 optimizer creation",A_LIMIT,B_CFG+":common_training_protocol.bn_adaptation"),
        row(50,"gradient clipping","UNRESOLVED",U,"global norm 5.0",A_LIMIT,B_CFG+":common_training_protocol.gradient_clip_norm"),
        row(51,"gradient clipping semantics/historical bug","UNRESOLVED",U,"unscale before clipping in controlled runner",A_LIMIT,"phase2_controlled_validation/phase2b_screen/train_one.py"),
        row(52,"AMP","UNRESOLVED",U,"True, float16",A_LIMIT,B_CFG+":common_training_protocol.amp/amp_dtype"),
        row(53,"GradScaler","UNRESOLVED",U,"enabled, init_scale=4096",A_LIMIT,B_CFG+":common_training_protocol.amp_grad_scaler_init_scale"),
        row(54,"validation transform","SAME","Resize299, ToTensor, ImageNet Normalize","same",A_RUN+":data.transform",B_CFG+":common_training_protocol.evaluation_rule"),
        row(55,"original view/TTA","SAME","original view only for the 97.5372 metric","original view only for 97.700648 metric",A_RUN+":tta_mode",B_CFG+":common_training_protocol.evaluation_rule"),
        row(56,"softmax averaging","SAME","none for original-view OOF","none for original-view OOF",A_RUN+":tta_views",B_CFG+":common_training_protocol.evaluation_rule"),
        row(57,"fold ensemble","SAME","none; each sample predicted only by its held-out fold model","none; each sample predicted only by its held-out fold model",A_RUN,B_CFG+":reporting.primary"),
        row(58,"OOF assembly rule","SAME","concatenate five mutually exclusive held-out folds to 18,353","same",A_RUN+":dataset_indices.npy",B_CFG+":reporting.primary"),
        row(59,"final metric source","SAME",f"pooled argmax accuracy from Phase2A checkpoint-only logits = {a_accuracy:.15f}",f"pooled accuracy from Phase2B saved Stage2 OOF predictions = {b_accuracy:.15f}","phase2_controlled_validation/inference/ours-ft/fold_*/original/dynamic/logits.npy","phase2_controlled_validation/phase2b_screen/#0/oof_predictions.npy"),
    ]
    if len(rows) != 59 or [r["index"] for r in rows] != list(range(1, 60)):
        raise RuntimeError("internal error: P0 table must contain exactly 59 ordered fields")
    with (OUT / "protocol_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("SAME", "DIFFERENT", "UNRESOLVED")}
    md = ["# P0 Phase2A Ours-FT versus Phase2B #0 protocol provenance", "",
          "Verdict: **C. NOT DIRECTLY COMPARABLE**", "",
          f"Directly measured OOF accuracies: Phase2A {a_accuracy:.15f} ({a_accuracy*100:.4f}%), Phase2B {b_accuracy:.15f} ({b_accuracy*100:.4f}%), difference {(b_accuracy-a_accuracy)*100:.4f} pp.", "",
          f"Field counts: SAME={counts['SAME']}, DIFFERENT={counts['DIFFERENT']}, UNRESOLVED={counts['UNRESOLVED']}.", "",
          "The dataset, exact folds, architecture, and original-view OOF rule align. However, historical Phase2A training seeds, optimizer/LR/augmentation/class-weight/BN/AMP details are not serialized in the selected checkpoints, while checkpoint-candidate and selection mechanisms differ. These protocol differences and unresolved fields prevent direct attribution of the +0.1634 pp gap.", "",
          "PHASE2A_BASELINE_PROVENANCE_NOT_FULLY_RECOVERED", "",
          "| # | Field | Status | Phase2A | Phase2B |", "|---:|---|---|---|---|"]
    for r in rows:
        clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        md.append(f"| {r['index']} | {clean(r['field'])} | {r['status']} | {clean(r['phase2a_value'])} | {clean(r['phase2b_value'])} |")
    (OUT / "protocol_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sources = {
        "phase": "Phase2C P0", "training_performed": False, "verdict": "C. NOT DIRECTLY COMPARABLE",
        "phase2a_accuracy": a_accuracy, "phase2b_accuracy": b_accuracy,
        "difference_phase2b_minus_phase2a_pp": (b_accuracy-a_accuracy)*100,
        "sample_count": 18_353, "sample_ids_labels_aligned": True, "exact_fold_membership_aligned": True,
        "status_counts": counts, "sources": sorted({str(r["phase2a_source"]) for r in rows} | {str(r["phase2b_source"]) for r in rows}),
        "provenance_flag": "PHASE2A_BASELINE_PROVENANCE_NOT_FULLY_RECOVERED",
    }
    (OUT / "provenance_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P0 COMPLETE")


if __name__ == "__main__":
    main()
