"""Aggregate the completed 60-job run into pooled OOF and paired statistics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "formal_run"
METHODS = ("l2_sp", "mc_loss", "cal", "ours_ft", "progressive", "dfag")
DATASETS = ("cub", "cars")
CARS_LOOKUP_IMPLEMENTATION = "one_pass_per_split_filename_index_v1"
RUNTIME_PATH = ROOT / "common" / "runtime.py"
PROTOCOL_CORE_PATH = ROOT.parent / "corrected_external_protocol_smoke" / "scripts" / "protocol_core.py"


def file_sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def metric(labels, predictions):
    return {"accuracy":float(accuracy_score(labels,predictions)),"macro_f1":float(f1_score(labels,predictions,average="macro",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(labels,predictions))}


def paired_bootstrap(reference_correct: np.ndarray, candidate_correct: np.ndarray, seed: int = 42) -> tuple[float,float]:
    differences=(candidate_correct.astype(np.float64)-reference_correct.astype(np.float64))*100.0
    rng=np.random.default_rng(seed); values=[]; n=len(differences)
    for start in range(0,100_000,1000):
        count=min(1000,100_000-start); indices=rng.integers(0,n,size=(count,n)); values.append(differences[indices].mean(axis=1))
    samples=np.concatenate(values); return float(np.percentile(samples,2.5)),float(np.percentile(samples,97.5))


def main() -> None:
    plan=json.loads((ROOT/"CONTROLLED_REIMPLEMENTATION_60_JOB_PLAN.json").read_text(encoding="utf-8")); jobs={row["job_id"]:row for row in plan["jobs"]}
    missing=[]; failed=[]; fold_rows=[]; pooled_rows=[]; pooled:dict[tuple[str,str],dict[str,np.ndarray]]={}
    expected_runtime_sha=file_sha(RUNTIME_PATH); expected_protocol_core_sha=file_sha(PROTOCOL_CORE_PATH)
    oof_root=RUN_ROOT/"pooled_oof"; oof_root.mkdir(parents=True,exist_ok=True)
    for dataset in DATASETS:
        for method in METHODS:
            parts=[]
            for fold in range(5):
                job_id=f"{method}_{dataset}_fold{fold}"; manifest_path=RUN_ROOT/"jobs"/job_id/"run_manifest.json"
                if not manifest_path.is_file(): missing.append(job_id); continue
                manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status")!="COMPLETE": failed.append(job_id); continue
                loader_identity=(manifest.get("runtime_sha256"),manifest.get("protocol_core_sha256"),manifest.get("cars_lookup_implementation"))
                expected_identity=(expected_runtime_sha,expected_protocol_core_sha,CARS_LOOKUP_IMPLEMENTATION)
                if loader_identity!=expected_identity: raise RuntimeError(f"loader identity drift {job_id}: {loader_identity} != {expected_identity}")
                archive=np.load(RUN_ROOT/"jobs"/job_id/"heldout_predictions.npz",allow_pickle=False); part={key:archive[key] for key in ("sample_ids","labels","predictions","dataset_indices","fold_ids","logits")}; parts.append(part)
                values=metric(part["labels"],part["predictions"]); fold_rows.append({"dataset":dataset,"method":method,"fold":fold,"count":len(part["labels"]),**values})
            if len(parts)!=5: continue
            combined={key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
            order=np.argsort(combined["sample_ids"].astype(str)); combined={key:value[order] for key,value in combined.items()}
            if len(set(combined["sample_ids"].astype(str).tolist()))!=len(combined["sample_ids"]): raise RuntimeError(f"duplicate pooled IDs {dataset}/{method}")
            expected=5994 if dataset=="cub" else 8144
            if len(combined["sample_ids"])!=expected or set(combined["fold_ids"].tolist())!=set(range(5)): raise RuntimeError(f"OOF coverage failure {dataset}/{method}")
            target=oof_root/f"{dataset}__{method}.npz"; np.savez_compressed(target,**combined); values=metric(combined["labels"],combined["predictions"])
            pooled_rows.append({"dataset":dataset,"method":method,"count":expected,"coverage":1.0,"oof_path":str(target),"oof_sha256":file_sha(target),**values}); pooled[(dataset,method)]=combined
    if missing or failed or len(pooled)!=12: raise RuntimeError(f"formal outputs incomplete missing={missing} failed={failed}")
    paired=[]
    for dataset in DATASETS:
        reference=pooled[(dataset,"ours_ft")]; ref_correct=reference["predictions"]==reference["labels"]
        for method in METHODS:
            if method=="ours_ft": continue
            candidate=pooled[(dataset,method)]
            if not np.array_equal(reference["sample_ids"].astype(str),candidate["sample_ids"].astype(str)) or not np.array_equal(reference["labels"],candidate["labels"]): raise RuntimeError("paired OOF alignment failure")
            cand_correct=candidate["predictions"]==candidate["labels"]; delta=float((cand_correct.mean()-ref_correct.mean())*100.0); low,high=paired_bootstrap(ref_correct,cand_correct)
            b=int(np.sum(ref_correct & ~cand_correct)); c=int(np.sum(~ref_correct & cand_correct)); p=1.0 if b+c==0 else float(binomtest(min(b,c),n=b+c,p=0.5,alternative="two-sided").pvalue)
            paired.append({"dataset":dataset,"candidate":method,"reference":"ours_ft","delta_accuracy_pp":delta,"bootstrap_replicates":100000,"ci_method":"paired percentile","ci95_low_pp":low,"ci95_high_pp":high,"mcnemar_b":b,"mcnemar_c":c,"mcnemar_exact_two_sided_p":p,"significance_unit":"pooled OOF sample"})
    write_csv(RUN_ROOT/"final_pooled_oof_metrics.csv",pooled_rows); write_csv(RUN_ROOT/"final_fold_descriptive_metrics.csv",fold_rows); write_csv(RUN_ROOT/"final_paired_statistics.csv",paired)
    loader_identity={"implementation":CARS_LOOKUP_IMPLEMENTATION,"runtime_sha256":expected_runtime_sha,"protocol_core_sha256":expected_protocol_core_sha,"equivalence_audit":"scripts/verify_cars_index_optimization.py","fold_manifest_sha256":"e2a18afd04c54105215b97cd02a2687b4ead9ac87c6dbe77a58b9508e02507df","class_map_sha256":"2cf8d881310afbf88fb3785191403a54ca1d4fff90d187fb68391286db646dcc"}
    result={"status":"FORMAL_RUN_COMPLETE","formal_jobs_complete":60,"formal_jobs_failed":0,"pooled_oof_coverage":"12/12 method-dataset combinations complete","protocol_deviations":[],"storage_lookup_implementation":loader_identity,"finalizer_sha256":file_sha(Path(__file__)),"pooled_metrics":pooled_rows,"paired_statistics":paired}
    (RUN_ROOT/"FINAL_CONTROLLED_REIMPLEMENTATION_RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report="# Final Controlled Reimplementation Results\n\n`FORMAL_RUN_COMPLETE`\n\nAll 60 jobs completed with complete five-fold pooled OOF coverage. Every job used the same one-pass Cars filename index; runtime and protocol-core SHA-256 identities were checked before aggregation. Its frozen assignments, folds, class map and path semantics were verified by `scripts/verify_cars_index_optimization.py`. See `final_pooled_oof_metrics.csv`, `final_fold_descriptive_metrics.csv`, and `final_paired_statistics.csv`. Paired statistics use 100,000 paired bootstrap resamples, 95% percentile intervals and exact two-sided McNemar tests on pooled OOF samples; folds are descriptive strata, not significance units.\n"
    (RUN_ROOT/"FINAL_CONTROLLED_REIMPLEMENTATION_RESULTS_AUDIT.md").write_text(report,encoding="utf-8")
    print(json.dumps({"status":"FORMAL_RUN_COMPLETE","jobs":60,"pooled":12}))


if __name__=="__main__": main()
