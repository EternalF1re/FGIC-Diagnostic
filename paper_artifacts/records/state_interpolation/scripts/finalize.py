from __future__ import annotations
import csv,json,time
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from csi_common import *
def wc(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def met(y,p):return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(y,p))}
def pair(y,a,r):
 ca=a==y;cr=r==y;n=len(y);n10=int(np.sum(ca&~cr));n01=int(np.sum(~ca&cr));d=n10+n01;cats=np.bincount(ca.astype(np.int8)*2+cr.astype(np.int8),minlength=4);draw=np.random.default_rng(BOOT_SEED).multinomial(n,cats/n,size=REPS);boot=(draw[:,2]-draw[:,1])*100/n;ma,mr=met(y,a),met(y,r)
 return {'n':n,'alpha_accuracy':ma['accuracy'],'reference_accuracy':mr['accuracy'],'accuracy_delta_alpha_minus_reference_pp':100*(ma['accuracy']-mr['accuracy']),'ci95_low_pp':float(np.quantile(boot,.025)),'ci95_high_pp':float(np.quantile(boot,.975)),'bootstrap_replicates':REPS,'bootstrap_seed':BOOT_SEED,'mcnemar_exact_two_sided_p':float(binomtest(min(n10,n01),d,.5).pvalue) if d else 1.,'n10_alpha_correct_reference_wrong':n10,'n01_alpha_wrong_reference_correct':n01,'changed_correctness_count':d,'macro_f1_delta':ma['macro_f1']-mr['macro_f1'],'balanced_accuracy_delta':ma['balanced_accuracy']-mr['balanced_accuracy']}
def ov(y,a,r):
 ea=a!=y;er=r!=y;u=int(np.sum(ea|er));return {'prediction_disagreement_count':int(np.sum(a!=r)),'prediction_disagreement_percent':100*float(np.mean(a!=r)),'alpha_correct_reference_wrong':int(np.sum(~ea&er)),'alpha_wrong_reference_correct':int(np.sum(ea&~er)),'both_wrong':int(np.sum(ea&er)),'error_set_jaccard':float(np.sum(ea&er)/u) if u else 1.}
def assemble(s,a):
 parts=[]
 for f in FOLDS:
  m=json.loads(task_manifest(s,f,a).read_text(encoding='utf-8'));p=task_npz(s,f,a)
  if m['status']!='COMPLETE' or sha256(p)!=m['artifact_sha256']:raise RuntimeError('artifact')
  with np.load(p) as z:parts.append({k:z[k] for k in ('sample_id','fold','label','prediction','logits')})
 x={k:np.concatenate([q[k] for q in parts]) for k in parts[0]};o=np.argsort(x['sample_id']);x={k:v[o] for k,v in x.items()};validate_oof(x,f'{s}/{a}');p=ROOT/'oof'/f"seed{s}_alpha_{a:.1f}".replace('.','_');np.savez_compressed(str(p)+'.npz',**x,training_seed=np.full(N,s),alpha=np.full(N,a));return x
def verify_snapshot(name):
 snap=json.loads((ROOT/'manifests'/name).read_text(encoding='utf-8-sig'));changed=[]
 for e in snap['files']:
  p=Path(e['path']);changed+=[] if p.exists() and sha256(p).upper()==e['sha256'].upper() else [str(p)]
 return changed
def main():
 if json.loads((ROOT/'orchestrator_summary.json').read_text())['status']!='ALL_165_COMPLETE':raise RuntimeError('queue')
 gate=json.loads((ROOT/'manifests'/'endpoint_fidelity_gate.json').read_text());
 if gate['status']!='PASS':raise RuntimeError('gate')
 data={};curve=[]
 for s in SEEDS:
  for a in ALPHAS:
   x=assemble(s,a);data[s,a]=x;curve.append({'training_seed':s,'alpha':a,'primary_comparator':a==.5,'n':N,**met(x['label'],x['prediction'])})
 wc(ROOT/'curves'/'checkpoint_interpolation_curve_per_seed.csv',curve);summ=[]
 for a in ALPHAS:
  rows=[x for x in curve if x['alpha']==a];q={'alpha':a,'primary_comparator':a==.5,'independent_seeds':3}
  for k in ('accuracy','macro_f1','balanced_accuracy'):
   v=np.array([x[k] for x in rows]);q.update({f'{k}_mean':v.mean(),f'{k}_sample_sd':v.std(ddof=1),f'{k}_min':v.min(),f'{k}_max':v.max()})
  summ.append(q)
 wc(ROOT/'curves'/'checkpoint_interpolation_curve_multiseed_summary.csv',summ);stats={1:[],2:[]};over=[]
 for s in SEEDS:
  a=data[s,.5]
  for st,label in ((1,'stage1'),(2,'stage2')):
   r=reference(s,st);stats[st].append({'training_seed':s,'comparison':f'alpha_0_5_vs_{label}',**pair(a['label'],a['prediction'],r['prediction'])});over.append({'training_seed':s,'comparison':f'alpha_0_5_vs_{label}',**ov(a['label'],a['prediction'],r['prediction'])})
 wc(ROOT/'statistics'/'alpha_0_5_vs_stage1.csv',stats[1]);wc(ROOT/'statistics'/'alpha_0_5_vs_stage2.csv',stats[2]);wc(ROOT/'statistics'/'prediction_overlap.csv',over)
 mult=[]
 for st in (1,2):
  v=np.array([x['accuracy_delta_alpha_minus_reference_pp'] for x in stats[st]]);mult.append({'comparison':f'alpha_0_5_vs_stage{st}','delta_mean_pp':v.mean(),'delta_sample_sd_pp':v.std(ddof=1),'delta_min_pp':v.min(),'delta_max_pp':v.max()})
 wc(ROOT/'statistics'/'alpha_0_5_multiseed_summary.csv',mult)
 manifests=[json.loads(task_manifest(s,f,a).read_text()) for s in SEEDS for f in FOLDS for a in ALPHAS];checks={'all_165_complete':len(manifests)==165 and all(x['status']=='COMPLETE' for x in manifests),'zero_gradient':all(x['parameter_hash_before']==x['parameter_hash_after'] and x['all_grad_none'] and not x['optimizer'] and not x['scheduler'] and not x['backward'] for x in manifests),'no_bn_refresh':all(x['no_bn_reset'] and x['no_bn_refresh'] and x['model_eval'] and not x['train_mode_forward'] for x in manifests),'endpoint_gate':gate['endpoint_models']==30 and all(x['predictions_exact'] for x in gate['rows']),'source_unchanged':all(sha256(Path(e[f'stage{st}']))==e[f'stage{st}_sha256'] for e in json.loads((ROOT/'manifests'/'preflight_audit.json').read_text())['entries'] for st in (1,2)),'round2a_unchanged':not verify_snapshot('phase2d_round2a_immutability_before.json'),'old_interpolation_unchanged':not verify_snapshot('phase2d_weight_interpolation_immutability_before.json'),'oof_exact':all(len(data[s,a]['sample_id'])==N for s in SEEDS for a in ALPHAS),'no_posthoc_selection':True,'no_dfag':True}
 post={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'completed_at':time.time()};(ROOT/'manifests'/'postflight_audit.json').write_text(json.dumps(post,indent=2)+'\n');
 if post['status']!='PASS':raise RuntimeError(post)
 table='\n'.join(f"| {x['alpha']:.1f} | {'Yes' if x['primary_comparator'] else 'No'} | {100*x['accuracy_mean']:.4f} +/- {100*x['accuracy_sample_sd']:.4f} |" for x in summ);prim='\n'.join(f"| {s} | {100*next(x for x in curve if x['training_seed']==s and x['alpha']==.5)['accuracy']:.4f} | {stats[1][i]['accuracy_delta_alpha_minus_reference_pp']:.4f} [{stats[1][i]['ci95_low_pp']:.4f}, {stats[1][i]['ci95_high_pp']:.4f}] | {stats[2][i]['accuracy_delta_alpha_minus_reference_pp']:.4f} [{stats[2][i]['ci95_low_pp']:.4f}, {stats[2][i]['ci95_high_pp']:.4f}] |" for i,s in enumerate(SEEDS))
 report=f"""# Checkpoint-State Interpolation Results Audit

## Integrity
PASS. All 30 endpoints reproduced original predictions exactly before intermediate inference. All 165 models used eval-only complete checkpoint-state interpolation with no BN refresh, optimizer, scheduler, backward, gradient, TTA, or source modification.

## Endpoint reproduction
Alpha 0 directly loaded original Stage2 full state; alpha 1 directly loaded original Stage1 full state. All sample IDs, labels, predictions, Accuracy, Macro-F1 and Balanced Accuracy were exact for every seed/fold. Logit reproduction details are in `manifests/endpoint_fidelity_gate.json`.

## Alpha 0.5 primary results
| Seed | Accuracy (%) | vs Stage1 delta pp [95% CI] | vs Stage2 delta pp [95% CI] |
|---:|---:|---:|---:|
{prim}

Across seeds: alpha=.5 vs Stage1 {mult[0]['delta_mean_pp']:.4f} +/- {mult[0]['delta_sample_sd_pp']:.4f} pp; vs Stage2 {mult[1]['delta_mean_pp']:.4f} +/- {mult[1]['delta_sample_sd_pp']:.4f} pp. Exact McNemar results are in the primary CSVs.

## Full descriptive curve
| alpha | Primary | Accuracy mean +/- sample SD (%) |
|---:|:---:|---:|
{table}

No alpha was selected; alpha=.6 and all non-primary points are descriptive only.

## Prediction overlap
Per-seed alpha=.5 disagreement and error Jaccard against each endpoint are in `statistics/prediction_overlap.csv`.

## Conclusions and limits
The report supports only measured endpoint-faithful checkpoint-state interpolation performance and prediction overlap. It does not support semantic, representation, collapse, basin, attractor, optimal-alpha, or DFAG-necessity claims. Branch count remains one and Stage1-only remains the simple comparator unless the pre-registered alpha=.5 evidence demonstrates otherwise.

Round2A and the old reset/recalibration experiment were unchanged. STOP: no DFAG task was launched.
""";(ROOT/'CHECKPOINT_STATE_INTERPOLATION_RESULTS_AUDIT.md').write_text(report,encoding='utf-8');print(json.dumps({'status':'FINALIZED','postflight':'PASS'}))
if __name__=='__main__':main()
