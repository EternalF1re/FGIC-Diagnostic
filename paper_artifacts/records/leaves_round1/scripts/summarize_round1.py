"""Build Round1 OOF files and all requested paired/multi-seed statistics."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from round1_common import ROUND_ROOT, VALIDATION_ROOT, ROUND_IDS, output_dir

REPS=100_000; SEED=20260807

def write_csv(path, data):
    with Path(path).open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),
            'balanced_accuracy':float(balanced_accuracy_score(y,p))}

def paired(y,a,b,seed=SEED):
    ca=a==y; cb=b==y; n=len(y); n11=int(np.sum(ca&cb)); n10=int(np.sum(ca&~cb)); n01=int(np.sum(~ca&cb)); n00=int(np.sum(~ca&~cb))
    cats=np.bincount(ca.astype(np.int8)*2+cb.astype(np.int8),minlength=4); draws=np.random.default_rng(seed).multinomial(n,cats/n,size=REPS)
    boot=(draws[:,1]-draws[:,2])*100/n; d=n10+n01
    ma,mb=metrics(y,a),metrics(y,b)
    return {'n':n,'accuracy_a':ma['accuracy'],'accuracy_b':mb['accuracy'],'delta_b_minus_a_pp':(mb['accuracy']-ma['accuracy'])*100,
            'ci95_low_pp':float(np.quantile(boot,.025)),'ci95_high_pp':float(np.quantile(boot,.975)),
            'bootstrap_replicates':REPS,'bootstrap_seed':seed,'mcnemar_exact_two_sided_p':float(binomtest(min(n10,n01),d,.5).pvalue) if d else 1.0,
            'prediction_changed_n':int(np.sum(a!=b)),'n11':n11,'n10':n10,'n01':n01,'n00':n00,'correctness_discordant_n':d,
            'macro_f1_a':ma['macro_f1'],'macro_f1_b':mb['macro_f1'],'macro_f1_delta':mb['macro_f1']-ma['macro_f1'],
            'balanced_accuracy_a':ma['balanced_accuracy'],'balanced_accuracy_b':mb['balanced_accuracy'],'balanced_accuracy_delta':mb['balanced_accuracy']-ma['balanced_accuracy']}

def assemble_new(run_id,stage):
    parts=[]
    for fold in range(5):
        d=output_dir(run_id,fold); prefix=f'stage{stage}_validation_'
        parts.append({k:np.load(d/f'{prefix}{k}.npy') for k in ('sample_ids','labels','predictions','logits') } | {'folds':np.full(len(np.load(d/f'{prefix}labels.npy')),fold,dtype=np.int8)})
    c={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}; o=np.argsort(c['sample_ids']); c={k:v[o] for k,v in c.items()}
    return c

def existing(key,stage):
    variant={'baseline':'#0','lambda_0_1':'#2','lambda_1_0':'#1'}[key]
    if stage==1:
        p=VALIDATION_ROOT/'phase2c_inference_diagnostic'/'p1_stage1_only_oof'/f"stage1_oof_predictions_{variant.replace('#','variant_')}.npz"
        with np.load(p) as z: return {'sample_ids':z['sample_id'],'labels':z['label'],'predictions':z['prediction'],'logits':z['logits'],'folds':z['fold']}
    base=VALIDATION_ROOT/'phase2b_screen'/variant
    folds=np.empty(18353,dtype=np.int8)
    for f in range(5): folds[np.load(base/f'fold_{f}'/'validation_sample_ids.npy')]=f
    return {'sample_ids':np.load(base/'oof_sample_ids.npy'),'labels':np.load(base/'oof_labels.npy'),'predictions':np.load(base/'oof_predictions.npy'),
            'logits':np.concatenate([np.load(base/f'fold_{f}'/'validation_logits.npy') for f in range(5)])[np.argsort(np.concatenate([np.load(base/f'fold_{f}'/'validation_sample_ids.npy') for f in range(5)]))], 'folds':folds}

def verify(data,label):
    ok=len(data['sample_ids'])==18353 and np.array_equal(data['sample_ids'],np.arange(18353)) and len(np.unique(data['sample_ids']))==18353
    ok &= np.isfinite(data['logits']).all() and np.array_equal(data['logits'].argmax(1),data['predictions']) and set(np.unique(data['folds']).tolist())=={0,1,2,3,4}
    if not ok: raise RuntimeError(f'OOF integrity failure: {label}')

def main():
    if not (ROUND_ROOT/'orchestrator_summary.json').exists() or json.loads((ROUND_ROOT/'orchestrator_summary.json').read_text())['status']!='ALL_20_COMPLETE': raise RuntimeError('20 runs not complete')
    data={}
    for rid in ROUND_IDS:
        for st in (1,2):
            x=assemble_new(rid,st); verify(x,f'{rid} stage{st}'); data[(rid,st)]=x
            np.savez_compressed(ROUND_ROOT/'oof'/f'{rid}_stage{st}_oof.npz',sample_id=x['sample_ids'],fold=x['folds'],label=x['labels'],prediction=x['predictions'],logits=x['logits'])
    metric_rows=[]
    for (rid,st),x in data.items():
        for scope,fold in [('pooled','ALL'),*[(f'fold_{f}',f) for f in range(5)]]:
            m=np.ones(18353,bool) if fold=='ALL' else x['folds']==fold; z=metrics(x['labels'][m],x['predictions'][m]); metric_rows.append({'run_id':rid,'stage':st,'scope':scope,'fold':fold,'n':int(m.sum()),**z})
    write_csv(ROUND_ROOT/'oof'/'round1_oof_metrics.csv',metric_rows)
    s12=[]; s12fold=[]
    for rid in ROUND_IDS:
        a,b=data[(rid,1)],data[(rid,2)];
        if not np.array_equal(a['labels'],b['labels']): raise RuntimeError('stage label mismatch')
        r={'run_id':rid,**paired(a['labels'],a['predictions'],b['predictions'])}; s12.append(r)
        for f in range(5):
            m=a['folds']==f; q=paired(a['labels'][m],a['predictions'][m],b['predictions'][m],SEED+f+1); d=q['delta_b_minus_a_pp']; s12fold.append({'run_id':rid,'fold':f,'direction':'positive' if d>0 else 'negative' if d<0 else 'equal',**q})
    write_csv(ROUND_ROOT/'statistics'/'stage1_vs_stage2_paired.csv',s12); write_csv(ROUND_ROOT/'statistics'/'stage1_vs_stage2_per_fold.csv',s12fold)

    # Baseline seed42/43/44, with seed42 reused from Phase2B/2C.
    baseline={42:{1:existing('baseline',1),2:existing('baseline',2)},43:{1:data[('baseline_seed43',1)],2:data[('baseline_seed43',2)]},44:{1:data[('baseline_seed44',1)],2:data[('baseline_seed44',2)]}}
    brows=[]
    for seed,stages in baseline.items():
        a,b=stages[1],stages[2]; q=paired(a['labels'],a['predictions'],b['predictions']); dirs=[]
        for f in range(5):
            m=a['folds']==f; d=float(np.mean(b['predictions'][m]==b['labels'][m])-np.mean(a['predictions'][m]==a['labels'][m])); dirs.append('+' if d>0 else '-' if d<0 else '=')
        brows.append({'scope':'seed','training_seed':seed,'stage1_accuracy':q['accuracy_a'],'stage2_accuracy':q['accuracy_b'],'delta_pp':q['delta_b_minus_a_pp'],
                      'ci95_low_pp':q['ci95_low_pp'],'ci95_high_pp':q['ci95_high_pp'],'mcnemar_p':q['mcnemar_exact_two_sided_p'],
                      'fold_positive':dirs.count('+'),'fold_negative':dirs.count('-'),'fold_equal':dirs.count('='),
                      'stage1_macro_f1':q['macro_f1_a'],'stage2_macro_f1':q['macro_f1_b'],'stage1_balanced_accuracy':q['balanced_accuracy_a'],'stage2_balanced_accuracy':q['balanced_accuracy_b']})
    agg={'scope':'three_seed_mean_std','training_seed':'42,43,44'}
    for k in ('stage1_accuracy','stage2_accuracy','delta_pp','stage1_macro_f1','stage2_macro_f1','stage1_balanced_accuracy','stage2_balanced_accuracy'):
        v=np.array([r[k] for r in brows]); agg[k+'_mean']=float(v.mean()); agg[k+'_std']=float(v.std(ddof=1))
    # Separate CSVs keep schemas explicit.
    write_csv(ROUND_ROOT/'statistics'/'baseline_multiseed_per_seed.csv',brows); write_csv(ROUND_ROOT/'statistics'/'baseline_multiseed_summary.csv',[agg])

    # Lambda/direct comparisons at both stages.
    variants={'baseline':{1:existing('baseline',1),2:existing('baseline',2)},'lambda_0_1':{1:existing('lambda_0_1',1),2:existing('lambda_0_1',2)},
              'lambda_0_7':{1:data[('lambda_0_7_seed42',1)],2:data[('lambda_0_7_seed42',2)]},'lambda_0_9':{1:data[('lambda_0_9_seed42',1)],2:data[('lambda_0_9_seed42',2)]},
              'lambda_1_0':{1:existing('lambda_1_0',1),2:existing('lambda_1_0',2)}}
    pairs=[('lambda_1_0','lambda_0_7'),('lambda_1_0','lambda_0_9'),('lambda_0_1','lambda_0_7'),('lambda_0_1','lambda_0_9'),('baseline','lambda_0_7'),('baseline','lambda_0_9')]
    prows=[]; pfolds=[]
    for stage in (1,2):
        for akey,bkey in pairs:
            a,b=variants[akey][stage],variants[bkey][stage]; q=paired(a['labels'],a['predictions'],b['predictions']); prows.append({'stage':stage,'a':akey,'b':bkey,'comparison':f'{akey}_vs_{bkey}',**q})
            for f in range(5):
                m=a['folds']==f; z=paired(a['labels'][m],a['predictions'][m],b['predictions'][m],SEED+f+1); d=z['delta_b_minus_a_pp']; pfolds.append({'stage':stage,'a':akey,'b':bkey,'fold':f,'direction':'positive' if d>0 else 'negative' if d<0 else 'equal',**z})
    write_csv(ROUND_ROOT/'statistics'/'lambda_direct_paired.csv',prows); write_csv(ROUND_ROOT/'statistics'/'lambda_direct_paired_per_fold.csv',pfolds)
    print(json.dumps({'status':'OOF_STATISTICS_COMPLETE','new_oof':8,'paired_comparisons':len(prows)}))

if __name__=='__main__': main()
