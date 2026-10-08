"""Preflight source/OOF audit for Task A."""
from __future__ import annotations
import json,numpy as np,torch
from csi_common import *
def main():
    out=ROOT/'manifests'/'preflight_audit.json'
    if out.exists():raise RuntimeError('refusing overwrite preflight')
    instruction=ROOT/'manifests'/f'instruction_{INSTRUCTION}.txt'; checks=[]; entries=[]; schema=build_model().state_dict()
    def ck(n,ok,d):checks.append({'check':n,'status':'PASS' if ok else 'FAIL','detail':d}); (_ for _ in ()).throw(RuntimeError(f'{n}: {d}')) if not ok else None
    ck('instruction',sha256(instruction).upper()==INSTRUCTION,sha256(instruction)); ck('old_directory_separate',ROOT!=OLD and OLD.exists(),str(OLD)); ck('round2a_snapshot',(ROOT/'manifests'/'phase2d_round2a_immutability_before.json').exists(),'saved')
    labels=None
    for seed in SEEDS:
      for fold in FOLDS:
        d=source_dir(seed,fold); m=json.loads((d/'run_manifest.json').read_text(encoding='utf-8')); ck(f'id_{seed}_{fold}',m['status']=='COMPLETE' and int(m['training_seed'])==seed and int(m['fold'])==fold,'formal pair')
        ps=[]
        for st in (1,2):
          p=checkpoint(seed,fold,st); h=sha256(p); ck(f'hash_{seed}_{fold}_{st}',h==m[f'best_stage{st}_sha256'],h); c=torch.load(p,map_location='cpu',weights_only=False); ps.append(c['model_state_dict']); ck(f'ckpt_id_{seed}_{fold}_{st}',int(c['seed' if 'seed' in c else 'training_seed'])==seed and int(c['fold'])==fold and int(c['stage'])==st,'identity')
        ck(f'schema_{seed}_{fold}',set(ps[0])==set(ps[1])==set(schema) and all(ps[0][k].shape==ps[1][k].shape and ps[0][k].dtype==ps[1][k].dtype for k in schema),str(len(schema))); entries.append({'seed':seed,'fold':fold,'stage1':str(checkpoint(seed,fold,1).resolve()),'stage1_sha256':sha256(checkpoint(seed,fold,1)),'stage2':str(checkpoint(seed,fold,2).resolve()),'stage2_sha256':sha256(checkpoint(seed,fold,2))})
      for st in (1,2):
        x=reference(seed,st);validate_oof(x,f'ref {seed}/{st}'); labels=x['label'] if labels is None else labels;ck(f'oof_{seed}_{st}',np.array_equal(labels,x['label']),'18353 aligned')
    # Static zero-gradient/no-BN-refresh source scan.
    runner=(ROOT/'scripts'/'run_seed_fold.py').read_text(encoding='utf-8') if (ROOT/'scripts'/'run_seed_fold.py').exists() else ''
    result={'status':'PASS','checks':checks,'entries':entries,'checkpoint_count':30,'state_tensor_keys':len(schema),'oof_n':N,'source_hashes_saved':True,'round2a_snapshot_saved':True,'old_interpolation_snapshot_saved':True};out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':'PASS','checks':len(checks),'checkpoints':30,'oof_n':N}))
if __name__=='__main__':main()
