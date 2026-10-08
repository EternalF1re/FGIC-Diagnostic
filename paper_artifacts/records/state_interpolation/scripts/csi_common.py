"""Common definitions for endpoint-faithful checkpoint-state interpolation."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]; VR=ROOT.parent; REPO=VR.parent; P2B=VR/'phase2b_screen'; R1=VR/'phase2d_round1'; P2C=VR/'phase2c_inference_diagnostic'; R2=VR/'phase2d_round2a'; OLD=VR/'phase2d_weight_interpolation'
if str(P2B) not in sys.path: sys.path.insert(0,str(P2B))
from screen_core import LeafDataset,ScreenModel,eval_transform,seed_everything,seed_worker,split_indices
from torch.utils.data import DataLoader,Subset
spec=importlib.util.spec_from_file_location('csi_train_one',P2B/'train_one.py'); TRAIN=importlib.util.module_from_spec(spec); spec.loader.exec_module(TRAIN)
SEEDS=(42,43,44); FOLDS=tuple(range(5)); ALPHAS=tuple(round(i/10,1) for i in range(11)); N=18353; REPS=100000; BOOT_SEED=20260807; INSTRUCTION='3643A4A9CB1951CE5601EFE72D918620060A17CDB8EDBB9FAE616E02832395FF'
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_dir(seed,fold): return P2B/'#0'/f'fold_{fold}' if seed==42 else R1/f'baseline_seed{seed}'/f'fold_{fold}'
def config_path(seed): return P2B/'configs'/'controlled_screen.json' if seed==42 else R1/'configs'/f'baseline_seed{seed}.json'
def load_config(seed): return json.loads(config_path(seed).read_text(encoding='utf-8'))
def checkpoint(seed,fold,stage): return source_dir(seed,fold)/f'best_stage{stage}.pth'
def build_model(): return ScreenModel('#0',176,pretrained=False)
def build_eval_loader(seed,fold):
    cfg=load_config(seed); d=cfg['dataset']; protocol=cfg['common_training_protocol']; dataset=LeafDataset(Path(d['train_csv']),Path(d['root']),eval_transform()); _,val_idx=split_indices(dataset.labels,fold,int(cfg['split']['split_random_state']))
    loader=DataLoader(Subset(dataset,val_idx.tolist()),batch_size=int(protocol['batch_size']),shuffle=False,drop_last=False,num_workers=int(protocol['num_workers']),pin_memory=True,worker_init_fn=seed_worker,persistent_workers=int(protocol['num_workers'])>0)
    return loader,val_idx
def tag(a): return f'alpha_{a:.1f}'.replace('.','_')
def task_dir(seed,fold): return ROOT/'tasks'/f'seed{seed}'/f'fold_{fold}'
def task_npz(seed,fold,a): return task_dir(seed,fold)/f'{tag(a)}.npz'
def task_manifest(seed,fold,a): return task_dir(seed,fold)/f'{tag(a)}.json'
def reference(seed,stage):
    if seed==42 and stage==1:
        with np.load(P2C/'p1_stage1_only_oof'/'stage1_oof_predictions_variant_0.npz') as z:return {k:z[k] for k in ('sample_id','fold','label','prediction','logits')}
    if seed==42:
        parts=[]
        for f in FOLDS:
            d=P2B/'#0'/f'fold_{f}'; ids=np.load(d/'validation_sample_ids.npy'); parts.append({'sample_id':ids,'fold':np.full(len(ids),f,np.int8),'label':np.load(d/'validation_labels.npy'),'prediction':np.load(d/'validation_predictions.npy'),'logits':np.load(d/'validation_logits.npy')})
        x={k:np.concatenate([q[k] for q in parts]) for k in parts[0]}; o=np.argsort(x['sample_id']); return {k:v[o] for k,v in x.items()}
    with np.load(R1/'oof'/f'baseline_seed{seed}_stage{stage}_oof.npz') as z:return {k:z[k] for k in ('sample_id','fold','label','prediction','logits')}
def validate_oof(x,label):
    if len(x['sample_id'])!=N or not np.array_equal(x['sample_id'],np.arange(N)) or len(np.unique(x['sample_id']))!=N:raise RuntimeError(f'{label}: OOF coverage')
    if not np.isfinite(x['logits']).all() or not np.array_equal(x['logits'].argmax(1),x['prediction']):raise RuntimeError(f'{label}: logits integrity')
def make_state(model,s1,s2,a):
    if a==0.: return {k:v.clone() for k,v in s2.items()},{'endpoint_direct_full_state':'Stage2'}
    if a==1.: return {k:v.clone() for k,v in s1.items()},{'endpoint_direct_full_state':'Stage1'}
    expected=model.state_dict()
    if set(s1)!=set(s2) or set(s1)!=set(expected):raise RuntimeError('state keys mismatch')
    out={}; floating=nonfloating=0
    for k in expected:
        x,y=s1[k],s2[k]
        if x.shape!=y.shape or x.dtype!=y.dtype:raise RuntimeError(k)
        if torch.is_floating_point(x):out[k]=x.mul(a).add(y,alpha=1-a);floating+=1
        else:out[k]=y.clone();nonfloating+=1
    return out,{'formula':'alpha*Stage1+(1-alpha)*Stage2','floating_state_tensors_interpolated':floating,'intermediate_nonfloating_buffer_rule':'copy Stage2','nonfloating_state_tensors_copied':nonfloating}



