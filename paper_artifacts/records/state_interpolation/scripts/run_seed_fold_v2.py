"""Endpoint-first checkpoint-state interpolation for one formal seed/fold."""
from __future__ import annotations
import argparse,gc,hashlib,json,time,traceback
import numpy as np,torch
from csi_common import *
ORDER=(0.0,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
def parameter_hash(model):
 d=hashlib.sha256()
 for n,p in model.named_parameters():d.update(n.encode());d.update(p.detach().cpu().contiguous().numpy().tobytes())
 return d.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=SEEDS);p.add_argument('--fold',type=int,required=True,choices=FOLDS);p.add_argument('--device',required=True,choices=('cuda:0','cuda:1'));a=p.parse_args();seed,fold,device=a.seed,a.fold,torch.device(a.device);out=task_dir(seed,fold)
 if out.exists():raise RuntimeError(f'refusing existing {out}')
 out.mkdir(parents=True); status=out/'task_status.json'; started=time.time(); rec={'status':'RUNNING','seed':seed,'fold':fold,'device':str(device),'endpoint_first_order':ORDER};status.write_text(json.dumps(rec,indent=2)+'\n')
 try:
  config=load_config(seed); c1=torch.load(checkpoint(seed,fold,1),map_location='cpu',weights_only=False);c2=torch.load(checkpoint(seed,fold,2),map_location='cpu',weights_only=False);s1,s2=c1['model_state_dict'],c2['model_state_dict'];refs={1:reference(seed,1),2:reference(seed,2)};done=[]
  for alpha in ORDER:
   began=time.time();seed_everything(seed);model=build_model();state,rule=make_state(model,s1,s2,alpha);model.load_state_dict(state,strict=True);del state
   for q in model.parameters():q.requires_grad_(False)
   before=parameter_hash(model);model.to(device);model.eval();val_loader,val_idx=build_eval_loader(seed,fold);result=TRAIN.evaluate(model,val_loader,device,collect=True);after=parameter_hash(model)
   if before!=after or any(q.grad is not None for q in model.parameters()) or model.training:raise RuntimeError('zero-gradient/eval integrity failure')
   if not np.array_equal(result['sample_ids'],val_idx) or not np.array_equal(result['prediction' if 'prediction' in result else 'predictions'],result['logits'].argmax(1)):raise RuntimeError('heldout integrity')
   pred=result['predictions']; endpoint=None
   if alpha in (0.,1.):
    stage=2 if alpha==0 else 1;ref=refs[stage];mask=ref['fold']==fold
    if not np.array_equal(result['sample_ids'],ref['sample_id'][mask]) or not np.array_equal(result['labels'],ref['label'][mask]) or not np.array_equal(pred,ref['prediction'][mask]):raise RuntimeError(f'ENDPOINT FIDELITY FAIL alpha={alpha}')
    endpoint={'stage':stage,'sample_ids_exact':True,'labels_exact':True,'predictions_exact':True,'logits_bitwise_equal':bool(np.array_equal(result['logits'],ref['logits'][mask])),'max_abs_logit_difference':float(np.max(np.abs(result['logits']-ref['logits'][mask])))}
   npz=task_npz(seed,fold,alpha);np.savez_compressed(npz,sample_id=result['sample_ids'].astype(np.int64),fold=np.full(len(val_idx),fold,np.int8),label=result['labels'].astype(np.int64),prediction=pred.astype(np.int64),logits=result['logits'].astype(np.float32))
   manifest={'status':'COMPLETE','seed':seed,'fold':fold,'alpha':alpha,'primary':alpha==.5,'device':str(device),'elapsed_seconds':time.time()-began,'stage1_checkpoint':str(checkpoint(seed,fold,1).resolve()),'stage1_sha256':sha256(checkpoint(seed,fold,1)),'stage2_checkpoint':str(checkpoint(seed,fold,2).resolve()),'stage2_sha256':sha256(checkpoint(seed,fold,2)),'state_rule':rule,'no_bn_reset':True,'no_bn_refresh':True,'model_eval':True,'train_mode_forward':False,'optimizer':False,'scheduler':False,'backward':False,'parameter_hash_before':before,'parameter_hash_after':after,'all_grad_none':all(q.grad is None for q in model.parameters()),'endpoint_fidelity':endpoint,'artifact':str(npz.resolve()),'artifact_sha256':sha256(npz),'accuracy':result['accuracy'],'macro_f1':result['macro_f1'],'balanced_accuracy':result['balanced_accuracy']};task_manifest(seed,fold,alpha).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');done.append(alpha);print(json.dumps({'status':'COMPLETE','seed':seed,'fold':fold,'alpha':alpha,'endpoint':endpoint,'elapsed':manifest['elapsed_seconds']}),flush=True);del model,val_loader,result;gc.collect();torch.cuda.empty_cache()
  rec.update({'status':'COMPLETE','completed_unix':time.time(),'elapsed_seconds':time.time()-started,'alphas':done});status.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({'status':'TASK_COMPLETE','seed':seed,'fold':fold,'alphas':len(done)}),flush=True)
 except Exception as e:rec.update({'status':'FAILED','error':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started});status.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');raise
if __name__=='__main__':main()


