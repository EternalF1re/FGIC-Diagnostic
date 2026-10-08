"""Global hard gate: all 30 endpoint predictions must reproduce before intermediates."""
from __future__ import annotations
import argparse,json,time
import numpy as np,torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from csi_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0',choices=('cuda:0','cuda:1'));a=p.parse_args();dev=torch.device(a.device);out=ROOT/'manifests'/'endpoint_fidelity_gate.json'
 if out.exists():raise RuntimeError('endpoint gate exists')
 rows=[];started=time.time()
 for seed in SEEDS:
  cfg=load_config(seed);refs={1:reference(seed,1),2:reference(seed,2)}
  for fold in FOLDS:
   for alpha,stage in ((0.,2),(1.,1)):
    seed_everything(seed);ck=torch.load(checkpoint(seed,fold,stage),map_location='cpu',weights_only=False);m=build_model();m.load_state_dict(ck['model_state_dict'],strict=True)
    for q in m.parameters():q.requires_grad_(False)
    m.to(dev);m.eval();loader,val_idx=build_eval_loader(seed,fold);r=TRAIN.evaluate(m,loader,dev,collect=True);ref=refs[stage];mask=ref['fold']==fold
    ids=bool(np.array_equal(r['sample_ids'],ref['sample_id'][mask]));labels=bool(np.array_equal(r['labels'],ref['label'][mask]));pred=bool(np.array_equal(r['predictions'],ref['prediction'][mask]));ma=(float(accuracy_score(r['labels'],r['predictions'])),float(f1_score(r['labels'],r['predictions'],average='macro')),float(balanced_accuracy_score(r['labels'],r['predictions'])));mr=(float(accuracy_score(ref['label'][mask],ref['prediction'][mask])),float(f1_score(ref['label'][mask],ref['prediction'][mask],average='macro')),float(balanced_accuracy_score(ref['label'][mask],ref['prediction'][mask])));metrics=ma==mr
    row={'seed':seed,'fold':fold,'alpha':alpha,'stage':stage,'sample_ids_exact':ids,'labels_exact':labels,'predictions_exact':pred,'metrics_exact':metrics,'logits_bitwise_equal':bool(np.array_equal(r['logits'],ref['logits'][mask])),'logits_allclose_1e_6':bool(np.allclose(r['logits'],ref['logits'][mask],atol=1e-6,rtol=1e-6)),'max_abs_logit_difference':float(np.max(np.abs(r['logits']-ref['logits'][mask])))};rows.append(row);print(json.dumps(row),flush=True)
    if not(ids and labels and pred and metrics):
     out.write_text(json.dumps({'status':'FAIL','rows':rows,'failed':row},indent=2)+'\n');raise RuntimeError(f'ENDPOINT HARD GATE FAIL {row}')
 result={'status':'PASS','endpoint_models':30,'all_sample_ids_exact':True,'all_labels_exact':True,'all_predictions_exact':True,'all_metrics_exact':True,'started':started,'completed':time.time(),'rows':rows};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','endpoint_models':30}),flush=True)
if __name__=='__main__':main()


