"""Static scope checks after source preflight and before endpoint gate."""
from __future__ import annotations
import json
from csi_common import *
def main():
 p=ROOT/'manifests'/'preflight_audit.json';x=json.loads(p.read_text(encoding='utf-8'));runner=(ROOT/'scripts'/'run_seed_fold.py').read_text(encoding='utf-8')
 checks=x['checks']
 for bad in ('torch.optim','backward(','reset_running_stats','adapt_batch_norm','model.train()'):
  ok=bad not in runner;checks.append({'check':f'forbidden_{bad}','status':'PASS' if ok else 'FAIL','detail':bad})
  if not ok:raise RuntimeError(bad)
 for req in ('model.eval()','alpha in (0.,1.)','ENDPOINT FIDELITY FAIL','make_state'):
  ok=req in runner;checks.append({'check':f'required_{req}','status':'PASS' if ok else 'FAIL','detail':req})
  if not ok:raise RuntimeError(req)
 x.update({'status':'PASS','checks':checks,'check_count':len(checks),'global_endpoint_gate_required':True,'intermediate_alpha_authorized_only_after_gate':True});p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':'PASS','checks':len(checks)}))
if __name__=='__main__':main()
