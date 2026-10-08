from __future__ import annotations
import argparse,json,sys,time
from csi_common import *
if str(VR) not in sys.path:sys.path.insert(0,str(VR))
from dynamic_gpu_queue import DynamicGpuQueue,SchedulerPaths,TaskSpec
def main():
 p=argparse.ArgumentParser();p.add_argument('--devices',nargs='+',default=('cuda:0',));p.add_argument('--poll-seconds',type=float,default=15);a=p.parse_args();pre=json.loads((ROOT/'manifests'/'preflight_audit.json').read_text(encoding='utf-8'))
 if pre['status']!='PASS':raise RuntimeError('preflight')
 gate=ROOT/'manifests'/'endpoint_fidelity_gate.json'
 if not gate.exists() or json.loads(gate.read_text(encoding='utf-8')).get('status')!='PASS':raise RuntimeError('endpoint hard gate')
 tasks=[];runner=ROOT/'scripts'/'run_seed_fold.py'
 for f in FOLDS:
  for s in SEEDS:
   if task_dir(s,f).exists():raise RuntimeError(f'existing {task_dir(s,f)}')
   tasks.append(TaskSpec(task_id=f'A_seed{s}_fold{f}',command=('{python}',str(runner),'--seed',str(s),'--fold',str(f),'--device','{device}'),estimated_seconds=420.,metadata={'task':'A','seed':s,'fold':f,'zero_gradient':True}))
 started=time.time();q=DynamicGpuQueue(tasks,devices=a.devices,paths=SchedulerPaths.under(ROOT/'scheduler'),cwd=REPO,poll_seconds=a.poll_seconds,longest_first=False);r=q.run();summary={'status':'ALL_165_COMPLETE','tasks':15,'models':165,'training_runs':0,'started':started,'completed':time.time(),'scheduler':r};(ROOT/'orchestrator_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

