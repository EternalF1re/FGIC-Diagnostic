"""Fail-closed sequencer: endpoints PASS -> Task A -> Task B."""
from __future__ import annotations
import json,subprocess,sys,time
from pathlib import Path
A=Path(__file__).resolve().parents[1];V=A.parent;B=V/'phase2d_bn_refresh_diagnostic'
def main():
 gate=A/'manifests'/'endpoint_fidelity_gate.json'
 while not gate.exists():time.sleep(15)
 g=json.loads(gate.read_text(encoding='utf-8'))
 if g.get('status')!='PASS':raise RuntimeError('endpoint hard gate failed; downstream paused')
 subprocess.run([sys.executable,str(A/'scripts'/'orchestrate.py'),'--devices','cuda:0','--poll-seconds','15'],cwd=str(V.parent),check=True)
 subprocess.run([sys.executable,str(B/'scripts'/'orchestrate.py'),'--devices','cuda:0','--poll-seconds','15'],cwd=str(V.parent),check=True)
 print(json.dumps({'status':'A_AND_B_INFERENCE_COMPLETE'}),flush=True)
if __name__=='__main__':main()
