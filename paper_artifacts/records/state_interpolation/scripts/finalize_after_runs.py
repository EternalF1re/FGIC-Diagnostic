from __future__ import annotations
import subprocess,sys,time
from pathlib import Path
A=Path(__file__).resolve().parents[1];B=A.parent/'phase2d_bn_refresh_diagnostic'
def main():
 while not (A/'orchestrator_summary.json').exists():time.sleep(20)
 subprocess.run([sys.executable,str(A/'scripts'/'finalize.py')],cwd=str(A.parent.parent),check=True)
 while not (B/'orchestrator_summary.json').exists():time.sleep(20)
 subprocess.run([sys.executable,str(B/'scripts'/'finalize.py')],cwd=str(B.parent.parent),check=True)
if __name__=='__main__':main()
