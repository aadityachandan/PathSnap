"""Reproduce the local project. Evaluator data is never used by this command."""
import argparse,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--trials',type=int,default=30);p.add_argument('--max-observations',type=int,default=400);p.add_argument('--skip-prepare',action='store_true');p.add_argument('--skip-train',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;os.chdir(root)
    env=os.environ.copy();env.setdefault('MPLCONFIGDIR',str(root/'work'/'mpl'));env.setdefault('OMP_NUM_THREADS','2');env.setdefault('OPENBLAS_NUM_THREADS','2')
    cmds=[]
    if not a.skip_prepare:cmds.append(['-m','member1_data.prepare','--data-root',a.data_root])
    if not a.skip_train:cmds.append(['-m','member2_models.train','--data-root',a.data_root,'--trials',str(a.trials),'--max-observations',str(a.max_observations)])
    cmds.extend([['-m','member3_decoder.tune','--data-root',a.data_root],['-m','member3_decoder.evaluate','--data-root',a.data_root],['-m','report.build','--data-root',a.data_root]])
    for cmd in cmds:subprocess.run([sys.executable,*cmd],check=True,env=env)
if __name__=='__main__':main()
