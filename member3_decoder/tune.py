"""Decoder development sweep (beta x backward tolerance); never uses evaluator data.

beta: how strongly the HMM penalises a mismatch between road-network distance and
GPS straight-line displacement. backward_tolerance: how many metres the projected
position may slip backwards on the same road (GPS noise) and still count as staying put.
"""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from pathsnap.data import load_trip
from pathsnap.context import model_input
from pathsnap.decoder import decode
from pathsnap.metrics import route_metrics

BETAS=[10.,20.,35.,70.,140.,350.]
TOLERANCES=[0.,5.,10.,20.,40.]

def run(root,cache,artifacts,out):
    artifacts=Path(artifacts);out=Path(out);out.mkdir(exist_ok=True,parents=True)
    model=joblib.load(artifacts/'final_model.joblib');meta=json.loads((artifacts/'metadata.json').read_text());records=[]
    start=time.perf_counter()
    for file in sorted((Path(cache)/'student_test').glob('*.npz')):
        tr=load_trip(Path(root)/'student_test'/file.stem);z=np.load(file)
        p=model.predict_proba(model_input(tr,z['ids'],z['features']))[:,1].reshape(z['ids'].shape)
        for tol in TOLERANCES:
            for beta in BETAS:
                pred,pieces,breaks=decode(tr,z['ids'],z['features'],p,beta=beta,gap_seconds=30.,backward_tolerance=tol)
                records.append(dict(trip=tr.id,beta=beta,backward_tolerance=tol,observations=len(pred),breaks=len(breaks),**route_metrics(tr,pred,z['labels'],z['reference'],[a for piece in pieces for a in piece])))
        print('decoder sweep',tr.id,flush=True)
    frame=pd.DataFrame(records);frame.to_csv(out/'decoder_trials.csv',index=False)
    means=frame.groupby(['backward_tolerance','beta']).top1_accuracy.mean();best_tol,best_beta=means.idxmax()
    summary={'selection_data':'10 student-test development trips; not an untouched test estimate','objective':'mean per-trip derived-reference Top-1',
        'beta_grid':BETAS,'backward_tolerance_grid_m':TOLERANCES,
        'mean_top1':{f'tol={t:g},beta={b:g}':v for (t,b),v in means.items()},
        'selected_beta':float(best_beta),'selected_backward_tolerance_m':float(best_tol),'seconds':time.perf_counter()-start}
    (out/'decoder_search.json').write_text(json.dumps(summary,indent=2))
    meta['decoder']={'beta':float(best_beta),'gap_seconds':30.,'backward_tolerance':float(best_tol)};meta['decoder_selection']=summary
    (artifacts/'metadata.json').write_text(json.dumps(meta,indent=2));print('selected beta',best_beta,'backward tolerance',best_tol,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--cache',default='work/cache');p.add_argument('--artifacts',default='artifacts');p.add_argument('--out',default='results');a=p.parse_args();run(a.data_root,a.cache,a.artifacts,a.out)
