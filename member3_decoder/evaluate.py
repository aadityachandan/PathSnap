"""Evaluate frozen classifier and decoder on all student-test observations."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from pathsnap.data import load_trip
from pathsnap.context import model_input
from pathsnap.decoder import decode,geometry
from pathsnap.metrics import route_metrics
from pathsnap.maps import make_map


def run(root,cache,artifacts,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'maps').mkdir(exist_ok=True);(out/'predictions').mkdir(exist_ok=True)
    model=joblib.load(Path(artifacts)/'final_model.joblib');meta=json.loads((Path(artifacts)/'metadata.json').read_text());rows=[]
    for file in sorted((Path(cache)/'student_test').glob('*.npz')):
        tr=load_trip(Path(root)/'student_test'/file.stem);z=np.load(file)
        start=time.perf_counter();p=model.predict_proba(model_input(tr,z['ids'],z['features']))[:,1].reshape(z['ids'].shape)
        raw=z['ids'][np.arange(len(p)),p.argmax(axis=1)];classifier_time=time.perf_counter()-start
        start=time.perf_counter();pred,pieces,breaks=decode(tr,z['ids'],z['features'],p,**meta['decoder']);decoder_time=time.perf_counter()-start
        for mode,values,seq in [('classifier',raw,None),('hmm',pred,[a for piece in pieces for a in piece])]:
            metrics=route_metrics(tr,values,z['labels'],z['reference'],seq)
            rows.append(dict(trip=tr.id,mode=mode,observations=len(pred),classifier_seconds=classifier_time,decoder_seconds=decoder_time if mode=='hmm' else 0,
                explicit_breaks=len(breaks) if mode=='hmm' else None,**metrics))
        payload={'trip_id':tr.id,'predicted_arc_ids':pred.tolist(),'route_pieces':pieces,'break_observations':breaks,'route_latlon':geometry(tr,pieces)}
        (out/'predictions'/f'{tr.id}.json').write_text(json.dumps(payload))
        make_map(tr,pieces,z['reference']).save(str(out/'maps'/f'{tr.id}.html'))
        pd.DataFrame(rows).to_csv(out/'decoder_comparison.csv',index=False)
        print(f'{tr.id}: raw {rows[-2]["top1_accuracy"]:.3f}, HMM {rows[-1]["top1_accuracy"]:.3f}, {len(breaks)} breaks, decode {decoder_time:.1f}s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--cache',default='work/cache');p.add_argument('--artifacts',default='artifacts');p.add_argument('--out',default='results');a=p.parse_args();run(a.data_root,a.cache,a.artifacts,a.out)
