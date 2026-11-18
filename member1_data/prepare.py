"""Run from project root: python -m member1_data.prepare --data-root ..."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import csv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap.data import load_trip,candidates,FEATURES
from pathsnap.reference import load_reference,align_reference


def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);inventory=[]
    for split in ['train','student_test']:
        dest=out/split;dest.mkdir(exist_ok=True)
        for folder in sorted((root/split).iterdir()):
            if not folder.is_dir():continue
            start=time.perf_counter();tr=load_trip(folder)
            reference=load_reference(folder,len(tr.arcs))
            target=dest/f'{tr.id}.npz'
            digest=hashlib.sha256()
            for ext in ['track','nodes','arcs','route']:digest.update((folder/f'{tr.id}.{ext}').read_bytes())
            fingerprint=digest.hexdigest()
            if target.exists():
                z=np.load(target)
                if str(z['source_sha256'])!=fingerprint:raise ValueError('Raw data changed; remove stale cache')
                ids=z['ids'];features=z['features'];labels=z['labels'];distance=z['reference_distance']
            else:
                ids,features=candidates(tr);labels,distance=align_reference(tr,reference)
                np.savez_compressed(target,ids=ids,features=features,labels=labels,
                    reference_distance=distance,reference=reference,source_sha256=fingerprint)
            g=csr_matrix((np.ones(len(tr.arcs)),(tr.arcs[:,0],tr.arcs[:,1])),shape=(len(tr.nodes),len(tr.nodes)))
            components,_=connected_components(g,directed=True,connection='weak')
            dt=np.diff(tr.track[:,2]);degree=np.bincount(tr.arcs.ravel(),minlength=len(tr.nodes))
            inventory.append(dict(split=split,trip=tr.id,gps_points=len(tr.track),nodes=len(tr.nodes),arcs=len(tr.arcs),route_arcs=len(reference),
                missing_values=0,nonpositive_steps=int(np.sum(dt<=0)),gaps_over_5s=int(np.sum(dt>5)),max_gap_s=float(dt.max()) if len(dt) else 0,
                median_step_s=float(np.median(dt)) if len(dt) else 0,weak_components=components,isolated_nodes=int(np.sum(degree==0)),
                gps_path_m=float(np.linalg.norm(np.diff(tr.gps,axis=0),axis=1).sum()),
                lon_min=float(tr.track[:,0].min()),lon_max=float(tr.track[:,0].max()),lat_min=float(tr.track[:,1].min()),lat_max=float(tr.track[:,1].max()),
                alignment_within_50m=float(np.mean(distance<=50)),candidate_miss_rate=float(np.mean(~np.any(ids==labels[:,None],axis=1))),source_sha256=fingerprint))
            print(f'{split}/{tr.id}: {len(tr.track)} points, miss {inventory[-1]["candidate_miss_rate"]:.3f}, {time.perf_counter()-start:.1f}s',flush=True)
    with open(out/'inventory.csv','w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    (out/'schema.json').write_text(json.dumps({'features':FEATURES,'label':'monotone reference alignment; derived, not independent truth','k':6},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--out',default='work/cache');a=p.parse_args();run(a.data_root,a.out)
