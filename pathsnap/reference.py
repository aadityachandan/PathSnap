"""Reference-only utilities. Never called by feature extraction or route prediction.
Pointwise labels are derived, NOT independently supplied GPS-level ground truth.
"""
from pathlib import Path
import numpy as np
from .data import project, motion


def load_reference(folder, n_arcs):
    folder=Path(folder)
    route=np.loadtxt(folder/f'{folder.name}.route',dtype=np.int64,ndmin=1)
    if not len(route) or route.min()<0 or route.max()>=n_arcs:
        raise ValueError('Invalid reference arc IDs')
    return route


def align_reference(trip, route):
    """Monotone DP over ordered reference-segment positions, allowing skipped arcs.
    Cost = (distance/20m)^2 + 2*(1-heading cosine) + .1*forward index jump.
    Start and end positions are free. Ties retain earlier reference position.
    """
    a=trip.xy[trip.arcs[route,0]]; b=trip.xy[trip.arcs[route,1]]
    directions,speed,_=motion(trip)
    vectors=(b-a)/np.maximum(trip.lengths[route,None],1e-9)
    m=len(route); n=len(trip.gps); back=np.empty((n,m),dtype=np.int32)
    index=np.arange(m); prev=np.zeros(m)
    for t,p in enumerate(trip.gps):
        d,_=project(p,a,b)
        emission=(d/20)**2
        if speed[t]>=0.5:emission+=2*(1-vectors@directions[t])
        if t==0:prev=emission;back[t]=index;continue
        adjusted=prev-0.1*index
        mins=np.minimum.accumulate(adjusted)
        best=np.maximum.accumulate(np.where(adjusted==mins,index,0))
        back[t]=best;prev=emission+mins+0.1*index
        prev-=prev.min()
    pos=np.empty(n,dtype=int);pos[-1]=int(np.argmin(prev))
    for t in range(n-1,0,-1):pos[t-1]=back[t,pos[t]]
    labels=route[pos]
    distance,_=project(trip.gps,trip.xy[trip.arcs[labels,0]],trip.xy[trip.arcs[labels,1]])
    return labels,distance
