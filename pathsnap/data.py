"""Member 1: raw inputs, metric geometry, candidate features; NO route access here."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

FEATURES = ['distance_m', 'heading_cos', 'segment_length_m', 'projection_fraction',
            'speed_mps', 'time_gap_s', 'distance_rank', 'nearest_margin_m']
K = 6

@dataclass
class Trip:
    id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    xy: np.ndarray
    gps: np.ndarray
    lengths: np.ndarray
    origin: np.ndarray
    scale: np.ndarray


def load_trip(folder):
    folder = Path(folder); tid = folder.name
    track = np.loadtxt(folder / f'{tid}.track', ndmin=2)
    nodes = np.loadtxt(folder / f'{tid}.nodes', ndmin=2)
    arcs = np.loadtxt(folder / f'{tid}.arcs', dtype=np.int64, ndmin=2)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f'{tid}: invalid file column count')
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f'{tid}: non-finite coordinates/timestamps')
    if len(track) == 0 or len(arcs) == 0 or np.any(np.diff(track[:, 2]) <= 0):
        raise ValueError(f'{tid}: empty input or timestamps not strictly increasing')
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError(f'{tid}: invalid node index')
    if np.any(np.abs(nodes[:,0])>180) or np.any(np.abs(nodes[:,1])>90):
        raise ValueError(f'{tid}: invalid longitude/latitude')
    origin = np.median(track[:, :2], axis=0)
    scale = np.array([111195.08 * np.cos(np.deg2rad(origin[1])), 111195.08])
    xy = (nodes-origin)*scale; gps = (track[:,:2]-origin)*scale
    lengths = np.linalg.norm(xy[arcs[:,1]]-xy[arcs[:,0]], axis=1)
    return Trip(tid, track, nodes, arcs, xy, gps, lengths, origin, scale)


def motion(trip):
    delta = np.diff(trip.gps, axis=0, prepend=trip.gps[:1])
    if len(delta)>1: delta[0] = delta[1]
    dt = np.diff(trip.track[:,2], prepend=trip.track[0,2]-1)
    speed = np.linalg.norm(delta, axis=1)/np.maximum(dt, 0.01)
    direction = delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-9)
    return direction, speed, dt


def project(points, starts, ends):
    vec=ends-starts
    frac=np.sum((points-starts)*vec,axis=-1)/np.maximum(np.sum(vec*vec,axis=-1),1e-12)
    frac=np.clip(frac,0,1)
    projected=starts+frac[...,None]*vec
    return np.linalg.norm(points-projected,axis=-1), frac


def candidates(trip, k=K):
    # Densification prevents long segments being missed by a midpoint-only index.
    a=trip.xy[trip.arcs[:,0]]; b=trip.xy[trip.arcs[:,1]]
    counts=np.maximum(1,np.ceil(trip.lengths/40).astype(int))
    ids=np.repeat(np.arange(len(a)),counts+1)
    offsets=np.repeat(np.cumsum(counts+1)-(counts+1),counts+1)
    fractions=(np.arange(len(ids))-offsets)/np.repeat(counts,counts+1)
    samples=a[ids]+fractions[:,None]*(b[ids]-a[ids])
    tree=cKDTree(samples)
    k=min(k,len(a)); result=np.empty((len(trip.gps),k),dtype=np.int32)
    # Exact top-k segment search: start from samples, then expand radius by the
    # maximum 20m sample-to-segment error. Every closer segment has a sample inside.
    for t,p in enumerate(trip.gps):
        q=min(64,len(samples))
        while True:
            _,ix=tree.query(p,k=q); pool=np.unique(ids[np.atleast_1d(ix)])
            if len(pool)>=k or q==len(samples):break
            q=min(q*2,len(samples))
        dist,_=project(p,a[pool],b[pool]); radius=np.partition(dist,k-1)[k-1]+20.000001
        pool=np.unique(ids[tree.query_ball_point(p,radius)])
        dist,_=project(p,a[pool],b[pool])
        result[t]=pool[np.lexsort((pool,dist))[:k]]
    starts=a[result]; ends=b[result]
    distance, fraction=project(trip.gps[:,None,:],starts,ends)
    direction,speed,dt=motion(trip)
    vec=(ends-starts)/np.maximum(trip.lengths[result][...,None],1e-9)
    cosine=np.sum(vec*direction[:,None,:],axis=2)
    cosine[speed<0.5]=0
    shape=result.shape
    features=np.stack([distance,cosine,trip.lengths[result],fraction,
        np.broadcast_to(speed[:,None],shape),np.broadcast_to(dt[:,None],shape),
        np.broadcast_to(np.arange(k),shape),distance-distance[:,:1]],axis=-1).astype(np.float32)
    return result,features
