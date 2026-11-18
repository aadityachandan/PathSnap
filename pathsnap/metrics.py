"""Evaluation on derived point labels and supplied route sequences."""
from difflib import SequenceMatcher
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix


def collapse(values):
    values=np.asarray(values)
    return values[np.r_[True,values[1:]!=values[:-1]]].tolist() if len(values) else []


def route_metrics(trip,pred,labels,reference,sequence=None):
    seq=collapse(pred) if sequence is None else sequence
    a=set(seq);b=set(reference.tolist());overlap=len(a&b)
    p=overlap/max(1,len(a));r=overlap/max(1,len(b))
    # This is strict adjacency, not reachability across intermediate segments.
    adjacent=(pred[:-1]==pred[1:]) | (trip.arcs[pred[:-1],1]==trip.arcs[pred[1:],0])
    return {'top1_accuracy':float(np.mean(pred==labels)),
      'strict_adjacency':float(np.mean(adjacent)) if len(adjacent) else 1.,
      'route_arc_set_f1':2*p*r/max(p+r,1e-12),
      'sequence_similarity':SequenceMatcher(None,seq,reference.tolist(),autojunk=False).ratio()}


def candidate_metrics(y,p):
    hard=p>=0.5
    return {'candidate_accuracy':accuracy_score(y,hard),
      'macro_precision':precision_score(y,hard,average='macro',zero_division=0),
      'macro_recall':recall_score(y,hard,average='macro',zero_division=0),
      'macro_f1':f1_score(y,hard,average='macro',zero_division=0),
      'log_loss':log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),
      'confusion_matrix':confusion_matrix(y,hard,labels=[0,1]).tolist()}


def top1_from_candidates(y,p,k=6):
    """Per-GPS-observation Top-1: rows are grouped k candidates per observation."""
    y=np.asarray(y).reshape(-1,k);p=np.asarray(p).reshape(-1,k)
    return float(np.mean(y[np.arange(len(y)),p.argmax(axis=1)]==1))
