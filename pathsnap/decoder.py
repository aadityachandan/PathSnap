"""Member 3: directed road graph and log-space Viterbi decoder."""
import heapq
from collections import OrderedDict
import numpy as np

class RoadGraph:
    def __init__(self,trip):
        self.trip=trip;self.adj=[[] for _ in trip.nodes];self.cache=OrderedDict()
        for arc,(u,v) in enumerate(trip.arcs):
            self.adj[u].append((int(v),float(max(trip.lengths[arc],1e-6)),arc))

    def search(self,source,limit):
        limit=int(np.ceil(limit/250)*250);key=(int(source),limit)
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        dist={int(source):0.};prev={};queue=[(0.,int(source))]
        while queue:
            cost,u=heapq.heappop(queue)
            if cost!=dist[u]:continue
            for v,w,arc in self.adj[u]:
                val=cost+w
                if val<=limit and val<dist.get(v,float('inf')):
                    dist[v]=val;prev[v]=(u,arc);heapq.heappush(queue,(val,v))
        self.cache[key]=(dist,prev)
        if len(self.cache)>4096:self.cache.popitem(last=False)
        return dist,prev

    def between(self,left,lf,right,rf,limit,with_path=False,backward_tolerance=20.):
        tr=self.trip
        # Staying on the same road: GPS noise can make the projected position slip slightly
        # backwards between seconds. Treat a backward slip up to backward_tolerance metres as
        # 'no movement' instead of forcing a loop around the network (which caused false breaks).
        if left==right and (rf-lf)*tr.lengths[left]>=-backward_tolerance:
            return max(0.,float((rf-lf)*tr.lengths[left])),[]
        source=int(tr.arcs[left,1]);target=int(tr.arcs[right,0])
        dist,prev=self.search(source,limit)
        if target not in dist:return float('inf'),[]
        length=(1-lf)*tr.lengths[left]+dist[target]+rf*tr.lengths[right]
        path=[]
        if with_path:
            u=target
            while u!=source:
                u0,arc=prev[u];path.append(arc);u=u0
            path.reverse()
        return float(length),path


def decode(trip,ids,features,probabilities,beta=35.,gap_seconds=30.,backward_tolerance=20.):
    """Impossible transitions are -inf. If all paths fail, restart explicitly.
    Returns separate connected pieces instead of drawing fictitious road links.
    """
    graph=RoadGraph(trip);n,k=ids.shape
    p=np.clip(probabilities,1e-9,1);p=p/p.sum(axis=1,keepdims=True)
    emission=np.log(p);dp=emission[0].copy()
    back=np.zeros((n,k),dtype=np.int16);reset=np.zeros(n,dtype=bool)
    for t in range(1,n):
        dt=trip.track[t,2]-trip.track[t-1,2]
        displacement=np.linalg.norm(trip.gps[t]-trip.gps[t-1]);limit=min(5000.,max(250.,displacement*3+100))
        trans=np.full((k,k),-np.inf)
        if dt<=gap_seconds:
            for i in range(k):
                for j in range(k):
                    length,_=graph.between(ids[t-1,i],features[t-1,i,3],ids[t,j],features[t,j,3],limit,backward_tolerance=backward_tolerance)
                    if np.isfinite(length) and length/max(dt,0.01)<=65:
                        trans[i,j]=-abs(length-displacement)/beta
        scores=dp[:,None]+trans
        best=scores.argmax(axis=0);new=scores[best,np.arange(k)]+emission[t]
        if not np.isfinite(new).any():
            reset[t]=True;back[t]=int(dp.argmax());new=emission[t].copy()
        else:back[t]=best
        dp=new-np.max(new)
    state=np.empty(n,dtype=int);state[-1]=int(dp.argmax())
    for t in range(n-1,0,-1):state[t-1]=back[t,state[t]]
    selected=ids[np.arange(n),state]
    pieces=[[int(selected[0])]];breaks=[]
    for t in range(1,n):
        if reset[t]:
            pieces.append([int(selected[t])]);breaks.append(t);continue
        displacement=np.linalg.norm(trip.gps[t]-trip.gps[t-1]);limit=min(5000.,max(250.,displacement*3+100))
        _,path=graph.between(selected[t-1],features[t-1,state[t-1],3],selected[t],features[t,state[t],3],limit,True,backward_tolerance)
        for arc in path+[int(selected[t])]:
            if arc!=pieces[-1][-1]:pieces[-1].append(arc)
    return selected,pieces,breaks


def geometry(trip,pieces):
    # Split any nonadjacent segment pair, even in classifier-only visualisations.
    output=[]
    for piece in pieces:
        line=[];last=None
        for arc in piece:
            u,v=trip.arcs[arc]
            if last is not None and u!=last:
                if line:output.append(line)
                line=[]
            if not line:line.append(trip.nodes[u,::-1].tolist())
            line.append(trip.nodes[v,::-1].tolist());last=v
        if line:output.append(line)
    return output
