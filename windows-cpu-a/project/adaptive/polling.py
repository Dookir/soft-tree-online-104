"""Evaluate new points on multiple transfer lengths; never interpolate labels."""
import heapq
import numpy as np
from .base import CONFIG
from .protocol import weight_key

STEPS=(.0005,.002,.008,.03,.08)


def legal_directions(center):
    center=center.reshape(3,7)
    groups=[]
    for b in range(3):
        groups.append([(b,int(i),j) for i in np.flatnonzero(center[b]>=.0005) for j in range(7) if i!=j])
    # Interleave buildings rather than scanning all directions of one first.
    result=[]
    for k in range(max(map(len,groups))):
        result.extend(g[k] for g in groups if k<len(g))
    return result


def moved(center,direction,requested):
    b,i,j=direction;w=center.reshape(3,7).copy();delta=min(float(requested),float(w[b,i]))
    if delta<.0005-1e-14:return None,None
    w[b,i]-=delta;w[b,j]+=delta
    return w.reshape(21),float(delta)


def direction_order(context,x,y,round_number,online,anchoring):
    center_id=int(np.argmin(y));center=x[center_id];directions=legal_directions(center)
    eligible=set(directions);success={}
    for row in online:
        if row['round']<round_number-3:continue
        m=row.get('acquisition',{});parent=m.get('center')
        if parent is None or parent>=len(y) or not all(k in m for k in ('building','donor','receiver')):continue
        direction=(int(m['building']),int(m['donor']),int(m['receiver']))
        gain=float(y[parent]-row['R'])
        if gain>=.02 and direction in eligible:success[direction]=max(success.get(direction,0.),gain)
    ordered=sorted(success,key=lambda d:(-success[d],d))[:2]
    labels={d:'observed_improvement' for d in ordered}
    probes=np.asarray([moved(center,d,.008)[0] for d in directions])
    raw=context.shadow(probes)
    scores=raw['anchor_score'] if anchoring else raw['raw_score']
    for i in np.argsort(scores,kind='stable'):
        if len(ordered)>=2:break
        d=directions[i]
        if d not in ordered:
            ordered.append(d);labels[d]='model_direction'
        if len(ordered)>=2:break
    offset=((round_number-1)*17)%len(directions)
    rotated=directions[offset:]+directions[:offset]
    for d in rotated:
        if d not in ordered:
            ordered.append(d);labels[d]='rotating_direction'
        if len(ordered)>=4:break
    for d in rotated:
        if d not in ordered:ordered.append(d);labels[d]='rotating_refill'
    return center_id,ordered,labels


def scan(context,x,y,round_number,online,seen,count=20,anchoring=False):
    parent,ordered,labels=direction_order(context,x,y,round_number,online,anchoring)
    center=x[parent];points=[];details=[];used=set(seen)
    def add(direction,step,selection):
        w,actual=moved(center,direction,step)
        if w is None:return False
        key=weight_key(w.tolist())
        if key in used:return False
        points.append(w);used.add(key)
        b,i,j=direction
        details.append(dict(center=parent,parent_R=float(y[parent]),building=b,donor=i,receiver=j,
                            requested_delta=float(step),delta=actual,selection=selection,
                            direction=list(direction),score_status='awaiting_real_evaluation'))
        return True
    for direction in ordered:
        for step in STEPS:
            add(direction,step,labels[direction])
            if len(points)==count:return np.asarray(points),details
    # All fixed levels may have been observed. Bisect largest remaining legal
    # intervals; these are new evaluation requests, not interpolated R values.
    intervals=[]
    for k,direction in enumerate(ordered):
        cap=float(center.reshape(3,7)[direction[0],direction[1]])
        levels=sorted({min(s,cap) for s in STEPS if min(s,cap)>=.0005-1e-14})
        for lo,hi in zip(levels,levels[1:]):
            if hi-lo>1e-10:heapq.heappush(intervals,(-(hi-lo),k,lo,hi))
    for _ in range(20000):
        if not intervals:break
        _,k,lo,hi=heapq.heappop(intervals);mid=(lo+hi)/2
        add(ordered[k],mid,'bisected_step_interval')
        if len(points)==count:return np.asarray(points),details
        for a,b in ((lo,mid),(mid,hi)):
            if b-a>1e-10:heapq.heappush(intervals,(-(b-a),k,a,b))
    raise RuntimeError('Cannot create20 unseen legal multi-step transfer probes')
