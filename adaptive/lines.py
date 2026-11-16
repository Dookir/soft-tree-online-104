"""Identity and real observations of a one-building simplex transfer line."""
import hashlib
import json
import numpy as np
from .protocol import weight_key


def normalized(weights):
    w=np.asarray(weights,dtype=float).reshape(3,7).copy()
    w/=w.max(1,keepdims=True);w/=w.sum(1,keepdims=True)
    return w


def identity(weights,building,i,j):
    i,j=sorted((int(i),int(j)));w=normalized(weights);total=float(w[building,i]+w[building,j])
    w[building,i]=total;w[building,j]=0.
    key=hashlib.sha256(json.dumps([int(building),i,j,np.round(w,12).tolist()],separators=(',',':')).encode()).hexdigest()
    return key,w,total


def point_on_line(base,building,i,j,t):
    i,j=sorted((int(i),int(j)));w=normalized(base);total=float(w[building,i]+w[building,j])
    if not 0<=t<=total:raise ValueError('Line coordinate outside simplex')
    w[building,i]=total-t;w[building,j]=t
    return w.reshape(21)


def transfer_metadata(parent,point,parent_index,y_parent,building,i,j):
    a=normalized(parent);b=normalized(point);i,j=sorted((int(i),int(j)))
    delta=float(b[building,j]-a[building,j]);donor,receiver=(i,j) if delta>=0 else (j,i)
    key,_,_=identity(a,building,i,j)
    return dict(center=int(parent_index),parent_R=float(y_parent),building=int(building),donor=donor,receiver=receiver,
                delta=abs(delta),line_key=key)


def registry(x,y,online):
    lines={}
    for position,row in enumerate(online):
        meta=row.get('acquisition',{});parent=meta.get('center')
        if parent is None or not all(k in meta for k in ('building','donor','receiver')):continue
        if not 0<=parent<2048+50*(row['round']-1):raise RuntimeError('Line parent is not past feedback')
        b=int(meta['building']);i,j=sorted((int(meta['donor']),int(meta['receiver'])))
        if i==j:continue
        key,base,total=identity(x[parent],b,i,j);key2,_,_=identity(row['weights'],b,i,j)
        if key!=key2:continue  # Never mix observations whose other weights changed.
        record=lines.setdefault(key,dict(key=key,base=base,building=b,i=i,j=j,total=total,observations={}))
        for index,weights,R in ((parent,x[parent],y[parent]),(2048+position,row['weights'],row['R'])):
            t=float(normalized(weights)[b,j]);record['observations'][weight_key(np.asarray(weights).tolist())]=dict(index=index,t=t,R=float(R))
    return lines


def boundary_candidates(x,y,online):
    candidates=[]
    for line in registry(x,y,online).values():
        observations=sorted(line['observations'].values(),key=lambda v:(v['t'],v['index']))
        for left,right in zip(observations,observations[1:]):
            width=right['t']-left['t'];gap=abs(right['R']-left['R'])
            if width<=1e-10 or width>.16 or gap<.02 or min(left['R'],right['R'])>float(y.min())+2.:continue
            t=(left['t']+right['t'])/2;w=point_on_line(line['base'],line['building'],line['i'],line['j'],t)
            parent=min((left,right),key=lambda v:(v['R'],v['index']))
            meta=transfer_metadata(x[parent['index']],w,parent['index'],parent['R'],line['building'],line['i'],line['j'])
            meta.update(selection='observed_R_interval_midpoint',interval_endpoints=[left,right],
                        interval_width=width,observed_R_gap=gap,utility=gap/np.sqrt(width+.0005))
            candidates.append((w,meta))
    return sorted(candidates,key=lambda pair:(-pair[1]['utility'],pair[1]['line_key']))
