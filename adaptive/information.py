"""Local contrast pairs, measured-change intervals and competitive disagreement."""
import numpy as np
import torch
from .base import CONFIG,predict,project
from .anchor import AnchorContext
from .lines import normalized,identity,transfer_metadata,boundary_candidates
from .protocol import weight_key


def splits(round_number):
    for lo,hi,exploit,learn in CONFIG['active_sampling']['role_schedule']:
        if lo<=round_number<=hi:return exploit,learn,tuple(CONFIG['active_sampling']['info_splits'][str(learn)])
    raise ValueError('Round must be1..20')


def good_centers(context,y):
    return [int(i) for i in context.ids if y[i]<=y.min()+2.]


def pair_pool(context,x,y,seen,round_number,rng):
    pool=[];metas=[]
    for center in good_centers(context,y):
        parent=normalized(x[center])
        for b in range(3):
            for i in range(7):
                for j in range(i+1,7):
                    line_key,_,_=identity(parent,b,i,j)
                    for nominal in CONFIG['active_sampling']['pair_steps']:
                        if min(parent[b,i],parent[b,j])>=.0005:
                            delta=min(nominal,float(parent[b,i]),float(parent[b,j]))
                            a=parent.copy();c=parent.copy();a[b,i]-=delta;a[b,j]+=delta;c[b,j]-=delta;c[b,i]+=delta
                            kind='symmetric'
                        else:
                            donor,receiver=(i,j) if parent[b,i]>=parent[b,j] else (j,i)
                            d1=min(nominal,float(parent[b,donor]));d2=min(2*nominal,float(parent[b,donor]))
                            if d1<.0005 or d2-d1<1e-10:continue
                            a=parent.copy();c=parent.copy();a[b,donor]-=d1;a[b,receiver]+=d1;c[b,donor]-=d2;c[b,receiver]+=d2
                            kind='one_sided'
                        points=np.array([a.reshape(21),c.reshape(21)])
                        keys=[weight_key(w.tolist()) for w in points]
                        if keys[0]==keys[1] or any(k in seen for k in keys):continue
                        if max(np.linalg.norm(w-x[center]) for w in points)>.2:continue
                        pool.extend(points);metas.append(dict(center=center,building=b,i=i,j=j,line_key=line_key,pair_kind=kind))
    if not metas:return []
    prediction=predict(context.models,np.asarray(pool)).reshape(5,-1,2)
    difference=prediction[:,:,0]-prediction[:,:,1]
    votes=(difference>1e-10).mean(0)+.5*(np.abs(difference)<=1e-10).mean(0)
    std=difference.std(0);utility=std*(1+4*votes*(1-votes))
    tie=rng.permutation(len(metas));order=np.lexsort((tie,-utility))
    result=[]
    for n in order:
        meta=metas[n];meta.update(utility=float(utility[n]),comparison_std=float(std[n]),positive_vote_fraction=float(votes[n]),
                                comparison_mean=float(difference[:,n].mean()))
        result.append((np.asarray(pool[2*n:2*n+2]),meta))
    return result


def competitive_pool(context,x,y,original_points,rng):
    points=list(original_points)
    for center in good_centers(context,y):
        for std in (.005,.02,.05):
            trials=x[center].reshape(1,3,7)+rng.normal(0,std,(256,3,7))
            points.extend(project(torch.tensor(trials)).numpy().reshape(-1,21))
    # Combinations are predicted as complete district weights, never separate R sums.
    ids=np.argsort(y,kind='stable')[:128];ids=ids[y[ids]<=y.min()+2.]
    blocks=[]
    for b in range(3):
        chosen=[]
        for i in ids:
            w=x[i].reshape(3,7)[b]
            if not chosen or np.linalg.norm(np.asarray(chosen)-w,axis=1).min()>=.01:
                chosen.append(w)
                if len(chosen)==8:break
        blocks.append(chosen)
    import itertools
    points.extend(np.concatenate(v) for v in itertools.product(*blocks))
    points=np.asarray(points);shadow=context.shadow(points)
    eligible=np.asarray(shadow['eligible'],dtype=bool)
    centers=np.asarray(shadow['center_indices'])
    eligible &= y[centers]<=y.min()+2.
    return points,shadow,eligible


def generate(models,x,y,round_number,online,seen,original_points,selected,anchoring=False):
    _,count,allocation=splits(round_number)
    if count==0:return np.empty((0,21)),[],[],dict(requested=0,actual={})
    rng=np.random.default_rng(CONFIG['active_sampling']['seed']+round_number*1009)
    context=AnchorContext(models,x,y);used=set(seen)
    points=[];sources=[];details=[]
    def add(w,source,meta):
        key=weight_key(w.tolist())
        if key in used:return False
        points.append(w.copy());sources.append(source);details.append(dict(meta,role='learn'))
        used.add(key);return True
    pair_target,boundary_target,competitive_target=allocation
    pair_lines=set();pair_count=0
    for pair,meta in pair_pool(context,x,y,used,round_number,rng):
        if pair_count>=pair_target:break
        if meta['line_key'] in pair_lines or any(weight_key(w.tolist()) in used for w in pair):continue
        pair_id=f'r{round_number:02d}-pair-{pair_count//2+1:02d}'
        for member,w in enumerate(pair):
            m=transfer_metadata(x[meta['center']],w,meta['center'],y[meta['center']],meta['building'],meta['i'],meta['j'])
            m.update(meta,pair_id=pair_id,pair_member=member,selection='committee_comparison_pair')
            add(w,'learn_pair',m)
        pair_lines.add(meta['line_key']);pair_count+=2
    # Pair construction is atomic. If all primary pairs were already measured,
    # fill with new legal paired steps; do not silently emit an unmatched pair.
    for _ in range(4000):
        if pair_count>=pair_target:break
        center=good_centers(context,y)[int(rng.integers(len(good_centers(context,y))))]
        b=int(rng.integers(3));parent=normalized(x[center]);donor=int(parent[b].argmax());receiver=(donor+1+int(rng.integers(6)))%7
        delta=float(rng.uniform(.001,.04));a=parent.copy();c=parent.copy()
        a[b,donor]-=delta;a[b,receiver]+=delta;c[b,donor]-=2*delta;c[b,receiver]+=2*delta
        pair=np.array([a.reshape(21),c.reshape(21)])
        if any(weight_key(w.tolist()) in used for w in pair):continue
        pair_id=f'r{round_number:02d}-pair-{pair_count//2+1:02d}'
        for member,w in enumerate(pair):
            m=transfer_metadata(x[center],w,center,y[center],b,donor,receiver)
            m.update(pair_id=pair_id,pair_member=member,pair_kind='one_sided',selection='legal_pair_refill')
            add(w,'learn_pair',m)
        pair_count+=2
    if pair_count!=pair_target:raise RuntimeError('Unable to create all requested information pairs')
    boundary_count=0;boundary_lines=set()
    for w,meta in boundary_candidates(x,y,online):
        if boundary_count>=boundary_target:break
        if meta['line_key'] in boundary_lines:continue
        if add(w,'learn_boundary',meta):boundary_count+=1;boundary_lines.add(meta['line_key'])
    # The required boundary slots may not exist before the first feedback.
    # Their explicit fallback is competitive local disagreement, not new labels.
    q,shadow,eligible=competitive_pool(context,x,y,original_points,rng)
    mean=np.asarray(shadow['anchor_mean'] if anchoring else shadow['raw_mean'])
    if eligible.any():
        threshold=float(y.min()+2.) if anchoring else float(mean[eligible].min()+2.)
        preferred=eligible & (mean<=threshold)
        if preferred.sum()>=count-len(points):eligible=preferred
    dispersion=np.asarray(shadow['anchor_predictions']).std(1)
    need=count-len(points)
    known=np.concatenate([np.asarray(selected).reshape(-1,21),np.asarray(points).reshape(-1,21)])
    distance=np.linalg.norm(q[:,None]-known[None],axis=2).min(1)
    for _ in range(need):
        utility=dispersion*(.25+.75*np.minimum(distance/.05,1.))
        order=np.lexsort((np.arange(len(q)),mean,-utility))
        picked=None
        for i in order:
            if eligible[i] and weight_key(q[i].tolist()) not in used:picked=int(i);break
        if picked is None:raise RuntimeError('Competitive information pool exhausted before50-point submission')
        i=picked;center=int(shadow['center_indices'][i])
        meta=dict(center=center,parent_R=float(y[center]),utility=float(utility[i]),difference_std=float(dispersion[i]),
                  center_distance=float(shadow['center_distance'][i]),selection='competitive_difference_disagreement',
                  anchor_used=anchoring,boundary_refill=boundary_count<boundary_target)
        add(q[i],'learn_competitive',meta)
        distance=np.minimum(distance,np.linalg.norm(q-q[i],axis=1));eligible[i]=False
    if len(points)!=count:raise RuntimeError('Information role count mismatch')
    return np.asarray(points),sources,details,dict(requested=count,requested_pair_points=pair_target,
        requested_boundary_points=boundary_target,requested_competitive_points=competitive_target,
        actual={s:sources.count(s) for s in sorted(set(sources))},
        boundary_shortfall=boundary_target-boundary_count,source_scope='original_and_current_experiment_only')
