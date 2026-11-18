"""Only prior: building 2 peak/ramping/resilience support is worth covering."""
import hashlib
import numpy as np
from .base import predict
from .acquisition_v1 import BINS,transfer_proposals
from .protocol import weight_key,atomic_json

SEED=7319026
SUPPORT=(2,5,6)
EXTRAS=(0,1,3,4)


def parent_key(w):
    return hashlib.sha256(np.asarray(w,dtype=np.float64).tobytes()).hexdigest()


def block_banks(x,y):
    order=np.lexsort((np.arange(len(y)),y));eligible=order[y[order]<=y.min()+2.]
    banks=[]
    for b in (0,2):
        selected=[]
        for i in eligible:
            w=x[i].reshape(3,7)[b]
            if not selected or np.linalg.norm(x[selected].reshape(-1,3,7)[:,b]-w,axis=1).min()>=.05:
                selected.append(int(i))
                if len(selected)==8:break
        banks.append(selected)
    return banks


def specialized_pool(x,y,number):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,number]))
    banks=block_banks(x,y);points=[];metadata=[]
    for size in (3,4):
        for concentration in (1.,.3):
            for _ in range(64):
                extra=None if size==3 else int(rng.choice(EXTRAS))
                active=list(SUPPORT)+([] if extra is None else [extra])
                block=np.zeros(7);block[active]=rng.dirichlet(np.full(size,concentration))
                left=int(rng.choice(banks[0]));right=int(rng.choice(banks[1]))
                points.append(np.concatenate([x[left].reshape(3,7)[0],block,x[right].reshape(3,7)[2]]))
                metadata.append(dict(support_size=size,concentration=concentration,extra_coordinate=extra,
                    block_parent_indices=[left,right],block_parent_R=[float(y[left]),float(y[right])],available_rows=len(y)))
    return np.asarray(points),metadata,banks


def transfer_pool(x,y,radius):
    parent=int(np.argmin(y));points=[];meta=[];unique=set()
    for bin_name in BINS:
        q,m=transfer_proposals(x[parent],bin_name,radius)
        for w,d in zip(q,m):
            if d['building']!=1 or d['donor'] not in SUPPORT or d['receiver'] not in SUPPORT:continue
            key=weight_key(w.tolist())
            if key in unique:continue
            unique.add(key);points.append(w)
            meta.append(dict(d,center=parent,parent_R=float(y[parent]),parent_key=parent_key(x[parent]),
                requested_delta=float(x[parent].reshape(3,7)[1,d['donor']]) if bin_name=='support' else {'small':.001,'medium':.01,'coarse':.08}[bin_name]*radius,
                direction=[1,d['donor'],d['receiver']],alpha=radius))
    return np.asarray(points).reshape(-1,21),meta


def successful_directions(x,y,online,number):
    parent=int(np.argmin(y));key=parent_key(x[parent]);success={}
    for row in online:
        if row['round']>=number:raise ValueError('Future feedback cannot guide selection')
        if row.get('source')!='specialized_transfer':continue
        m=row.get('acquisition',{});index=m['center']
        if not 0<=index<len(y):raise ValueError('Unknown historical parent')
        if m['parent_key']!=parent_key(x[index]):raise ValueError('Parent fingerprint mismatch')
        if m['parent_key']!=key:continue
        gain=float(y[index]-row['R'])
        if gain>1e-12:
            d=tuple(m['direction']);success[d]=max(success.get(d,0.),gain)
    return success


def select(models,x,y,number,radius,online,seen,anchors,folder=None):
    if any(row['round']>=number for row in online):raise ValueError('Future feedback')
    pool,metadata,banks=specialized_pool(x,y,number)
    pred=predict(models,pool);risk=pred.mean(0)+.5*pred.std(0)
    used=set(seen);selected=[];sources=[];details=[];pool_indices=[]
    def add(w,source,m):
        key=weight_key(w.tolist())
        if key in used:return False
        used.add(key);selected.append(w);sources.append(source);details.append(dict(m,role='exploit'));return True
    order=np.lexsort((np.arange(len(pool)),risk));ranked=2 if number<=18 else 4
    for i in order:
        if len(selected)>=ranked:break
        if add(pool[i],'specialized_model',dict(metadata[i],pool_index=int(i),selection='mean_plus_half_std',selection_score=float(risk[i]))):pool_indices.append(int(i))
    if number<=18:
        available=[i for i,w in enumerate(pool) if weight_key(w.tolist()) not in used]
        while len(selected)<4 and available:
            anchor=np.asarray(list(anchors)+selected).reshape(-1,21)
            distances=np.linalg.norm(pool[available,None]-anchor[None],axis=2).min(1)
            k=int(np.lexsort((np.asarray(available),-distances))[0]);i=available.pop(k)
            if add(pool[i],'specialized_coverage',dict(metadata[i],pool_index=i,selection='max_min_distance_no_model_gate',selection_distance=float(distances[k]))):pool_indices.append(i)
    tq,tm=transfer_pool(x,y,radius)
    tp=predict(models,tq) if len(tq) else np.empty((len(models),0))
    tr=tp.mean(0)+.5*tp.std(0) if len(tq) else np.empty(0)
    success=successful_directions(x,y,online,number);transfer_selected=[]
    def take(indices,reason):
        for i in indices:
            m=tm[i]
            if add(tq[i],'specialized_transfer',dict(m,transfer_pool_index=int(i),selection=reason,
                     same_parent_observed_gain=success.get(tuple(m['direction'])),selection_score=float(tr[i]))):
                transfer_selected.append(int(i));return True
        return False
    if number>=19 and success:
        follow=[i for i,m in enumerate(tm) if tuple(m['direction']) in success]
        follow=sorted(follow,key=lambda i:(-success[tuple(tm[i]['direction'])],tr[i],i))
        while len(transfer_selected)<2 and take(follow,'same_exact_parent_improvement'):pass
    if len(transfer_selected)<2:
        take(np.lexsort((np.arange(len(tq)),tr)).tolist(),'model_ranked_transfer')
    if len(transfer_selected)<2 and len(tq):
        # Fixed directed-pair x step-bin rotation; no prediction gate.
        fixed=sorted(range(len(tq)),key=lambda i:(tm[i]['donor'],tm[i]['receiver'],BINS.index(tm[i]['bin'])))
        offset=(number-1)%len(fixed);take(fixed[offset:]+fixed[:offset],'fixed_rotation_no_model_gate')
    report=dict(seed=SEED,round_seed_components=[SEED,number],bank_row_indices=banks,generated_candidates=256,
        requested=6,actual=len(selected),shortfall=6-len(selected),sample_actual=len(pool_indices),transfer_actual=len(transfer_selected),
        selected_pool_indices=pool_indices,selected_transfer_indices=transfer_selected,alpha=radius,
        structural_prior='building2 peak/ramping/resilience; no known parameter ratios')
    if folder is not None:
        atomic_json(folder/'specialized-pool.json',dict(weights=pool.reshape(-1,3,7).tolist(),metadata=metadata,predictions=pred.T.tolist(),
            transfers=tq.reshape(-1,3,7).tolist(),transfer_metadata=tm,transfer_predictions=tp.T.tolist(),report=report))
    return selected,sources,details,report
