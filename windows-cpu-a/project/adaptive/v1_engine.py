"""Fixed-budget, scale-stratified acquisition; only weights and scalar R enter."""
import itertools
from pathlib import Path

import numpy as np
import torch

from .base import CONFIG, configure, descend, distinct_order, predict, project
from .protocol import atomic_json, weight_key

BINS = ('small','medium','coarse','support')
QUOTAS = dict(surrogate=10,transfer=20,recombination=10,joint=8,disagreement=2)


def search_centers(x,y):
    order = np.argsort(y,kind='stable')
    good = order[y[order] <= y[order[0]]+CONFIG['near_best_margin']]
    picked = [int(order[0])]
    for size in (64,256):
        for i in good[:size]:
            if i not in picked and np.linalg.norm(x[picked]-x[i],axis=1).min() >= .15:
                picked.append(int(i))
                if len(picked) == 3:
                    return np.asarray(picked)
    return np.asarray(picked)


def transfer_proposals(center,bin_name,alpha):
    center = center.reshape(3,7)
    proposals, metadata = [], []
    amount = dict(small=.001,medium=.01,coarse=.08).get(bin_name,0.)*alpha
    for b in range(3):
        for donor in np.flatnonzero(center[b] > 0):
            for receiver in range(7):
                if receiver == donor:
                    continue
                delta = center[b,donor] if bin_name == 'support' else min(amount,center[b,donor])
                # Exclude tiny donors from a nominally coarse movement bin.
                if bin_name != 'support' and delta < min(amount,float(center[b].max()))-1e-14:
                    continue
                trial = center.copy()
                trial[b,donor] -= delta
                trial[b,receiver] += delta
                proposals.append(trial.reshape(21))
                metadata.append(dict(building=b,donor=int(donor),receiver=receiver,delta=float(delta),bin=bin_name))
    return np.asarray(proposals),metadata


def building_recombinations(x,y):
    elite = np.argsort(y,kind='stable')[:64]
    blocks = []
    for b in range(3):
        selected = []
        for i in elite:
            w = x[i].reshape(3,7)[b]
            if not selected or np.linalg.norm(np.asarray(selected)-w,axis=1).min() >= .05:
                selected.append(w.copy())
                if len(selected) == 8:
                    break
        blocks.append(selected)
    combinations = list(itertools.product(*[range(len(v)) for v in blocks]))
    return (np.asarray([np.concatenate([blocks[b][c[b]] for b in range(3)]) for c in combinations]),
            [dict(block_indices=list(c)) for c in combinations], [len(v) for v in blocks])


def joint_pair(center,mask,noise):
    masked = np.zeros((3,7))
    masked[list(mask)] = noise[list(mask)]
    return project(torch.tensor(np.stack([center.reshape(3,7)+masked,
                                         center.reshape(3,7)-masked]))).numpy().reshape(2,21)


def update_alpha(state,scores,previous_best,new_best):
    improvement = max(0.,float(previous_best-new_best))
    fraction = len({round(float(r),8) for r in scores})/len(scores)
    state['small_improvement_streak'] = state.get('small_improvement_streak',0)+1 if improvement < .02 else 0
    state['good_improvement_streak'] = state.get('good_improvement_streak',0)+1 if improvement >= .02 else 0
    old = state['radius']
    reason = 'unchanged'
    if fraction < .70 or state['small_improvement_streak'] >= 2:
        state['radius'] = min(4.,old*2.)
        state['small_improvement_streak'] = state['good_improvement_streak'] = 0
        reason = 'expand_duplicate_R' if fraction < .70 else 'expand_stagnation'
    elif state['good_improvement_streak'] >= 2:
        state['radius'] = max(.5,old*.7)
        state['good_improvement_streak'] = 0
        reason = 'contract_after_two_improvements'
    return dict(alpha_used=old,alpha_next=state['radius'],unique_rounded_R_fraction=fraction,
                improvement=improvement,reason=reason)


def generate_batch(models,x,y,round_number,seen,folder,radius=1.,search_steps=None,local_ranker=None):
    configure()
    rng = np.random.default_rng(CONFIG['search_seed']+round_number*1009)
    random = np.concatenate([rng.dirichlet(np.full(7,a),size=(8192,3)).reshape(-1,21) for a in (1.,.3)])
    corners = np.asarray([np.eye(7)[list(c)].reshape(21) for c in itertools.product(range(7),repeat=3)])
    pool = np.concatenate([random,x[np.argsort(y,kind='stable')[:64]],corners,np.full((1,21),1/7)])
    p = predict(models,pool)
    low = distinct_order(pool,np.argsort(p.mean(0),kind='stable'),256,.1)
    starts = np.concatenate([pool[low],random[rng.choice(len(random),256,replace=False)],corners,np.full((1,21),1/7)])
    _,unique = np.unique(starts,axis=0,return_index=True)
    starts = starts[np.sort(unique)]
    steps = search_steps or 1000
    mean_points = descend(models,starts,'mean',steps,[(0,.02),(int(steps*.4),.005),(int(steps*.7),.001)])
    max_points = descend(models,mean_points,'max',steps,[(0,.01),(int(steps*.4),.002),(int(steps*.7),.0004)])
    pm,px = predict(models,mean_points),predict(models,max_points)
    center_ids = search_centers(x,y)
    blocks,block_meta,block_sizes = building_recombinations(x,y)
    pb = predict(models,blocks)  # Always predict the complete district combination.
    chosen,sources,details = [],[],[]
    used = set(seen)

    def eligible(w,distance=0.):
        return (weight_key(w.tolist()) not in used and
                (not chosen or distance == 0 or np.linalg.norm(np.asarray(chosen)-w,axis=1).min() >= distance))

    def add(w,source,detail,distance=0.):
        if not eligible(w,distance):
            return False
        chosen.append(w.copy());sources.append(source);details.append(detail)
        used.add(weight_key(w.tolist()))
        return True

    # Keep the five mean and five max slots independent; collapsed endpoints
    # remain missing slots until the structured fallback, never global coverage.
    for label,points,pp in [('mean',mean_points,pm),('max',max_points,px)]:
        score = pp.mean(0) if label == 'mean' else pp.max(0)
        count = 0
        for i in np.argsort(score,kind='stable'):
            if add(points[i],'surrogate',dict(objective=label),.02):
                count += 1
                if count == 5:
                    break

    transfer_cache = []
    allocations = [5] if len(center_ids)==1 else ([4,1] if len(center_ids)==2 else [3,1,1])
    for bin_index,bin_name in enumerate(BINS):
        for center_rank,(center_id,count) in enumerate(zip(center_ids,allocations)):
            proposals,metas = transfer_proposals(x[center_id],bin_name,radius)
            pred = predict(models,proposals)
            score = local_ranker(proposals,pred) if local_ranker else pred.mean(0)+.5*pred.std(0)
            transfer_cache.extend((w,dict(meta,center=int(center_id),center_rank=center_rank)) for w,meta in zip(proposals,metas))
            ranked_count = count-1 if center_rank==0 else count
            obtained = 0
            for i in np.argsort(score,kind='stable'):
                if obtained >= ranked_count:
                    break
                if add(proposals[i],'transfer',dict(metas[i],center=int(center_id),center_rank=center_rank,selection='ranked')):
                    obtained += 1
            if center_rank == 0:
                # Round/bin rotation is independent of all model predictions.
                offset = ((round_number-1)*17+bin_index*31)%len(proposals)
                for i in np.roll(np.arange(len(proposals)),-offset):
                    if add(proposals[i],'transfer',dict(metas[i],center=int(center_id),center_rank=0,selection='rotating')):
                        obtained += 1
                        break
            # If directions collide across bins, exhaust this same bin first.
            for i in np.argsort(score,kind='stable'):
                if obtained >= count:
                    break
                if add(proposals[i],'transfer',dict(metas[i],center=int(center_id),center_rank=center_rank,selection='ranked_refill')):
                    obtained += 1

    count = 0
    for i in np.argsort(local_ranker(blocks,pb) if local_ranker else pb.mean(0)+.5*pb.std(0),kind='stable'):
        if add(blocks[i],'recombination',block_meta[i],.02):
            count += 1
            if count == 10:
                break

    pair_id = 0
    def add_joint(mask,source='joint',slot=None):
        nonlocal pair_id
        for _ in range(500):
            noise = rng.normal(0,.04*radius,size=(3,7))
            pair = joint_pair(x[center_ids[0]],mask,noise)
            if (all(eligible(w) for w in pair) and weight_key(pair[0].tolist()) != weight_key(pair[1].tolist())):
                pair_id += 1
                for sign,w in zip((1,-1),pair):
                    add(w,source,dict(mask=list(mask),sign=sign,pair=pair_id,center=int(center_ids[0]),
                                      selection='unranked',noise_std=.04*radius,requested_slot=slot))
                return True
        return False

    for mask in ((0,1),(0,2),(1,2),(0,1,2)):
        if not add_joint(mask):
            raise RuntimeError('Cannot create an unseen antithetic joint pair')

    structured = np.asarray([w for w,_ in transfer_cache])
    competitive_pool = np.concatenate([pool,mean_points,max_points,blocks,structured])
    pc = predict(models,competitive_pool)
    mean,std = pc.mean(0),pc.std(0)
    competitive = np.flatnonzero(mean <= mean.min()+CONFIG['near_best_margin'])
    count = 0
    for i in competitive[np.argsort(-std[competitive],kind='stable')]:
        if add(competitive_pool[i],'disagreement',dict(mean=float(mean[i]),std=float(std[i]),
                competitive_min=float(mean.min()),epsilon=CONFIG['near_best_margin']),.02):
            count += 1
            if count == 2:
                break

    nominal_counts = {s:sources.count(s) for s in QUOTAS}
    # Fallback priority: coarse transfers, building recombination, then local
    # joint perturbations. Never spend a real evaluation on farthest randoms.
    for source,target in QUOTAS.items():
        missing = target-nominal_counts[source]
        for bin_name in ('coarse','support'):
            for w,meta in transfer_cache:
                if missing <= 0:
                    break
                if meta['bin'] == bin_name and add(w,'fallback_transfer',dict(meta,requested_slot=source)):
                    missing -= 1
        for i in np.argsort(local_ranker(blocks,pb) if local_ranker else pb.mean(0)+.5*pb.std(0),kind='stable'):
            if missing <= 0:
                break
            if add(blocks[i],'fallback_recombination',dict(block_meta[i],requested_slot=source),.02):
                missing -= 1
        for _ in range(500):
            if missing <= 0:
                break
            trial = joint_pair(x[center_ids[0]],(0,1,2),rng.normal(0,.04*radius,(3,7)))[0]
            if add(trial,'fallback_joint',dict(requested_slot=source,selection='unranked',noise_std=.04*radius)):
                missing -= 1
        if missing:
            raise RuntimeError('Structured candidates exhausted before evaluation')
    batch = np.asarray(chosen)
    if len(batch)!=50 or len({weight_key(w.tolist()) for w in batch})!=50:
        raise RuntimeError('Acquisition did not produce exactly 50 unique points')
    predictions = predict(models,batch)
    folder = Path(folder);folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'candidates.json',dict(weights=batch.reshape(-1,3,7).tolist(),sources=sources,
        metadata=details,predictions=predictions.T.tolist(),round=round_number,alpha=radius,starts=len(starts),
        center_indices=center_ids.tolist(),center_R=y[center_ids].tolist(),block_sizes=block_sizes,
        requested_quotas=QUOTAS,nominal_counts=nominal_counts,actual_counts={s:sources.count(s) for s in sorted(set(sources))},
        transfer_bin_counts={b:sum(s=='transfer' and d['bin']==b for s,d in zip(sources,details)) for b in BINS},
        unranked_count=sum(d.get('selection') in ('rotating','unranked') for d in details)))
    np.savez_compressed(folder/'search.npz',mean_points=mean_points,max_points=max_points,
                        mean_predictions=pm,max_predictions=px)
    return batch,sources,predictions
