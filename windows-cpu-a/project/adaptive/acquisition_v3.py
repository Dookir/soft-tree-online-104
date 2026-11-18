"""Frozen v1 until a real-feedback stall; then30 backbone +20 line probes."""
from pathlib import Path
import json
import numpy as np
from . import acquisition_v1 as frozen
from . import v1_engine
from .anchor import AnchorContext,update_feedback
from .base import predict
from .polling import scan
from .protocol import atomic_json,weight_key

QUOTAS=dict(surrogate=4,transfer=12,recombination=6,joint=8,polish_scan=20)


def update_alpha(state,scores,previous_best,new_best,batch=None):
    # Radius updates remain exactly v1; new line steps do not share its radius.
    return frozen.update_alpha(state,scores,previous_best,new_best)


def backbone_indices(batch,count=30):
    sources=batch['sources'];metadata=batch['metadata'];chosen=[]
    def take(indices,n):
        added=0
        for i in indices:
            if i in chosen:continue
            chosen.append(i);added+=1
            if added==n:break
    for objective in ('mean','max'):
        take([i for i,s in enumerate(sources) if s=='surrogate' and metadata[i].get('objective')==objective],2)
    for bin_name in frozen.BINS:
        ids=[i for i,s in enumerate(sources) if s=='transfer' and metadata[i].get('bin')==bin_name]
        ranked=[i for i in ids if metadata[i].get('center_rank')==0 and metadata[i].get('selection')=='ranked']
        rotating=[i for i in ids if metadata[i].get('selection')=='rotating']
        alternatives=[i for i in ids if metadata[i].get('center_rank',0)>0]
        # Best-center exploitation, one rotating probe, and an alternative center.
        priority=ranked[:1]+rotating[:1]
        if alternatives:priority.append(alternatives[(batch['round']-1)%len(alternatives)])
        priority+=ids
        take(priority,3)
    take([i for i,s in enumerate(sources) if s=='recombination'],6)
    take([i for i,s in enumerate(sources) if s=='joint'],8)
    for family in (('transfer','fallback_transfer'),('recombination','fallback_recombination'),('joint','fallback_joint')):
        if len(chosen)>=count:break
        take([i for i,s in enumerate(sources) if s in family],count-len(chosen))
    if len(chosen)!=count:raise RuntimeError('Cannot form30 structured v1 backbone candidates')
    return chosen


def generate_batch(models,x,y,round_number,seen,folder,radius=1.,search_steps=None,search_state=None,online=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    state=search_state or {};polish=bool(state.get('polish_active',False))
    anchoring=bool(polish and state.get('anchor_enabled',False))
    if not polish:
        # Never add ranking hooks or consume different RNGs in the v1 path.
        weights,sources,pred=frozen.generate_batch(models,x,y,round_number,seen,folder,radius,search_steps)
        batch=json.loads((folder/'candidates.json').read_text())
        context=AnchorContext(models,x,y)
        batch.update(phase='v1_discovery',anchor_used_for_selection=False,
                     shadow=context.shadow(weights,pred),prediction_model_version=round_number-1)
    else:
        context=AnchorContext(models,x,y)
        v1_engine.generate_batch(models,x,y,round_number,seen,folder/'v1-pool',radius,search_steps,
                                 local_ranker=context.rank if anchoring else None)
        original=json.loads((folder/'v1-pool/candidates.json').read_text())
        original['proposals_only_not_submitted_as_a_batch']=True
        atomic_json(folder/'v1-pool/candidates.json',original)
        ids=backbone_indices(original)
        backbone=np.asarray(original['weights']).reshape(-1,21)[ids]
        used=set(seen)|{weight_key(w.tolist()) for w in backbone}
        probes,probe_meta=scan(context,x,y,round_number,online or [],used,20,anchoring)
        weights=np.concatenate([backbone,probes])
        sources=[original['sources'][i] for i in ids]+['polish_scan']*20
        meta=[original['metadata'][i] for i in ids]+probe_meta
        pred=predict(models,weights)
        batch=dict(weights=weights.reshape(-1,3,7).tolist(),sources=sources,metadata=meta,
                   predictions=pred.T.tolist(),round=round_number,alpha=radius,phase='feedback_triggered_polish',
                   center_indices=context.ids.tolist(),center_R=y[context.ids].tolist(),
                   requested_quotas=QUOTAS,actual_counts={s:sources.count(s) for s in sorted(set(sources))},
                   transfer_bin_counts={b:sum(s=='transfer' and m.get('bin')==b for s,m in zip(sources,meta)) for b in frozen.BINS},
                   unranked_count=sum(m.get('selection') in ('rotating','unranked','observed_improvement',
                         'rotating_direction','rotating_refill','bisected_step_interval') for m in meta),
                   anchor_used_for_selection=anchoring,shadow=context.shadow(weights,pred),prediction_model_version=round_number-1)
    if len(weights)!=50 or len({weight_key(w.tolist()) for w in weights})!=50:
        raise RuntimeError('Exactly50 distinct candidates required')
    if any(weight_key(w.tolist()) in seen for w in weights):raise RuntimeError('Known candidate was resubmitted')
    atomic_json(folder/'candidates.json',batch)
    return weights,sources,pred
