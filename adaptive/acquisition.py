"""Fifty real queries: source-diverse exploitation plus scheduled information."""
from pathlib import Path
import json
import numpy as np
from . import acquisition_v3 as prior
from .information import generate as information_batch,splits
from .anchor import AnchorContext
from .base import predict
from .polling import scan
from .protocol import atomic_json,weight_key

update_alpha=prior.update_alpha


def exploitation_indices(batch,target):
    if target==50:
        return [i for i,s in enumerate(batch['sources']) if s!='disagreement']
    chosen=prior.backbone_indices(batch)
    sources=batch['sources'];metadata=batch['metadata']
    # Round-robin late line probes; otherwise preserve transfer/recombination
    # diversity instead of sorting every extra slot by the same low prediction.
    scans={}
    for i,s in enumerate(sources):
        if s=='polish_scan':
            m=metadata[i];scans.setdefault((m['building'],m['donor'],m['receiver']),[]).append(i)
    for k in range(max([len(v) for v in scans.values()]+[0])):
        for group in scans.values():
            if len(chosen)>=target:return chosen
            if k<len(group) and group[k] not in chosen:chosen.append(group[k])
    for family in (('transfer','fallback_transfer'),('recombination','fallback_recombination'),
                   ('joint','fallback_joint'),('surrogate',),('disagreement',)):
        for i,s in enumerate(sources):
            if len(chosen)>=target:return chosen
            if s in family and i not in chosen:chosen.append(i)
    if len(chosen)!=target:raise RuntimeError('Cannot fill exploitation quota from v3 pool')
    return chosen


def generate_batch(models,x,y,round_number,seen,folder,radius=1.,search_steps=None,search_state=None,online=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);online=online or [];state=search_state or {}
    exploit_count,learn_count,_=splits(round_number)
    prior.generate_batch(models,x,y,round_number,seen,folder/'v3-pool',radius,search_steps,search_state=state,online=online)
    pool=json.loads((folder/'v3-pool/candidates.json').read_text())
    pool['proposals_only_not_submitted_as_a_batch']=True;atomic_json(folder/'v3-pool/candidates.json',pool)
    ids=exploitation_indices(pool,exploit_count)
    points=list(np.asarray(pool['weights']).reshape(-1,21)[ids]);sources=[pool['sources'][i] for i in ids]
    details=[dict(pool['metadata'][i],role='exploit') for i in ids]
    used=set(seen)|{weight_key(w.tolist()) for w in points}
    context=AnchorContext(models,x,y);anchoring=bool(state.get('polish_active') and state.get('anchor_enabled'))
    if len(points)<exploit_count:
        extra,meta=scan(context,x,y,round_number,online,used,exploit_count-len(points),anchoring)
        for w,m in zip(extra,meta):
            points.append(w);sources.append('exploit_final_scan');details.append(dict(m,role='exploit'));used.add(weight_key(w.tolist()))
    info,info_sources,info_meta,info_report=information_batch(models,x,y,round_number,online,used,
                np.asarray(pool['weights']).reshape(-1,21),np.asarray(points),anchoring)
    points.extend(info);sources.extend(info_sources);details.extend(info_meta)
    weights=np.asarray(points);roles=['exploit']*exploit_count+['learn']*learn_count
    if len(weights)!=50 or len({weight_key(w.tolist()) for w in weights})!=50:
        raise RuntimeError('Active batch must contain50 unique weights')
    if any(weight_key(w.tolist()) in seen for w in weights):raise RuntimeError('Known point cannot consume another query')
    for meta in details:
        parent=meta.get('center')
        if parent is not None:
            if not 0<=parent<len(y):raise RuntimeError('Future sample parent')
            meta['parent_R']=float(y[parent])
    prediction=predict(models,weights)
    batch=dict(weights=weights.reshape(-1,3,7).tolist(),predictions=prediction.T.tolist(),sources=sources,metadata=details,
        roles=roles,role_counts={'exploit':roles.count('exploit'),'learn':roles.count('learn')},information=info_report,
        round=round_number,phase=pool['phase'],alpha=radius,anchor_used_for_selection=pool['anchor_used_for_selection'],
        shadow=context.shadow(weights,prediction),prediction_model_version=round_number-1,
        requested_quotas={'exploit':exploit_count,'learn':learn_count},
        actual_counts={s:sources.count(s) for s in sorted(set(sources))},
        transfer_bin_counts={b:sum(s=='transfer' and m.get('bin')==b for s,m in zip(sources,details)) for b in prior.frozen.BINS},
        unranked_count=sum(m.get('selection') in ('rotating','unranked','observed_improvement','rotating_direction',
                   'rotating_refill','bisected_step_interval') for m in details),
        old_experiment_data_used=False,information_utility_is_heuristic=True)
    atomic_json(folder/'candidates.json',batch)
    return weights,sources,prediction
