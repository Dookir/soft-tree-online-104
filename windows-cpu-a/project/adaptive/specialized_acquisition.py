"""Six structural-prior exploit slots; all other choices follow original v4.0.0."""
import json
from pathlib import Path
import numpy as np
from . import acquisition_original as original
from .base import predict
from .anchor import AnchorContext
from .structure_search import select
from .protocol import atomic_json,weight_key

update_alpha=original.update_alpha
exploitation_indices=original.exploitation_indices


def generate_batch(models,x,y,round_number,seen,folder,radius=1.,search_steps=None,search_state=None,online=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    original.generate_batch(models,x,y,round_number,seen,folder/'original-pool',radius,search_steps,search_state,online)
    batch=json.loads((folder/'original-pool/candidates.json').read_text())
    weights=np.asarray(batch['weights']).reshape(-1,21);count=batch['role_counts']['exploit']
    blocked=set(seen)|{weight_key(w.tolist()) for w in weights}
    points,sources,metadata,report=select(models,x,y,round_number,radius,online or [],blocked,list(weights[:count-6]),folder)
    n=len(points);slots=list(range(count-n,count))
    for i,w,s,m in zip(slots,points,sources,metadata):
        weights[i]=w;batch['sources'][i]=s;batch['metadata'][i]=m
    prediction=predict(models,weights)
    batch.update(weights=weights.reshape(-1,3,7).tolist(),predictions=prediction.T.tolist(),
        shadow=AnchorContext(models,x,y).shadow(weights,prediction),specialization=dict(report,replacement_slots=slots),
        exploration_radius_multiplier=1.0,base_alpha=radius,
        actual_counts={s:batch['sources'].count(s) for s in sorted(set(batch['sources']))})
    batch['transfer_bin_counts']={b:sum(s=='transfer' and m.get('bin')==b for s,m in zip(batch['sources'],batch['metadata'])) for b in original.prior.frozen.BINS}
    batch['unranked_count']=sum(m.get('selection') in ('rotating','unranked','observed_improvement','rotating_direction','rotating_refill','bisected_step_interval','max_min_distance_no_model_gate','fixed_rotation_no_model_gate') for m in batch['metadata'])
    keys=[weight_key(w.tolist()) for w in weights]
    if len(keys)!=50 or len(set(keys))!=50 or any(k in seen for k in keys):raise RuntimeError('Batch must be 50 unseen unique inputs')
    atomic_json(folder/'candidates.json',batch)
    return weights,batch['sources'],prediction
