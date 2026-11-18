
"""Formal batch composition with preserved original learning and polish."""

import json
from pathlib import Path

import numpy as np

from . import acquisition_original as original
from . import specialized_acquisition as opening
from .anchor import AnchorContext
from .base import predict
from .paired_selection import choose
from .protocol import atomic_json, weight_key

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'specialization.json').read_text())
STRATEGY = CONFIG['strategy']
START_ROUND = CONFIG['strategy_starts_round']
PROTECTED = {'polish_scan', 'exploit_final_scan'}
update_alpha = original.update_alpha
exploitation_indices = original.exploitation_indices


def compose(models, x, y, number, radius, online, seen, base, strategy=STRATEGY):
    weights = np.asarray(base['weights']).reshape(-1, 21).copy()
    sources = list(base['sources'])
    metadata = list(base['metadata'])
    exploit = base['role_counts']['exploit']
    reserved = (2 if number >= 19 else 4) if strategy == 'recombination' else 6
    replaceable = [i for i in range(exploit - 1, -1, -1) if sources[i] not in PROTECTED][:reserved]
    blocked = set(seen) | {weight_key(w.tolist()) for w in weights}
    if replaceable:
        selected = choose(models, x, y, number, radius, online, blocked,
                          list(weights[:exploit]), strategy, len(replaceable))
    else:
        selected = []
    slots = sorted(replaceable[:len(selected)])
    for i, (weights_new, source, details) in zip(slots, selected):
        weights[i] = weights_new
        sources[i] = source
        metadata[i] = details

    keys = {weight_key(w.tolist()) for w in weights}
    if (len(weights) != 50 or len(keys) != 50 or keys & seen or np.any(weights < 0)
            or not np.allclose(weights.reshape(-1, 3, 7).sum(2), 1, atol=1e-12, rtol=0)):
        raise RuntimeError('Batch must be 50 legal, unseen and unique inputs')
    prediction = predict(models, weights)
    sources_count = {s: sources.count(s) for s in sorted(set(sources))}
    report = dict(strategy=strategy, requested=reserved, actual=len(slots),
                  shortfall=reserved-len(slots), replacement_slots=slots,
                  original_sources_replaced=[base['sources'][i] for i in slots],
                  protected_polish=sum(s in PROTECTED for s in base['sources']))
    batch = dict(base, weights=weights.reshape(-1, 3, 7).tolist(), sources=sources,
                 metadata=metadata, predictions=prediction.T.tolist(),
                 shadow=AnchorContext(models, x, y).shadow(weights, prediction),
                 specialization=report, base_alpha=radius, exploration_radius_multiplier=1.0,
                 actual_counts=sources_count)
    batch['transfer_bin_counts'] = {
        bin_name: sum(s == 'transfer' and m.get('bin') == bin_name
                      for s, m in zip(sources, metadata))
        for bin_name in original.prior.frozen.BINS
    }
    unranked = ('rotating', 'unranked', 'observed_improvement', 'rotating_direction',
                'rotating_refill', 'bisected_step_interval', 'untried_parent_direction_rotation',
                'same_step_different_donor', 'same_direction_two_steps', 'confirmed_direction_step')
    batch['unranked_count'] = sum(m.get('selection') in unranked for m in metadata)
    return weights, sources, prediction, batch


def generate_batch(models, x, y, round_number, seen, folder, radius=1.,
                   search_steps=None, search_state=None, online=None):
    if round_number < START_ROUND:
        return opening.generate_batch(models, x, y, round_number, seen, folder, radius,
                                      search_steps, search_state, online)
    folder = Path(folder)
    base_path = folder / 'original-pool' / 'candidates.json'
    original.generate_batch(models, x, y, round_number, seen, base_path.parent, radius,
                            search_steps, search_state, online)
    base = json.loads(base_path.read_text(encoding='utf-8'))
    weights, sources, prediction, batch = compose(models, x, y, round_number, radius,
                                                  online or [], seen, base)
    atomic_json(folder / 'candidates.json', batch)
    return weights, sources, prediction
