"""Select extra full-district queries from visible scores only."""

import itertools

import numpy as np

from .acquisition_v1 import BINS, transfer_proposals
from .base import predict
from .protocol import weight_key
from .structure_search import SUPPORT, block_banks, parent_key, specialized_pool

ORIGINAL_ROWS = 2048
MIN_GAIN = 0.01
FAMILIES = ('recombination', 'fallback_recombination', 'specialized_model',
            'specialized_coverage', 'specialized_combination')


def risk_order(models, points):
    if not len(points):
        return np.empty(0, dtype=int), np.empty(0)
    pred = predict(models, np.asarray(points))
    risk = pred.mean(0) + 0.5 * pred.std(0)
    return np.argsort(risk, kind='stable'), risk


def parent_fields(x, y, online, index):
    return dict(center=index, parent_R=float(y[index]), parent_key=parent_key(x[index]),
                parent_id=online[index - ORIGINAL_ROWS]['id'] if index >= ORIGINAL_ROWS else None)


def move(x, parent, direction, delta):
    building, donor, receiver = direction
    weights = x[parent].reshape(3, 7).copy()
    weights[building, donor] -= delta
    weights[building, receiver] += delta
    return weights.reshape(21)


def transfer(x, y, online, parent, direction, delta, reason, **extra):
    building, donor, receiver = direction
    return move(x, parent, direction, delta), dict(
        parent_fields(x, y, online, parent), direction=list(direction),
        building=building, donor=donor, receiver=receiver, delta=float(delta),
        selection=reason, **extra)


def legal_transfers(x, y, online, parent, radius, structural=False):
    for bin_name in BINS:
        points, metadata = transfer_proposals(x[parent], bin_name, radius)
        for weights, item in zip(points, metadata):
            b, d, r = (item[k] for k in ('building', 'donor', 'receiver'))
            if structural and (b != 1 or d not in SUPPORT or r not in SUPPORT):
                continue
            yield weights, dict(parent_fields(x, y, online, parent),
                                **item, direction=[b, d, r])


def histories(x, y, online, number):
    """Use only verified, single-direction moves from completed feedback."""
    results = {}
    for i, row in enumerate(online):
        if row['round'] >= number:
            raise ValueError('Future feedback cannot guide selection')
        item = row.get('acquisition', {})
        fields = ('center', 'building', 'donor', 'receiver', 'delta')
        if not all(k in item for k in fields):
            continue
        parent, b, d, r, delta = (item[k] for k in fields)
        child = ORIGINAL_ROWS + i
        if not 0 <= parent < child or (parent >= ORIGINAL_ROWS and
                                       online[parent - ORIGINAL_ROWS]['round'] >= row['round']):
            raise ValueError('Transfer parent was not visible before its child')
        if item.get('parent_key', parent_key(x[parent])) != parent_key(x[parent]):
            raise ValueError('Transfer parent fingerprint mismatch')
        if (d == r or not 0 < delta <= x[parent].reshape(3, 7)[b, d] + 1e-14 or
                not np.allclose(move(x, parent, (b, d, r), delta), x[child], rtol=0, atol=1e-14)):
            continue
        key = (parent, b, d, r)
        results.setdefault(key, []).append((float(delta), float(y[parent] - y[child]), row))
    return results


def followups(x, y, online, number, radius, history, structural):
    """Keep the successful exact parent available for two subsequent rounds."""
    gains = []
    for key, observations in history.items():
        parent, b, d, r = key
        if structural and (b != 1 or d not in SUPPORT or r not in SUPPORT):
            continue
        recent = [o for o in observations if o[1] >= MIN_GAIN and o[2]['round'] >= number - 2]
        if recent:
            gains.append((key, max(recent, key=lambda o: (o[1], -o[0]))))
    gains.sort(key=lambda entry: (-entry[1][1], y[entry[0][0]], entry[0]))
    for (parent, *direction), (step, gain, child) in gains:
        b, d, _ = direction
        limit = x[parent].reshape(3, 7)[b, d]
        tried = history[(parent, *direction)]
        steps = sorted({0.0, float(limit), *(t[0] for t in tried),
                        *(s * radius for s in (.001, .01, .08) if s * radius < limit)})
        lower = max(s for s in steps if s < step)
        higher = min((s for s in steps if s > step), default=None)
        options = [(lower + step) / 2]
        if higher is not None:
            options.append((step + higher) / 2)
        if len(options) < 2:
            options.extend(s for s in sorted(steps, key=lambda t: (abs(t - step), t))
                           if 0 < s < step)
        yield [transfer(x, y, online, parent, direction, s, 'confirmed_direction_step',
                        reference_child_id=child['id'], reference_delta=step,
                        same_parent_observed_gain=gain) for s in options if 0 < s <= limit]


def signature(weights):
    values = weights.reshape(3, 7)[1, list(SUPPORT)]
    return tuple(values <= 1e-12), int(np.argmax(values))


def choose(models, x, y, number, radius, online, blocked, anchors, strategy, capacity):
    if len(x) != ORIGINAL_ROWS + len(online):
        raise ValueError('Visible training rows and feedback differ')
    history = histories(x, y, online, number)
    best = int(np.argmin(y))
    blocked = set(blocked)
    selected = []

    def add_group(options, source):
        options = list(options)
        keys = {weight_key(w.tolist()) for w, _ in options}
        if not options or len(options) + len(selected) > capacity:
            return False
        if len(keys) != len(options) or keys & blocked:
            return False
        selected.extend((w, 'specialized_' + source, dict(m, role='exploit')) for w, m in options)
        blocked.update(keys)
        return True

    def remaining(options):
        return [(w, m) for w, m in options if weight_key(w.tolist()) not in blocked]

    def model_first(options, source, reason):
        options = remaining(options)
        if not options:
            return None
        order, risk = risk_order(models, [w for w, _ in options])
        i = int(order[0])
        w, m = options[i]
        if add_group([(w, dict(m, selection=reason, selection_score=float(risk[i])))], source):
            return m
        return None

    if strategy == 'recombination':
        if number >= 19:
            for options in followups(x, y, online, number, radius, history, structural=False):
                for option in options:
                    add_group([option], 'step_followup')
                if len(selected) == capacity:
                    break
            return selected

        parents = [ORIGINAL_ROWS + i for i, row in enumerate(online)
                   if row['round'] >= number - 2 and row['source'] in FAMILIES
                   and best != ORIGINAL_ROWS + i and row['R'] <= y[best] + 1.0]
        tried = {row['acquisition']['center'] for row in online
                 if row['source'] == 'specialized_recombination_transfer'}
        parents.sort(key=lambda p: (p in tried, y[p], -p))
        for parent in parents:
            options = list(legal_transfers(x, y, online, parent, radius))
            first = model_first(options, 'recombination_transfer', 'parent_model_rank')
            if first is None:
                continue
            fresh = [(w, m) for w, m in remaining(options)
                     if m['direction'] != first['direction'] and
                     (parent, *m['direction']) not in history]
            fresh.sort(key=lambda item: (item[1]['direction'], BINS.index(item[1]['bin'])))
            if fresh:
                offset = (number - 1) % len(fresh)
                fresh = fresh[offset:] + fresh[:offset]
            fresh.sort(key=lambda item: (item[1]['building'] == first['building'],
                                         item[1]['bin'] == first['bin']))
            if fresh:
                w, m = fresh[0]
                add_group([(w, dict(m, selection='untried_parent_direction_rotation'))],
                          'recombination_transfer')
            break

        model_first(legal_transfers(x, y, online, best, radius, structural=True),
                    'incumbent_transfer', 'incumbent_model_rank')
        pool, metadata, _ = specialized_pool(x, y, number)
        options = remaining(zip(pool, metadata))
        if number % 2 or not options:
            model_first(options, 'combination', 'model_ranked_combination')
        else:
            order, risk = risk_order(models, [w for w, _ in options])
            shortlist = order[:(len(order) + 3) // 4]
            references = np.asarray([*anchors, *(w for w, _, _ in selected)])
            distances = np.linalg.norm(np.asarray([options[i][0] for i in shortlist])[:, None]
                                       - references[None], axis=2).min(axis=1)
            i = int(shortlist[int(np.argmax(distances))])
            w, m = options[i]
            add_group([(w, dict(m, selection='distance_within_top_prediction_quartile',
                                selection_score=float(risk[i])))], 'combination')
        return selected

    if strategy != 'boundary':
        raise ValueError('Unknown paired experiment strategy')
    parent = best
    if number % 2 == 0:
        other = [i for i in np.flatnonzero(y <= y[best] + 1.0)
                 if signature(x[i]) != signature(x[best])]
        if other:
            parent = int(min(other, key=lambda i: (y[i], i)))

    block = x[parent].reshape(3, 7)[1]
    receivers = list(SUPPORT)
    offset = (number - 7) % len(receivers)
    receivers = receivers[offset:] + receivers[:offset]
    receivers.sort(key=lambda r: block[r] > 1e-12)
    # Rotate the original three scales independently from the receiver cycle.
    steps = np.roll(np.array([.001, .01, .08]) * radius, -((number - 7) // 3) % 3)
    for receiver, delta in itertools.product(receivers, steps):
        donors = [d for d in SUPPORT if d != receiver and block[d] >= delta]
        if len(donors) != 2:
            continue
        pair = [transfer(x, y, online, parent, (1, donor, receiver), delta,
                         'same_step_different_donor', pair_member=i)
                for i, donor in enumerate(donors)]
        if add_group(pair, 'donor_comparison'):
            break

    confirmed = False
    for options in followups(x, y, online, number, radius, history, structural=True):
        pair = remaining(options)[:2]
        if len(pair) == 2 and add_group(pair, 'step_followup'):
            confirmed = True
            break
    if not confirmed:
        options = remaining(legal_transfers(x, y, online, parent, radius, structural=True))
        by_direction = {}
        for w, m in options:
            by_direction.setdefault(tuple(m['direction']), {})[m['delta']] = (w, m)
        directions = sorted(by_direction, key=lambda d: (len(history.get((parent, *d), [])), d))
        if directions:
            min_count = len(history.get((parent, *directions[0]), []))
            tied = [d for d in directions if len(history.get((parent, *d), [])) == min_count]
            offset = (number - 7) % len(tied)
            directions = tied[offset:] + tied[:offset] + directions[len(tied):]
        for direction in directions:
            candidates = by_direction[direction]
            if len(candidates) >= 2:
                pair = [(w, dict(m, selection='same_direction_two_steps', pair_member=i))
                        for i, (w, m) in enumerate(candidates[d] for d in (min(candidates), max(candidates)))]
                if add_group(pair, 'step_comparison'):
                    break

    for b, bank in zip((0, 2), block_banks(x, y)):
        options = []
        for donor in bank:
            if donor == parent:
                continue
            weights = x[parent].reshape(3, 7).copy()
            weights[b] = x[donor].reshape(3, 7)[b]
            options.append((weights.reshape(21), dict(parent_fields(x, y, online, parent),
                           building=b, block_parent=int(donor), block_parent_R=float(y[donor]),
                           block_parent_key=parent_key(x[donor]))))
        model_first(options, 'single_building_swap', 'single_building_model_rank')
    return selected
