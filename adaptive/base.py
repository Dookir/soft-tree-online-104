"""Weighted full-data refits and diverse batch acquisition; never reads an evaluator."""
import os
from .determinism import prepare_environment, configure_runtime, seed_all, settings
prepare_environment()

from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import DistrictSurrogate
from .protocol import atomic_json, weight_key

CONFIG = json.loads((Path(__file__).parent/'recipe.json').read_text())
CONFIGURED = False
DEVICE = os.environ.get('SOFTTREE_DEVICE', 'cpu')


def configure():
    global CONFIGURED
    if not CONFIGURED:
        configure_runtime()
        if DEVICE.startswith('cuda'):
            budget = float(os.environ.get('SOFTTREE_GPU_MEMORY_GIB', '2'))*1024**3
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(budget/total, .5), 0)
        CONFIGURED = True


def project(weights):
    rows = weights.reshape(-1, 7)
    ordered = rows.sort(dim=-1, descending=True, stable=True).values
    cumulative = ordered.cumsum(-1) - 1
    k = torch.arange(1, 8, dtype=rows.dtype, device=rows.device)
    rho = (ordered - cumulative/k > 0).sum(-1).clamp(min=1)-1
    theta = cumulative.gather(1, rho[:, None])/(rho+1)[:, None]
    return (rows-theta).clamp(min=0).reshape(weights.shape)


def centers(x, y, count=5, distance=.05):
    ids = []
    for i in np.argsort(y, kind='stable'):
        if not ids or np.linalg.norm(x[ids]-x[i], axis=1).min() >= distance:
            ids.append(int(i))
            if len(ids) == count:
                break
    return np.asarray(ids, dtype=int)


def sample_weights(x, y, original_rows):
    elite = centers(x, y)
    distance = np.linalg.norm(x[:, None]-x[elite][None], axis=2).min(axis=1)
    q20, q50 = np.quantile(y, [.2, .5])
    low_bonus = 1 + .5*np.clip((q50-y)/max(q50-q20, 1e-12), 0, 1)
    region_bonus = 1 + .5*np.exp(-.5*(distance/.2)**2)
    source_bonus = np.where(np.arange(len(y)) >= original_rows, 1.5, 1.)
    importance = np.minimum(low_bonus*region_bonus*source_bonus, 3.)
    active = np.ones(len(y), dtype=bool)
    # Conservative pruning: only near-duplicate, mutually high-R, far-away rows
    # with essentially the same label. All records remain in the history.
    high = np.flatnonzero((y >= np.quantile(y, .8)) & (distance >= .3))
    removed = []
    for i in high[np.argsort(y[high], kind='stable')[::-1]]:
        if len(removed) >= int(.05*len(y)):
            break
        others = high[(high != i) & active[high]]
        if len(others):
            d = np.linalg.norm(x[others]-x[i], axis=1)
            j = int(others[d.argmin()])
            if d.min() <= .03 and abs(y[i]-y[j]) <= .01*np.median(y):
                active[i] = False
                removed.append(int(i))
    importance[~active] = 0
    report = dict(total_rows=len(y), original_rows=original_rows, online_rows=len(y)-original_rows,
                  active_rows=int(active.sum()), pruned_indices=removed, elite_indices=elite.tolist(),
                  weighting='source(offline=1,online=1.5)*low_R[1,1.5]*region[1,1.5], cap=3',
                  active_weight_min=float(importance[active].min()), active_weight_max=float(importance.max()),
                  online_weight_share=float(importance[original_rows:].sum()/importance.sum()),
                  pruning='high-R top20%, elite distance>=0.3, neighbor distance<=0.03, label gap<=1% median; at most5%')
    return importance, active, report


def train_one(seed, x, y, importance, scale, folder, epochs):
    configure()
    seed_all(seed)
    started = time.monotonic()
    model = DistrictSurrogate(scale, seed).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                                  betas=(.9,.999), eps=1e-8, foreach=False, fused=False)
    xx, yy, ww = torch.tensor(x,device=DEVICE), torch.tensor(y/scale,device=DEVICE), torch.tensor(importance,device=DEVICE)
    threshold = float(np.quantile(y, .2)/scale)
    rng = np.random.default_rng(seed+100000)
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        losses = []
        for offset in range(0, len(y), 128):
            ids = torch.tensor(order[offset:offset+128],device=DEVICE)
            p = model.normalized_prediction(xx[ids])
            target, weight = yy[ids], ww[ids]
            regression = ((p-target).square()*weight).sum()/weight.sum()
            delta = target[:, None]-target[None, :]
            low = target <= threshold
            mask = torch.triu((delta != 0) & (low[:, None] | low[None, :]), diagonal=1)
            pair_weight = torch.sqrt(weight[:, None]*weight[None, :])
            rank = F.softplus(-delta.sign()*(p[:, None]-p[None, :])/.1)
            ranking = (rank[mask]*pair_weight[mask]).sum()/pair_weight[mask].sum() if mask.any() else p.sum()*0
            loss = regression+.02*ranking
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch+1) % 100 == 0 or epoch+1 == epochs:
            print(f'train seed={seed} epoch={epoch+1}/{epochs} loss={np.mean(losses):.7f}', file=sys.stderr, flush=True)
    path = Path(folder)/f'seed{seed}.npz'
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as out:
        np.savez_compressed(out, **{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    tmp.replace(path)
    return dict(seed=seed, epochs=epochs, seconds=time.monotonic()-started,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def train_ensemble(x, y, original_rows, folder, jobs=2, epoch_override=None):
    configure()
    folder = Path(folder)
    fingerprint = hashlib.sha256(x.tobytes()+y.tobytes()+str(epoch_override).encode()).hexdigest()
    if (folder/'training.json').exists():
        report = json.loads((folder/'training.json').read_text())
        if report.get('determinism') != settings():
            raise RuntimeError('Cached training uses a different numerical policy; choose a new output directory')
        if report['data_fingerprint'] != fingerprint:
            raise RuntimeError('Training resume fingerprint mismatch')
        return load_ensemble(folder)
    folder.mkdir(parents=True, exist_ok=True)
    importance, active, report = sample_weights(x, y, original_rows)
    scale = float(np.median(y[:original_rows]))
    np.savez_compressed(folder/'sample_weights.npz', weights=importance, active=active)
    atomic_json(folder/'weighting.json', report)
    args = [(seed, x[active], y[active], importance[active], scale, folder,
             epoch_override or CONFIG['epochs'][str(seed)]) for seed in CONFIG['seeds']]
    if jobs == 1:
        trained = [train_one(*a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(train_one, *a) for a in args]
            trained = [f.result() for f in futures]
    atomic_json(folder/'training.json', dict(data_fingerprint=fingerprint, scale=scale, models=trained,
                **report, initialization='fresh_fixed_seeds', device=DEVICE, torch=str(torch.__version__), numpy=np.__version__,
                determinism=settings()))
    return load_ensemble(folder)


def load_ensemble(folder):
    models = []
    report = json.loads((Path(folder)/'training.json').read_text())
    for item in report['models']:
        path = Path(folder)/f'seed{item["seed"]}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Model checksum mismatch')
        model = DistrictSurrogate(report['scale'], item['seed'])
        with np.load(path, allow_pickle=False) as z:
            model.load_state_dict({k:torch.tensor(z[k]) for k in z.files})
        model.to(DEVICE).eval().requires_grad_(False)
        models.append(model)
    return models


def predict(models, x):
    with torch.no_grad():
        parts = []
        for start in range(0, len(x), 256):
            w = torch.tensor(x[start:start+256],device=DEVICE)
            parts.append(torch.stack([m(w) for m in models]).cpu().numpy())
    return np.concatenate(parts, axis=1)


def descend(models, starts, kind, steps, rates):
    weights = torch.nn.Parameter(torch.tensor(starts,device=DEVICE))
    optimizer = torch.optim.Adam([weights], lr=rates[0][1], betas=(.9,.999), eps=1e-8,
                                 foreach=False, fused=False)
    best = weights.detach().clone()
    best_score = torch.full((len(starts),), float('inf'), dtype=torch.float64,device=DEVICE)
    for step in range(steps+1):
        if step in dict(rates):
            optimizer.param_groups[0]['lr'] = dict(rates)[step]
        optimizer.zero_grad(set_to_none=True)
        p = torch.stack([m(weights) for m in models])
        score = p.mean(0) if kind == 'mean' else p.max(0).values
        if not torch.isfinite(score).all():
            raise RuntimeError('Nonfinite search score')
        with torch.no_grad():
            improved = score < best_score
            best[improved], best_score[improved] = weights[improved], score[improved]
        if step % 250 == 0:
            print(f'search {kind} step={step}/{steps} best={best_score.min():.7f}', file=sys.stderr, flush=True)
        if step == steps:
            break
        (score.sum()/models[0].scale).backward()
        optimizer.step()
        with torch.no_grad():
            weights.copy_(project(weights))
    return best.cpu().numpy()


def distinct_order(x, order, count, distance):
    picked = []
    for i in order:
        if not picked or np.linalg.norm(x[picked]-x[i], axis=1).min() >= distance:
            picked.append(int(i))
            if len(picked) == count:
                break
    return picked


def generate_batch(models, x, y, round_number, seen, folder, radius=.01, search_steps=None):
    configure()
    rng = np.random.default_rng(20261111+round_number*1009)
    random = np.concatenate([rng.dirichlet(np.full(7, a), size=(8192, 3)).reshape(-1,21) for a in (1., .3)])
    corners = np.asarray([np.eye(7)[list(c)].reshape(21) for c in itertools.product(range(7), repeat=3)])
    pool = np.concatenate([random, x[np.argsort(y, kind='stable')[:64]], corners, np.full((1,21),1/7)])
    p = predict(models, pool)
    low = distinct_order(pool, np.argsort(p.mean(0), kind='stable'), 256, .1)
    starts = np.concatenate([pool[low], random[rng.choice(len(random),256,replace=False)],corners,np.full((1,21),1/7)])
    _, unique = np.unique(starts, axis=0, return_index=True)
    starts = starts[np.sort(unique)]
    steps = search_steps or 1000
    rates1 = [(0,.02),(int(steps*.4),.005),(int(steps*.7),.001)]
    rates2 = [(0,.01),(int(steps*.4),.002),(int(steps*.7),.0004)]
    mean_points = descend(models, starts, 'mean', steps, rates1)
    max_points = descend(models, mean_points, 'max', steps, rates2)
    points = np.concatenate([mean_points,max_points])
    pp = predict(models, points)
    elite = x[centers(x,y)]
    neighbors = []
    scales = [.0001, .001, radius, max(.05, 2*radius)]
    for center in elite:
        for scale in scales:
            for b in (-1,0,1,2):
                trial = np.tile(center,(24,1)).reshape(-1,3,7)
                noise = rng.normal(0,scale,trial.shape)
                if b >= 0:
                    mask = np.zeros((1,3,1)); mask[:,b]=1
                    noise *= mask
                neighbors.extend(project(torch.tensor(trial+noise)).numpy().reshape(-1,21))
        # Explicitly activate and deactivate coordinates to cross action regions.
        trial = np.tile(center,(42,1)).reshape(-1,3,7)
        for j in range(21):
            trial[j].flat[j] = 0
            trial[21+j].flat[j] += .05
        neighbors.extend(project(torch.tensor(trial)).numpy().reshape(-1,21))
    neighbors = np.asarray(neighbors)
    pn = predict(models, neighbors)
    if round_number <= 4:
        quota = [20,15,10,5]
    elif round_number <= 14:
        quota = [20,20,5,5]
    else:
        quota = [15,30,3,2]
    chosen, sources = [], []
    used = set(seen)
    def add(weights, label, count, min_distance):
        added = 0
        for w in weights:
            key = weight_key(w.tolist())
            if key in used or (chosen and np.linalg.norm(np.asarray(chosen)-w,axis=1).min() < min_distance):
                continue
            chosen.append(w.copy()); sources.append(label); used.add(key); added+=1
            if added == count:
                break
        return added
    # Half mean, half conservative, selected from multiple basins.
    mean_order = distinct_order(points,np.argsort(pp.mean(0),kind='stable'),100,.02)
    max_order = distinct_order(points,np.argsort(pp.max(0),kind='stable'),100,.02)
    combined = [v for pair in itertools.zip_longest(max_order,mean_order) for v in pair if v is not None]
    add(points[combined], 'surrogate', quota[0], .01)
    # Select multiple scales and centers, instead of exclusively tiny perturbations.
    neighbor_order = np.argsort(pn.mean(0)+.5*pn.std(0),kind='stable')
    add(neighbors[neighbor_order], 'local', quota[1], 1e-5)
    competitive = np.flatnonzero(p.mean(0) <= np.quantile(p.mean(0), .3))
    uncertainty = competitive[np.argsort(-p[:,competitive].std(0),kind='stable')]
    add(pool[uncertainty], 'disagreement', quota[2], .05)
    # Global coverage uses only distances, no extra labels.
    target = 50-len(chosen)
    global_pool = random[rng.choice(len(random),512,replace=False)]
    known = x.copy()
    for _ in range(target):
        dist = np.full(len(global_pool), np.inf)
        for offset in range(0,len(known),256):
            dist = np.minimum(dist,np.linalg.norm(global_pool[:,None]-known[None,offset:offset+256],axis=2).min(1))
        idx = int(dist.argmax())
        add(global_pool[idx:idx+1], 'global', 1, 1e-5)
        known = np.concatenate([known,global_pool[idx:idx+1]])
        global_pool = np.delete(global_pool,idx,axis=0)
    if len(chosen) < 50:
        add(random, 'fallback-diverse', 50-len(chosen), 1e-6)
    batch = np.asarray(chosen[:50])
    if len(batch) != 50 or len({weight_key(w.tolist()) for w in batch}) != 50:
        raise RuntimeError('Unable to create 50 new unique candidates')
    predictions = predict(models,batch)
    folder = Path(folder)
    atomic_json(folder/'candidates.json',dict(weights=batch.reshape(-1,3,7).tolist(),sources=sources[:50],
                predictions=predictions.T.tolist(),round=round_number,starts=len(starts),radius=radius,
                requested_quotas=quota,actual_counts={s:sources[:50].count(s) for s in set(sources[:50])}))
    np.savez_compressed(folder/'search.npz',mean_points=mean_points,max_points=max_points,predictions=pp)
    return batch, sources[:50], predictions
