"""Offline release verification, fixed-feedback replay, training and proposals."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

from .protocol import atomic_json, digest, parse_request, parse_response, weight_key

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/formal-v4'
DATA=ROOT/'adaptive/data/train.csv'
DATA_HASH='877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023'


def verify_manifest():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or digest(path)!=expected:
            raise RuntimeError('Release file changed: '+name)
    return len(manifest['files'])


def records():
    if digest(DATA)!=DATA_HASH:
        raise RuntimeError('Original 2048-row data changed')
    with DATA.open(encoding='utf-8-sig',newline='') as handle:
        rows=sorted(csv.DictReader(handle),key=lambda r:r['candidate_id'])
    if len(rows)!=2048 or len({r['candidate_id'] for r in rows})!=2048:
        raise RuntimeError('Expected 2048 unique original rows')
    names=json.loads((ROOT/'adaptive/recipe.json').read_text())['input_feature_order']
    original=[dict(id=r['candidate_id'],weights=[[float(r[k]) for k in names[b*7:(b+1)*7]] for b in range(3)],
                   R=float(r['ramping'])) for r in rows]
    return original


def verify_report():
    checked=verify_manifest()
    original=records();known={weight_key(r['weights']) for r in original};online=[]
    best=min(original,key=lambda r:r['R'])
    visible_R=[r['R'] for r in original]
    recipe=json.loads((ROOT/'adaptive/recipe.json').read_text())
    role_schedule=recipe['active_sampling']['role_schedule']
    audit_rounds=recipe['information_audit']['after_feedback_rounds']
    role_totals={'exploit':0,'learn':0}
    listed_before_feedback=[]
    archived_rows=json.loads((RESULT/'online.json').read_text())
    for number in range(1,21):
        folder=RESULT/'rounds'/f'{number:02d}'
        request=parse_request(json.loads((folder/'request.json').read_text()))
        scores=parse_response(json.loads((folder/'stdout.json').read_text()),50)
        batch=json.loads((folder/'candidates.json').read_text())
        shadow=batch['shadow']
        if (shadow['available_rows']!=2048+50*(number-1) or not shadow['generated_before_feedback']
                or batch['prediction_model_version']!=number-1
                or shadow['raw_predictions']!=batch['predictions']):
            raise RuntimeError('Shadow predictions do not match pre-feedback model/data version')
        for index,value in zip(shadow['center_indices'],shadow['center_R']):
            if not 0<=index<len(visible_R) or visible_R[index]!=value:
                raise RuntimeError('Shadow center refers to future or mismatched feedback')
        target=next((e,l) for lo,hi,e,l in role_schedule if lo<=number<=hi)
        if (batch['roles'].count('exploit'),batch['roles'].count('learn'))!=target:
            raise RuntimeError('Archived role quotas do not match frozen schedule')
        pair_members={}
        for role,meta in zip(batch['roles'],batch['metadata']):
            if meta['role']!=role:raise RuntimeError('Role/source pairing mismatch')
            role_totals[role]+=1
            if 'pair_id' in meta:pair_members.setdefault(meta['pair_id'],[]).append(meta['pair_member'])
            for endpoint in meta.get('interval_endpoints',[]):
                if not 0<=endpoint['index']<len(visible_R) or visible_R[endpoint['index']]!=endpoint['R']:
                    raise RuntimeError('Boundary query used future or mismatched truth')
        if any(sorted(v)!=[0,1] for v in pair_members.values()):raise RuntimeError('Unmatched contrast pair')
        if number-1 in audit_rounds:
            comparison=batch['information_model_comparison']
            expected_ids=[r['id'] for r in listed_before_feedback if r['round']==number-1 and r['role']=='learn']
            if comparison['excluded_sample_ids']!=expected_ids or not comparison['generated_before_feedback'] or comparison['used_for_acquisition']:
                raise RuntimeError('Training-value diagnostic used wrong information group')
            if comparison['after_feedback_round']!=number-1 or comparison['full_rows']!=len(visible_R):
                raise RuntimeError('Training-value diagnostic used wrong data version')
        if request!=batch['weights'] or len(request)!=50:
            raise RuntimeError('Recorded weight/score pairing mismatch')
        for i,(w,r) in enumerate(zip(request,scores)):
            key=weight_key(w)
            if key in known:
                raise RuntimeError('Repeated weight in recorded run')
            known.add(key)
            online.append(dict(id=f'r{number:02d}-{i+1:02d}',weights=w,R=r,round=number,index=i+1))
        listed_before_feedback.extend(archived_rows[(number-1)*50:number*50])
        visible_R.extend(scores)
        best=min([best,*online[-50:]],key=lambda r:r['R'])
        feedback=json.loads((folder/'feedback.json').read_text())
        if feedback['batch_min_R']!=min(scores) or feedback['cumulative_best_R']!=best['R']:
            raise RuntimeError('Recorded progress does not match feedback')
    saved=json.loads((RESULT/'best.json').read_text())
    listed=json.loads((RESULT/'online.json').read_text())
    if len(listed)!=1000 or len(online)!=1000:
        raise RuntimeError('Expected exactly 1000 archived feedback rows')
    for a,b in zip(online,listed):
        if any(a[k]!=b[k] for k in ('id','weights','R','round','index')):
            raise RuntimeError('Archive index / response pairing mismatch')
    if saved['R']!=best['R'] or saved['weights']!=best['weights'] or saved['origin']!=best['id']:
        raise RuntimeError('Saved recommendation is not the best observed point')
    state=json.loads((RESULT/'state.json').read_text())
    if state['completed_rounds']!=20 or state['attempted_calls']!=20 or state['status']!='complete':
        raise RuntimeError('Experiment was not a completed20x50 run')
    import math
    trace=json.loads((RESULT/'verification/best-net-traces.json').read_text())
    if len(trace['weeks'])!=8 or trace['source_case']!=best['id']:raise RuntimeError('Best trace identity mismatch')
    weekly=[]
    for week in trace['weeks']:
        net=week['net'];buildings=week['building_nets']
        if len(net)!=168 or len(buildings)!=3 or any(len(b)!=168 for b in buildings):raise RuntimeError('Invalid trace shape')
        if any(abs(sum(b[t] for b in buildings)-net[t])>1e-9 for t in range(168)):raise RuntimeError('Building/district trace mismatch')
        weekly.append(math.fsum(abs(net[t]-net[t-1]) for t in range(1,168)))
    recomputed=math.fsum(weekly)/8
    if abs(recomputed-best['R'])>1e-9 or trace['R']!=best['R']:raise RuntimeError('Archived trace does not reproduce true R')
    return dict(version='4.0.2',experiment_version='4.0.0',files_checked=checked,original_rows=2048,feedback_rows=1000,
                real_R=best['R'],origin=best['id'],weights=best['weights'],target_R=104.,
                target_achieved=best['R']<=104.,score_type='archived_real_evaluation',
                evaluator_called=False,models_retrained=False,role_totals=role_totals,trajectory_recomputed_R=recomputed)


def environment(device='cpu',gpu=None):
    if sys.version_info[:3]!=(3,10,21):
        raise RuntimeError('Use the pinned Python3.10.21 training environment; run bootstrap.py')
    from .runner import select_device
    # Device must be selected before importing modules that bind base.DEVICE.
    allocation=select_device(argparse.Namespace(device=device,gpu=gpu,allow_shared_gpu=False,
                                                wait_for_gpu=False,exclude_gpu='',train_jobs=1))
    from .base import configure
    import numpy as np
    import torch
    configure()
    if np.__version__!='1.26.4' or str(torch.__version__).split('+')[0]!='2.6.0':
        raise RuntimeError('Expected NumPy1.26.4 and Torch2.6.0')
    return allocation


def training_data(round_number):
    import numpy as np
    rows=records()
    if round_number:
        feedback=json.loads((RESULT/'online.json').read_text())
        rows.extend(r for r in feedback if r['round']<=round_number)
    return np.asarray([r['weights'] for r in rows]).reshape(-1,21),np.asarray([r['R'] for r in rows])


def train(args):
    verify_manifest();allocation=environment(args.device,args.gpu)
    from . import base,learning
    x,y=training_data(args.round)
    output=args.output.absolute()
    output.mkdir(parents=True,exist_ok=True)
    from .determinism import settings
    if (output/'training.json').exists():
        cached=json.loads((output/'training.json').read_text())
        if cached.get('determinism')!=settings():
            raise RuntimeError('Cached training uses a different numerical policy; choose a new output directory')
    function=learning.train_ensemble
    started=time.monotonic();models=function(x,y,2048,output,1)
    reference=RESULT/'models'/f'v{args.round:02d}'
    original_models=base.load_ensemble(reference)
    import numpy as np
    probes=x[:64]
    gap=np.abs(base.predict(models,probes)-base.predict(original_models,probes))
    report=dict(round=args.round,training_rows=len(y),original_rows=2048,online_rows=len(y)-2048,
                initialization='fresh_fixed_seeds',
                reference_probe_prediction_max_abs_difference=float(gap.max()),
                reference_probe_prediction_MAE=float(gap.mean()),device=allocation,
                seconds=time.monotonic()-started,evaluator_called=False,determinism=settings(),
                note='CPU/GPU training paths may differ; reported drift is measured, not a bitwise guarantee')
    atomic_json(output/'reproduction.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('doctor','test','verify-report'):
        sub.add_parser(name)
    reproduction=sub.add_parser('reproduce',help='Verify and export archived real results without an oracle')
    reproduction.add_argument('--output',type=Path,default=ROOT/'outputs/reproduced')
    stability=sub.add_parser('check-determinism',help='Two fresh CPU training/acquisition processes; no evaluator')
    from .repeatability import positive
    stability.add_argument('--epochs',type=positive,default=3)
    stability.add_argument('--search-steps',type=positive,default=8)
    stability.add_argument('--output',type=Path,default=ROOT/'outputs/determinism')
    stability.add_argument('--reference',type=Path,help='Optional signature.npz from another reference computer')
    fitting=sub.add_parser('train',help='Train on original rows plus feedback through --round')
    fitting.add_argument('--round',type=int,choices=range(21),default=0)
    fitting.add_argument('--from-scratch',action='store_true')
    fitting.add_argument('--output',type=Path,default=ROOT/'outputs/trained')
    replay=sub.add_parser('replay',help='Retrain every version using the exact archived feedback sequence')
    replay.add_argument('--output',type=Path,default=ROOT/'outputs/replay')
    replay.add_argument('--from-scratch',action='store_true',help='Also retrain the original-only v00')
    optimize=sub.add_parser('optimize',help='Generate50 unscored proposals from the frozen acquisition rule')
    optimize.add_argument('--round',type=int,choices=range(21),default=20)
    optimize.add_argument('--model',type=Path)
    optimize.add_argument('--output',type=Path,default=ROOT/'outputs/proposals')
    for p in (fitting,replay,optimize):
        p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        p.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.command in ('verify-report','reproduce'):
        report=verify_report()
        if args.command=='reproduce':
            args.output.mkdir(parents=True,exist_ok=True)
            atomic_json(args.output/'best.json',report)
            atomic_json(args.output/'weights.json',dict(weights=[report['weights']]))
        print(json.dumps(report,ensure_ascii=False,indent=2));return 0
    if args.command=='doctor':
        checked=verify_manifest();allocation=environment()
        import numpy as np,torch
        from .learning import verify_original_data
        x,y=training_data(0);verify_original_data(x,y)
        from .determinism import settings
        print(json.dumps(dict(version='4.0.2',files_checked=checked,python=platform.python_version(),
                             numpy=np.__version__,torch=str(torch.__version__),allocation=allocation,
                             determinism=settings()),indent=2));return 0
    if args.command=='check-determinism':
        from .repeatability import check
        return check(args)
    if args.command=='test':
        environment('cpu')
        import unittest
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'))).wasSuccessful() else 1
    if args.command=='train':
        train(args);return 0
    if args.command=='replay':
        verify_report()
        reports=[]
        root=args.output
        for number in range(21):
            item=argparse.Namespace(**vars(args));item.round=number;item.output=root/'models'/f'v{number:02d}'
            reports.append(train(item))
        atomic_json(root/'replay-summary.json',dict(feedback_replayed=1000,oracle_calls=0,versions=reports,
                    explanation='Exact archived input/score pairs replayed; no scores assigned to newly generated proposals'))
        return 0
    if args.command=='optimize':
        verify_manifest();environment(args.device,args.gpu)
        from .base import load_ensemble
        from .acquisition import generate_batch
        x,y=training_data(args.round)
        models=load_ensemble(args.model or RESULT/'models'/f'v{args.round:02d}')
        alpha=1. if args.round==0 else json.loads((RESULT/'rounds'/f'{args.round:02d}'/'feedback.json').read_text())['scale_update']['alpha_next']
        # A free offline proposal computation; never requests an extra evaluation.
        if args.round:
            f=json.loads((RESULT/'rounds'/f'{args.round:02d}'/'feedback.json').read_text())['v3_transition']
            policy_state={'polish_active':f['polish_active_next'],'anchor_enabled':f['anchor_enabled_next']}
        else:policy_state={'polish_active':False,'anchor_enabled':False}
        history=[r for r in json.loads((RESULT/'online.json').read_text()) if r['round']<=args.round]
        proposal_round=min(args.round+1,20)
        generate_batch(models,x,y,proposal_round,{weight_key(w.tolist()) for w in x},args.output,alpha,
                       search_state=policy_state,online=history)
        from .determinism import settings
        atomic_json(args.output/'score-status.json',dict(predictions_only=True,oracle_calls=0,determinism=settings(),
                    true_R_available_for_new_proposals=False,proposal_generator_round=proposal_round,training_feedback_round=args.round,formal_round_submission=False))
        print(json.dumps(dict(output=str(args.output),candidates=50,score_type='model_predictions',oracle_calls=0)));return 0
    return 0
