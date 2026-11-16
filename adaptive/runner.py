"""Stateful twenty-round optimization, called by the original run.py entry."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .protocol import atomic_json, digest, parse_request, parse_response, weight_key
from .budget import claim_run, validate_mode, reserve_call, complete_call

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'adaptive/data/train.csv'
DATA_HASH = '877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solver-command', type=Path, help='JSON executable argument array')
    p.add_argument('--session', type=Path, default=ROOT/'sessions'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--backend', choices=('recovered','external'), default='external')
    p.add_argument('--evaluator-python', type=Path, default=ROOT.parent/'evaluator-workflow/.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))
    p.add_argument('--workflow', type=Path, default=ROOT/'evaluator-workflow')
    p.add_argument('--wine', type=Path)
    p.add_argument('--wine-prefix', type=Path)
    p.add_argument('--executable', type=Path, default=ROOT/'solver.exe')
    p.add_argument('--eval-workers', type=int, choices=range(1,9), default=8)
    p.add_argument('--train-jobs', type=int, choices=(1,2), default=1)
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='cpu')
    p.add_argument('--gpu', type=int, help='Physical GPU index; omitted to select a low-utilization GPU')
    p.add_argument('--wait-for-gpu', action='store_true')
    p.add_argument('--exclude-gpu', default='')
    p.add_argument('--allow-shared-gpu', action='store_true')
    p.add_argument('--rounds', type=int, choices=range(1,21), default=20)
    p.add_argument('--timeout', type=float, default=3600.)
    p.add_argument('--test-epochs', type=int, help='Smoke-test only; omitted in scientific runs')
    p.add_argument('--test-search-steps', type=int, help='Smoke-test only; omitted in scientific runs')
    p.add_argument('--dry-run', action='store_true', help='Check configuration without training or evaluation')
    return p


def solver_command(args, round_folder):
    if args.backend == 'external':
        if args.solver_command is None:raise ValueError('--backend external requires --solver-command')
        command=json.loads(args.solver_command.read_text(encoding='utf-8'))
        if not isinstance(command,list) or not command or any(not isinstance(v,str) or not v for v in command):
            raise ValueError('Solver command must be a nonempty JSON array of strings')
        return command
    if args.backend == 'native':
        return [str(args.executable)]
    if args.backend == 'wine':
        if not args.wine or not args.wine_prefix:
            raise ValueError('Wine requires --wine and --wine-prefix')
        return [str(args.wine), str(args.executable)]
    return [str(args.evaluator_python),str(ROOT/'solver_bridge.py'),'--workflow',str(args.workflow),
            '--workers',str(args.eval_workers),'--records',str(round_folder/'evaluator')]


def source_fingerprint():
    h = hashlib.sha256()
    for p in [ROOT/'run.py',ROOT/'solver_bridge.py',ROOT/'supervise.py',*sorted((ROOT/'adaptive').glob('*.py')),ROOT/'adaptive/recipe.json']:
        h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def select_device(args):
    from .resources import select_device as automatic
    return automatic(args)


def record_feedback(session, state, folder):
    import numpy as np
    batch = json.loads((folder/'candidates.json').read_text())
    scores = parse_response(json.loads((folder/'stdout.json').read_bytes()),50)
    request = parse_request(json.loads((folder/'request.json').read_text()))
    if (batch['weights'] != request or len(batch['weights']) != 50 or
            len(batch['sources']) != 50 or len(batch['predictions']) != 50 or len(batch['metadata']) != 50 or len(batch['roles']) != 50):
        raise RuntimeError('Candidate / request / output pairing mismatch')
    from .information import splits
    expected_exploit,expected_learn,_=splits(batch['round'])
    if batch['roles'].count('exploit')!=expected_exploit or batch['roles'].count('learn')!=expected_learn:
        raise RuntimeError('Active sampling role quota mismatch')
    if any(m.get('role')!=r for m,r in zip(batch['metadata'],batch['roles'])):
        raise RuntimeError('Role metadata pairing mismatch')
    pairs={}
    for m in batch['metadata']:
        if 'pair_id' in m:pairs.setdefault(m['pair_id'],[]).append(m.get('pair_member'))
    if any(sorted(v)!=[0,1] for v in pairs.values()):raise RuntimeError('Information pair is incomplete')
    if digest(folder/'request.json') != state['pending']['input_sha256']:
        raise RuntimeError('Pending request fingerprint changed')
    if state['pending'].get('candidates_sha256') is not None and digest(folder/'candidates.json') != state['pending']['candidates_sha256']:
        raise RuntimeError('Pre-feedback candidates/shadow fingerprint changed')
    online = json.loads((session/'online.json').read_text()) if (session/'online.json').exists() else []
    existing = {r['id'] for r in online}
    rows = []
    for i,(w,r,source,pred) in enumerate(zip(batch['weights'],scores,batch['sources'],batch['predictions'])):
        rows.append(dict(id=f'r{batch["round"]:02d}-{i+1:02d}',round=batch['round'],index=i+1,
                         weights=w,R=r,source=source,role=batch['roles'][i],predictions_before_feedback=pred,acquisition=batch['metadata'][i]))
    online.extend(r for r in rows if r['id'] not in existing)
    atomic_json(session/'online.json',online)
    with (session/'online.csv.tmp').open('w',encoding='utf-8',newline='') as out:
        writer=csv.writer(out); writer.writerow(['id','round','index','source','role',*[f'b{b}_{j}' for b in range(1,4) for j in range(1,8)],'R'])
        for row in online:
            writer.writerow([row['id'],row['round'],row['index'],row['source'],row['role'],*[v for b in row['weights'] for v in b],row['R']])
    (session/'online.csv.tmp').replace(session/'online.csv')
    best = min(online,key=lambda r:(r['R'],r['id']))
    offline = state['offline_best']
    overall = best if best['R'] < offline['R'] else offline
    atomic_json(session/'best.json',dict(R=overall['R'],weights=overall['weights'],origin=overall['id'],
                online_best=dict(R=best['R'],weights=best['weights'],round=best['round'],index=best['index']),
                offline_best=offline,backend=state['backend'],true_score=True,
                official_executable_used=state['backend'] in ('native','wine'),observations=len(online)))
    pred = np.asarray(batch['predictions']).mean(1)
    truth = np.asarray(scores)
    summary = dict(round=batch['round'],batch_min_R=float(truth.min()),best_online_R=best['R'],
                   cumulative_best_R=overall['R'],prediction_MAE=float(np.abs(pred-truth).mean()),
                   prediction_RMSE=float(np.sqrt(((pred-truth)**2).mean())),
                   source_counts={s:batch['sources'].count(s) for s in sorted(set(batch['sources']))},
                   successful_observations=len(online),model_version=batch['round']-1)
    atomic_json(folder/'feedback.json',summary)
    previous = state.get('cumulative_best_R',state['offline_best']['R'])
    summary['role_counts']={role:batch['roles'].count(role) for role in sorted(set(batch['roles']))}
    summary['role_best_R']={role:min(r['R'] for r in rows if r['role']==role) for role in sorted(set(batch['roles']))}
    summary['role_improvements_over_previous_best']={role:sum(r['role']==role and r['R']<previous-.02 for r in rows) for role in sorted(set(batch['roles']))}
    from .training_audit import assess
    audit=assess(batch,scores)
    if audit is not None:
        summary['information_training_audit']=audit
        audit_folder=session/'learning-audits'/f'after-r{audit["after_feedback_round"]:02d}'
        atomic_json(audit_folder/f'evaluation-on-r{batch["round"]:02d}.json',audit)
    from .acquisition import update_alpha
    summary['scale_update'] = update_alpha(state,scores,previous,overall['R'],batch=batch)
    from .anchor import update_feedback
    summary['v3_transition'] = update_feedback(state,batch,scores,previous,overall['R'])
    summary['phase'] = batch['phase']
    summary['anchor_used_for_selection'] = batch['anchor_used_for_selection']
    summary['source_best_R'] = {source:min(r['R'] for r in rows if r['source']==source) for source in sorted(set(batch['sources']))}
    summary['transfer_bin_counts'] = batch['transfer_bin_counts']
    summary['unranked_count'] = batch['unranked_count']
    summary['local_pair_accuracy'] = pair_accuracy(truth,pred)
    atomic_json(folder/'feedback.json',summary)
    state['cumulative_best_R'] = overall['R']
    state['best_online_R']=best['R']
    state['completed_rounds']=batch['round']
    state['pending']=None
    state['status']='ready'
    atomic_json(session/'state.json',state)
    summaries=[json.loads(p.read_text()) for p in sorted((session/'rounds').glob('*/feedback.json'))]
    with (session/'progress.csv').open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=[k for k,v in summary.items() if not isinstance(v,(dict,list))])
        writer.writeheader(); writer.writerows({k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in summaries)
    print(f'ROUND {batch["round"]}: min={truth.min():.12f} best={overall["R"]:.12f} MAE={summary["prediction_MAE"]:.5f}',file=sys.stderr,flush=True)


def pair_accuracy(truth,prediction):
    import numpy as np
    truth,prediction=np.asarray(truth),np.asarray(prediction)
    delta=truth[:,None]-truth[None,:]
    mask=np.triu(delta!=0,1)
    if not mask.any():
        return None
    return float(((prediction[:,None]-prediction[None,:])[mask]*delta[mask]>0).mean())


def diagnose(models,x,y,predict,path):
    import numpy as np
    p=predict(models,x).mean(0)
    low=y<=np.quantile(y,.1)
    atomic_json(path,dict(original_MAE=float(np.abs(p-y).mean()),
                         original_low10_MAE=float(np.abs(p[low]-y[low]).mean()),
                         original_low10_pair_accuracy=pair_accuracy(y[low],p[low])))


def finalize_report(session,state,allocation,started,args):
    summaries=[json.loads(p.read_text()) for p in sorted((session/'rounds').glob('*/feedback.json'))]
    profiles=[json.loads(p.read_text()) for p in sorted((session/'rounds').glob('*/timings.json'))]
    metadata=[json.loads(p.read_text()) for p in sorted((session/'models').glob('*/training.json'))]
    timing=dict(training_seconds=sum(m['wall_seconds'] for m in metadata),
                search_seconds=sum(p.get('search_seconds',0.) for p in profiles),
                evaluation_seconds=sum(p.get('evaluation_seconds',0.) for p in profiles),
                worker_seconds_this_invocation=time.monotonic()-started,
                session_wall_seconds=(datetime.fromisoformat(state['completed_at'])-datetime.fromisoformat(state['started_at'])).total_seconds(),
                GPU_wait_seconds=allocation['first_GPU_wait_seconds'],
                reused_pretraining_seconds=sum(m.get('historical_pretraining_seconds',0.) for m in metadata))
    audit_metadata=[json.loads(p.read_text()) for p in sorted((session/'learning-audits').glob('*/training.json'))]
    timing['diagnostic_training_seconds']=sum(p['wall_seconds'] for p in audit_metadata)
    timing['all_training_seconds']=timing['training_seconds']+timing['diagnostic_training_seconds']
    timing['total_with_GPU_wait_seconds']=timing['session_wall_seconds']+timing['GPU_wait_seconds']
    atomic_json(session/'timing-summary.json',timing)
    best=json.loads((session/'best.json').read_text())
    outcome=dict(target_R=104.,achieved=best['R']<=104.,best_real_R=best['R'],
                 calls=state['attempted_calls'],evaluations=state['completed_rounds']*50,
                 extra_evaluations=0,recommendation_is_observed=True)
    outcome.update(polish_trigger_round=state.get('polish_trigger_round'),
                   polish_rounds=sum(r.get('phase')=='feedback_triggered_polish' for r in summaries),
                   anchor_enabled_rounds=[r['round'] for r in summaries if r.get('anchor_used_for_selection')])
    all_rows=json.loads((session/'online.json').read_text())
    outcome['role_totals']={role:sum(r['role']==role for r in all_rows) for role in ('exploit','learn')}
    outcome['best_R_by_role']={role:min(r['R'] for r in all_rows if r['role']==role) for role in ('exploit','learn') if any(r['role']==role for r in all_rows)}
    outcome['information_training_audits']=[r['information_training_audit'] for r in summaries if 'information_training_audit' in r]
    atomic_json(session/'outcome.json',outcome)
    with (session/'REPORT.md').open('a') as output:
        output.write('\n目标 R ≤104：'+('达到' if outcome['achieved'] else '未达到')+'。预算用完后不追加评估。\n')
        output.write('\n训练 / 寻优 / 真实评估秒数：'+str([timing[k] for k in ('training_seconds','search_seconds','evaluation_seconds')])+'。\n')
        output.write('初始模型从原始2048行从头训练；复用预训练时间为 '+str(timing['reused_pretraining_seconds'])+' 秒。\n')
    with (session/'REPORT.md').open('a') as output:
        output.write('\n冲分/补信息名额合计：'+json.dumps(outcome['role_totals'],ensure_ascii=False)+'；分组最低真实R：'+json.dumps(outcome['best_R_by_role'],ensure_ascii=False)+'。\n')
        output.write('诊断影子训练耗时单独记录为 '+str(timing['diagnostic_training_seconds'])+' 秒；不参与选点，不新增真实评估。\n')
        for audit in outcome['information_training_audits']:
            output.write('\n第'+str(audit['evaluated_round'])+'轮条件性诊断，加入信息组后的MAE改善：'+str(audit['MAE_improvement_from_information'])+'；不能解释为全空间因果结论。\n')
    # This file is explicitly denied by the audit hook until all20 rounds finish.
    reference=ROOT/'baseline_reference.json'
    if args.rounds==20 and args.test_epochs is None and reference.exists():
        old=json.loads(reference.read_text())
        comparison=dict(references=old,new=outcome,timing=timing,
            same_evaluation_count=[dict(round=r['round'],new_R=r['cumulative_best_R'],
                baseline_R=old['v3_frozen']['best_R_by_round'][r['round']-1]) for r in summaries])
        atomic_json(session/'comparison.json',comparison)
        with (session/'REPORT.md').open('a') as output:
            output.write('\n历史结果仅用于完成后的对照：'+json.dumps({k:v['best_R'] for k,v in old.items()},ensure_ascii=False)+'。\n')


def main():
    args=parser().parse_args()
    if args.backend=='auto':
        args.backend='native' if os.name=='nt' else 'recovered'
    for key in ('session','evaluator_python','workflow','executable','wine','wine_prefix'):
        if getattr(args,key) is not None:
            # Preserve a virtualenv's interpreter symlink: resolving it would
            # run the base interpreter without that environment's packages.
            setattr(args,key,getattr(args,key).absolute())
    if digest(DATA)!=DATA_HASH:
        raise RuntimeError('Original CSV changed')
    recipe=json.loads((ROOT/'adaptive/recipe.json').read_text())
    if digest(ROOT/'adaptive/acquisition_v1.py')!=recipe['frozen_v1_acquisition_sha256']:
        raise RuntimeError('Frozen v1 acquisition changed')
    if args.backend=='recovered' and (not args.evaluator_python.exists() or not (args.workflow/'src/pipeline.py').exists()):
        raise RuntimeError('Missing recovered evaluation environment or workflow')
    if args.backend in ('native','wine') and not args.executable.exists():
        raise RuntimeError('Missing official executable')
    if args.dry_run:
        print(json.dumps(dict(backend=args.backend,rounds=args.rounds,batch_size=50,training_data=str(DATA),
                              evaluation_command=solver_command(args,args.session/'rounds/01')),ensure_ascii=False))
        return 0
    if sys.version_info[:2]!=(3,10):
        raise RuntimeError('Training entry requires Python3.10 / PyTorch2.6 / NumPy1.26.4; evaluator uses a separate Python3.11')
    if args.backend=='external':solver_command(args,args.session/'rounds/01')
    validate_mode(args)
    claim_run(args)
    worker_started=time.monotonic()
    allocation=select_device(args)
    from .learning import configure, train_ensemble, generate_batch, verify_original_data, predict
    import numpy as np
    import torch
    configure()
    if np.__version__!='1.26.4' or str(torch.__version__).split('+')[0]!='2.6.0':
        raise RuntimeError('Use the pinned training environment')
    with DATA.open(encoding='utf-8-sig',newline='') as f:
        original=sorted(list(csv.DictReader(f)),key=lambda r:r['candidate_id'])
    names=json.loads((ROOT/'adaptive/recipe.json').read_text())['input_feature_order']
    original_x=np.asarray([[float(row[k]) for k in names] for row in original])
    original_y=np.asarray([float(row['ramping']) for row in original])
    verify_original_data(original_x,original_y)
    from .determinism import settings
    session=args.session
    config=dict(backend=args.backend,rounds=args.rounds,batch_size=50,data_sha256=DATA_HASH,
                solver_command=solver_command(args,args.session/'rounds/01') if args.backend=='external' else None,
                source_sha256=source_fingerprint(),eval_workers=args.eval_workers,train_jobs=args.train_jobs,
                evaluator_python=str(args.evaluator_python),workflow=str(args.workflow),executable=str(args.executable),
                wine=str(args.wine) if args.wine else None,wine_prefix=str(args.wine_prefix) if args.wine_prefix else None,
                test_epochs=args.test_epochs,test_search_steps=args.test_search_steps,
                training_environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__),
                determinism=settings(),
                requested_device=args.device,gpu=args.gpu,allow_shared_gpu=args.allow_shared_gpu,exclude_gpu=args.exclude_gpu,
                acquisition=json.loads((ROOT/'adaptive/recipe.json').read_text()),
                initialization='fresh_fixed_seeds_original_only',
                historical_extra_labels_used=False)
    if (session/'state.json').exists():
        if not args.resume:
            raise RuntimeError('Session exists; use --resume')
        saved=json.loads((session/'config.json').read_text())
        if saved!=config:
            raise RuntimeError('Resume config/source changed')
        state=json.loads((session/'state.json').read_text())
    else:
        session.mkdir(parents=True,exist_ok=False)
        i=int(original_y.argmin())
        offline_best=dict(id=original[i]['candidate_id'],R=float(original_y[i]),weights=original_x[i].reshape(3,7).tolist())
        state=dict(status='ready',backend=args.backend,completed_rounds=0,attempted_calls=0,pending=None,
                   offline_best=offline_best,radius=1.,small_improvement_streak=0,good_improvement_streak=0,
                   cumulative_best_R=offline_best['R'],target_R=104.,polish_active=False,anchor_enabled=False,
                   polish_stall_streak=0,anchor_history=[],started_at=datetime.now().isoformat())
        atomic_json(session/'config.json',config); atomic_json(session/'state.json',state)
        snapshot=session/'source'; snapshot.mkdir()
        for path in [ROOT/'run.py',ROOT/'solver_bridge.py',ROOT/'supervise.py',ROOT/'launch_background.py',*sorted((ROOT/'adaptive').glob('*.py')),ROOT/'adaptive/recipe.json',DATA]:
            target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    previous_allocation=json.loads((session/'allocation.json').read_text()) if (session/'allocation.json').exists() else {}
    allocation['first_GPU_wait_seconds']=previous_allocation.get('first_GPU_wait_seconds',allocation.get('wait_gpu_seconds',0.))
    atomic_json(session/'allocation.json',allocation)
    print('Training allocation: '+json.dumps(allocation),file=sys.stderr,flush=True)
    lock=(session/'session.lock').open('a+')
    if os.name=='nt':
        import msvcrt
        lock.seek(0); lock.write('0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if state['pending']:
        folder=session/'rounds'/f'{state["pending"]["round"]:02d}'
        if not (folder/'returncode.json').exists() or json.loads((folder/'returncode.json').read_text())['returncode']!=0:
            raise RuntimeError('Previous evaluator call unresolved; no automatic retry or budget reset')
        record_feedback(session,state,folder)
        complete_call(args,state['completed_rounds'],digest(folder/'request.json'))
    while state['completed_rounds']<args.rounds:
        round_number=state['completed_rounds']+1
        if state['attempted_calls']>=args.rounds:
            raise RuntimeError('Evaluation budget exhausted')
        online=json.loads((session/'online.json').read_text()) if (session/'online.json').exists() else []
        if online:
            x=np.concatenate([original_x,np.asarray([r['weights'] for r in online]).reshape(-1,21)])
            y=np.concatenate([original_y,np.asarray([r['R'] for r in online])])
        else:
            x,y=original_x,original_y
        folder=session/'rounds'/f'{round_number:02d}'; folder.mkdir(parents=True,exist_ok=True)
        print(f'ROUND {round_number}/{args.rounds}: train rows={len(y)}',file=sys.stderr,flush=True)
        state['status']='training'; atomic_json(session/'state.json',state)
        models=train_ensemble(x,y,len(original_y),session/'models'/f'v{round_number-1:02d}',args.train_jobs,args.test_epochs)
        folder=session/'rounds'/f'{round_number:02d}'; folder.mkdir(parents=True,exist_ok=True)
        from .training_audit import prepare,attach,AFTER_ROUNDS
        if round_number-1 in AFTER_ROUNDS:
            state['status']='diagnostic_training';atomic_json(session/'state.json',state)
        diagnostic,diagnostic_seconds=prepare(session,models,x,y,online,round_number,args.test_epochs)
        training=json.loads((session/'models'/f'v{round_number-1:02d}'/'training.json').read_text())
        timings=json.loads((folder/'timings.json').read_text()) if (folder/'timings.json').exists() else {}
        timings['training_seconds']=training['wall_seconds']
        timings['diagnostic_training_seconds']=diagnostic_seconds
        metrics_path=session/'models'/f'v{round_number-1:02d}'/'diagnostics.json'
        if not metrics_path.exists():
            diagnose(models,original_x,original_y,predict,metrics_path)
        state['status']='searching'; atomic_json(session/'state.json',state)
        seen={weight_key(w.tolist()) for w in x}
        if (folder/'candidates.json').exists():
            batch=json.loads((folder/'candidates.json').read_text())
            weights=np.asarray(batch['weights']).reshape(-1,21)
        else:
            search_started=time.monotonic()
            weights,_,_=generate_batch(models,x,y,round_number,seen,folder,state['radius'],args.test_search_steps,
                                       search_state=state,online=online)
            timings['search_seconds']=time.monotonic()-search_started
        batch=json.loads((folder/'candidates.json').read_text())
        if diagnostic is not None and 'information_model_comparison' not in batch:
            batch=attach(batch,models,diagnostic);atomic_json(folder/'candidates.json',batch)
        atomic_json(folder/'timings.json',timings)
        request=dict(weights=weights.reshape(-1,3,7).tolist())
        assert len(parse_request(request))==50
        atomic_json(folder/'request.json',request)
        atomic_json(ROOT/'weight.json',request)
        if len({weight_key(w) for w in request['weights']}) != 50 or any(weight_key(w) in seen for w in request['weights']):
            raise RuntimeError('Previously evaluated or duplicate candidates blocked before call')
        state['attempted_calls']+=1
        state['pending']=dict(round=round_number,input_sha256=digest(folder/'request.json'),
                              candidates_sha256=digest(folder/'candidates.json'))
        state['status']='evaluating'
        atomic_json(session/'state.json',state)
        reserve_call(args,round_number,state['pending']['input_sha256'])
        env=os.environ.copy()
        if args.backend=='wine':
            env.update(WINEPREFIX=str(args.wine_prefix),WINEDEBUG='-all',WINEDLLOVERRIDES='winegstreamer=;winemenubuilder.exe=')
        started=time.monotonic()
        print(f'ROUND {round_number}: evaluate 50 weights backend={args.backend}',file=sys.stderr,flush=True)
        with (folder/'request.json').open('rb') as stdin,(folder/'stdout.json').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
            completed=subprocess.run(solver_command(args,folder),stdin=stdin,stdout=stdout,stderr=stderr,
                                     cwd=ROOT,env=env,timeout=args.timeout)
        atomic_json(folder/'returncode.json',dict(returncode=completed.returncode,seconds=time.monotonic()-started))
        timings['evaluation_seconds']=time.monotonic()-started
        atomic_json(folder/'timings.json',timings)
        if completed.returncode!=0:
            raise RuntimeError(f'Evaluator failed ({completed.returncode}); outputs saved; no automatic retry')
        record_feedback(session,state,folder)
        complete_call(args,round_number,digest(folder/'request.json'))
    # Include the last returned batch in the delivered model as well. This
    # refit never submits another batch and cannot consume a twenty-first call.
    state['status']='final_training'
    atomic_json(session/'state.json',state)
    online=json.loads((session/'online.json').read_text())
    final_x=np.concatenate([original_x,np.asarray([r['weights'] for r in online]).reshape(-1,21)])
    final_y=np.concatenate([original_y,np.asarray([r['R'] for r in online])])
    train_ensemble(final_x,final_y,len(original_y),session/'models'/f'v{args.rounds:02d}',args.train_jobs,args.test_epochs)
    from .base import load_ensemble
    diagnose(load_ensemble(session/'models'/f'v{args.rounds:02d}'),original_x,original_y,predict,
             session/'models'/f'v{args.rounds:02d}'/'diagnostics.json')
    best=json.loads((session/'best.json').read_text())
    summaries=[json.loads(p.read_text()) for p in sorted((session/'rounds').glob('*/feedback.json'))]
    lines=['# 软决策树 104-v4 主动采样独立实验结果','',f'后端：{args.backend}；正式调用：{state["attempted_calls"]}；本次新标签：{len(online)}。',
           f'最终最低真实 R：**{best["R"]:.12f}**，来源：{best["origin"]}。',
           f'原始最好 R：{state["offline_best"]["R"]:.12f}；本次线上最好 R：{best["online_best"]["R"]:.12f}。','',
           '| 轮次 | 本轮最低 R | 累计最低 R | 提交前预测 MAE |','|---:|---:|---:|---:|']
    lines.extend(f'| {r["round"]} | {r["batch_min_R"]:.9f} | {r["cumulative_best_R"]:.9f} | {r["prediction_MAE"]:.5f} |' for r in summaries)
    lines.extend(['','最终权重：','```json',json.dumps(best['weights'],ensure_ascii=False,indent=2),'```','',
                  '全部结果来自本次正式反馈或原始 CSV；提交前预测与真实 R 分开保存。逆向后端不声称运行了 Windows EXE。'])
    (session/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state['status']='complete'; state['completed_at']=datetime.now().isoformat()
    atomic_json(session/'state.json',state)
    finalize_report(session,state,allocation,worker_started,args)
    print((session/'best.json').read_text(),flush=True)
    return 0
