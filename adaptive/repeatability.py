"""Independent-process smoke check of training and acquisition; zero oracle calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .protocol import atomic_json, weight_key

ROOT = Path(__file__).resolve().parents[1]


def array_digest(arrays):
    import numpy as np
    h=hashlib.sha256()
    for key,value in sorted(arrays.items()):
        array=np.ascontiguousarray(value,dtype='<f8')
        h.update(key.encode());h.update(str(array.shape).encode());h.update(array.tobytes())
    return h.hexdigest()


def compare(first,second):
    import numpy as np
    if set(first)!=set(second):raise ValueError('Signature array names differ')
    result={}
    for name in sorted(first):
        a,b=first[name],second[name]
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid signature shape or values: '+name)
        result[name]=dict(exact=bool(np.array_equal(a,b)),max_abs_difference=float(np.max(np.abs(a-b))))
    return result


def worker(folder,epochs,steps,perturb):
    from .release import environment,training_data
    environment('cpu')
    from . import base
    from .acquisition import generate_batch
    from .determinism import settings
    import numpy as np
    import torch
    import random
    # Model-local seeds must also override unrelated application RNG consumption.
    random.seed(perturb);np.random.seed(perturb);torch.manual_seed(perturb)
    x,y=training_data(0)
    models=base.train_ensemble(x,y,2048,folder/'models',jobs=1,epoch_override=epochs)
    weights,_,_=generate_batch(models,x,y,1,{weight_key(w.tolist()) for w in x},folder/'proposals',
                              search_steps=steps,search_state={'polish_active':False,'anchor_enabled':False},online=[])
    parameters=np.concatenate([value.detach().cpu().numpy().reshape(-1)
                                for model in models for _,value in sorted(model.state_dict().items())])
    arrays=dict(model_state=parameters,probe_predictions=base.predict(models,x[:64]),
                candidate_weights=weights.reshape(50,3,7),candidate_predictions=base.predict(models,weights))
    np.savez_compressed(folder/'signature.npz',**arrays)
    batch=json.loads((folder/'proposals/candidates.json').read_text())
    report=dict(numerical_digest=array_digest(arrays),runtime=settings(),original_rows=2048,
                seeds=base.CONFIG['seeds'],epochs_per_seed=epochs,search_steps_per_phase=steps,
                oracle_calls=0,candidate_sources=batch['sources'],candidate_roles=batch['roles'])
    atomic_json(folder/'result.json',report)


def check(args):
    from .release import verify_manifest
    verify_manifest()
    import numpy as np
    output=args.output.absolute();output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='independent-runs-',dir=output) as temp:
        reports=[];signatures=[]
        for number in (1,2):
            folder=Path(temp)/str(number)
            env=os.environ.copy()
            env.update(PYTHONHASHSEED=str(number*111),OMP_NUM_THREADS=str(number+2),
                       MKL_NUM_THREADS=str(number+2),CUBLAS_WORKSPACE_CONFIG=':16:8')
            command=[sys.executable,'-m','adaptive.repeatability','--worker',str(folder),
                     '--epochs',str(args.epochs),'--search-steps',str(args.search_steps),'--perturb',str(number*99)]
            done=subprocess.run(command,cwd=ROOT,env=env,text=True,encoding='utf-8',capture_output=True,timeout=600)
            (output/f'run-{number}.log').write_text(done.stdout+done.stderr,encoding='utf-8')
            if done.returncode:raise RuntimeError('Independent worker failed; see '+str(output/f'run-{number}.log'))
            reports.append(json.loads((folder/'result.json').read_text()))
            with np.load(folder/'signature.npz',allow_pickle=False) as archive:
                signatures.append({k:archive[k].copy() for k in archive.files})
            shutil.copyfile(folder/'signature.npz',output/f'run-{number}-signature.npz')
            shutil.copyfile(folder/'proposals/candidates.json',output/f'run-{number}-candidates.json')
        gaps=compare(*signatures)
        same_roles_and_sources=all(reports[0][k]==reports[1][k] for k in ('candidate_sources','candidate_roles'))
        exact=all(v['exact'] for v in gaps.values()) and same_roles_and_sources
        report=dict(policy='strict-v1',passed=exact,comparison=gaps,
                    candidate_roles_and_sources_equal=same_roles_and_sources,runs=reports,oracle_calls=0,
                    scope='Short training and acquisition repeatability, not full20-round or cross-OS performance validation')
        if args.reference:
            with np.load(args.reference,allow_pickle=False) as archive:
                reference={k:archive[k].copy() for k in archive.files}
            report['reference_comparison']=compare(reference,signatures[0])
            report['reference_note']='Reported differences are measurements, not an automatic cross-device guarantee'
        atomic_json(output/'report.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if exact else 1


def positive(value):
    number=int(value)
    if number<=0:raise argparse.ArgumentTypeError('Expected a positive integer')
    return number


if __name__=='__main__':
    from .determinism import bootstrap_cli
    bootstrap_cli()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--epochs',type=positive,required=True)
    parser.add_argument('--search-steps',type=positive,required=True)
    parser.add_argument('--perturb',type=int,required=True)
    args=parser.parse_args()
    args.worker.mkdir(parents=True,exist_ok=False)
    worker(args.worker,args.epochs,args.search_steps,args.perturb)
