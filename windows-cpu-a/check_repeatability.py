"""Two independent short CPU runs, no native evaluator invocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'project'))


def worker(folder):
    import numpy as np
    from adaptive import base
    from adaptive.acquisition import generate_batch
    from adaptive.release import training_data
    from adaptive.protocol import weight_key
    folder.mkdir();base.configure();x,y=training_data(0)
    models=base.train_ensemble(x,y,2048,folder/'models',jobs=1,epoch_override=3)
    signature=dict(parameters=np.concatenate([v.detach().cpu().numpy().reshape(-1) for model in models for _,v in sorted(model.state_dict().items())]))
    for number,polish in ((1,False),(7,False),(8,False),(19,True)):
        q,_,p=generate_batch(models,x,y,number,{weight_key(w.tolist()) for w in x},folder/f'r{number}',search_steps=4,
            search_state=dict(polish_active=polish,anchor_enabled=polish),online=[])
        signature[f'weights{number}']=q;signature[f'predictions{number}']=p
    np.savez_compressed(folder/'signature.npz',**signature)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path);args=parser.parse_args()
    if args.worker:worker(args.worker);return
    import numpy as np
    signatures=[];out=ROOT/'verification/repeatability';out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='structure-repeat-') as tmp:
        for n in (1,2):
            path=Path(tmp)/str(n);env=os.environ.copy();env['PYTHONHASHSEED']=str(n*137)
            with (out/f'run{n}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(Path(__file__)), '--worker',str(path)],env=env,stdout=log,stderr=log,check=True,timeout=600)
            with np.load(path/'signature.npz') as z:signatures.append({k:z[k].copy() for k in z.files})
    comparison={k:dict(exact=bool(np.array_equal(signatures[0][k],signatures[1][k])),max_abs=float(np.abs(signatures[0][k]-signatures[1][k]).max())) for k in signatures[0]}
    result=dict(passed=all(v['exact'] for v in comparison.values()),comparison=comparison,epochs_per_seed=3,search_steps=4,native_calls=0)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Short flow is not repeatable')


if __name__=='__main__':main()
