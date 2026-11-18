"""Single-seed diagnostics of the latest information group, never acquisition."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from . import base
from .protocol import atomic_json,digest

AFTER_ROUNDS=(4,8,12,16)


def prepare(session,models,x,y,online,round_number,epoch_override=None):
    after=round_number-1
    if after not in AFTER_ROUNDS:return None,0.
    excluded=[2048+i for i,r in enumerate(online) if r['round']==after and r['role']=='learn']
    if not excluded:raise RuntimeError('Scheduled information audit has no latest information samples')
    keep=np.ones(len(y),dtype=bool);keep[excluded]=False;xx,yy=x[keep],y[keep]
    epochs=epoch_override or base.CONFIG['epochs']['2026'];folder=Path(session)/'learning-audits'/f'after-r{after:02d}'
    fingerprint=hashlib.sha256(xx.tobytes()+yy.tobytes()+str(epochs).encode()).hexdigest()
    if (folder/'training.json').exists():
        report=json.loads((folder/'training.json').read_text())
        if report['data_fingerprint']!=fingerprint:raise RuntimeError('Diagnostic resume data mismatch')
        return dict(model=base.load_ensemble(folder)[0],folder=folder,report=report),report['wall_seconds']
    folder.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    importance,active,weighting=base.sample_weights(xx,yy,2048);scale=float(np.median(y[:2048]))
    np.savez_compressed(folder/'training_samples.npz',x=xx,y=yy,retained_global_indices=np.flatnonzero(keep),excluded_global_indices=excluded)
    np.savez_compressed(folder/'sample_weights.npz',weights=importance,active=active)
    atomic_json(folder/'weighting.json',weighting)
    trained=base.train_one(2026,xx[active],yy[active],importance[active],scale,folder,epochs)
    report=dict(data_fingerprint=fingerprint,scale=scale,models=[trained],after_feedback_round=after,
        full_rows=len(y),without_information_rows=len(yy),excluded_global_indices=excluded,
        excluded_sample_ids=[online[i-2048]['id'] for i in excluded],epochs=epochs,seed=2026,
        wall_seconds=time.monotonic()-started,used_for_acquisition=False,extra_true_evaluations=0)
    atomic_json(folder/'training.json',report)
    return dict(model=base.load_ensemble(folder)[0],folder=folder,report=report),report['wall_seconds']


def attach(batch,models,audit):
    if audit is None:return batch
    report=audit['report'];weights=np.asarray(batch['weights']).reshape(-1,21)
    comparison=dict(after_feedback_round=report['after_feedback_round'],evaluated_round=batch['round'],seed=2026,
        full_rows=report['full_rows'],without_information_rows=report['without_information_rows'],
        excluded_sample_ids=report['excluded_sample_ids'],shadow_model_sha256=report['models'][0]['sha256'],
        full_predictions=base.predict([models[0]],weights)[0].tolist(),
        without_information_predictions=base.predict([audit['model']],weights)[0].tolist(),
        generated_before_feedback=True,used_for_acquisition=False)
    batch['information_model_comparison']=comparison
    return batch


def pair_accuracy(truth,prediction):
    delta=truth[:,None]-truth[None,:];mask=np.triu(delta!=0,1)
    return float(((prediction[:,None]-prediction[None,:])[mask]*delta[mask]>0).mean()) if mask.any() else None


def assess(batch,scores):
    if 'information_model_comparison' not in batch:return None
    comparison=batch['information_model_comparison'];truth=np.asarray(scores,dtype=float)
    if not comparison['generated_before_feedback'] or comparison['evaluated_round']!=batch['round']:
        raise RuntimeError('Diagnostic predictions must precede their feedback')
    if comparison['after_feedback_round']!=batch['round']-1:raise RuntimeError('Diagnostic uses wrong feedback version')
    result={k:v for k,v in comparison.items() if not k.endswith('predictions')}
    low=truth<=np.quantile(truth,.2)
    for name,key in (('full','full_predictions'),('without_information','without_information_predictions')):
        prediction=np.asarray(comparison[key]);
        if prediction.shape!=truth.shape:raise RuntimeError('Diagnostic pairing mismatch')
        result[name]=dict(MAE=float(np.abs(prediction-truth).mean()),low20_MAE=float(np.abs(prediction[low]-truth[low]).mean()),
                         pair_accuracy=pair_accuracy(truth,prediction),
                         selected_true_R=float(truth[prediction.argmin()]),selection_regret=float(truth[prediction.argmin()]-truth.min()))
    result['MAE_improvement_from_information']=result['without_information']['MAE']-result['full']['MAE']
    result['interpretation']='conditional on next actually queried candidates; not a whole-space causal claim'
    result['extra_true_evaluations']=0
    return result
