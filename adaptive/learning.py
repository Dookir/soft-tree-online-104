"""Independent original-only cold start and unchanged full-data weighted refits."""
import hashlib
import json
from pathlib import Path
import time
from . import base
from .base import configure,predict,sample_weights
from .protocol import atomic_json
ROOT=Path(__file__).resolve().parents[1]
CONFIG=base.CONFIG


def verify_original_data(x,y):
    import numpy as np
    if len(y)!=2048 or x.shape!=(2048,21) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError('Only the original2048 rows may initialize this run')
    if (x<0).any() or not np.allclose(x.reshape(-1,3,7).sum(-1),1.,atol=1e-12):
        raise RuntimeError('Invalid original weights')
    if any((ROOT/p).exists() for p in ('pretrained_original','seed_feedback.json','seed_models')):
        raise RuntimeError('Independent experiment must not stage pretrained or historical feedback')


def train_ensemble(x,y,original_rows,folder,jobs=1,epoch_override=None):
    started=time.monotonic();folder=Path(folder)
    if (folder/'training.json').exists():
        report=json.loads((folder/'training.json').read_text())
        expected=hashlib.sha256(x.tobytes()+y.tobytes()+str(epoch_override).encode()).hexdigest()
        if report['data_fingerprint']!=expected:raise RuntimeError('Resume data mismatch')
        return base.load_ensemble(folder)
    if len(y)==original_rows:verify_original_data(x,y)
    folder.mkdir(parents=True,exist_ok=True)
    base.np.savez_compressed(folder/'training_samples.npz',x=x,y=y,original_rows=original_rows)
    models=base.train_ensemble(x,y,original_rows,folder,jobs,epoch_override)
    report=json.loads((folder/'training.json').read_text())
    report.update(wall_seconds=time.monotonic()-started,optimizer_steps=sum(m['epochs'] for m in report['models'])*int(base.np.ceil(report['active_rows']/128)),
                  old_models_loaded=False,old_online_labels_used=False)
    atomic_json(folder/'training.json',report)
    return models

from .acquisition import generate_batch
