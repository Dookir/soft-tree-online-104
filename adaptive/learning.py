"""Verified original-only initialization, then unchanged baseline full refits."""
import hashlib
import json
from pathlib import Path
import shutil
import time

from . import base
from .base import configure, predict, sample_weights
from .protocol import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = base.CONFIG


def verify_pretrained(x, y):
    directory = ROOT/'pretrained_original'
    manifest = json.loads((directory/'provenance.json').read_text())
    if manifest['data_sha256'] != CONFIG['data_sha256']:
        raise RuntimeError('Pretrained original-data provenance mismatch')
    for relative, expected in manifest['files'].items():
        path = (directory/relative).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise RuntimeError('Pretrained allowlist checksum mismatch: '+relative)
    for name in ('model.py', 'base.py'):
        if digest(ROOT/'adaptive'/name) != manifest['runtime_sources'][name]:
            raise RuntimeError('Original model/training source changed')
    report = json.loads((directory/'training.json').read_text())
    fingerprint = hashlib.sha256(x.tobytes()+y.tobytes()+b'None').hexdigest()
    if (len(y) != 2048 or report['original_rows'] != 2048 or report['online_rows'] != 0
            or report['total_rows'] != 2048 or report['data_fingerprint'] != fingerprint
            or report['scale'] != float(base.np.median(y))
            or [m['seed'] for m in report['models']] != CONFIG['seeds']):
        raise RuntimeError('Checkpoint must contain only the original 2048 rows')
    margin = json.loads((directory/'original-oof-verification.json').read_text())
    if (margin['data_sha256'] != CONFIG['data_sha256'] or margin['held_out_coverage'] != 2048
            or abs(margin['epsilon']-CONFIG['near_best_margin']) > 1e-9):
        raise RuntimeError('Original-only OOF margin verification failed')
    # load_state_dict enforces all parameter and buffer shapes; compare dtypes too.
    for item in report['models']:
        if item['epochs'] != CONFIG['epochs'][str(item['seed'])]:
            raise RuntimeError('Original training epochs mismatch')
        with base.np.load(directory/f'seed{item["seed"]}.npz', allow_pickle=False) as z:
            reference = base.DistrictSurrogate(report['scale'], item['seed']).state_dict()
            if set(z.files) != set(reference):
                raise RuntimeError('Pretrained parameter keys mismatch')
            for key, tensor in reference.items():
                value = z[key]
                if (value.shape != tuple(tensor.shape) or value.dtype != tensor.numpy().dtype
                        or not base.np.isfinite(value).all()):
                    raise RuntimeError('Invalid pretrained parameter '+key)
                if key not in ('gate_weight','gate_bias','leaves','offset') and not base.np.array_equal(value,tensor.numpy()):
                    raise RuntimeError('Original preprocessing buffer changed')
    return manifest


def train_ensemble(x, y, original_rows, folder, jobs=1, epoch_override=None):
    started = time.monotonic()
    folder = Path(folder)
    if (folder/'training.json').exists():
        report = json.loads((folder/'training.json').read_text())
        expected = hashlib.sha256(x.tobytes()+y.tobytes()+str(epoch_override).encode()).hexdigest()
        if report['data_fingerprint'] != expected:
            raise RuntimeError('Training resume fingerprint mismatch')
        return base.load_ensemble(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Every version retains the exact ordered fitting data and weighting.
    base.np.savez_compressed(folder/'training_samples.npz', x=x, y=y,
                            original_rows=original_rows)
    if len(y) == original_rows and epoch_override is None:
        verify_pretrained(x, y)
        origin = ROOT/'pretrained_original'
        report = json.loads((origin/'training.json').read_text())
        for item in report['models']:
            shutil.copy2(origin/f'seed{item["seed"]}.npz', folder)
        importance, active, weighting = sample_weights(x,y,original_rows)
        base.np.savez_compressed(folder/'sample_weights.npz', weights=importance, active=active)
        atomic_json(folder/'weighting.json',weighting)
        report.update(initialization='verified_original_only_v00', wall_seconds=time.monotonic()-started,
                      historical_pretraining_seconds=sum(m['seconds'] for m in report['models']),
                      optimizer_steps=0, pretrained_provenance_sha256=digest(origin/'provenance.json'))
        atomic_json(folder/'training.json',report)
        return base.load_ensemble(folder)
    models = base.train_ensemble(x,y,original_rows,folder,jobs,epoch_override)
    report = json.loads((folder/'training.json').read_text())
    report.update(wall_seconds=time.monotonic()-started,
                  optimizer_steps=sum(m['epochs'] for m in report['models'])*int(base.np.ceil(report['active_rows']/128)))
    atomic_json(folder/'training.json',report)
    return models


from .acquisition import generate_batch
