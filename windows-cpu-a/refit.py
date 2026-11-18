"""Refit one archived dataset version on CPU and compare with its published models."""

import argparse
import json
from pathlib import Path

import numpy as np

from submission import ROOT, check
from adaptive import base, learning
from adaptive.protocol import atomic_json
from adaptive.release import training_data
from finalize_submission import verify_archive_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, choices=range(21), default=20)
    args = parser.parse_args()
    check()
    archive = ROOT / 'evidence'
    verify_archive_manifest(archive)
    x, y = training_data(0)
    if args.round:
        records = json.loads((archive / 'formal/online.json').read_text(encoding='utf-8'))
        rows = [row for row in records if row['round'] <= args.round]
        if len(rows) != 50 * args.round:
            raise RuntimeError('Archived feedback count mismatch')
        x = np.concatenate([x, np.asarray([r['weights'] for r in rows]).reshape(-1, 21)])
        y = np.concatenate([y, np.asarray([r['R'] for r in rows])])
    version = f'v{args.round:02d}'
    target = ROOT / 'outputs' / 'refit' / version
    target.mkdir(parents=True, exist_ok=False)
    models = learning.train_ensemble(x, y, 2048, target, jobs=2)
    reference_dir = archive / 'formal' / 'models' / version
    reference = base.load_ensemble(reference_dir)
    difference = np.abs(base.predict(models, x[:64]) - base.predict(reference, x[:64]))
    original = json.loads((reference_dir / 'training.json').read_text(encoding='utf-8'))
    trained = json.loads((target / 'training.json').read_text(encoding='utf-8'))
    if original['data_fingerprint'] != trained['data_fingerprint']:
        raise RuntimeError('Refit dataset differs from archived training data')
    same = {str(a['seed']): a['sha256'] == b['sha256']
            for a, b in zip(original['models'], trained['models'])}
    result = dict(version=version, training_rows=len(y), evaluator_calls=0,
                  prediction_max_abs_difference=float(difference.max()), checkpoint_sha256_equal=same)
    atomic_json(target / 'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
