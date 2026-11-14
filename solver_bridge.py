#!/usr/bin/env python3
"""Recovered backend: same JSON stdin/stdout as solver.exe; CPython 3.11."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import gzip
import json
import multiprocessing
from pathlib import Path
import sys

from adaptive.protocol import atomic_json, parse_request


def evaluate_case(index, weights, workflow, records):
    sys.path.insert(0, str(Path(workflow) / 'src'))
    import pipeline
    with redirect_stdout(sys.stderr):
        result = pipeline.evaluate(weights, f'case-{index+1:03d}')
    if records:
        folder = Path(records)
        filename = folder / f'case-{index+1:03d}.json.gz'
        tmp = filename.with_suffix('.gz.tmp')
        with gzip.open(tmp, 'wt', encoding='utf-8') as out:
            json.dump(result, out, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        tmp.replace(filename)
    print(f'completed case={index+1} R={result["R"]:.12f}', file=sys.stderr, flush=True)
    return index, result['R'], result['audit']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, default=Path(__file__).resolve().parent.parent/'evaluator-workflow')
    parser.add_argument('--records', type=Path)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    try:
        groups = parse_request(json.load(sys.stdin))
        if sys.version_info[:2] != (3, 11):
            raise ValueError('Recovered bytecode requires Python 3.11')
        if args.records:
            args.records.mkdir(parents=True, exist_ok=True)
        scores, audits = [None]*len(groups), [None]*len(groups)
        if args.workers == 1:
            results = [evaluate_case(i, w, args.workflow, args.records) for i, w in enumerate(groups)]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(evaluate_case, i, w, args.workflow, args.records) for i, w in enumerate(groups)]
                for future in as_completed(futures):
                    results.append(future.result())
        for index, score, audit in results:
            scores[index], audits[index] = score, audit
        if args.records:
            atomic_json(args.records/'index.json', dict(R=scores, audits=audits, backend='recovered',
                        official_executable_used=False, source_executable_sha256=pipeline_source_hash(args.workflow)))
        print(json.dumps({'R': scores}, allow_nan=False, separators=(',', ':')), flush=True)
        return 0
    except Exception as error:
        print(f'Evaluation failed: {error}', file=sys.stderr, flush=True)
        return 2


def pipeline_source_hash(workflow):
    manifest = json.loads((Path(workflow)/'data/bundle-manifest.json').read_text())
    return manifest['source_executable_sha256']


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
