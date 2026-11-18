"""Submission entry: offline checks or a fresh official 20 x 50 experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'project'
sys.path.insert(0, str(PROJECT))
from adaptive.protocol import atomic_json


def interpreter():
    path = PROJECT / '.venv/Scripts/python.exe'
    if not path.exists():
        raise RuntimeError('Run 01_SETUP.bat first')
    return path


def check(*, require_executable=False):
    from adaptive.release import verify_manifest,records
    import numpy as np
    import torch
    if sys.version_info[:3]!=(3,10,21) or np.__version__!='1.26.4' or str(torch.__version__)!='2.6.0+cpu':
        raise RuntimeError('Expected original CPU environment')
    source=json.loads((ROOT/'SOURCE.json').read_text())
    exe = ROOT / 'teacher_solver/solver.exe'
    actual = None
    if exe.is_file():
        from adaptive.protocol import digest
        actual = digest(exe)
        if actual != source['teacher_executable_sha256']:
            raise RuntimeError('Original solver.exe SHA-256 mismatch')
    elif require_executable:
        raise RuntimeError('Place the original course scorer in teacher_solver/solver.exe first')
    baseline=json.loads((ROOT/'verification/BASELINE_HASHES.json').read_text())
    for name,sha in baseline.items():
        if name not in ('adaptive/acquisition.py','adaptive/runner.py') and hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Original runtime changed: '+name)
    if hashlib.sha256((PROJECT/'adaptive/acquisition_original.py').read_bytes()).hexdigest()!=baseline['adaptive/acquisition.py']:
        raise RuntimeError('Original selection rule changed')
    longest=PROJECT/'sessions/formal/rounds/20/original-pool/v3-pool/v1-pool/candidates.json.tmp'
    if len(str(longest.resolve()))>=240:
        raise RuntimeError('Move the complete package to a shorter path before starting; path preflight failed')
    report=dict(files_checked=verify_manifest(),original_rows=len(records()),original_kernel_verified=True,
                exploration_radius_multiplier=1.0,teacher_executable_sha256=actual,evaluator_called=False)
    atomic_json(ROOT/'verification/runtime-check.json',report)
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'formal', 'resume', 'status'])
    args = parser.parse_args()
    if args.action == 'check':
        check()
        return 0
    session = PROJECT / 'sessions/formal'
    if args.action == 'status':
        path = session / 'state.json'
        print(path.read_text(encoding='utf-8') if path.exists() else 'Formal experiment has not started')
        return 0
    check(require_executable=True)
    python = interpreter()
    subprocess.run([str(python), str(PROJECT / 'run.py'), 'doctor'], cwd=PROJECT, check=True)
    if args.action == 'formal':
        if session.exists() or (PROJECT / 'formal-budget.json').exists():
            raise RuntimeError('Experiment already started; use resume; never reset an active experiment')
        if (ROOT / 'teacher_solver/usage.count').exists() or (PROJECT / 'usage.count').exists():
            raise RuntimeError('Teacher solver already has usage.count; preserve and review existing usage')
    argv = [str(python), str(PROJECT / 'run.py'), 'online', '--backend', 'native',
            '--executable', str(ROOT / 'teacher_solver/solver.exe'), '--session', str(session),
            '--device', 'cpu', '--train-jobs', '2', '--timeout', '3600']
    if args.action == 'resume':
        argv.append('--resume')
    print('Starting original-only cold start, 20 rounds x 50 official evaluations.', flush=True)
    return subprocess.run(argv, cwd=PROJECT).returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'Submission stopped: {error}', file=sys.stderr, flush=True)
        raise SystemExit(2)
