"""Exactly one formal run; synthetic tests can never reach the real workflow."""
import json
import os
from pathlib import Path
from .protocol import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def validate_mode(args):
    if args.test_epochs is not None or args.test_search_steps is not None:
        manifest = json.loads((args.workflow/'data/bundle-manifest.json').read_text())
        if (args.test_epochs is None or args.test_epochs <= 0 or args.test_search_steps is None
                or args.test_search_steps <= 0 or manifest.get('source_executable_sha256') != 'SYNTHETIC_TEST_ONLY'):
            raise RuntimeError('Test overrides require the synthetic backend, never the real evaluator')
    elif args.rounds != 20:
        raise RuntimeError('The sole formal experiment has exactly 20 rounds of 50')


def claim_run(args):
    if args.test_epochs is not None:
        return
    path = ROOT/'formal-budget.json'
    claim = dict(session=str(args.session),max_calls=20,batch_size=50,max_evaluations=1000)
    try:
        fd = os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        saved=json.loads(path.read_text())
        if any(saved.get(k)!=v for k,v in claim.items()) or not args.resume:
            raise RuntimeError('Formal budget already belongs to one session; no additional experiment is permitted')
    else:
        with os.fdopen(fd,'w') as output:
            json.dump(dict(claim,attempts=[]),output,indent=2)


def reserve_call(args,round_number,input_sha256):
    if args.test_epochs is not None:
        return
    path=ROOT/'formal-budget.json'
    ledger=json.loads(path.read_text())
    attempts=ledger['attempts']
    if len(attempts)>=20 or round_number!=len(attempts)+1:
        raise RuntimeError('Persistent real-evaluation budget exhausted or inconsistent')
    attempts.append(dict(round=round_number,input_sha256=input_sha256,count=50,status='pending'))
    atomic_json(path,ledger)


def complete_call(args,round_number,input_sha256):
    if args.test_epochs is not None:
        return
    path=ROOT/'formal-budget.json'
    ledger=json.loads(path.read_text())
    record=ledger['attempts'][round_number-1]
    if record['input_sha256']!=input_sha256:
        raise RuntimeError('Persistent budget input mismatch')
    record['status']='complete'
    atomic_json(path,ledger)
