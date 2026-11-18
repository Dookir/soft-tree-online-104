"""One unattended20x50 run, training-only CUDA recovery, never oracle retries."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time
from adaptive.protocol import atomic_json


def may_retry(state,text,count):
    cuda=bool(re.search(r'CUDA out of memory|CUDA error|CUDA driver|CUBLAS_STATUS|GPU has fallen',text,re.I))
    safe=state.get('pending') is None and state.get('status') in ('training','diagnostic_training','final_training','searching')
    return cuda and safe and count<3


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--status',type=Path,required=True);parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--session',type=Path,required=True);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[0]=='--' else args.command
    args.log.parent.mkdir(parents=True,exist_ok=True)
    attempts=[];start=time.monotonic()
    status=dict(status='running',started_at=datetime.now().isoformat(),command=command,log=str(args.log),restarts=attempts)
    atomic_json(args.status,status)
    while True:
        position=args.log.stat().st_size if args.log.exists() else 0
        with args.log.open('ab',buffering=0) as log:
            result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state=json.loads((args.session/'state.json').read_text()) if (args.session/'state.json').exists() else {}
        text=args.log.read_bytes()[position:].decode(errors='replace')
        if result.returncode and may_retry(state,text,len(attempts)):
            attempts.append(dict(at=datetime.now().isoformat(),returncode=result.returncode,phase=state.get('status'),
                                 completed_rounds=state.get('completed_rounds'),reason='recoverable CUDA failure before evaluator call'))
            status.update(status='recovering_GPU',restarts=attempts);atomic_json(args.status,status)
            time.sleep(60)
            status['status']='running';atomic_json(args.status,status)
            continue
        status.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,
                      finished_at=datetime.now().isoformat(),seconds=time.monotonic()-start,
                      completed_rounds=state.get('completed_rounds',0),best_real_R=state.get('cumulative_best_R'),
                      unresolved_evaluation=state.get('pending') is not None)
        atomic_json(args.status,status)
        return result.returncode


if __name__=='__main__':raise SystemExit(main())
