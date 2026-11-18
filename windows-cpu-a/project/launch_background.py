#!/usr/bin/env python3
"""One unattended job, not a polling monitor; record the eventual exit status."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from adaptive.protocol import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',type=Path,required=True)
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command and args.command[0]=='--' else args.command
    if not command:
        raise ValueError('Missing worker command')
    args.log.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    status=dict(status='running',started_at=datetime.now().isoformat(),command=command,log=str(args.log.absolute()))
    atomic_json(args.status,status)
    try:
        with args.log.open('ab',buffering=0) as log:
            result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
    except Exception as error:
        status.update(status='failed',returncode=2,error=str(error))
    status.update(finished_at=datetime.now().isoformat(),seconds=time.monotonic()-started)
    atomic_json(args.status,status)
    return status['returncode']


if __name__=='__main__':
    raise SystemExit(main())
