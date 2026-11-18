"""Windows console entry for a fresh formal run or a safe resume."""

from datetime import datetime, timezone
import ctypes
import json
import msvcrt
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'project'


def call(script, *args):
    return subprocess.run([sys.executable, '-u', str(ROOT / script), *args], cwd=ROOT).returncode


def main():
    if sys.argv[1:] not in (['formal'], ['resume']):
        raise RuntimeError('Use formal or resume')
    action = sys.argv[1]
    session = PROJECT / 'sessions/formal'
    output = ROOT / 'verification/EXIT.json'
    with (ROOT / 'formal-launch.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if action == 'formal' and (session.exists() or (PROJECT / 'formal-budget.json').exists()):
            raise RuntimeError('A formal run already exists. Use 04_RESUME.bat if interrupted.')
        if action == 'resume' and not (PROJECT / 'formal-budget.json').is_file():
            raise RuntimeError('No formal run to resume. Use 03_FORMAL_TRAIN.bat.')
        from submission import atomic_json
        start = datetime.now(timezone.utc).isoformat()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            result = call('submission.py', action)
            code = result or call('finalize_submission.py')
            atomic_json(output, dict(started_at=start, ended_at=datetime.now(timezone.utc).isoformat(),
                                      experiment_returncode=result, returncode=code))
            return code
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print(f'Formal launch stopped: {error}', file=sys.stderr)
        raise SystemExit(2)
