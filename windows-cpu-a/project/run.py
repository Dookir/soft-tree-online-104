#!/usr/bin/env python3
"""Portable release entry; use --help for offline commands, online --help for live runs."""
import multiprocessing
import subprocess
import sys


def main():
    if len(sys.argv)>1 and sys.argv[1]=='online':
        sys.argv.pop(1)
        from adaptive.audit import install
        install()
        from adaptive.runner import main as online
        return online()
    from adaptive.release import main as release
    return release()


if __name__ in ('__main__','__mp_main__'):
    from adaptive.input_guard import install
    install()


if __name__=='__main__':
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
        print('Failed: '+str(error),file=sys.stderr,flush=True)
        raise SystemExit(2)
