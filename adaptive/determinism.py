"""Strict numerical policy; set environment before importing numerical libraries."""
import os
import subprocess
import sys

SEED = 2026
ENVIRONMENT = {
    'PYTHONHASHSEED': str(SEED),
    'PYTHONUTF8': '1',
    'PYTHONIOENCODING': 'utf-8',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1',
    'OMP_DYNAMIC': 'FALSE',
    'MKL_DYNAMIC': 'FALSE',
    'MKL_CBWR': 'COMPATIBLE',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'NVIDIA_TF32_OVERRIDE': '0',
}


def prepare_environment():
    # Assignment is intentional: inherited shell settings must not override policy.
    os.environ.update(ENVIRONMENT)


def bootstrap_cli():
    """Hashing and text encoding require a fresh interpreter, not just os.environ."""
    if sys.flags.ignore_environment or sys.flags.isolated or sys._xoptions.get('utf8') == '0':
        raise SystemExit('Strict startup requires environment variables and UTF-8; remove -E, -I or -X utf8=0.')
    restart = (os.environ.get('PYTHONHASHSEED') != ENVIRONMENT['PYTHONHASHSEED']
               or not sys.flags.utf8_mode)
    prepare_environment()
    if restart:
        raise SystemExit(subprocess.call([sys.executable, *sys.orig_argv[1:]], env=os.environ.copy()))


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if os.environ.get('SOFTTREE_DEVICE', 'cpu').startswith('cuda'):
        torch.cuda.manual_seed_all(seed)


def configure_runtime():
    import torch
    prepare_environment()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.mkldnn.enabled = False
    try:
        import torch.backends.opt_einsum as einsum
    except ImportError:
        pass
    else:
        einsum.enabled = False
    seed_all(SEED)


def settings():
    import platform
    import numpy as np
    import torch
    return dict(policy='strict-v1',global_seed=SEED,
                python_hash_seed=os.environ.get('PYTHONHASHSEED'),utf8_mode=bool(sys.flags.utf8_mode),
                environment={k:os.environ.get(k) for k in ENVIRONMENT},
                threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                default_dtype=str(torch.get_default_dtype()),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_enabled=torch.backends.cudnn.enabled,cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                mkldnn_enabled=torch.backends.mkldnn.enabled,
                optimizer_foreach=False,optimizer_fused=False,
                python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
                platform=platform.system(),machine=platform.machine(),
                device=os.environ.get('SOFTTREE_DEVICE','cpu'),
                scope='Same-environment repeatability; cross-platform equality is measured, not guaranteed')
