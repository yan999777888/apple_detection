"""Local execution explicitly authorized by the user on 2026-10-19; F: only."""
import os
from pathlib import Path
import platform
import tempfile

ROOT=Path(__file__).resolve().parents[1]
ALLOWED_HOST='YAN'


def configure_paths():
    if platform.node().upper()!=ALLOWED_HOST:
        raise RuntimeError('This local trial is authorized only on host YAN')
    if ROOT.drive.upper()!='F:':
        raise RuntimeError('The user requires the complete local project on F:')
    directories={
        'TEMP':ROOT/'temp','TMP':ROOT/'temp',
        'TORCH_HOME':ROOT/'cache/torch','CUDA_CACHE_PATH':ROOT/'cache/cuda',
        'XDG_CACHE_HOME':ROOT/'cache','HF_HOME':ROOT/'cache/huggingface',
        'MPLCONFIGDIR':ROOT/'cache/matplotlib','PIP_CACHE_DIR':ROOT/'cache/pip',
        'NUMBA_CACHE_DIR':ROOT/'cache/numba','PYTHONPYCACHEPREFIX':ROOT/'cache/pycache',
    }
    for key,directory in directories.items():
        directory.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(directory)
    tempfile.tempdir=str(ROOT/'temp')
    os.environ['PYTHONUNBUFFERED']='1'
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['MKL_NUM_THREADS']='4'
    for name in ['data','reports','logs','runs']:
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    return ROOT


def authorized_guard():
    configure_paths()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable on the authorized local computer')
    properties=torch.cuda.get_device_properties(0)
    if '4060' not in properties.name:
        raise RuntimeError(f'Expected the inspected RTX4060, found {properties.name}')
    return torch
