"""Wait for the running verifier pilot to exit, then run the geometry-only attribution pilot.

Diagnostic only: no gate and no formal launch. It answers one question with one
variable changed - does the differentiable A-box refinement alone reproduce the
4-5 point drop that both verifier pilots measured, or does the verifier cause it?
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"C:\Users\216\Apple_YOLO26_RTX3080_v1\env\python.exe"
WATCH_PID = 29076
CONFIG = ROOT / 'configs/od_mask_geometry_attr_pilot_3080.json'
CONFIG_SHA = '8dc471107c80be5635a23c11171e8b7f35c7548ddf73bcab12bcb5770610d2f7'
RUN = ROOT / 'runs/od_mask_geometry_attr_pilot_11'
LOG = ROOT / 'reports/attribution_pilot_train_stdout.log'
STATE = ROOT / 'reports/attribution_pilot_chain.json'
MAX_WAIT_SECONDS = 6 * 3600


def sha(path):
    import hashlib
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(stage, detail, **extra):
    payload = dict(schema='od_attribution_pilot_chain_v1',
                   updated_utc=datetime.now(timezone.utc).isoformat(),
                   pid=os.getpid(), stage=stage, detail=detail, **extra)
    temporary = STATE.with_name(STATE.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)
        stream.write('\n')
    os.replace(temporary, STATE)
    print(json.dumps({'stage': stage, 'detail': detail}, ensure_ascii=False), flush=True)


def pid_live(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error != 87:
            raise RuntimeError(f'Unresolved native process query pid={pid} winerror={error}')
        return False
    try:
        wait = kernel.WaitForSingleObject(handle, 0)
        if wait not in (0, 258):
            raise RuntimeError(f'Unresolved liveness pid={pid} wait={wait}')
        return wait == 258
    finally:
        kernel.CloseHandle(handle)


def main():
    assert sha(CONFIG) == CONFIG_SHA, 'Attribution config changed after review'
    started = time.monotonic()
    write('waiting_for_verifier_pilot', f'watch pid {WATCH_PID}')
    while pid_live(WATCH_PID):
        if time.monotonic() - started > MAX_WAIT_SECONDS:
            write('timeout', f'pid {WATCH_PID} still live')
            return 4
        time.sleep(120)
    if RUN.exists() and any(RUN.iterdir()):
        write('output_not_fresh', str(RUN))
        return 5
    environment = dict(os.environ)
    for key in [name for name in environment if name.startswith('CODEBUDDY_SAFE_DELETE')]:
        environment.pop(key, None)
    environment['PATH'] = os.pathsep.join(
        part for part in environment.get('PATH', '').split(os.pathsep) if 'safe-bin' not in part.lower())
    environment.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0')
    with LOG.open('a', encoding='utf-8') as stream:
        stream.write('\n=== attribution pilot start ' + datetime.now(timezone.utc).isoformat() + ' ===\n')
        stream.flush()
        process = subprocess.Popen([PYTHON, '-u', '-B', 'scripts/train_od.py', '--config',
                                    str(CONFIG.relative_to(ROOT)), '--device', 'cuda'],
                                   cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        write('training_started', f'pid {process.pid}', training_pid=process.pid,
              config_sha256=CONFIG_SHA, log=str(LOG))
        code = process.wait()
    write('training_finished', f'exit {code}', training_exit_code=code, log=str(LOG),
          elapsed_seconds=time.monotonic() - started)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
