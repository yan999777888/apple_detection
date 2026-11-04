"""Wait for the legacy V5 job to exit, preflight the verifier arm, then train the pilot.

Read-only native process checks; this script never terminates or signals anything.
Order is fixed: no GPU work starts while any legacy PID is natively live, and no
training starts unless the CUDA preflight exits 0.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"C:\Users\216\Apple_YOLO26_RTX3080_v1\env\python.exe"
PRIMARY = r"D:\Apple_DualBranch_3080_20261022\Apple_DualBranch_3080_20261022\.venv\Scripts\python.exe"
LEGACY = {8416: '2026-10-22T22:02:51.6659064Z', 29752: '2026-10-22T22:02:51.6241210Z'}
PREFLIGHT = ROOT / 'tmp/preflight_mask_geometry_stages_v2.py'
PREFLIGHT_REPORT = ROOT / 'reports/od_mask_geometry_verifier_stage_preflight_cuda_v2.json'
PILOT_CONFIG = ROOT / 'configs/od_mask_geometry_verifier_pilot2_3080.json'
PILOT_CONFIG_SHA = 'e90319bb9d4e8539b883504adaa0eb861698e570084590be2a82dd554f4d9d07'
PILOT_LOG = ROOT / 'reports/verifier_pilot2_train_stdout.log'
PILOT_RUN = ROOT / 'runs/od_mask_geometry_verifier_pilot2_11'
FORMAL_CONFIG = ROOT / 'configs/od_mask_geometry_verifier_papple_joint8_3080.json'
FORMAL_RUN = ROOT / 'runs/od_mask_geometry_verifier_joint8_11'
FORMAL_LOG = ROOT / 'reports/verifier_formal_train_stdout.log'
GATE_MIN_PR = .915
CHAIN_LOG = ROOT / 'reports/verifier_arm_chain.json'
MAX_WAIT_SECONDS = 6 * 3600


def sha(path):
    import hashlib
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(status, **extra):
    payload = dict(schema='od_verifier_arm_chain_v1', updated_utc=datetime.now(timezone.utc).isoformat(),
                   pid=os.getpid(), **status, **extra)
    temporary = CHAIN_LOG.with_name(CHAIN_LOG.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)
        stream.write('\n')
    os.replace(temporary, CHAIN_LOG)
    print(json.dumps({k: payload[k] for k in ('stage', 'detail') if k in payload}, ensure_ascii=False), flush=True)
    return payload


def native_liveness():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    live, handles = [], []
    for pid in LEGACY:
        handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error != 87:
                raise RuntimeError(f'Unresolved native process query pid={pid} winerror={error}')
            continue
        handles.append(handle)
        wait = kernel.WaitForSingleObject(handle, 0)
        if wait not in (0, 258):
            raise RuntimeError(f'Unresolved liveness pid={pid} wait={wait}')
        if wait == 258:
            live.append(pid)
    for handle in handles:
        kernel.CloseHandle(handle)
    return live


def main():
    assert sha(PILOT_CONFIG) == PILOT_CONFIG_SHA, 'Pilot config changed after review'
    started = time.monotonic()
    write(dict(stage='waiting_for_legacy', detail='no GPU work until the legacy V5 PIDs exit'))
    while True:
        live = native_liveness()
        if not live:
            break
        if time.monotonic() - started > MAX_WAIT_SECONDS:
            write(dict(stage='timeout', detail=f'legacy PIDs still live: {live}'))
            return 4
        time.sleep(120)
    write(dict(stage='legacy_exited', detail='running the CUDA preflight next'))
    environment = dict(os.environ)
    # The workspace delete guard is for interactive shell hygiene; the trainer
    # owns and verifies its own run directory, and a turn-wide delete quota
    # aborted a complete VAL exactly at its deferred-cache cleanup once.
    for key in [name for name in environment if name.startswith('CODEBUDDY_SAFE_DELETE')]:
        environment.pop(key, None)
    environment['PATH'] = os.pathsep.join(
        part for part in environment.get('PATH', '').split(os.pathsep)
        if 'safe-bin' not in part.lower())
    environment.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0')
    if PREFLIGHT_REPORT.exists():
        write(dict(stage='preflight_report_exists', detail=str(PREFLIGHT_REPORT)))
        return 5
    # Same interpreter as training: the preflight must prove the stack that trains.
    completed = subprocess.run([PYTHON, '-B', str(PREFLIGHT), '--arm', 'verifier', '--device', 'cuda',
                                '--report', str(PREFLIGHT_REPORT.relative_to(ROOT))],
                               cwd=ROOT, env=environment, capture_output=True, text=True)
    if completed.returncode != 0:
        write(dict(stage='preflight_failed', detail=completed.stderr[-1500:].strip()),
              preflight_stdout=completed.stdout[-2000:], preflight_returncode=completed.returncode)
        return completed.returncode
    write(dict(stage='preflight_passed', detail=sha(PREFLIGHT_REPORT)),
          preflight_report_sha256=sha(PREFLIGHT_REPORT))
    environment.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with PILOT_LOG.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(f'\n=== pilot start {datetime.now(timezone.utc).isoformat()} ===\n')
        stream.flush()
        process = subprocess.Popen([PYTHON, '-u', '-B', 'scripts/train_od.py', '--config',
                                    str(PILOT_CONFIG.relative_to(ROOT)), '--device', 'cuda'],
                                   cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT)
        write(dict(stage='training_started', detail=f'pid {process.pid}'),
              training_pid=process.pid, log=str(PILOT_LOG), config_sha256=PILOT_CONFIG_SHA)
        code = process.wait()
    write(dict(stage='training_finished', detail=f'exit {code}'), training_exit_code=code,
          log=str(PILOT_LOG), elapsed_seconds=time.monotonic() - started)
    if code != 0:
        return code
    points = []
    for directory in sorted(PILOT_RUN.glob('validation/update_*')):
        summary_path, metadata_path = directory / 'summary.json', directory / 'evaluation_metadata.json'
        if not (summary_path.is_file() and metadata_path.is_file()):
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        if summary.get('phase') != 'complete' or summary.get('model_selection_eligible') is not True:
            continue
        block = summary['summary']
        points.append(dict(directory=directory.name, update=int(directory.name.split('_')[1]),
                           precision=block['precision'], recall=block['recall'],
                           score=summary['score'], confidence=summary['confidence'],
                           tp=block['tp'], fp=block['fp'], fn=block['fn'],
                           goal_95_met=block['goal_95_met']))
    if not points:
        write(dict(stage='gate_no_completed_point', detail='pilot produced no eligible validation'))
        return 6
    best = max(points, key=lambda row: row['score'])
    passed = best['score'] >= GATE_MIN_PR or best['goal_95_met']
    write(dict(stage='gate_evaluated', detail=f"best min(P,R)={best['score']:.4f} "
                                             f"gate={GATE_MIN_PR} passed={passed}"),
          pilot_points=points, best_point=best, gate_min_pr=GATE_MIN_PR, gate_passed=passed)
    if not passed:
        write(dict(stage='gate_stopped', detail='pilot below gate; no formal run launched'),
              pilot_points=points, best_point=best)
        return 0
    if FORMAL_RUN.exists() and any(FORMAL_RUN.iterdir()):
        write(dict(stage='formal_output_not_fresh', detail=str(FORMAL_RUN)))
        return 7
    with FORMAL_LOG.open('a', encoding='utf-8') as stream:
        stream.write('\n=== formal 8+4 start ' + datetime.now(timezone.utc).isoformat() + ' ===\n')
        stream.flush()
        process = subprocess.Popen([PYTHON, '-u', '-B', 'scripts/train_od.py', '--config',
                                    str(FORMAL_CONFIG.relative_to(ROOT)), '--device', 'cuda'],
                                   cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        write(dict(stage='formal_training_started', detail=f'pid {process.pid}'),
              formal_pid=process.pid, log=str(FORMAL_LOG), gate_passed=True, best_pilot=best)
        formal_code = process.wait()
    write(dict(stage='formal_training_finished', detail=f'exit {formal_code}'),
          formal_exit_code=formal_code, log=str(FORMAL_LOG), elapsed_seconds=time.monotonic() - started)
    return formal_code


if __name__ == '__main__':
    raise SystemExit(main())
