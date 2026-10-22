"""Read-only WDDM desktop-baseline identity gate; no GPU-idle claim.

Windows may report ordinary desktop apps as C+G in the compute-apps inventory.
Only identities explicitly reviewed in one SHA-pinned baseline are accepted.
Unknown processes, PID reuse or inaccessible identities fail closed. This gate
does not replace the caller's model-command/PID/lock/STOP/resource checks.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import ntpath
from pathlib import Path
import platform
import re
import subprocess

SCHEMA = 'verified_wddm_desktop_process_baseline_v1'
TIME_TOLERANCE = 1e-6
DWM = r'C:\Windows\System32\dwm.exe'
GPU_COMMAND = ['nvidia-smi', '--query-gpu=index,uuid,driver_version,driver_model.current', '--format=csv,noheader,nounits']
PROCESS_COMMAND = ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits']


def canonical_exe(value):
    if not isinstance(value, str) or not ntpath.isabs(value) or not re.fullmatch(r'[A-Za-z]:', ntpath.splitdrive(value)[0]):
        raise ValueError('A full Windows drive-qualified executable path is required')
    return ntpath.normcase(ntpath.normpath(value))


def load_baseline(path, expected_sha256):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Full immutable baseline SHA256 required')
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Desktop baseline bytes changed')
    baseline = json.loads(content.decode('utf-8-sig'))
    gpu = baseline.get('gpu', {})
    if (baseline.get('schema') != SCHEMA or baseline.get('phase') != 'root_reviewed_snapshot' or
            baseline.get('host', '').upper() != 'YAN' or gpu.get('index') != 0 or gpu.get('driver_model') != 'WDDM' or
            not re.fullmatch(r'GPU-[0-9A-Fa-f-]+', gpu.get('uuid', '')) or
            not isinstance(gpu.get('driver_version'), str) or not baseline.get('captured_utc')):
        raise ValueError('Reviewed YAN/GPU0/WDDM baseline metadata required')
    rows = baseline.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Explicit reviewed desktop identity rows are required')
    index = {}
    for row in rows:
        pid, created = row.get('pid'), row.get('create_time')
        if (type(pid) is not int or pid < 1 or pid in index or not isinstance(created, (float, int)) or
                not math.isfinite(created) or created <= 0 or row.get('classification') != 'verified_desktop' or
                not isinstance(row.get('verified_reason'), str) or not row['verified_reason'].strip()):
            raise ValueError('Each desktop identity must be unique and individually reviewed')
        executable = canonical_exe(row.get('exe'))
        if ntpath.basename(executable) in {'python.exe', 'pythonw.exe'}:
            raise ValueError('A Python compute process cannot enter the desktop baseline')
        if row.get('cmdline_collected') is not False or row.get('cmdline') is not None:
            raise ValueError('This reviewed baseline deliberately omits desktop command lines for privacy')
        index[pid] = row
    return baseline, index


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=15,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode != 0:
        raise RuntimeError('Read-only NVIDIA inventory failed: '+result.stderr.strip())
    return result.stdout


def inspect_desktop_baseline(psutil, baseline_path, baseline_sha256, *, query_fn=None):
    """Snapshot NVIDIA PID inventory then verify only exact baseline identities.

    query_fn is for pure fixture injection. Production uses the two fixed
    read-only NVIDIA commands. It never updates the baseline or stops a process.
    """
    baseline, known = load_baseline(baseline_path, baseline_sha256)
    if platform.node().upper() != baseline['host'].upper():
        raise RuntimeError('Desktop baseline belongs to a different host')
    perform_query = query if query_fn is None else query_fn
    gpu_text = perform_query(GPU_COMMAND)
    gpu_rows = [[field.strip() for field in row] for row in csv.reader(io.StringIO(gpu_text)) if row]
    expected_gpu = [str(baseline['gpu']['index']), baseline['gpu']['uuid'],
                    baseline['gpu']['driver_version'], baseline['gpu']['driver_model']]
    if gpu_rows != [expected_gpu]:
        raise RuntimeError('The exact single-GPU index/UUID/driver baseline changed or inventory is unsupported')
    process_text = perform_query(PROCESS_COMMAND)
    rows = []
    for row in csv.reader(io.StringIO(process_text)):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) < 2 or not row[0].strip().isdigit():
            raise RuntimeError('Unparseable NVIDIA process inventory; refusing to guess')
        pid = int(row[0].strip())
        name = ','.join(row[1:]).strip()
        if not name or pid < 1 or any(item['pid'] == pid for item in rows):
            raise RuntimeError('Invalid or duplicated NVIDIA PID inventory')
        if pid not in known:
            raise RuntimeError(f'Unknown/new NVIDIA process PID{pid}; a newly reviewed baseline is required')
        rows.append({'pid': pid, 'reported_process_name': name})
    verified, exited = [], []
    for item in rows:
        pid = item['pid']
        expected = known[pid]
        try:
            process = psutil.Process(pid)
            if not process.is_running():
                exited.append(pid)
                continue
            created = float(process.create_time())
            executable = process.exe()
            if (not math.isclose(created, expected['create_time'], rel_tol=0, abs_tol=TIME_TOLERANCE) or
                    canonical_exe(executable) != canonical_exe(expected['exe'])):
                raise RuntimeError(f'Desktop baseline identity changed or PID{pid} was reused')
            verified.append({**item, 'create_time': created, 'exe': executable,
                             'classification': 'verified_desktop_baseline_identity',
                             'desktop_cmdline_collected': False,
                             'identity_basis': 'exact_PID_create_time_and_full_executable_path',
                             'protected_system_dwm_cmdline_not_verified': canonical_exe(executable) == canonical_exe(DWM),
                             'baseline_reason': expected['verified_reason']})
        except psutil.NoSuchProcess:
            exited.append(pid)
        except psutil.AccessDenied as error:
            raise RuntimeError(f'Cannot verify required desktop identity fields for PID{pid}') from error
    # Detect any replacement of the supposedly immutable baseline while this
    # snapshot was checked. No broad executable-name allowlist is used.
    load_baseline(baseline_path, baseline_sha256)
    return {'policy': 'no_other_model_work_and_only_reviewed_WDDM_desktop_identities',
            'baseline_path': str(Path(baseline_path)), 'baseline_sha256': baseline_sha256,
            'checked_utc': datetime.now(timezone.utc).isoformat(), 'gpu_identity': expected_gpu,
            'inventory_count': len(rows), 'verified_desktop_count': len(verified),
            'verified_desktop_processes': verified, 'known_processes_exited_after_snapshot': exited,
            'NVIDIA_inventory_query': PROCESS_COMMAND, 'NVIDIA_process_inventory_stdout': process_text,
            'GPU_idle_claim': False, 'no_GPU_compute_or_shader_activity_claim': False,
            'interpretation': 'WDDM desktop C+G activity is permitted only for reviewed PID/create_time/full-exe identities. This does not mean the GPU is empty or idle.',
            'baseline_updated': False, 'processes_terminated': False}
