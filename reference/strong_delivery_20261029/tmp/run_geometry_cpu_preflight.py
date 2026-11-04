"""Capture the actual CPU helper exit and output; never start a GPU process."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
receipt_path = ROOT / 'reports/od_mask_geometry_real_train_cpu_execution.json'
stdout_path = ROOT / 'reports/od_mask_geometry_real_train_cpu_execution.stdout.txt'
assert not receipt_path.exists() and not stdout_path.exists()
started = datetime.now(timezone.utc).isoformat()
helper = ROOT / 'tmp/preflight_mask_geometry_real_train_cpu.py'
with stdout_path.open('x', encoding='utf-8') as stream:
    process = subprocess.Popen([sys.executable, '-B', str(helper)], cwd=ROOT,
        stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ))
    print(json.dumps(dict(child_pid=process.pid, started_utc=started)), flush=True)
    code = process.wait()
report = ROOT / 'reports/od_mask_geometry_real_train_cpu_preflight.json'
receipt = dict(schema='od_mask_geometry_real_train_cpu_execution_v1', actual_child_pid=process.pid,
    actual_exit_code=code, started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
    device='cpu', CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
    helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
    stdout_path=str(stdout_path), stdout_sha256=hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
    report_path=str(report), report_sha256=hashlib.sha256(report.read_bytes()).hexdigest() if report.is_file() else None)
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt), flush=True)
raise SystemExit(code)
