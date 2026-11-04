"""Serial, bounded local training/evaluation stages. Never launches parallel GPU jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    output = (ROOT / config['output']).resolve()
    for path in (ROOT, Path(sys.executable), config_path, output):
        if path.drive.upper() != 'F:':
            raise ValueError(f'Runtime must stay on F: {path}')
    if config['schema'] != 'local_visible_formal_pipeline_v1':
        raise ValueError('Unknown pipeline schema')
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'pipeline_status.json'
    if status_path.exists():
        raise FileExistsError('Existing pipeline evidence; inspect it instead of restarting')
    lock = ROOT / 'reports/local_formal_pipeline.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({'pid': os.getpid(), 'output': str(output), 'created_utc': now()}).encode())
    os.close(fd)
    status = {'phase': 'initializing', 'pid': os.getpid(), 'created_utc': now(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'config_path': str(config_path), 'config_sha256': sha(config_path),
              'source_sha256': {p: sha(ROOT / p) for p in config['frozen_sources']},
              'completed_stages': [], 'child_pid': None, 'original_global_models_modified': False,
              'test_split_used': False, 'automatic_model_promotion': False}
    write(status_path, status)
    def event(kind, **values):
        with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'utc': now(), 'kind': kind, **values}, ensure_ascii=False) + '\n')
    child = None
    try:
        gate = json.loads((ROOT / config['engineering_gate']).read_text(encoding='utf-8'))
        if gate['phase'] != 'engineering_gate_complete' or gate['global_step'] != 2:
            raise ValueError('Formal two-update GPU engineering gate has not passed')
        probe = json.loads((ROOT / config['evaluation_probe']).read_text(encoding='utf-8'))
        if probe['phase'] != 'complete' or probe['images'] != 2:
            raise ValueError('Two-image native-mask GPU evaluation probe has not passed')
        for index, stage in enumerate(config['stages']):
            if sha(config_path) != status['config_sha256']:
                raise ValueError('Pipeline configuration changed during execution')
            for path, digest in status['source_sha256'].items():
                if sha(ROOT / path) != digest:
                    raise ValueError(f'Frozen source changed during execution: {path}')
            for requirement in stage.get('requires', []):
                if not (ROOT / requirement).is_file():
                    raise FileNotFoundError(requirement)
            command = [sys.executable, str(ROOT / stage['script']), *stage['args']]
            stdout = output / (stage['name'] + '.stdout.log')
            stderr = output / (stage['name'] + '.stderr.log')
            status.update(phase='running', stage=stage['name'], stage_index=index,
                          stage_count=len(config['stages']), updated_utc=now(), command=command)
            event('stage_started', stage=stage['name'], command=command)
            with stdout.open('w', encoding='utf-8') as out, stderr.open('w', encoding='utf-8') as err:
                child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=out, stderr=err)
                status['child_pid'] = child.pid
                write(status_path, status)
                while child.poll() is None:
                    time.sleep(5)
                    status['updated_utc'] = now()
                    write(status_path, status)
                code = child.returncode
            status['child_pid'] = None
            if code:
                raise RuntimeError(f"Stage {stage['name']} exited {code}; inspect its stderr log")
            checks = {}
            for required in stage['results']:
                path = ROOT / required['path']
                value = json.loads(path.read_text(encoding='utf-8'))
                for key, expected in required.get('equals', {}).items():
                    if value.get(key) != expected:
                        raise ValueError(f'Result contract failed: {path} {key}')
                checks[required['path']] = sha(path)
            record = {'stage': stage['name'], 'completed_utc': now(), 'result_sha256': checks}
            status['completed_stages'].append(record)
            write(status_path, status)
            event('stage_complete', **record)
        status.update(phase='complete', updated_utc=now(), child_pid=None)
        write(status_path, status)
        event('pipeline_complete', stages=len(config['stages']))
    except BaseException as exc:
        if child is not None and child.poll() is None:
            if os.name == 'nt':
                # Windows venv launchers may own a second Python process.
                # Stop this owned child tree so a failed controller leaves no GPU orphan.
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                child.terminate()
            child.wait(timeout=30)
        status.update(phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      updated_utc=now(), child_pid=None, error=repr(exc))
        write(status_path, status)
        event(status['phase'], error=repr(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
