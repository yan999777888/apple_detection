"""Bounded query dual-mask GPU-smoke supervisor; preparation is not execution.

Run only with the exact F: native falldet environment and ``-m watch_query_dual_gpu_native``.
The unchanged pinned production supervisor supplies 84C, 2GiB available RAM,
five-second monitoring, STOP, wall limit, exclusive lock and owned-child cleanup.
Module scope and preparation are standard-library only; no Torch/model imports.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT / 'scripts'
PYTHON = Path('F:/anaconda/envs/falldet/python.exe')
QUERY_SITE = ROOT / 'envs/query_dual_v1/Lib/site-packages'
TORCH_SITE = Path('F:/anaconda/envs/falldet/Lib/site-packages/torch')
RUNNER = HERE / 'probe_query_dual_gpu_native.py'
RUNNER_SHA = '2e83146d1b94f75cc221d156ee0a6b2aaf7c2c0494f08764e7bb2e0642066a34'
BASE = SCRIPTS / 'watch_refinement.py'
BASE_SHA = '4f673f671cf1055408da2fd684b68dcbd15bb6f4f9606098e8a89c3b56248b60'
ATOMIC = SCRIPTS / 'atomic_io.py'
ATOMIC_SHA = '3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380'
BOX = ROOT / 'temp/context_dual_quality_eval_candidate_20261021/evaluate_fixed_quality_powers_wddm_retry.py'
BOX_SHA = '1d923931447fd0147873dab29f678933410ed6bd7fd8d77d27dbc6445f9f17a5'
WDDM = BOX.with_name('wddm_process_guard.py')
WDDM_SHA = '67b3d86f751149f45f0989ebd78ab30abd1dda34aa3df597d4c5e6aac5c45ebe'
DESKTOP = ROOT / 'reports/wddm_desktop_gpu_baseline_20261021_01.json'
DESKTOP_SHA = '0ebeff5fbf560055c6e3f0c0cbfedb3b4a0ee951c91f817e4bdbb249a8318896'
PAPPLE_SUPERVISOR = ROOT / 'reports/context_dual_papple03_e006_full01_supervisor.json'
PAPPLE_OUTPUT = ROOT / 'temp/context_dual_papple_quality_candidate_20261021/papple03_e006_full01'
PAPPLE_IDENTITIES = ({'pid': 4904, 'create_time': 1792594740.6636336},
                     {'pid': 23336, 'create_time': 1792594742.6573884})
STATE = {}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


SELF_SHA = sha(__file__)


def configure_native_imports():
    """Validate exact external package overlay using standard-library metadata."""
    if 'torch' in sys.modules:
        raise RuntimeError('Native import preparation must precede Torch import')
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('Use the exact native F: falldet Python process')
    if os.environ.get('PYTHONPATH') != str(QUERY_SITE):
        raise RuntimeError('PYTHONPATH must be exactly the one query_dual_v1 site-packages entry')
    os.environ['PYTHONPATH'] = str(QUERY_SITE)
    origins = {}
    versions = {}
    requirements = {'transformers': '4.57.1', 'safetensors': '0.8.0', 'tokenizers': '0.22.2',
                    'huggingface_hub': '0.36.2', 'regex': '2026.9.29', 'tqdm': '4.70.1',
                    'torch': '2.12.1+cu132'}
    for name, expected in requirements.items():
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise RuntimeError('Missing package origin: ' + name)
        origin = Path(spec.origin).resolve()
        required_root = TORCH_SITE if name == 'torch' else QUERY_SITE
        if not origin.is_relative_to(required_root.resolve()):
            raise RuntimeError('Package resolved outside its exact F: source: ' + name)
        actual = version(name)
        if actual != expected:
            raise RuntimeError(f'Native package version differs for {name}: {actual}')
        origins[name] = str(origin)
        versions[name] = actual
    if 'torch' in sys.modules:
        raise RuntimeError('Native import preparation unexpectedly imported Torch')
    return {'python_executable': str(PYTHON), 'PYTHONPATH': str(QUERY_SITE),
            'origins': origins, 'versions': versions, 'torch_imported': False}


def load_module(path, name, expected):
    if sha(path) != expected:
        raise RuntimeError('Pinned source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if 'torch' in sys.modules:
        raise RuntimeError('Supervisor must never import Torch')
    return module


def lexical(value):
    path = Path(value).expanduser()
    path = Path(os.path.abspath(path if path.is_absolute() else ROOT / path))
    if path.drive.upper() != 'F:':
        raise ValueError('All paths must remain on F:')
    return path


def parse_child(arguments, root=ROOT):
    if Path(root) != ROOT:
        raise ValueError('Fixed local_amodal root only')
    arguments = list(arguments)
    if arguments and arguments[0] == '--':
        arguments = arguments[1:]
    if len(arguments) != 3 or lexical(arguments[0]) != RUNNER or arguments[1] != '--output':
        raise ValueError('Only the pinned probe with one explicit --output is permitted')
    output = lexical(arguments[2])
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if (output.parent != HERE or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', output.name)
            or output.name.endswith('.') or output.name.split('.')[0].upper() in reserved or output.exists()
            or output.resolve() != output):
        raise ValueError('Fresh directly named candidate-local output required')
    return [str(PYTHON), '-u', str(RUNNER), '--output', str(output)], {'--output': str(output)}


def load_runner():
    if sha(__file__) != SELF_SHA or sha(BASE) != BASE_SHA or sha(ATOMIC) != ATOMIC_SHA:
        raise RuntimeError('Pinned adapter/production/atomic source changed')
    return load_module(RUNNER, 'query_gpu_smoke_metadata_only', RUNNER_SHA)


def validate_command(arguments, root):
    command, paths = parse_child(arguments, root)
    metadata = load_runner().metadata(types.SimpleNamespace(output=Path(paths['--output'])))
    expected = metadata['source_sha256_expected']
    if (not isinstance(expected, dict) or expected.get(str(RUNNER)) != RUNNER_SHA
            or any(lexical(path) != Path(path) or not re.fullmatch('[0-9a-f]{64}', value) for path, value in expected.items())):
        raise ValueError('Complete pinned F: actual-child source map required')
    parent = {str(Path(__file__)): SELF_SHA, str(BASE): BASE_SHA, str(ATOMIC): ATOMIC_SHA,
              str(BOX): BOX_SHA, str(WDDM): WDDM_SHA, str(DESKTOP): DESKTOP_SHA}
    if any(path in expected and expected[path] != value for path, value in parent.items()):
        raise ValueError('Parent/child source pins conflict')
    prepared = {'command': command, 'paths': paths, 'identity': metadata['identity'], 'planned_cases': metadata['planned_cases'],
                'expected_child_sources': expected, 'parent_only_sources': parent}
    if STATE.get('prepared') is not None and STATE['prepared'] != prepared:
        raise RuntimeError('Fixed child recipe/source map changed before launch')
    STATE['prepared'] = prepared
    return command, RUNNER, paths


def papple_complete(psutil, box):
    """Require verified complete full814/3449 at all powers and exact processes gone."""
    supervisor = read(PAPPLE_SUPERVISOR)
    if (supervisor.get('phase') != 'complete' or supervisor.get('exit_code') != 0
            or supervisor.get('supervisor') != PAPPLE_IDENTITIES[0] or supervisor.get('child') != PAPPLE_IDENTITIES[1]
            or supervisor.get('output') != str(PAPPLE_OUTPUT)):
        raise RuntimeError('The exact current PApple supervisor must first complete with exit0')
    final = supervisor.get('final_evidence', {})
    if (final.get('phase') != 'verified' or final.get('mode') != 'full' or final.get('epoch') != 6
            or final.get('images') != 814 or final.get('GT') != 3449
            or final.get('source_sha256_before') != final.get('source_sha256_after')
            or not final.get('source_sha256_before')):
        raise RuntimeError('Current PApple final verified full814/3449 evidence required')
    for identity in PAPPLE_IDENTITIES:
        if box.is_live(psutil, identity['pid'], identity['create_time']):
            raise RuntimeError('Exact current PApple supervisor/child identity is still alive')
    report = read(PAPPLE_OUTPUT / 'report.json')
    if (report.get('phase') != 'complete_papple_fullval' or report.get('mode') != 'full' or report.get('epoch') != 6
            or report.get('images_planned') != 814 or report.get('images_completed') != 814 or report.get('GT') != 3449
            or report.get('scope') != 'PApple_val814_GT3449_true_dual_masks_fixed_three_powers'
            or set(report.get('results', {})) != {'0.0', '0.5', '1.0'}
            or report.get('source_before_after_identical') is not True):
        raise RuntimeError('Current PApple actual full report must contain all three complete powers')
    signatures = final.get('output_sha256_before_after', {})
    if str(PAPPLE_OUTPUT / 'report.json') not in signatures or not signatures:
        raise RuntimeError('PApple final verification must bind its actual output bytes')
    for path, value in signatures.items():
        if not Path(path).is_relative_to(PAPPLE_OUTPUT) or sha(path) != value:
            raise RuntimeError('PApple verified output bytes changed')
    for power in (0., .5, 1.):
        path = PAPPLE_OUTPUT / f'power_{power:g}/summary.json'
        summary = read(path)
        if str(path) not in signatures or summary['summary']['images'] != 814 or summary['summary']['gt_count'] != 3449:
            raise RuntimeError('Each verified PApple power must contain the full814/3449')
    return {'supervisor_report': str(PAPPLE_SUPERVISOR), 'phase': 'verified_complete_and_exact_processes_exited',
            'images': 814, 'GT': 3449, 'powers': [0., .5, 1.]}


def scan_other_model_commands(psutil):
    # Same conservative model-script scan as pinned box.no_other_work. The
    # environment check is adapted explicitly to this new F: query venv.
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid == os.getpid():
            continue
        try:
            command = process.info['cmdline'] or []
            scripts = [Path(item).name.lower() for item in command if item.lower().endswith('.py')]
            if any(name == RUNNER.name.lower() or re.search(r'(^|_)(train|training|eval|evaluate|inference|predict|smoke|verify)(_|\.)', name) for name in scripts):
                raise RuntimeError(f'Another potentially GPU-using script is alive: PID {process.pid}, {scripts}')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            if (process.info.get('name') or '').lower().startswith('python'):
                raise RuntimeError('Cannot inspect another Python process; concurrency check fails closed')


def guard(psutil):
    configure_native_imports()
    if platform.node().upper() != 'YAN' or Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('Exact YAN/F: native falldet Python required')
    if (ROOT / 'STOP').exists() or (ROOT / 'STOP_LOCAL_TRAINING').exists():
        raise InterruptedError('STOP is present; never cleared automatically')
    box = load_module(BOX, 'pinned_query_existing_box_guard_metadata_only', BOX_SHA)
    old = read(box.RUN / 'status.json')
    if old.get('phase') != 'complete' or old.get('epoch') != 6:
        raise RuntimeError('Pinned six-epoch context trainer must be complete')
    for pid, created in ((53748, 1792584803.285806), (12480, 1792584800.7768795)):
        if box.is_live(psutil, pid, created):
            raise RuntimeError('Pinned original context trainer/supervisor is still alive')
    papple = papple_complete(psutil, box)
    current = ROOT / 'reports/current_job.json'
    if current.is_file():
        child = read(current).get('child') or {}
        if child.get('pid') != os.getpid() and child.get('pid') and box.is_live(psutil, child['pid'], child.get('create_time')):
            raise RuntimeError('An existing supervised compute child is still alive')
    scan_other_model_commands(psutil)
    wddm = load_module(WDDM, 'pinned_query_existing_wddm_guard', WDDM_SHA)
    inventory = wddm.inspect_desktop_baseline(psutil, DESKTOP, DESKTOP_SHA)
    if inventory['known_processes_exited_after_snapshot']:
        scan_other_model_commands(psutil)
    return {'papple_full_gate': papple, 'trainer_complete_epoch': 6,
            'GPU_inventory_policy': 'no_other_model_work_plus_exact_reviewed_desktop_baseline',
            'GPU_idle_claim': False, 'WDDM_inventory': inventory}


def preflight(psutil):
    command = psutil.Process(os.getpid()).cmdline()
    supervisor = (any(command[i:i + 2] == ['-m', 'watch_query_dual_gpu_native'] for i in range(len(command) - 1))
                  and not any(item.lower().endswith('.py') for item in command))
    child = (len(command) == 5 and lexical(command[0]) == PYTHON and command[1] == '-u'
             and lexical(command[2]) == RUNNER and command[3] == '--output')
    if not supervisor and not child:
        raise RuntimeError('Only the -m watch_query_dual_gpu_native supervisor or its exact probe CLI is allowed')
    if child:
        parent = psutil.Process(os.getppid())
        parent_command = parent.cmdline()
        if (Path(parent.exe()).resolve() != PYTHON.resolve()
                or not any(parent_command[i:i + 2] == ['-m', 'watch_query_dual_gpu_native'] for i in range(len(parent_command) - 1))
                or any(item.lower().endswith('.py') for item in parent_command)):
            raise RuntimeError('Smoke child must belong to its exact native falldet module supervisor')
    return guard(psutil)


def guarded_spawn(command, *args, **kwargs):
    import psutil
    configure_native_imports()
    guard_result = preflight(psutil)
    expected, _, _ = validate_command(command[2:], ROOT)
    if (command != expected or command[:2] != [str(PYTHON), '-u']
            or Path(kwargs.get('cwd', '')) != ROOT or kwargs.get('shell', False) or kwargs.get('env') is not None):
        raise ValueError('Only exact pinned query-smoke child may spawn')
    STATE['guard_before_spawn'] = guard_result
    return subprocess.Popen(command, *args, **kwargs)


def verify_success(supervisor):
    prepared = STATE['prepared']
    if supervisor.get('exit_code') != 0 or not supervisor.get('child', {}).get('pid') or not supervisor['child'].get('create_time'):
        raise ValueError('Actual owned smoke child exit0 required')
    output = Path(prepared['paths']['--output'])
    report_path = output / 'report.json'
    report_sha = sha(report_path)
    report = read(report_path)
    if (report.get('phase') != 'gpu_smoke_passed' or report.get('identity') != prepared['identity']
            or report.get('native_import_environment') != configure_native_imports()
            or report.get('source_before_after_identical') is not True
            or report.get('source_sha256_before') != prepared['expected_child_sources']
            or report.get('source_sha256_after') != prepared['expected_child_sources']
            or report.get('checkpoint_roundtrip_passed') is not True or report.get('postprocess_bounds_passed') is not True
            or report.get('accuracy_reported') is not False or report.get('test_used') is not False
            or report.get('formal_training_resumable') is not False or report.get('future_formal_warm_start_allowed') is not False
            or report.get('planned_cases') != prepared['planned_cases'] or report.get('completed_smoke_updates') != 6
            or report.get('model_loaded') is not True or report.get('torch_imported') is not True
            or report.get('strict_load') != {'strict': True, 'missing_keys': [], 'unexpected_keys': []}
            or report.get('initialization') != {'copied_classifier_rows': [47, 80],
                'AV_MLP_equal_and_independent': True, 'shared_model_by_reference': True}):
        raise ValueError('Actual smoke/checkpoint/postprocess/source evidence rejected')
    stages = report.get('stage_results', [])
    if (report.get('stages') != [512, 768] or report.get('cases_per_stage') != 3
            or [stage.get('input_size') for stage in stages] != [512, 768]
            or any(len(stage.get('cases', [])) != 3 for stage in stages)):
        raise ValueError('Both explicit pixel stages and all three bounded cases must pass')
    if any([case.get('case_kind') for case in stage['cases']] != ['ordinary', 'max_gt', 'empty'] for stage in stages):
        raise ValueError('Exact ordinary/max_gt/empty cases required at each pixel stage')
    update = 0
    for stage in stages:
        for case, planned in zip(stage['cases'], prepared['planned_cases'], strict=True):
            update += 1
            if ({key: case.get(key) for key in planned} != planned or case.get('finite') is not True
                    or case.get('optimizer_update_verified') is not True or case.get('optimizer_update_number') != update
                    or case.get('GTunchanged') is not True or case.get('GT_shapes') != [planned['GT_count'], 1300, 1300]
                    or case.get('input_shape') != [1, 3, stage['input_size'], stage['input_size']]):
                raise ValueError('Actual fixed-case identity/full original GT/finite update evidence differs')
            gradients = case.get('gradients_abs_sum', {})
            if set(gradients) != {'new_amodal_MLP', 'visible_MLP', 'classifier', 'shared_image_feature_parameter'}:
                raise ValueError('Independent A/V/classifier/shared gradient evidence required')
            for branch, values in gradients.items():
                if (not isinstance(values, dict) or not values
                        or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values.values())):
                    raise ValueError('Finite gradient absolute sums required')
                total = sum(values.values())
                if branch in {'new_amodal_MLP', 'visible_MLP'} and planned['GT_count'] == 0:
                    if total != 0:
                        raise ValueError('Empty-GT mask gradients must be exactly zero')
                elif total <= 0:
                    raise ValueError('Nonzero A/V/classifier/shared gradients required')
            losses = case.get('losses', {})
            if (len(losses) != 5 or any(type(value) not in (int, float) or not math.isfinite(value) for value in losses.values())):
                raise ValueError('Five finite real dual-mask losses required')
    all_sources = {**prepared['expected_child_sources'], **prepared['parent_only_sources']}
    actual = {path: sha(path) for path in all_sources}
    if actual != all_sources or sha(report_path) != report_sha:
        raise ValueError('Smoke source/output bytes changed during final verification')
    checkpoint = output / 'smoke.partial.pt'
    if (not checkpoint.is_file() or checkpoint.stat().st_size <= 0
            or report.get('checkpoint_path') != str(checkpoint) or report.get('checkpoint_sha256') != sha(checkpoint)):
        raise ValueError('Actual partial smoke checkpoint missing')
    if report.get('checkpoint') != {'path': str(checkpoint), 'sha256': sha(checkpoint), 'strict_model': True,
            'optimizer_scaler_RNG_restored': True, 'formal_training_resumable': False,
            'future_formal_warm_start_allowed': False, 'completed_smoke_updates': 6}:
        raise ValueError('Strict own partial model/optimizer/scaler/RNG roundtrip evidence required')
    return {'phase': 'verified', 'scope': 'six_bounded_train_cases_and_original_coordinate_postprocess_only',
            'pixel_sizes': [512, 768], 'accuracy_reported': False, 'formal_training_resumable': False,
            'source_sha256_before': all_sources, 'source_sha256_after': actual,
            'output_sha256': {str(report_path): report_sha, str(checkpoint): sha(checkpoint)}}


def load_base():
    load_runner()
    sys.path.insert(0, str(SCRIPTS))
    base = load_module(BASE, 'pinned_query_gpu_production_supervisor', BASE_SHA)
    if list(inspect.signature(base.run).parameters) != ['args', 'root', 'psutil']:
        raise RuntimeError('Pinned production supervisor run contract changed')
    atomic_json = base.atomic_json
    def validated_save(path, report):
        prepared = STATE.get('prepared')
        if prepared is not None and 'log_name' in report and 'monitor_summary' in report:
            report['pinned_adapter_sha256'] = SELF_SHA
            report['pinned_probe_sha256'] = RUNNER_SHA
            report['native_import_environment'] = configure_native_imports()
            report['expected_actual_child_sources'] = prepared['expected_child_sources']
            report['query_smoke_guard'] = STATE.get('guard_before_spawn')
            report['accuracy_note'] = 'This bounded smoke establishes execution evidence only; no accuracy or future formal-training resume claim.'
            try:
                progress = read(Path(prepared['paths']['--output']) / 'report.json')
                report['runner_evidence'] = {key: progress[key] for key in ('phase', 'active_input_size', 'active_case', 'error') if key in progress}
            except (OSError, ValueError):
                report['runner_evidence'] = {'available': False}
            if report.get('phase') == 'complete' and not STATE.get('final_verified'):
                try:
                    report['final_evidence'] = verify_success(report)
                    STATE['final_verified'] = True
                except BaseException as error:
                    report.update(phase='failed', failure_reason='actual_query_smoke_evidence_rejected', final_evidence_error=repr(error))
        atomic_json(path, report)
    base.validate_command = validate_command
    base.SCRIPT_RULES = {RUNNER.name: (set(), set(), set(), 'report.json')}
    base.atomic_json = validated_save
    base.subprocess = types.SimpleNamespace(run=subprocess.run, Popen=guarded_spawn,
        TimeoutExpired=subprocess.TimeoutExpired, CREATE_NO_WINDOW=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-name', required=True)
    parser.add_argument('--wall-time-seconds', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.wall_time_seconds <= 28800:
        parser.error('Explicit wall limit must be 1..28800 seconds')
    child = [str(RUNNER), '--output', args.output]
    parse_child(child)
    import psutil
    preflight(psutil)
    base = load_base()
    from runtime import configure_paths
    if Path(configure_paths()).resolve() != ROOT:
        raise RuntimeError('Runtime root changed')
    return base.run(types.SimpleNamespace(log_name=args.log_name, wall_time_seconds=args.wall_time_seconds,
                                          command=child), ROOT, psutil)


if __name__ == '__main__':
    raise SystemExit(main())
