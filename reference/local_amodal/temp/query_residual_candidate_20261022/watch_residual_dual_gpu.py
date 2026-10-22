"""Bounded query dual-mask GPU-smoke supervisor; preparation is not execution.

Run only with the exact F: native falldet environment and ``-m watch_residual_dual_gpu``.
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
RUNNER = HERE / 'probe_residual_dual_gpu.py'
RUNNER_SHA = '32c3804889afdafa13bff05461785970d89ee727ac51c580bef3da3f3b805b8a'
BASE = SCRIPTS / 'watch_refinement.py'
BASE_SHA = '4f673f671cf1055408da2fd684b68dcbd15bb6f4f9606098e8a89c3b56248b60'
ATOMIC = SCRIPTS / 'atomic_io.py'
ATOMIC_SHA = '3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380'
BOX = ROOT / 'temp/context_dual_quality_eval_candidate_20261021/evaluate_fixed_quality_powers_wddm_retry.py'
BOX_SHA = '1d923931447fd0147873dab29f678933410ed6bd7fd8d77d27dbc6445f9f17a5'
WDDM = BOX.with_name('wddm_process_guard.py')
WDDM_SHA = '67b3d86f751149f45f0989ebd78ab30abd1dda34aa3df597d4c5e6aac5c45ebe'
DESKTOP = ROOT / 'reports/wddm_desktop_gpu_baseline_20261022_01.json'
DESKTOP_SHA = 'f9467bf7a991370b082e30700247ac44210c09bc0ecac0c682a0358ba8c973bc'
PAPPLE_SUPERVISOR = ROOT / 'reports/query_dual_papple_train01_20261021_supervisor.json'
PAPPLE_OUTPUT = ROOT / 'runs/query_dual_papple_train01_20261021'
PAPPLE_IDENTITIES = ({'pid': 12600, 'create_time': 1792597177.922958},
                     {'pid': 38160, 'create_time': 1792597188.4334152})
LEGACY_NATIVE = ROOT / 'temp/query_dual_mask_candidate_20261021/watch_query_dual_gpu_native.py'
LEGACY_NATIVE_SHA = 'a0c2445a76104680221a65f18e9ffbea9f0ac5973a66a29b6b2779dda1e18686'
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
    if (output != HERE / 'residual_gpu_smoke01' or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', output.name)
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
              str(BOX): BOX_SHA, str(WDDM): WDDM_SHA, str(DESKTOP): DESKTOP_SHA,
              str(LEGACY_NATIVE): LEGACY_NATIVE_SHA}
    if any(path in expected and expected[path] != value for path, value in parent.items()):
        raise ValueError('Parent/child source pins conflict')
    prepared = {'command': command, 'paths': paths, 'identity': metadata['identity'], 'planned_cases': metadata['planned_cases'],
                'expected_child_sources': expected, 'parent_only_sources': parent, 'parent_checkpoint': metadata['parent_checkpoint']}
    if STATE.get('prepared') is not None and STATE['prepared'] != prepared:
        raise RuntimeError('Fixed child recipe/source map changed before launch')
    STATE['prepared'] = prepared
    return command, RUNNER, paths


def papple_complete(psutil, box):
    """Require this exact completed six-epoch parent and both exact identities gone."""
    supervisor=read(PAPPLE_SUPERVISOR)
    final=supervisor.get('final_evidence',{})
    if (supervisor.get('phase')!='complete' or supervisor.get('exit_code')!=0
            or supervisor.get('supervisor')!=PAPPLE_IDENTITIES[0] or supervisor.get('child')!=PAPPLE_IDENTITIES[1]
            or supervisor.get('output')!=str(PAPPLE_OUTPUT)
            or final.get('phase')!='verified' or final.get('epochs_completed')!=6
            or final.get('run_id')!='a49fea73-dbfa-4c42-936c-a3449d6199d2'
            or final.get('test_used') is not False):
        raise RuntimeError('Exact complete/exit0/final-verified six-epoch query parent required')
    for identity in PAPPLE_IDENTITIES:
        if box.is_live(psutil,identity['pid'],identity['create_time']):
            raise RuntimeError('Exact query parent supervisor/child identity is still alive')
    summary_path=PAPPLE_OUTPUT/'training_summary.json'
    summary=read(summary_path)
    if (summary.get('phase')!='complete' or summary.get('epochs_completed')!=6
            or summary.get('source_before_after_identical') is not True
            or summary.get('source_sha256_before')!=summary.get('source_sha256_after')
            or final.get('output_sha256',{}).get(str(summary_path))!=sha(summary_path)):
        raise RuntimeError('Parent committed full-epoch summary bytes/evidence changed')
    rows=summary.get('history',[])
    if len(rows)!=6 or any(row.get('train_images')!=2304 or row.get('validation',{}).get('summary',{}).get('images')!=814
                          or row.get('validation',{}).get('summary',{}).get('gt_count')!=3449 for row in rows):
        raise RuntimeError('Parent must contain six complete TRAIN2304/fullVAL814/GT3449 epochs')
    return {'supervisor_report':str(PAPPLE_SUPERVISOR),'phase':'verified_complete_and_exact_processes_exited',
            'epochs_completed':6,'images':814,'GT':3449,'exact_parent_identities':list(PAPPLE_IDENTITIES)}


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
    if platform.node().upper()!='YAN' or Path(sys.executable).resolve()!=PYTHON.resolve():
        raise RuntimeError('Exact YAN/F: native falldet Python required')
    if (ROOT/'STOP').exists() or (ROOT/'STOP_LOCAL_TRAINING').exists():
        raise InterruptedError('STOP is present; never cleared automatically')
    legacy=load_module(LEGACY_NATIVE,'pinned_residual_legacy_native_guard',LEGACY_NATIVE_SHA)
    # The prior guard's process/telemetry policy is retained. The reviewed
    # snapshot is explicitly replaced by the new20261022 inventory, never an idle claim.
    legacy.DESKTOP=DESKTOP
    legacy.DESKTOP_SHA=DESKTOP_SHA
    inherited=legacy.guard(psutil)
    box=load_module(BOX,'pinned_residual_query_parent_liveness_guard',BOX_SHA)
    parent=papple_complete(psutil,box)
    scan_other_model_commands(psutil)
    return {**inherited,'parent_query_training_gate':parent,
            'reviewed_desktop_override':{'path':str(DESKTOP),'sha256':DESKTOP_SHA,
                'basis':'root verified35/35 exact desktop process identities for20261022'},
            'GPU_idle_claim':False}


def preflight(psutil):
    command = psutil.Process(os.getpid()).cmdline()
    supervisor = (any(command[i:i + 2] == ['-m', 'watch_residual_dual_gpu'] for i in range(len(command) - 1))
                  and not any(item.lower().endswith('.py') for item in command))
    child = (len(command) == 5 and lexical(command[0]) == PYTHON and command[1] == '-u'
             and lexical(command[2]) == RUNNER and command[3] == '--output')
    if not supervisor and not child:
        raise RuntimeError('Only the -m watch_residual_dual_gpu supervisor or its exact probe CLI is allowed')
    if child:
        parent = psutil.Process(os.getppid())
        parent_command = parent.cmdline()
        if (Path(parent.exe()).resolve() != PYTHON.resolve()
                or not any(parent_command[i:i + 2] == ['-m', 'watch_residual_dual_gpu'] for i in range(len(parent_command) - 1))
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
    prepared=STATE['prepared']
    if supervisor.get('exit_code')!=0 or not supervisor.get('child',{}).get('pid') or not supervisor['child'].get('create_time'):
        raise ValueError('Actual owned smoke child exit0 required')
    output=Path(prepared['paths']['--output'])
    report_path=output/'report.json'
    report_sha=sha(report_path)
    report=read(report_path)
    if (report.get('phase')!='residual_gpu_smoke_passed' or report.get('identity')!=prepared['identity']
            or report.get('native_import_environment')!=configure_native_imports()
            or report.get('input_size')!=1024 or report.get('completed_smoke_updates')!=3
            or report.get('source_before_after_identical') is not True
            or report.get('source_sha256_before')!=prepared['expected_child_sources']
            or report.get('source_sha256_after')!=prepared['expected_child_sources']
            or report.get('checkpoint_roundtrip_passed') is not True or report.get('postprocess_bounds_passed') is not True
            or report.get('accuracy_reported') is not False or report.get('test_used') is not False
            or report.get('formal_training_resumable') is not False or report.get('future_formal_warm_start_allowed') is not False
            or report.get('planned_cases')!=prepared['planned_cases'] or report.get('strict_parent_model_load') is not True
            or report.get('model_loaded') is not True or report.get('torch_imported') is not True):
        raise ValueError('Actual1024 residual smoke/checkpoint/postprocess/source evidence rejected')
    parent=report.get('parent_checkpoint',{})
    expected_parent=prepared['parent_checkpoint']
    if (any(parent.get(key)!=expected_parent[key] for key in ('path','sha256','run_id','completed_epoch'))
            or parent.get('checkpoint_payload_verified') is not True or parent.get('strict_parent_model_load') is not True
            or parent.get('complete_train_epoch') is not True or parent.get('complete_full_val') is not True
            or parent.get('optimizer_restored') is not False):
        raise ValueError('Strict fixed complete parent MODEL-only restoration proof required')
    criterion=report.get('criterion',{})
    if criterion.get('match_points')!=12544 or criterion.get('loss_points')!=12544 or criterion.get('auxiliary_weight')!=0.5:
        raise ValueError('Full12544-point final/penultimate criterion required')
    groups=report.get('optimizer_groups',[])
    if ([(g.get('name'),g.get('lr'),g.get('weight_decay')) for g in groups]
            !=[('encoder',1e-5,.05),('old_other',5e-5,.05),('residual_box_head',2e-4,.05)]):
        raise ValueError('Fresh formal-recipe three-group AdamW required')
    cases=report.get('cases',[])
    if len(cases)!=3 or [c.get('case_kind') for c in cases]!=['ordinary','max_gt','empty']:
        raise ValueError('Exactly three ordinary/maxGT/empty1024 cases required')
    final_keys={'loss_class','loss_visible_bce','loss_visible_dice','loss_amodal_bce','loss_amodal_dice','loss_bbox','loss_giou'}
    expected_loss_keys=final_keys|{key+'_aux' for key in final_keys}
    for update,(case,planned) in enumerate(zip(cases,prepared['planned_cases'],strict=True),start=1):
        if ({key:case.get(key) for key in planned}!=planned or case.get('finite') is not True
                or case.get('optimizer_update_verified') is not True or case.get('optimizer_update_number')!=update
                or case.get('GTunchanged') is not True or case.get('GT_shapes')!=[planned['GT_count'],1300,1300]
                or case.get('box_shape')!=[planned['GT_count'],4] or case.get('input_shape')!=[1,3,1024,1024]
                or case.get('auxiliary_stages')!=1 or case.get('original_GT_sha256_before')!=case.get('original_GT_sha256_after')
                or not case.get('original_GT_sha256_before')):
            raise ValueError('Actual fixed-case/full original GT/finite update evidence differs')
        gradients=case.get('gradients_abs_sum',{})
        if set(gradients)!={'new_amodal_MLP','visible_MLP','classifier','residual_box_head','shared_image_feature_parameter'}:
            raise ValueError('Independent A/V/class/box/shared gradient evidence required')
        for branch,values in gradients.items():
            if not isinstance(values,dict) or not values or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values.values()):
                raise ValueError('Finite gradient absolute sums required')
            total=sum(values.values())
            if branch in {'new_amodal_MLP','visible_MLP','residual_box_head'} and planned['GT_count']==0:
                if total!=0:
                    raise ValueError('Empty-GT mask/box gradients must be exactly zero')
            elif total<=0:
                raise ValueError('Nonempty mask/box and class/shared gradients must have positive total')
        losses=case.get('losses',{})
        if set(losses)!=expected_loss_keys or any(type(v) not in (int,float) or not math.isfinite(v) for v in losses.values()):
            raise ValueError('Fourteen finite final/penultimate scalar losses required')
        if planned['GT_count']==0 and any(v!=0 for key,v in losses.items() if key not in {'loss_class','loss_class_aux'}):
            raise ValueError('Empty-GT mask/box losses must be exactly zero')
    all_sources={**prepared['expected_child_sources'],**prepared['parent_only_sources']}
    actual={path:sha(path) for path in all_sources}
    if actual!=all_sources or sha(report_path)!=report_sha:
        raise ValueError('Smoke source/output bytes changed during final verification')
    checkpoint=output/'smoke.partial.pt'
    if (not checkpoint.is_file() or checkpoint.stat().st_size<=0
            or report.get('checkpoint_path')!=str(checkpoint) or report.get('checkpoint_sha256')!=sha(checkpoint)):
        raise ValueError('Actual own partial checkpoint required')
    if report.get('checkpoint')!={'path':str(checkpoint),'sha256':sha(checkpoint),'strict_model':True,
            'optimizer_scaler_RNG_restored':True,'formal_training_resumable':False,
            'future_formal_warm_start_allowed':False,'completed_smoke_updates':3}:
        raise ValueError('Strict own partial model/optimizer/scaler/RNG roundtrip evidence required')
    return {'phase':'verified','scope':'three_bounded_1024_real_train_cases_and_original_coordinate_postprocess_only',
            'input_size':1024,'cases':3,'losses_per_case':14,'completed_smoke_updates':3,
            'strict_parent_model_load':True,'accuracy_reported':False,'formal_training_resumable':False,
            'source_sha256_before':all_sources,'source_sha256_after':actual,
            'output_sha256':{str(report_path):report_sha,str(checkpoint):sha(checkpoint)}}


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
    if not 1 <= args.wall_time_seconds <= 1800:
        parser.error('Explicit wall limit must be 1..1800 seconds')
    if args.log_name != 'residual_gpu_smoke01':
        parser.error('Use the exact residual_gpu_smoke01 log identity')
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
