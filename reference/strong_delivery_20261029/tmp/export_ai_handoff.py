"""Export factual working context, exact code history and small evidence only."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
OLD = PARENT/'od-joint-mask-supervision'
DEST = Path('C:/Users/216/Desktop/apple_detection_handoff_20261023')
assert DEST.is_dir() and not (DEST/'state.json').exists()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(tree,*args):
    return subprocess.check_output(['git','-C',str(tree),*args])


copied = []
def copy(source,relative):
    assert source.is_file() and source.suffix.lower() in ('.json','.xml','.md','.py','.ps1','.txt','.bundle','.path')
    target = DEST/relative
    assert target.is_relative_to(DEST) and not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    assert sha(source) == sha(target)
    copied.append(dict(source=str(source),exported=str(target.relative_to(DEST)),bytes=target.stat().st_size,sha256=sha(target)))


guard_path = PARENT/'od-visible-proposals/tmp/preflight_visible_proposals_stages_v8.py'
spec = importlib.util.spec_from_file_location('native_handoff_guard_only',guard_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
guard = module.LegacyProcessGuard()
try:
    native = guard.observe()
finally:
    guard.close()
assert all(row['original_identity'] and row['live'] for row in native)
status = read(OLD/'runs/od_joint_visible_masks_only_r50_05/training_status.json')
monitor = read(OLD/'tmp/v5_full_protocol_monitor.json')
assert status['pid'] == 8416 and not monitor['errors'] and not monitor['STOP_written_by_this_monitor']
bundle = read(OLD/'tmp/latest_git_bundle.json')
assert bundle['complete_history_verified'] and bundle['verification_returncode'] == 0
assert sha(Path(bundle['path'])) == bundle['sha256']
assert 'af14f4fe1d6788e0603e266f4da9b52e1e4e8e51 refs/heads/experiment/od-mask-geometry' in bundle['refs']
copy(Path(bundle['path']),'apple_detection_code_history.bundle')
verified = subprocess.run(['git','bundle','verify',str(DEST/'apple_detection_code_history.bundle')],cwd=ROOT,
    capture_output=True,text=True,encoding='utf-8')
assert verified.returncode == 0
trees = []
for block in git(ROOT,'worktree','list','--porcelain').decode('utf-8').strip().split('\n\n'):
    lines = dict(line.split(' ',1) for line in block.splitlines() if ' ' in line)
    tree = Path(lines['worktree'])
    trees.append(dict(path=str(tree),HEAD=lines.get('HEAD'),branch=lines.get('branch'),
        tracked_status=git(tree,'status','--porcelain','--untracked-files=no').decode('utf-8')))
assert not any(row['tracked_status'].strip() for row in trees)
geometry = read(ROOT/'reports/od_mask_geometry_real_train_cpu_preflight.json')
assert geometry['outcome'] == 'pass' and sha(ROOT/'reports/od_mask_geometry_real_train_cpu_preflight.json') == '4ab2cf0adba0894bbf9097506f6a2191033a473015d1d3aebdee6ccec235bf31'
sources = geometry['source']['source17_before']
assert sources == geometry['source']['source17_after'] and len(sources) == 17
for name,digest in sources.items():
    assert sha(ROOT/'src/apple_seg'/name) == digest
    assert hashlib.sha256(git(ROOT,'show','HEAD:src/apple_seg/'+name)).hexdigest() == digest
profiles = {}
for name in ('od_mask_geometry_papple_3080.json','od_mask_geometry_control_3080.json'):
    path = ROOT/'configs'/name
    assert path.read_bytes() == git(ROOT,'show','HEAD:configs/'+name)
    profiles[name] = dict(path=str(path),sha256=sha(path))
copy(Path('E:/\u5efa\u8bae.txt'),'reference/\u5efa\u8bae.txt')
for label in ('od-joint-mask-supervision','od-hard-negative-gate','od-context-foreground','od-visible-proposals','od-shallow-detail','od-mask-geometry'):
    tree = PARENT/label
    for folder,patterns in [('reports',('*published_metadata_review*.json','*real_train_cpu*.json','*committed_source_binding*.json','*package_refresh_receipt*.json','*CPU*.xml','*components*.xml','*geometry*.xml')),
                            ('benchmarks',('od_*.json',)),('docs/experiments',('od_*.md',))]:
        selected = set()
        for pattern in patterns:
            selected.update((tree/folder).glob(pattern))
        for source in sorted(selected):
            if source.is_file():
                copy(source,Path('evidence')/label/folder/source.name)
    if (tree/'README.md').is_file():
        copy(tree/'README.md',Path('evidence')/label/'README.md')
    for pattern in ('preflight*.py','run_*preflight*.py','bind_*committed_source.py'):
        for source in (tree/'tmp').glob(pattern):
            copy(source,Path('helpers')/label/source.name)
copy(ROOT/'HANDOFF_MASK_GEOMETRY_WIP.md','evidence/HANDOFF_MASK_GEOMETRY_WIP.md')
for name in ('run_v5_full_protocol_monitor.ps1','probe_completed_v5_goal_point.py','v5_full_protocol_monitor.json'):
    copy(OLD/'tmp'/name,Path('helpers/active_V5')/name)
copy(OLD/'runs/od_joint_visible_masks_only_r50_05/training_status.json','evidence/active_V5_training_status.json')
copy(OLD/'tmp/current_training_handoff.json','evidence/legacy_current_training_handoff_STALE_BEFORE5760_AND_WIP.json')
copy(OLD/'package_manifest.json','evidence/V5_incremental_package_manifest.json')
latest = read(OLD/'reports/od_joint_visible_only_update5760_published_metadata_review.json')
state = dict(schema='apple_detection_AI_handoff_v1',exported_utc=datetime.now(timezone.utc).isoformat(),
    purpose='User requested desktop working-memory export to continue training with another AI',
    user_goal='Original full814/3449 BOX precision and recall both >=.95 at one global threshold; report same-threshold V/A and fixed groups',
    goal_achieved=False,existing_GPU_job_left_running=True,new_GPU_jobs_started_during_export=False,
    original_process_native_observations=native,active_training_status=status,sole_monitor_snapshot=monitor,
    active_worktree=str(OLD),active_consumed_source='4392ac53efa7c17b478e5ef5436fa9de3ad79af8',
    active_config_sha256='fcc8c711090a1dc555f5f71d9daaaa9433acd55042de785b5499f32b2bb6686d',
    worktrees=trees,latest_completed_original_validation=dict(update=5760,confidence=latest['confidence_exact'],
        metrics=latest['metrics'],metadata_report_sha256=sha(OLD/'reports/od_joint_visible_only_update5760_published_metadata_review.json'),
        scope=latest['scope'],limitations=latest['limitations']),best_completed_own_validation_update=4608,
    newest_WIP=dict(worktree=str(ROOT),commit='af14f4fe1d6788e0603e266f4da9b52e1e4e8e51',
        branch='experiment/od-mask-geometry',source17_sha256=sources,profiles=profiles,
        current_source_equals_real_CPU_before_after_and_Git=True,
        CPU_actual_exit_code=0,CPU_child_pid=13548,CPU_model_states=401,
        CPU_report_sha256='4ab2cf0adba0894bbf9097506f6a2191033a473015d1d3aebdee6ccec235bf31',
        geometry_learning_proof=geometry['geometry_learning_proof'],
        full_unified_CPU_regression_done=False,GPU_preflight_done=False,own_fullVAL0_done=False,formal_training_started=False,
        known_blocker='Evaluator type-only source snapshot misses inherited od_context_model.py; strict formal loader then rejects geometry variant. Fix local MRO/actual method source capture, test and run complete relevant CPU regression first.'),
    latest_completed_unified_CPU_candidate=dict(worktree=str(PARENT/'od-shallow-detail'),commit='9ebc41a5d93fd48c6ca16b8a808ee2d01391d5c4',passed=727,GPU_tested=False),
    Git_bundle=dict(**bundle,desktop_verification_returncode=verified.returncode,desktop_path=str(DEST/'apple_detection_code_history.bundle')),
    legacy_cache_warning='tmp/current_training_handoff.json predates5760 publication and geometry WIP; use this fresh snapshot and actual current state.',
    copy_scope='Small JSON/XML/docs/helper evidence and code history only. Original datasets, weights, live checkpoint/raw files stay at existing D drive locations.',
    no_checkpoint_or_dataset_bytes_copied=True,exported_files=copied)
(DEST/'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
manifest = [dict(path=str(p.relative_to(DEST)),bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'manifest.json']
(DEST/'manifest.json').write_text(json.dumps(dict(files=manifest,total_files=len(manifest),
    total_bytes=sum(row['bytes'] for row in manifest)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
for row in manifest:
    assert sha(DEST/row['path']) == row['sha256']
print(json.dumps(dict(destination=str(DEST),files=len(manifest)+1,bytes=sum(row['bytes'] for row in manifest),
    state_sha256=sha(DEST/'state.json'),manifest_sha256=sha(DEST/'manifest.json'),
    live_phase=status['phase'],live_update=status['updates'],bundle_verified=True),ensure_ascii=False))
