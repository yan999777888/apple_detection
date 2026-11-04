"""Bounded mask-geometry preflight: fixed CUDA1024, real TRAIN, isolated rebuild RNG.

Scope: two joint updates and two classifier updates on genuine PApple TRAIN
images with the formal constructor. It is a gradient/plumbing proof, never an
accuracy, convergence or VAL claim, and it is not the formal 8+4 protocol.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'e059c9a590d8732962b9f948cd9e6d935e907004'
CONFIG = ROOT / 'configs/od_mask_geometry_papple_3080.json'
CONFIG_SHA = 'dc292dd1a7562ac93c34fcea5fd1f93f9bc5f837f25b6c94cfddd9ad945f80ca'
PARENT_SHA = '9b44a9e7416fe13b9788f813161139094217bdbfc4cd62ec9d23ec719e3cb479'
BINDING = ROOT / 'reports/od_mask_geometry_committed_source_binding.json'
BINDING_SHA = 'e6f64f23b65f7af785945f09f47344f79eb35c2c017df409ece4e5b9b321da4b'
SOURCE_COUNT = 17
STATE_COUNT = 401
GEOMETRY_PREFIXES = ('roi_heads.mask_geometry_head.', 'roi_heads.mask_geometry_refiner.')
NEW_PREFIXES = ('roi_heads.context_fusion.', 'roi_heads.context_cascade.',
                'roi_heads.foreground_head.', 'roi_heads.foreground_fusion.',
                'visible_proposal_branch.rpn.', 'visible_proposal_branch.converter.',
                'shallow_detail_fusion.') + GEOMETRY_PREFIXES
GATE = 'roi_heads.gated_classifier.'
DETAIL_PREFIX = 'shallow_detail_fusion.'
VISIBLE_PROPOSAL_PREFIX = 'visible_proposal_branch.'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_report(path, value):
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    os.replace(temporary, path)


def filetime_ticks(iso):
    seconds, fraction = iso.rstrip('Z').split('.')
    delta = (datetime.fromisoformat(seconds).replace(tzinfo=timezone.utc)
             - datetime(1601, 1, 1, tzinfo=timezone.utc))
    return (delta.days * 86400 + delta.seconds) * 10_000_000 + int(fraction.ljust(7, '0'))


class LegacyProcessGuard:
    """Read/query/synchronize handles only; no terminate, signal or statefile."""

    expected = {8416: '2026-10-22T22:02:51.6659064Z',
                29752: '2026-10-22T22:02:51.6241210Z'}

    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('CUDA preflight requires the reviewed Windows native process guard')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.kernel.GetProcessTimes.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handles = []

    def observe(self):
        result = []
        for pid, expected in self.expected.items():
            handle = self.kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
            if not handle:
                error = ctypes.get_last_error()
                if error != 87:  # Only ERROR_INVALID_PARAMETER proves the PID absent.
                    raise RuntimeError(f'Native process query unresolved: pid={pid} winerror={error}')
                result.append(dict(pid=pid, state='PID_absent_ERROR_INVALID_PARAMETER', live=False))
                continue
            self.handles.append(handle)
            times = [wintypes.FILETIME() for _ in range(4)]
            if not self.kernel.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
                raise RuntimeError(f'GetProcessTimes failed for {pid}: {ctypes.get_last_error()}')
            creation = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            wait = self.kernel.WaitForSingleObject(handle, 0)
            if wait not in (0, 258):
                raise RuntimeError(f'Native liveness unresolved for {pid}: wait={wait}')
            result.append(dict(pid=pid, creation_FILETIME_ticks=creation, expected_native_utc=expected,
                               original_identity=creation == filetime_ticks(expected), live=wait == 258,
                               state='native_live' if wait == 258 else 'native_signaled_exit'))
        return result

    def close(self):
        for handle in self.handles:
            self.kernel.CloseHandle(handle)
        self.handles.clear()


def run(args, report):
    assert len(COMMIT) == 40 and len(CONFIG_SHA) == 64, 'Root committed pins have not been supplied'
    # This guard precedes the torch import and every possible CUDA runtime call.
    guard = LegacyProcessGuard() if args.device == 'cuda' else None
    try:
        def check_gpu_slot():
            if guard is not None:
                observations = guard.observe()
                report.setdefault('native_guard_observations', []).append(observations)
                if any(row['live'] for row in observations):
                    report['outcome'] = 'refused_live_legacy_PID_or_reused_PID'
                    raise RuntimeError('CUDA refused: a queried legacy PID is natively live; no process is controlled')
        check_gpu_slot()
        if args.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        sys.path.insert(0, str(ROOT / 'src'))
        import numpy as np
        import torch
        from apple_seg import od_training as training
        from apple_seg.data import PAppleDataset
        from apple_seg.od_data import ODInputDataset
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.training import _same
        report['torch_imported'] = True
        torch.set_num_threads(2)
        cfg = training.load_od_config(CONFIG)
        recipe = cfg['recipe']
        source_before = training.source_snapshot()
        binding = json.loads(BINDING.read_text(encoding='utf-8'))
        assert sha(BINDING) == BINDING_SHA, 'Reviewed source binding report changed'
        assert binding['commit'] == COMMIT and binding['tracked_files_clean'] is True
        assert len(source_before) == SOURCE_COUNT == binding['source17_count']
        assert source_before == binding['source17_sha256']
        assert sha(CONFIG) == CONFIG_SHA and cfg['raw']['parent_sha256'] == PARENT_SHA
        assert binding['config_sha256'] == CONFIG_SHA
        # Read-only commit blobs, independent of a later docs-only HEAD move.
        for relative, digest in [('src/apple_seg/' + name, value)
                                 for name, value in source_before.items()]:
            blob = subprocess.run(['git', 'show', COMMIT + ':' + relative], cwd=ROOT,
                                  capture_output=True, check=True)
            assert hashlib.sha256(blob.stdout).hexdigest() == digest, relative
        observed_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       capture_output=True, text=True, check=True).stdout.strip()
        tracked_status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=ROOT, capture_output=True, text=True, check=True).stdout
        if args.device == 'cuda':
            assert not tracked_status.strip(), 'Formal-constructor CUDA preflight requires clean tracked files'
        report['source_binding'] = dict(implementation_ref=COMMIT, observed_HEAD=observed_head,
                                        tracked_files_clean=not bool(tracked_status.strip()),
                                        source17_before=source_before,
                                        source_Git_blobs_current_equal=True,
                                        binding_report_sha256=BINDING_SHA, config_sha256=CONFIG_SHA)
        pins = dict(config=CONFIG, parent=cfg['paths']['parent_checkpoint'],
                    provenance=cfg['paths']['papple_provenance'],
                    train_annotation_visible=None, train_annotation_amodal=None)

        def pin_files():
            return {key: dict(path=str(path), sha256=sha(path))
                    for key, path in pins.items() if path is not None}

        report['files_before'] = pin_files()
        assert report['files_before']['parent']['sha256'] == PARENT_SHA
        seed = recipe['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        device = torch.device(args.device)
        amp = args.amp or ('bfloat16' if args.device == 'cuda' else 'float32')
        if args.device == 'cpu' and amp != 'float32':
            raise ValueError('CPU preflight uses FP32 only; CUDA BF16 requires the guarded CUDA entry')
        if args.device == 'cuda':
            assert amp == recipe['amp_dtype'] == 'bfloat16', 'CUDA uses the formal BF16 recipe'
            check_gpu_slot()
            assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            report['GPU_used'] = True
        options = dict(cfg['raw']['model'])
        if args.device == 'cpu':
            options.update(rpn_post_nms_top_n_train=16, rpn_post_nms_top_n_test=16,
                           box_batch_size_per_image=8)
        report['execution'] = dict(
            device=args.device, amp=amp, cpu_threads=2,
            source_order=list(training._training_source_order(recipe)), per_stage_updates=2,
            formal_constructor=args.device == 'cuda', uses_allowed_formal_scale=args.device == 'cuda',
            formal_resolution_set_fully_covered=False,
            CUDA_resolution_scope='fixed 1024 subset of allowed formal scales; not a random training trajectory',
            formal_model_options=cfg['raw']['model'], actual_model_options=options,
            short_sides=[1024] if args.device == 'cuda' else [64],
            max_size=recipe['max_size'] if args.device == 'cuda' else 64,
            augmentation_retained=recipe['augment'], config_written=False)
        assert report['execution']['source_order'] == ['PApple', 'PApple']
        if args.device == 'cuda':
            assert 1024 in recipe['short_sides'] and recipe['max_size'] >= 1024
        print('PREFLIGHT source17/commit/config/parent verified; building real R50 geometry model',
              flush=True)
        model = build_od_amodal_detector(weights=None, checkpoint_path=cfg['paths']['parent_checkpoint'],
                                         expected_sha256=PARENT_SHA, **options)
        report['R50_warmstart'] = deepcopy(model.checkpoint_provenance)
        assert model.checkpoint_provenance['all_source_tensors_equal']
        assert model.checkpoint_provenance['transferred_shared_tensors_equal']
        assert len(model.state_dict()) == STATE_COUNT, 'Geometry branch must expose exactly 401 states'
        assert model.constructor_config['mask_box_refinement'] is True
        assert model.constructor_config['amodal_base'] is True
        geometry_names = sorted(name for name in model.state_dict()
                                if name.startswith(GEOMETRY_PREFIXES))
        detail_names = sorted(name for name in model.state_dict() if name.startswith(DETAIL_PREFIX))
        proposal_names = sorted(name for name in model.state_dict()
                                if name.startswith(VISIBLE_PROPOSAL_PREFIX))
        assert len(geometry_names) == 13 and len(detail_names) == 4 and len(proposal_names) == 12
        assert [name for name in geometry_names
                if name.startswith('roi_heads.mask_geometry_refiner.')] == [
                    'roi_heads.mask_geometry_refiner.mix_logit']
        report['geometry_architecture'] = dict(
            state_count=STATE_COUNT, geometry_state_names=geometry_names,
            geometry_states=len(geometry_names), detail_states=len(detail_names),
            visible_proposal_states=len(proposal_names),
            location='expanded-support A head -> differentiable first/last-occupancy soft box '
                     '-> tanh BoxCoder residual -> final-ROI V/H/A and classifier',
            zero_initialized_mix_logit=True)
        native = PAppleDataset(cfg['paths']['papple_root'], 'train', input_size=32, augment=False)
        dataset = ODInputDataset(native, short_sides=tuple(report['execution']['short_sides']),
                                max_size=report['execution']['max_size'], augment=recipe['augment'])
        sampler = training.CyclingOrder(len(dataset), seed)
        pins.update({'train_' + name: path for name, path in native.annotation_paths.items()})
        report['files_before'] = pin_files()
        report['sampler_before'] = sampler.state_dict()
        report['native_original_GT_proofs'] = []
        report['stages'] = []
        check_gpu_slot()
        model.to(device)

        def tensor_sha(value):
            return hashlib.sha256(memoryview(value.detach().cpu().contiguous().numpy())).hexdigest()

        def component(name):
            for prefix in NEW_PREFIXES + (GATE, 'backbone.', 'rpn.', 'roi_heads.box_head.',
                                          'roi_heads.box_predictor.', 'roi_heads.od_mask_head.'):
                if name.startswith(prefix):
                    return prefix[:-1]
            return 'other'

        def gradients():
            result = {}
            for name, value in model.named_parameters():
                row = result.setdefault(component(name), dict(nonzero=[], zero=[], absent=[]))
                if value.grad is None:
                    row['absent'].append(name)
                else:
                    assert bool(torch.isfinite(value.grad).all()), name
                    row['nonzero' if bool(value.grad.ne(0).any()) else 'zero'].append(name)
            return result

        def compare(initial):
            result = {}
            for name, value in model.state_dict().items():
                value = value.detach().cpu()
                assert bool(torch.isfinite(value).all()), name
                row = result.setdefault(component(name), dict(changed=[], unchanged=[]))
                row['unchanged' if torch.equal(value, initial[name]) else 'changed'].append(name)
            return result

        miner_events, sample_events = [], []
        original_miner = model.roi_heads._negative_classifier_losses
        original_sampler = model.roi_heads._classifier_samples
        current = {}

        def miner(*values):
            negative = values[-1]
            assert bool((current['labels'][negative] == 0).all())
            scores = original_miner(*values)
            assert not scores.requires_grad and bool(torch.isfinite(scores).all())
            current['ranked_hard'] = negative[torch.argsort(scores, descending=True, stable=True)
                                              [:current['hard_count']]]
            miner_events.append(dict(negative_count=len(negative), hard_count=current['hard_count'],
                                     hard_indices=current['ranked_hard'].tolist(),
                                     max_current_gate_CE=float(scores.max()),
                                     all_post_ignore_label0=True, no_grad=True))
            return scores

        def samples(features, boxes, image_shapes, base, labels):
            cpu_before = torch.get_rng_state()
            cuda_before = torch.cuda.get_rng_state() if args.device == 'cuda' else None
            positive, negative = torch.where(labels >= 1)[0], torch.where(labels == 0)[0]
            specification = model.roi_heads.fg_bg_sampler
            positive_count = min(len(positive), int(specification.batch_size_per_image
                                                    * specification.positive_fraction))
            negative_count = min(len(negative), specification.batch_size_per_image - positive_count)
            current.clear()
            current.update(labels=labels, hard_count=negative_count // 2)
            selected = original_sampler(features, boxes, image_shapes, base, labels)
            cpu_after = torch.get_rng_state()
            cuda_after = torch.cuda.get_rng_state() if args.device == 'cuda' else None
            try:
                torch.set_rng_state(cpu_before)
                if cuda_before is not None:
                    torch.cuda.set_rng_state(cuda_before)
                expected_positive = positive[torch.randperm(len(positive),
                                                            device=positive.device)[:positive_count]]
                random_negative = negative[torch.randperm(len(negative),
                                                          device=negative.device)[:negative_count]]
                assert torch.equal(torch.get_rng_state(), cpu_after)
                if cuda_after is not None:
                    assert torch.equal(torch.cuda.get_rng_state(), cuda_after)
            finally:
                torch.set_rng_state(cpu_after)
                if cuda_after is not None:
                    torch.cuda.set_rng_state(cuda_after)
            assert torch.equal(selected[labels[selected] > 0].sort().values,
                               expected_positive.sort().values)
            assert int((labels[selected] == 0).sum()) == negative_count
            assert bool((labels[selected] >= 0).all()) and len(selected.unique()) == len(selected)
            hard = current.get('ranked_hard', negative[:0])
            assert bool(torch.isin(hard, selected).all())
            sample_events.append(dict(candidate_count=len(labels), positive_count=positive_count,
                                      negative_count=negative_count,
                                      selected_indices=selected.tolist(),
                                      random_negative_indices=random_negative.tolist(),
                                      hard_indices=hard.tolist(),
                                      original_random_positive_set_and_negative_quota_equal=True,
                                      persistent_RNG_equal=True, ignored_excluded=True,
                                      hard_scoring_executed='ranked_hard' in current))
            return selected

        model.roi_heads._negative_classifier_losses = miner
        model.roi_heads._classifier_samples = samples
        smoke_dir = ROOT / 'tmp' / ('mask_geometry_stage_preflight_' + str(time.time_ns()))
        smoke_dir.mkdir()
        stages = ('joint', 'classifier')
        for stage in stages:
            model.zero_grad(set_to_none=True)  # Discard joint buffers before proving isolation.
            optimizer = training.make_optimizer(model, recipe, stage)
            model.train()
            initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            trainable = [name for name, value in model.named_parameters() if value.requires_grad]
            if stage == 'classifier':
                assert len(trainable) == 6 and all(name.startswith(GATE) for name in trainable)
                assert not model.roi_heads.mask_geometry_head.training
                assert not model.roi_heads.mask_geometry_refiner.training
            else:
                rates = {id(value): group['base_lr'] for group in optimizer.param_groups
                         for value in group['params']}
                assert all(rates[id(value)] == recipe['new_head_lr'] for name, value
                           in model.named_parameters() if name.startswith(NEW_PREFIXES))
            stage_report = dict(stage=stage, trainable_parameters=trainable, updates=[],
                                stale_gradient_buffers_cleared_on_stage_entry=True,
                                optimizer_groups=[dict(name=group['name'], base_lr=group['base_lr'],
                                                       parameter_tensor_count=len(group['params']))
                                                  for group in optimizer.param_groups])
            report['stages'].append(stage_report)
            if args.device == 'cuda':
                check_gpu_slot()
                torch.cuda.reset_peak_memory_stats()
            for update in range(2):
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * ((update + 1) / recipe['warmup_updates'])
                optimizer.zero_grad(set_to_none=True)
                row = dict(update=update + 1, images=[], normalized_loss=0.)
                for source in training._training_source_order(recipe):
                    check_gpu_slot()
                    index = sampler.next()
                    native_target = native[index]['target']
                    native_fp = training.target_fingerprint(native_target)
                    del native_target
                    item = dataset[index]
                    target = item['target']
                    assert target['source']['split'] == 'train' and source == 'PApple'
                    actual_input_hw = list(item['pixel_values'].shape[-2:])
                    if args.device == 'cuda':
                        assert tuple(target['original_hw']) == (1300, 1300), 'Unexpected original TRAIN dimensions'
                        assert actual_input_hw == [1024, 1024], 'Actual CUDA input must be 1024 square'
                    fingerprint = training.target_fingerprint(target)
                    input_sha = tensor_sha(item['pixel_values'])
                    assert input_sha == target['source']['normalized_input_sha256']
                    rgb_path = Path(target['source']['rgb_path'])
                    rgb_sha = sha(rgb_path)
                    print(f'PREFLIGHT {stage} update={update + 1} TRAIN index={index} '
                          f'input={actual_input_hw}', flush=True)
                    pixels = item['pixel_values'].unsqueeze(0).to(device)
                    mask = item['pixel_mask'].unsqueeze(0).to(device)
                    with torch.autocast(device_type=args.device, dtype=torch.bfloat16,
                                        enabled=amp == 'bfloat16'):
                        losses = model(pixels, mask, targets=[target])
                        loss, _, omitted = training.supervised_loss(
                            losses, source, cfg['raw']['loss_weights'], stage=stage,
                            unknown_amodal_joint_policy=recipe['unknown_amodal_joint_policy'])
                    assert not omitted and bool(torch.isfinite(loss))
                    (loss / 2).backward()
                    assert training.target_fingerprint(target) == fingerprint
                    assert tensor_sha(item['pixel_values']) == input_sha and sha(rgb_path) == rgb_sha
                    row['normalized_loss'] += float(loss.detach()) / 2
                    row['images'].append(dict(index=index, source=target['source'],
                                              actual_input_hw=actual_input_hw,
                                              GT_before=fingerprint,
                                              GT_after=training.target_fingerprint(target),
                                              input_sha256=input_sha, rgb_sha256=rgb_sha,
                                              raw_losses={key: float(value.detach())
                                                          for key, value in losses.items()}))
                    report['native_original_GT_proofs'].append(dict(index=index,
                                                                    native_GT_before=native_fp,
                                                                    rgb_path=str(rgb_path),
                                                                    rgb_sha256_before=rgb_sha))
                    del item, target, pixels, mask, losses, loss
                row['gradients'] = gradients()
                row['gradient_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), recipe['gradient_clip_norm'], error_if_nonfinite=True))
                optimizer.step()
                row['changes_from_stage_start'] = compare(initial)
                if stage == 'classifier':
                    assert all(not value['changed'] for name, value
                               in row['changes_from_stage_start'].items() if name != GATE[:-1])
                    assert all(value.grad is None for name, value in model.named_parameters()
                               if not name.startswith(GATE))
                    row['all395_non_gate_states_bitwise_frozen'] = True
                    row['thirteen_geometry_states_bitwise_frozen'] = True
                    assert all(torch.equal(model.state_dict()[name].detach().cpu(), initial[name])
                               for name in geometry_names)
                elif update == 1:
                    geometry_gradients = row['gradients']['roi_heads.mask_geometry_head']
                    refiner_gradients = row['gradients']['roi_heads.mask_geometry_refiner']
                    observed = sorted(geometry_gradients['nonzero'] + refiner_gradients['nonzero'])
                    assert observed == geometry_names, 'Second joint step must exercise all 13 geometry states'
                    assert not geometry_gradients['zero'] and not geometry_gradients['absent']
                    assert not refiner_gradients['zero'] and not refiner_gradients['absent']
                    changed = sorted(row['changes_from_stage_start']['roi_heads.mask_geometry_head']['changed']
                                     + row['changes_from_stage_start']['roi_heads.mask_geometry_refiner']['changed'])
                    assert changed == geometry_names, 'All 13 geometry states must change by the second joint step'
                    row['geometry_second_step_learning_proof'] = dict(
                        all13_gradients_finite_nonzero=True,
                        all13_states_changed_from_stage_start=True,
                        scope='bounded real TRAIN algorithm gradient proof, not accuracy/convergence')
                stage_report['updates'].append(row)
            stage_report['final_changes'] = compare(initial)
            if stage == 'classifier':
                assert stage_report['final_changes'][GATE[:-1]]['changed'], 'No real classifier state updated'
            if args.device == 'cuda':
                torch.cuda.synchronize()
                stage_report['CUDA_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                stage_report['CUDA_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            checkpoint = smoke_dir / (stage + '.pt')
            payload = dict(schema=training.SCHEMA, smoke=True, stage=stage, updates=2,
                           model=model.state_dict(), optimizer=optimizer.state_dict(),
                           constructor_config=model.constructor_config, source_commit=COMMIT,
                           source_sha256=source_before, config_sha256=CONFIG_SHA)
            roundtrip_rng_before = training._rng_state(torch, np, generator, args.device == 'cuda')
            try:
                digest = training.save_checkpoint(checkpoint, payload)
                loaded = training.load_checkpoint(checkpoint)
                restored = build_od_amodal_detector(weights=None, **loaded['constructor_config'])
                restored.load_state_dict(loaded['model'], strict=True)
                assert len(restored.state_dict()) == STATE_COUNT
                restored_optimizer = training.make_optimizer(restored, recipe, stage)
                restored_optimizer.load_state_dict(loaded['optimizer'])
                assert all(torch.equal(value.detach().cpu(), restored.state_dict()[name])
                           for name, value in model.state_dict().items())
                assert _same(optimizer.state_dict(), loaded['optimizer'], torch, np)
                assert _same({key: value for key, value in loaded['optimizer'].items()},
                             restored_optimizer.state_dict(), torch, np)
            finally:
                training._restore_rng(roundtrip_rng_before, torch, np, generator,
                                      args.device == 'cuda')
            roundtrip_rng_after = training._rng_state(torch, np, generator, args.device == 'cuda')
            stage_report['roundtrip_rng_isolation'] = {
                key: _same(value, roundtrip_rng_after[key], torch, np)
                for key, value in roundtrip_rng_before.items()}
            assert all(stage_report['roundtrip_rng_isolation'].values()), 'Rebuild changed persistent RNG'
            stage_report['checkpoint_roundtrip'] = dict(
                path=str(checkpoint), sha256=digest, bytes_hashed=True,
                strict_model_401_states_equal=True, optimizer_state_equal=True,
                constructor=loaded['constructor_config'])
            del initial, loaded, restored, restored_optimizer, payload
            gc.collect()
        report['classifier_miner_events'] = miner_events
        report['classifier_sampler_events'] = sample_events
        assert miner_events and any(row['hard_scoring_executed'] for row in sample_events), \
            'No actual hard-negative ranking exercised'
        report['classifier_positive_coverage'] = dict(
            actual_positive_counts=[row['positive_count'] for row in sample_events],
            at_least_one_real_positive=any(row['positive_count'] > 0 for row in sample_events),
            required_for_CUDA_pass=args.device == 'cuda', GT_or_predictions_injected=False)
        if args.device == 'cuda':
            assert report['classifier_positive_coverage']['at_least_one_real_positive'], \
                'No real classifier positive exercised'
            observed = [row['actual_input_hw'] for stage in report['stages']
                        for update in stage['updates'] for row in update['images']]
            assert len(observed) == 8 and all(shape == [1024, 1024] for shape in observed)
            report['CUDA_memory_input_coverage'] = dict(
                actual_shapes=observed, actual_shape_covered=[1024, 1024],
                all_formal_scales_tested=False,
                scope='reported allocated/reserved peaks cover only these actual 1024 square TRAIN samples')
        for proof in report['native_original_GT_proofs']:
            proof['native_GT_after'] = training.target_fingerprint(native[proof['index']]['target'])
            proof['rgb_sha256_after'] = sha(proof['rgb_path'])
            assert proof['native_GT_before'] == proof['native_GT_after']
            assert proof['rgb_sha256_before'] == proof['rgb_sha256_after']
        report['source_binding']['source17_after'] = training.source_snapshot()
        report['files_after'] = pin_files()
        assert report['source_binding']['source17_after'] == source_before
        assert report['files_after'] == report['files_before']
        report.update(outcome='pass', sampler_after=sampler.state_dict(),
                      actual_constructor=model.constructor_config,
                      accuracy_evaluated=False, formal_training_started=False,
                      VAL_pixels_read=False, Minne_pixels_read=False)
    finally:
        if guard is not None:
            guard.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--amp', choices=('float32', 'bfloat16'))
    parser.add_argument('--report', type=Path, required=True,
                        help='A fresh path under this tree reports/')
    args = parser.parse_args()
    args.report = (ROOT / args.report).resolve()
    assert args.report.is_relative_to(ROOT / 'reports') and not args.report.exists()
    started = time.monotonic()
    report = dict(schema='od_mask_geometry_two_stage_preflight_v1', outcome='running',
                  device_requested=args.device, torch_imported=False, GPU_used=False,
                  helper_sha256=sha(__file__), process_controlled=False,
                  scope='bounded genuine TRAIN smoke, never full VAL/accuracy or formal  8+4',
                  warnings=['Per-stage 2 updates only; no convergence/precision/recall conclusion.',
                            'CUDA fixes allowed formal short 1024 and verifies actual 1024 square inputs; '
                            'no random training trajectory or complete scale coverage is claimed.',
                            'CPU uses 64px/post16/batch8 with reduced RPN; CPU FP32 does not validate CUDA BF16.',
                            'Per-parameter zero/absent gradients are evidence, never promoted to successful learning.'])
    code = 0
    try:
        run(args, report)
    except Exception as error:
        code = 3 if report['outcome'].startswith('refused') else 1
        if report['outcome'] == 'running':
            report['outcome'] = 'fail'
        report.update(error=repr(error), traceback=traceback.format_exc())
    report.update(actual_helper_exit_code=code, elapsed_seconds=time.monotonic() - started)
    write_report(args.report, report)
    print(json.dumps(dict(outcome=report['outcome'], exit_code=code, report=str(args.report),
                          report_sha256=sha(args.report))), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
