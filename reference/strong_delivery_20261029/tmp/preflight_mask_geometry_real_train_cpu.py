"""Bounded real PApple TRAIN CPU smoke, never a validation/accuracy experiment."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import psutil
import torch

from apple_seg import od_training as training
from apple_seg.data import PAppleDataset
from apple_seg.od_data import ODInputDataset
from apple_seg.od_model import build_od_amodal_detector
from apple_seg.training import _same


REPORT = ROOT / 'reports/od_mask_geometry_real_train_cpu_preflight.json'
CONFIG = ROOT / 'configs/od_mask_geometry_papple_3080.json'
NEW_PREFIXES = ('roi_heads.context_fusion.', 'roi_heads.context_cascade.',
                'roi_heads.foreground_head.', 'roi_heads.foreground_fusion.',
                'visible_proposal_branch.rpn.', 'visible_proposal_branch.converter.', 'shallow_detail_fusion.', 'roi_heads.mask_geometry_head.', 'roi_heads.mask_geometry_refiner.')


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def component(name):
    for prefix in NEW_PREFIXES:
        if name.startswith(prefix):
            return prefix[:-1]
    for prefix in ('backbone.', 'rpn.', 'roi_heads.box_head.', 'roi_heads.box_predictor.',
                   'roi_heads.gated_classifier.', 'roi_heads.od_mask_head.'):
        if name.startswith(prefix):
            return prefix[:-1]
    return 'other'


def gradient_evidence(model):
    result = {}
    for name, parameter in model.named_parameters():
        row = result.setdefault(component(name), dict(parameter_count=0, gradient_present=[],
            gradient_absent=[], gradient_nonzero=[], gradient_all_zero=[], gradient_l1=0., finite=True))
        row['parameter_count'] += 1
        if parameter.grad is None:
            row['gradient_absent'].append(name)
            continue
        grad = parameter.grad.detach()
        row['gradient_present'].append(name)
        finite = bool(torch.isfinite(grad).all())
        row['finite'] = row['finite'] and finite
        if not finite:
            raise RuntimeError('Nonfinite gradient: ' + name)
        row['gradient_l1'] += float(grad.abs().sum())
        row['gradient_nonzero' if bool(grad.ne(0).any()) else 'gradient_all_zero'].append(name)
    return result


def changes(model, initial):
    result = {}
    for name, value in model.state_dict().items():
        row = result.setdefault(component(name), dict(state_count=0, changed=[], unchanged=[], finite=True))
        row['state_count'] += 1
        row['finite'] = row['finite'] and bool(torch.isfinite(value).all())
        row['unchanged' if torch.equal(value, initial[name]) else 'changed'].append(name)
        if not row['finite']:
            raise RuntimeError('Nonfinite model state: ' + name)
    return result


def hashes(paths):
    return {name: dict(path=str(path), sha256=training.sha(path)) for name, path in paths.items()}


def main():
    assert not REPORT.exists(), 'Refuse to overwrite a prior evidence report'
    torch.set_num_threads(2)
    started = time.monotonic()
    cfg = training.load_od_config(CONFIG)
    recipe = cfg['recipe']
    seed = recipe['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    original_sources = training.source_snapshot()
    assert len(original_sources) == 17
    original_config = CONFIG.read_bytes()
    order = training._training_source_order(recipe)
    assert order == ('PApple', 'PApple')
    pins = dict(config=CONFIG, parent=cfg['paths']['parent_checkpoint'],
                manifest=cfg['paths']['minne_manifest'], provenance=cfg['paths']['papple_provenance'])
    if cfg['paths']['parent_checkpoint'].with_suffix('.pt.sha256').exists():
        pins['parent_sidecar'] = cfg['paths']['parent_checkpoint'].with_suffix('.pt.sha256')
    file_before = hashes(pins)
    assert file_before['parent']['sha256'] == cfg['raw']['parent_sha256']
    options = dict(cfg['raw']['model'])
    options.update(rpn_post_nms_top_n_train=16, rpn_post_nms_top_n_test=16,
                   box_batch_size_per_image=8)
    report = dict(schema='od_mask_geometry_real_train_cpu_smoke_v1', outcome='running',
        scope='real R50-parent weights and real PApple TRAIN; four sampled images at <=64px; no VAL or accuracy evaluation',
        source=dict(base_commit_reference='9ebc41a5d93fd48c6ca16b8a808ee2d01391d5c4', uncommitted_consumed_source=True,
                    source17_before=original_sources),
        preflight_overrides=dict(device='cpu', cpu_threads=2, amp='disabled_float32',
            short_sides=[64], max_size=64, rpn_post_nms_top_n_train=16,
            rpn_post_nms_top_n_test=16, box_batch_size_per_image=8,
            optimizer_updates=2, images_per_optimizer_update=2,
            augmentation_retained=recipe['augment'], formal_config_written=False),
        files_before=file_before, source_order=list(order), formal_recipe=recipe,
        formal_model_options=cfg['raw']['model'], actual_build_options=options,
        warnings=['Smoke updates are not formal8+4 training or full validation.',
            'R50 source restoration and warm-start transfers are audited by the builder; this is not a347-state OD strictcopy.',
            'Changed states can reflect AdamW decay; actual gradients are reported independently.',
            'Zero initializers and proposal matches can legitimately leave some modules with zero/absent gradients.'])
    print('PREFLIGHT pins verified; building real R50 context model', flush=True)
    model = build_od_amodal_detector(weights=None, checkpoint_path=cfg['paths']['parent_checkpoint'],
        expected_sha256=cfg['raw']['parent_sha256'], **options)
    provenance = model.checkpoint_provenance
    assert provenance['strict_source_tensor_load'] and provenance['all_source_tensors_equal']
    assert provenance['transferred_shared_tensors_equal']
    assert provenance['checkpoint_sha256'] == cfg['raw']['parent_sha256']
    report['base_warmstart_provenance'] = provenance
    assert model.visible_proposal_branch.initialization_provenance['all_source_tensors_equal']
    assert model.visible_proposal_branch.initialization_provenance['all_copied_tensor_storages_independent']
    assert all(torch.equal(v, model.visible_proposal_branch.rpn.state_dict()[k]) for k,v in model.rpn.state_dict().items())
    assert model.shallow_detail and len(model.shallow_detail_fusion.state_dict()) == 4
    assert model.constructor_config['mask_box_refinement'] is True
    assert model.constructor_config['amodal_base'] is True
    geometry = model.roi_heads.mask_geometry_head
    source_A = model.roi_heads.od_mask_head
    for copied, original in ((geometry.mask_head,source_A.amodal_base_head),
                             (geometry.mask_predictor,source_A.amodal_base_predictor)):
        for name,tensor in copied.state_dict().items():
            assert torch.equal(tensor,original.state_dict()[name])
            assert tensor.data_ptr() != original.state_dict()[name].data_ptr()
    report['independent_A_geometry_initialization'] = geometry.initialization_provenance
    report['actual_constructor'] = model.constructor_config
    report['actual_model_config'] = model.model_config
    state = model.state_dict()
    initial = {name: value.detach().clone() for name, value in state.items()}
    new_keys = sorted(name for name in state if name.startswith(NEW_PREFIXES))
    # Topology only: no R50 reload, no OD checkpoint, no inherited calibration.
    baseline_options = dict(model.constructor_config)
    for name in ('context_cascade', 'context_expansion', 'cascade_iou_thresholds', 'visible_proposals', 'visible_proposal_limit', 'shallow_detail', 'mask_box_refinement'):
        baseline_options.pop(name)
    baseline = build_od_amodal_detector(weights=None, **baseline_options)
    old_keys = set(baseline.state_dict())
    assert set(state) - old_keys == set(new_keys) and not old_keys - set(state)
    assert all(state[name].shape == value.shape for name, value in baseline.state_dict().items())
    report['architecture_inventory'] = dict(actual_state_count=len(state),
        legacy_topology_state_count=len(old_keys), new_state_count=len(new_keys),
        new_keys=new_keys, legacy_keyset_subset=True, legacy_shapes_match=True,
        baseline_scope='random topology-only legacy model used solely for key/shape comparison; no OD weights loaded')
    del baseline, state
    gc.collect()
    # Preserve training RNG after the optional topology-only inventory builder.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator.manual_seed(seed)
    optimizer = training.make_optimizer(model, recipe, 'joint')
    model.train()
    rates = {id(parameter): group for group in optimizer.param_groups for parameter in group['params']}
    assert all(rates[id(parameter)]['name'] == 'new_head'
        and rates[id(parameter)]['base_lr'] == recipe['new_head_lr']
        for name, parameter in model.named_parameters() if name.startswith(NEW_PREFIXES))
    report['optimizer_groups'] = [dict(name=group['name'], base_lr=group['base_lr'],
        parameter_tensor_count=len(group['params']), scalar_count=sum(p.numel() for p in group['params']))
        for group in optimizer.param_groups]
    native = PAppleDataset(cfg['paths']['papple_root'], 'train', input_size=32, augment=False)
    dataset = ODInputDataset(native, short_sides=(64,), max_size=64, augment=recipe['augment'])
    assert native.split == 'train' and dataset.split == 'train'
    pins.update({f'train_{name}_annotation': path for name, path in native.annotation_paths.items()})
    file_before.update(hashes({name: path for name, path in pins.items() if name not in file_before}))
    report['dataset'] = dict(items=len(dataset), split='train', metadata=dataset.metadata(),
        minne_dataset_constructed=False, minne_pixels_read=False, validation_dataset_constructed=False)
    sampler = training.CyclingOrder(len(dataset), seed)
    report['sampler_before'] = sampler.state_dict()
    report['updates'] = []
    native_proofs = []
    for update in range(2):
        if time.monotonic() - started > 240:
            raise RuntimeError('CPU preflight exceeded240s before next bounded update')
        factor = (update + 1) / recipe['warmup_updates']
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        optimizer.zero_grad(set_to_none=True)
        row = dict(update=update + 1, images=[], normalized_total_loss=0.,
            mean_unweighted_losses={}, accumulation_divisor=len(order), warmup_factor=factor)
        for source in order:
            assert source == 'PApple'
            index = sampler.next()
            native_target = native[index]['target']
            native_fp = training.target_fingerprint(native_target)
            assert native_target['visible_masks'].device.type == 'cpu'
            assert native_target['amodal_masks'].device.type == 'cpu'
            del native_target
            item = dataset[index]
            target = item['target']
            before = training.target_fingerprint(target)
            input_hash = tensor_sha(item['pixel_values'])
            assert input_hash == target['source']['normalized_input_sha256']
            rgb_path = Path(target['source']['rgb_path'])
            rgb_hash = training.sha(rgb_path)
            pixels = item['pixel_values'].unsqueeze(0)
            pixel_mask = item['pixel_mask'].unsqueeze(0)
            assert pixels.device.type == 'cpu' and max(pixels.shape[-2:]) <= 64
            assert target['source']['dataset'] == 'PApple' and target['source']['split'] == 'train'
            print(f'PREFLIGHT forward update={update + 1} index={index} input={list(pixels.shape[-2:])} GT={len(target["instance_ids"])}', flush=True)
            losses = model(pixels, pixel_mask, targets=[target])
            objective, optimized, omitted = training.supervised_loss(losses, source, cfg['raw']['loss_weights'],
                stage='joint', unknown_amodal_joint_policy=recipe['unknown_amodal_joint_policy'])
            assert not omitted and objective.ndim == 0 and bool(torch.isfinite(objective))
            (objective / len(order)).backward()
            after = training.target_fingerprint(target)
            assert after == before, 'Model changed delivered native-resolution CPU GT'
            assert tensor_sha(item['pixel_values']) == input_hash, 'Model changed normalized input'
            assert training.sha(rgb_path) == rgb_hash
            sample = dict(dataset=source, index=index, image_id=target['source']['image_id'],
                source=target['source'], input_hw=list(pixels.shape[-2:]), input_sha256_before=input_hash,
                input_sha256_after=tensor_sha(item['pixel_values']), rgb_sha256_before=rgb_hash,
                rgb_sha256_after=training.sha(rgb_path), GT_count=len(target['instance_ids']),
                delivered_GT_before=before, delivered_GT_after=after,
                losses={name: float(value.detach()) for name, value in losses.items()},
                weighted_objective=float(objective.detach()), optimized_loss_names=list(optimized))
            row['images'].append(sample)
            row['normalized_total_loss'] += float(objective.detach()) / len(order)
            for name, value in optimized.items():
                row['mean_unweighted_losses'][name] = row['mean_unweighted_losses'].get(name, 0.) + float(value.detach()) / len(order)
            native_proofs.append(dict(index=index, image_id=target['source']['image_id'],
                rgb_path=str(rgb_path), rgb_sha256_before=rgb_hash, native_GT_before=native_fp))
            del losses, optimized, objective, target, pixels, pixel_mask, item
            gc.collect()
        row['gradients_before_clip'] = gradient_evidence(model)
        row['gradient_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
            recipe['gradient_clip_norm'], error_if_nonfinite=True))
        optimizer.step()
        row['state_changes_from_initial'] = changes(model, initial)
        report['updates'].append(row)
        print(f'PREFLIGHT optimizer update={update + 1} loss={row["normalized_total_loss"]:.7g} gradnorm={row["gradient_norm_before_clip"]:.7g}', flush=True)
    detail_grad = report['updates'][-1]['gradients_before_clip']['shallow_detail_fusion']
    detail_change = changes(model, initial)['shallow_detail_fusion']
    assert detail_grad['finite'] and len(detail_grad['gradient_nonzero']) == 4
    assert not detail_grad['gradient_absent'] and not detail_grad['gradient_all_zero']
    assert detail_change['finite'] and len(detail_change['changed']) == 4
    assert not detail_change['unchanged']
    assert len(model.state_dict()) == 401
    report['detail_learning_proof'] = dict(second_update_all4_finite_nonzero_gradients=True,
        final_all4_states_changed=True, new_parameter_count=sum(p.numel() for p in model.shallow_detail_fusion.parameters()),
        source='actual genuine TRAIN losses through full two-RPN/shared-ROI model, not synthetic feature loss')
    geometry_proof = {}
    for prefix, count in (('roi_heads.mask_geometry_head',12),('roi_heads.mask_geometry_refiner',1)):
        grad = report['updates'][-1]['gradients_before_clip'][prefix]
        change = changes(model, initial)[prefix]
        assert grad['finite'] and len(grad['gradient_nonzero']) == count
        assert not grad['gradient_absent'] and not grad['gradient_all_zero']
        assert change['finite'] and len(change['changed']) == count and not change['unchanged']
        geometry_proof[prefix] = dict(second_update_all_finite_nonzero_gradients=True,
            final_all_states_changed=True, actual_parameter_tensors=count)
    report['geometry_learning_proof'] = geometry_proof
    report['sampler_after'] = sampler.state_dict()
    report['actual_optimizer_updates'] = 2
    report['actual_training_images'] = 4
    report['actual_distinct_training_images'] = len(set(proof['index'] for proof in native_proofs))
    for proof in native_proofs:
        proof['native_GT_after'] = training.target_fingerprint(native[proof['index']]['target'])
        proof['rgb_sha256_after'] = training.sha(proof['rgb_path'])
        assert proof['native_GT_before'] == proof['native_GT_after']
        assert proof['rgb_sha256_before'] == proof['rgb_sha256_after']
    report['independently_reloaded_original_TRAIN_GT_proofs'] = native_proofs
    report['final_state_changes_from_initial'] = changes(model, initial)
    smoke_dir = ROOT / 'tmp' / f'mask_geometry_real_train_cpu_{time.time_ns()}'
    smoke_dir.mkdir()
    checkpoint = smoke_dir / 'smoke.pt'
    payload = dict(schema=training.SCHEMA, smoke=True, phase='complete', stage='joint', updates=2,
        scope=report['scope'], model=model.state_dict(), optimizer=optimizer.state_dict(),
        constructor_config=model.constructor_config, model_config=model.model_config,
        parent_provenance=provenance, identity=dict(source_sha256=original_sources,
            recipe=recipe, source_order=list(order), preflight_overrides=report['preflight_overrides']),
        sampler=sampler.state_dict(), rng=training._rng_state(torch, np, generator, False))
    smoke_sha = training.save_checkpoint(checkpoint, payload)
    loaded = training.load_checkpoint(checkpoint)
    restored = build_od_amodal_detector(weights=None, **loaded['constructor_config'])
    restored.load_state_dict(loaded['model'], strict=True)
    restored_optimizer = training.make_optimizer(restored, recipe, 'joint')
    restored_optimizer.load_state_dict(loaded['optimizer'])
    state_equal = all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    optimizer_equal = _same(optimizer.state_dict(), restored_optimizer.state_dict(), torch, np)
    assert state_equal and optimizer_equal
    report['smoke_checkpoint_roundtrip'] = dict(path=str(checkpoint), sha256=smoke_sha,
        checkpoint_bytes_hashed=True, sidecar_verified_on_load=True, strict_state_count=len(restored.state_dict()),
        all_model_tensors_equal=state_equal, optimizer_state_equal=optimizer_equal,
        scope='temporary smoke only; actual constructor includes reduced proposals/ROI batch, not formal constructor')
    report['source']['source17_after'] = training.source_snapshot()
    report['files_after'] = hashes(pins)
    assert report['source']['source17_after'] == original_sources
    assert CONFIG.read_bytes() == original_config
    assert report['files_after'] == file_before
    report.update(outcome='pass', goal95_claim=False, accuracy_evaluated=False,
        validation_images_read=0, minne_images_read=0, elapsed_seconds=time.monotonic() - started,
        process_rss_bytes=psutil.Process().memory_info().rss,
        helper_sha256=training.sha(Path(__file__)), torch_version=torch.__version__,
        python_version=sys.version, execution_device='cpu', threads=torch.get_num_threads())
    training.atomic_json(REPORT, report)
    print(json.dumps(dict(report=str(REPORT), report_sha256=training.sha(REPORT), outcome=report['outcome'],
        elapsed_seconds=report['elapsed_seconds'], actual_state_count=report['architecture_inventory']['actual_state_count'])), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Failure is explicit and cannot be mistaken for completed formal work.
        if not REPORT.exists():
            training.atomic_json(REPORT, dict(outcome='fail', error=repr(error), traceback=traceback.format_exc(),
                scope='bounded real TRAIN CPU smoke; no accuracy/validation result', helper_sha256=training.sha(Path(__file__))))
        raise
