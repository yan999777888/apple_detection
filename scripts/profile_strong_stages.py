"""Small-sample, synchronized diagnostic of the unchanged original strong Base.

The nested timers are inclusive; only sibling stages may be added. This is a
diagnostic with synchronization overhead, not a replacement latency benchmark.
No training, threshold changes, candidate truncation, or source edits occur.
"""
import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import statistics
import time
import traceback

from reproduce_strong_baseline import sha, source_inventory, write


class TemporaryStageTimers(AbstractContextManager):
    """Wrap instance methods only, then restore their original lookup behavior."""

    def __init__(self, sync):
        self.sync = sync
        self.restorations = []
        self.rows = {}
        self.enabled = False

    def reset(self):
        self.rows = {}

    def wrap(self, owner, method_name, label, describe=None):
        if not hasattr(owner, method_name):
            return
        original = getattr(owner, method_name)
        had_instance_attribute = method_name in owner.__dict__
        instance_attribute = owner.__dict__.get(method_name)

        def timed(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            self.sync()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            self.sync()
            elapsed = (time.perf_counter() - started) * 1000
            row = self.rows.setdefault(label, {'inclusive_ms': 0., 'calls': 0, 'details': []})
            row['inclusive_ms'] += elapsed
            row['calls'] += 1
            if describe is not None:
                row['details'].append(describe(args, kwargs, result))
            return result

        setattr(owner, method_name, timed)
        self.restorations.append((owner, method_name, had_instance_attribute, instance_attribute))

    def __exit__(self, *exc):
        for owner, name, had_attribute, value in reversed(self.restorations):
            if had_attribute:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self.restorations.clear()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    assert args.count >= 3 and args.warmup in (1, 2)
    cfgpath, sampled, out = args.config.resolve(), args.sample_dir.resolve(), args.output.resolve()
    assert out.drive.upper() == 'F:', 'All output must remain on F drive'
    cfg = json.loads(cfgpath.read_text(encoding='utf-8-sig'))
    root = (cfgpath.parent / cfg['path_base']).resolve()
    paths = {key: (root / value).resolve() for key, value in cfg['paths'].items()}
    manifest_path = sampled / 'sample_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    samples = manifest['samples'][:args.count]
    assert len(samples) == args.count and cfg['short_side'] == 1024
    references = {row['image_id']: row['pred'] for row in json.loads(
        (sampled / 'base/original_predictions.json').read_text(encoding='utf-8'))}
    out.mkdir(parents=True, exist_ok=False)
    inventory, script_sha = source_inventory(), sha(__file__)
    status = {'schema': 'strong_synchronized_stage_profile_v1', 'phase': 'initializing',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sample_count': len(samples), 'selection': 'first N of the existing fixed random 8-image manifest',
              'sample_manifest_sha256': sha(manifest_path), 'samples': samples,
              'config_sha256': sha(cfgpath), 'source_sha256': inventory, 'script_sha256': script_sha,
              'batch_size': 1, 'FP32': True, 'short_side': cfg['short_side'], 'max_size': cfg['max_size'],
              'warmup_frames': args.warmup, 'platform': platform.platform(),
              'timing_warning': 'Diagnostic synchronized wall-clock: nested timers are inclusive, synchronization adds overhead. Small sample, no throughput claim.',
              'timer_hierarchy': {'model_call': ['backbone_fpn', 'rpn', 'roi_total', 'model_other'],
                  'roi_total': ['roi_predictions', 'final_nms', 'cascade_boxes_if_active', 'foreground_if_active', 'roi_other'],
                  'roi_predictions': ['pooled_roi_features', 'mask_predictions', 'gated_classifier_if_active',
                                      'classification_evidence', 'occlusion_verifier_if_active', 'roi_predictions_other']},
              'excluded': ['weight/model load and validation', 'output JSON writes', 'GT', 'rendering', 'camera and robot I/O']}
    write(out / 'profile_status.json', status)
    model = None
    torch = None
    try:
        import numpy as np
        import torch
        from PIL import Image
        from apple_seg.od_data import legacy_rgb_tensor
        from apple_seg.od_training import load_checkpoint
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.od_inference import _strict_restore
        from apple_seg.od_evaluate import _validate_prediction
        from apple_seg.od_roi import paste_roi_mask
        torch.set_num_threads(cfg['cpu_threads'])
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available()
        status.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                      tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                      tf32_cudnn=torch.backends.cudnn.allow_tf32)
        checkpoint = paths['base_checkpoint']
        assert sha(checkpoint) == cfg['base_checkpoint_sha256']
        print('PROFILE loading original Base, FP32, first ' + str(len(samples)) + ' fixed images', flush=True)
        payload = load_checkpoint(checkpoint)
        model = build_od_amodal_detector(weights=None, **payload['constructor_config'])
        restored = _strict_restore(model, payload['model'])
        assert model.constructor_config == payload['constructor_config']
        status.update(checkpoint=str(checkpoint), checkpoint_sha256=cfg['base_checkpoint_sha256'],
                      strict_restore=restored, constructor_config=payload['constructor_config'])
        del payload
        gc.collect()
        model.requires_grad_(False).eval().to('cuda')
        roi = model.roi_heads
        status.update(model_type=f'{type(model).__module__}.{type(model).__qualname__}',
                      roi_type=f'{type(roi).__module__}.{type(roi).__qualname__}',
                      backbone_type=f'{type(model.backbone).__module__}.{type(model.backbone).__qualname__}',
                      roi_configuration=roi.configuration, roi_chunk_size=roi.roi_chunk_size,
                      active={'context_cascade': hasattr(roi, 'context_cascade'),
                              'mask_geometry_head': hasattr(roi, 'mask_geometry_head'),
                              'visible_proposals': bool(model.visible_proposals),
                              'shallow_detail': bool(model.shallow_detail)})
        print('PROFILE_ACTIVE ' + json.dumps({key: status[key] for key in ('roi_type', 'roi_chunk_size', 'active')}, ensure_ascii=False), flush=True)
        weight_versions = {key: value._version for key, value in model.state_dict(keep_vars=True).items()}

        def sync():
            torch.cuda.synchronize()

        def tick():
            sync()
            return time.perf_counter()

        def shapes(value):
            return {key: list(item.shape) for key, item in value.items() if isinstance(item, torch.Tensor)}

        with TemporaryStageTimers(sync) as timers:
            timers.wrap(model.backbone, 'forward', 'backbone_fpn', lambda a, k, r: {'output_shapes': shapes(r)})
            timers.wrap(model.rpn, 'forward', 'rpn', lambda a, k, r: {'proposals_per_image': [len(p) for p in r[0]]})
            timers.wrap(roi, 'forward', 'roi_total', lambda a, k, r: {'input_proposals': [len(p) for p in a[1]], 'output_candidates': len(r[0][0]['scores'])})
            timers.wrap(roi, '_roi_predictions', 'roi_predictions', lambda a, k, r: {'proposals': [len(p) for p in a[1]], 'mask_shapes': shapes(r[0])})
            timers.wrap(roi, '_final_nms', 'final_nms', lambda a, k, r: {'input_candidates': len(a[1]), 'kept_before_final_output_cap': len(r)})
            timers.wrap(roi, '_pooled_roi_features', 'pooled_roi_features')
            timers.wrap(roi, '_mask_predictions', 'mask_predictions')
            timers.wrap(roi, '_fuse_classification_evidence', 'classification_evidence')
            if roi.gated_classifier is not None and roi.use_visible_gate:
                timers.wrap(roi.gated_classifier, 'forward', 'gated_classifier_if_active')
            if roi.occlusion_verifier is not None:
                timers.wrap(roi.occlusion_verifier, 'forward', 'occlusion_verifier_if_active')
            timers.wrap(roi, '_cascade_boxes', 'cascade_boxes_if_active')
            timers.wrap(roi, '_with_foreground', 'foreground_if_active')
            if hasattr(roi, 'mask_geometry_head'):
                timers.wrap(roi.mask_geometry_head, 'forward', 'geometry_head_nested_in_cascade')
            if hasattr(roi, 'context_cascade'):
                timers.wrap(roi.context_cascade, 'forward_stage', 'context_stages_nested_in_cascade')

            def infer(sample, measured):
                timers.enabled = measured
                timers.reset()
                start = tick()
                with Image.open(sample['rgb_path']) as image:
                    rgb = image.convert('RGB')
                after_read = tick()
                assert rgb.size == (sample['width'], sample['height'])
                pixels = legacy_rgb_tensor(rgb, cfg['short_side'], cfg['max_size'])
                after_preprocess = tick()
                input_hw, hw = tuple(pixels.shape[-2:]), (rgb.height, rgb.width)
                gpu_pixels = pixels[None].to('cuda')
                pixel_mask = torch.ones((1, *input_hw), dtype=torch.bool, device='cuda')
                after_transfer = tick()
                raw = model(gpu_pixels, pixel_mask=pixel_mask)
                after_model = tick()
                raw_shapes = shapes(raw[0])
                pred = _validate_prediction(raw, input_hw, hw, mask_chunk_size=8)
                after_decode = tick()
                selected = [i for i, score in enumerate(pred['scores']) if score >= .75]
                masks = [paste_roi_mask(pred[branch][i], pred['boxes'][i], hw, mask_threshold=.5)
                         for i in selected for branch in ('visible', 'amodal')]
                after_masks = tick()
                difference = {}
                for key in ('boxes', 'scores'):
                    actual = pred[key].numpy() if isinstance(pred[key], torch.Tensor) else np.asarray(pred[key])
                    expected = np.asarray(references[sample['image_id']][key])
                    if key == 'boxes':
                        actual, expected = actual.reshape(-1, 4), expected.reshape(-1, 4)
                    assert actual.shape == expected.shape
                    difference[key] = float(np.max(np.abs(actual - expected), initial=0))
                assert max(difference.values()) <= 1e-4, difference
                row = {'image_id': sample['image_id'], 'input_hw': list(input_hw), 'original_hw': list(hw),
                       'stages': timers.rows, 'raw_output_shapes': raw_shapes, 'raw_candidates': len(pred['scores']),
                       'selected_at_confidence_0_75': len(selected), 'pasted_mask_crops': len(masks),
                       'reference_max_abs_difference': difference,
                       'read_rgb_ms': (after_read-start)*1000,
                       'preprocess_ms': (after_preprocess-after_read)*1000,
                       'input_transfer_ms': (after_transfer-after_preprocess)*1000,
                       'model_call_ms': (after_model-after_transfer)*1000,
                       'output_decode_ms': (after_decode-after_model)*1000,
                       'native_selected_va_paste_ms': (after_masks-after_decode)*1000,
                       'read_to_masks_ms': (after_masks-start)*1000}
                if measured:
                    def ms(name):
                        return timers.rows.get(name, {}).get('inclusive_ms', 0.)
                    row['residual_ms'] = {
                        'model_other': row['model_call_ms'] - sum(ms(s) for s in ('backbone_fpn', 'rpn', 'roi_total')),
                        'roi_other': ms('roi_total') - sum(ms(s) for s in ('roi_predictions', 'final_nms', 'cascade_boxes_if_active', 'foreground_if_active')),
                        'roi_predictions_other': ms('roi_predictions') - sum(ms(s) for s in ('pooled_roi_features', 'mask_predictions', 'gated_classifier_if_active', 'classification_evidence', 'occlusion_verifier_if_active'))}
                    assert min(row['residual_ms'].values()) >= -.01, 'Nested timer accounting failed'
                    outer_sum = sum(row[key] for key in ('read_rgb_ms', 'preprocess_ms', 'input_transfer_ms', 'model_call_ms', 'output_decode_ms', 'native_selected_va_paste_ms'))
                    row['outer_accounting_error_ms'] = row['read_to_masks_ms'] - outer_sum
                    assert abs(row['outer_accounting_error_ms']) < .01
                del raw, pred, masks, pixels, gpu_pixels, pixel_mask, rgb
                return row

            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                for index in range(args.warmup):
                    infer(samples[index % len(samples)], measured=False)
                    print(f'PROFILE warmup {index+1}/{args.warmup}', flush=True)
                torch.cuda.reset_peak_memory_stats()
                rows = []
                for index, sample in enumerate(samples):
                    rows.append(infer(sample, measured=True))
                    status.update(phase='profiling', completed=index+1, frames=rows)
                    write(out / 'profile_status.json', status)
                    print('PROFILE_FRAME ' + json.dumps(rows[-1], ensure_ascii=False), flush=True)
            timers.enabled = False
        assert all(model.state_dict(keep_vars=True)[key]._version == version for key, version in weight_versions.items()), 'Model tensors were modified'
        assert source_inventory() == inventory and sha(__file__) == script_sha
        assert sha(manifest_path) == status['sample_manifest_sha256']
        assert sha(checkpoint) == cfg['base_checkpoint_sha256']
        stage_labels = sorted({key for row in rows for key in row['stages']})
        summary = {key: {'mean_inclusive_ms': statistics.mean(row['stages'].get(key, {}).get('inclusive_ms', 0.) for row in rows),
                         'calls_per_image': [row['stages'].get(key, {}).get('calls', 0) for row in rows]}
                   for key in stage_labels}
        status.update(phase='complete', frames=rows, stage_summary=summary,
                      residual_mean_ms={key: statistics.mean(row['residual_ms'][key] for row in rows) for key in rows[0]['residual_ms']},
                      outer_mean_ms={key: statistics.mean(row[key] for row in rows) for key in rows[0] if key.endswith('_ms') and isinstance(rows[0][key], (int, float))},
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      source_before_after_identical=True, checkpoint_before_after_identical=True,
                      state_tensor_versions_unchanged=True, temporary_wrappers_restored=True,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'profile_results.json', status)
        write(out / 'profile_status.json', status)
        print('PROFILE_RESULT ' + json.dumps({'active': status['active'], 'roi_type': status['roi_type'], 'stages': summary, 'residual_mean_ms': status['residual_mean_ms'], 'outer_mean_ms': status['outer_mean_ms']}, ensure_ascii=False), flush=True)
    except BaseException as error:
        status.update(phase='failed', error=str(error), traceback=traceback.format_exc())
        write(out / 'profile_status.json', status)
        raise
    finally:
        model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print('PROFILE_GPU_RELEASED process ending; no training or other GPU work started', flush=True)


if __name__ == '__main__':
    main()
