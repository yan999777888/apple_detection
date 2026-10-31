"""Synchronized module profile of original Base with the audited fast adapter.

Fixed candidate: FP16 autocast, ROI64, pre-NMS .35, final .75, skip-dead-A
disabled. Reuses the original profiler's reversible timers and fixed samples.
No training or modification of the model/runtime source occurs.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

from reproduce_strong_baseline import REPO, sha, source_inventory, write
from profile_strong_stages import TemporaryStageTimers

sys.path.insert(0, str(REPO))


def reference_difference(actual_boxes, actual_scores, reference, numpy):
    boxes = numpy.asarray(actual_boxes, dtype=float).reshape(-1, 4)
    expected_boxes = numpy.asarray(reference['endpoint_boxes'], dtype=float).reshape(-1, 4)
    scores = numpy.asarray(actual_scores, dtype=float)
    expected_scores = numpy.asarray(reference['endpoint_scores'], dtype=float)
    if boxes.shape != expected_boxes.shape or scores.shape != expected_scores.shape:
        raise AssertionError('Fast benchmark/profile endpoint candidate counts differ')
    differences = {'boxes': float(numpy.max(numpy.abs(boxes-expected_boxes), initial=0)),
                   'scores': float(numpy.max(numpy.abs(scores-expected_scores), initial=0))}
    if max(differences.values()) > 1e-4:
        raise AssertionError('Fast benchmark/profile endpoint drift: '+str(differences))
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--warmup', type=int, choices=(1, 2), default=2)
    parser.add_argument('--baseline-profile', type=Path, default=REPO/'reports/fast_deploy_20261031/profile_fp32/profile_results.json')
    parser.add_argument('--reference-benchmark', type=Path, default=REPO/'reports/fast_deploy_20261031/benchmark_fp16_v1/benchmark_results.json')
    args = parser.parse_args()
    cfgpath, sampled, out = args.config.resolve(), args.sample_dir.resolve(), args.output.resolve()
    assert out.drive.upper() == 'F:' and args.count >= 3
    cfg = json.loads(cfgpath.read_text(encoding='utf-8-sig'))
    root = (cfgpath.parent/cfg['path_base']).resolve()
    paths = {key: (root/value).resolve() for key, value in cfg['paths'].items()}
    assert cfg['short_side'] == 1024 and cfg['max_size'] == 1280
    manifest_path = sampled/'sample_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    samples = manifest['samples'][:args.count]
    assert len(samples) == args.count
    baseline_path, benchmark_path = args.baseline_profile.resolve(), args.reference_benchmark.resolve()
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    benchmark = json.loads(benchmark_path.read_text(encoding='utf-8'))
    assert baseline['phase'] == 'complete' and baseline['checkpoint_sha256'] == cfg['base_checkpoint_sha256']
    assert baseline['sample_manifest_sha256'] == sha(manifest_path)
    assert [row['image_id'] for row in baseline['frames'][:args.count]] == [sample['image_id'] for sample in samples]
    assert benchmark['cases']['base1024']['phase'] == 'complete'
    assert benchmark['precision'] == 'fp16' and benchmark['roi_chunk_size'] == 64
    assert benchmark['roi_precision'] == 'autocast' and benchmark['skip_dead_amodal'] is False
    assert benchmark['sample_manifest_sha256'] == sha(manifest_path)
    assert benchmark['cases']['base1024']['spec']['pre_nms_threshold'] == .35
    references = {row['image_id']: row for row in benchmark['cases']['base1024']['frames']}
    out.mkdir(parents=True, exist_ok=False)
    inventory, adapter_sha, script_sha = source_inventory(), sha(REPO/'deployment/strong_fast.py'), sha(__file__)
    assert benchmark['adapter_sha256'] == adapter_sha and benchmark['source_sha256'] == inventory
    status = {'schema': 'strong_fast_synchronized_stage_profile_v1', 'phase': 'initializing',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'samples': samples, 'sample_count': len(samples),
              'warmup_frames': args.warmup, 'precision': 'fp16', 'roi_precision': 'autocast',
              'roi_chunk_size': 64, 'pre_nms_threshold': .35, 'final_threshold': .75,
              'skip_dead_amodal': False, 'sample_manifest_sha256': sha(manifest_path),
              'source_sha256': inventory, 'adapter_sha256': adapter_sha, 'script_sha256': script_sha,
              'timer_helper_sha256': sha(REPO/'scripts/profile_strong_stages.py'),
              'config_sha256': sha(cfgpath), 'baseline_profile_sha256': sha(baseline_path),
              'reference_benchmark_sha256': sha(benchmark_path),
              'timing_warning': 'Small-sample synchronized module diagnostic; nested times are inclusive and synchronization adds overhead. Not the warmed whole-frame throughput benchmark.',
              'comparison_scope': 'Same original Base checkpoint and fixed native images; deployment arithmetic/chunk/cutoff differ. Accuracy requires independent full-VAL audit.',
              'timer_hierarchy': baseline['timer_hierarchy'], 'training': False}
    write(out/'profile_status.json', status)
    model, torch = None, None
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
        from deployment.strong_fast import FastDeploymentAdapter
        torch.set_num_threads(cfg['cpu_threads'])
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available()
        status.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                      tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                      tf32_cudnn=torch.backends.cudnn.allow_tf32)
        checkpoint = paths['base_checkpoint']
        assert sha(checkpoint) == cfg['base_checkpoint_sha256']
        print('FAST_PROFILE loading original Base; FP16/ROI64/cutoff.35, no dead-A bypass', flush=True)
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
        assert roi.use_amodal_refinement is True
        status.update(model_type=f'{type(model).__module__}.{type(model).__qualname__}',
                      roi_type=f'{type(roi).__module__}.{type(roi).__qualname__}',
                      active={'context_cascade': hasattr(roi, 'context_cascade'),
                              'mask_geometry_head': hasattr(roi, 'mask_geometry_head'),
                              'visible_proposals': bool(model.visible_proposals), 'shallow_detail': bool(model.shallow_detail)})
        versions = {key: value._version for key, value in model.state_dict(keep_vars=True).items()}

        def sync():
            torch.cuda.synchronize()

        def tick():
            sync()
            return time.perf_counter()

        def shapes(values):
            return {key: list(value.shape) for key, value in values.items() if isinstance(value, torch.Tensor)}

        # Enter the adapter first. Timer teardown restores its bound overrides;
        # adapter teardown subsequently restores the untouched original methods.
        with FastDeploymentAdapter(model, pre_nms_threshold=.35, roi_chunk_size=64,
                                   skip_dead_amodal=False, roi_precision='autocast') as adapter:
            with TemporaryStageTimers(sync) as timers:
                timers.wrap(model.backbone, 'forward', 'backbone_fpn', lambda a, k, r: {'output_shapes': shapes(r)})
                timers.wrap(model.rpn, 'forward', 'rpn', lambda a, k, r: {'proposals_per_image': [len(p) for p in r[0]]})
                timers.wrap(roi, 'forward', 'roi_total', lambda a, k, r: {'input_proposals': [len(p) for p in a[1]], 'output_candidates': len(r[0][0]['scores'])})
                timers.wrap(roi, '_roi_predictions', 'roi_predictions', lambda a, k, r: {'proposals': [len(p) for p in a[1]], 'mask_shapes': shapes(r[0])})
                timers.wrap(roi, '_final_nms', 'final_nms', lambda a, k, r: {'input_candidates_before_adapter_cutoff': len(a[1]), 'kept_before_final_output_cap': len(r)})
                timers.wrap(roi, '_pooled_roi_features', 'pooled_roi_features')
                timers.wrap(roi, '_mask_predictions', 'mask_predictions')
                timers.wrap(roi, '_fuse_classification_evidence', 'classification_evidence')
                if roi.gated_classifier is not None and roi.use_visible_gate:
                    timers.wrap(roi.gated_classifier, 'forward', 'gated_classifier_if_active')
                if roi.occlusion_verifier is not None:
                    timers.wrap(roi.occlusion_verifier, 'forward', 'occlusion_verifier_if_active')

                def infer(sample, measured):
                    timers.enabled = measured
                    timers.reset()
                    t0 = tick()
                    with Image.open(sample['rgb_path']) as image:
                        rgb = image.convert('RGB')
                    t1 = tick()
                    pixels = legacy_rgb_tensor(rgb, cfg['short_side'], cfg['max_size'])
                    t2 = tick()
                    hw, input_hw = (rgb.height, rgb.width), tuple(pixels.shape[-2:])
                    inputs = pixels[None].to('cuda')
                    mask = torch.ones((1, *input_hw), dtype=torch.bool, device='cuda')
                    t3 = tick()
                    with torch.autocast('cuda', dtype=torch.float16):
                        raw = adapter.predict_views(inputs, mask)
                    t4 = tick()
                    pred = _validate_prediction(raw, input_hw, hw, mask_chunk_size=8)
                    t5 = tick()
                    selected = [index for index, score in enumerate(pred['scores']) if score >= .75]
                    boxes = pred['boxes'][selected].tolist()
                    scores = [float(pred['scores'][index]) for index in selected]
                    t6 = tick()
                    masks = [paste_roi_mask(pred[branch][index], pred['boxes'][index], hw, mask_threshold=.5)
                             for index in selected for branch in ('visible', 'amodal')]
                    t7 = tick()
                    difference = reference_difference(boxes, scores, references[sample['image_id']], np)
                    row = {'image_id': sample['image_id'], 'input_hw': list(input_hw), 'original_hw': list(hw),
                           'stages': timers.rows, 'raw_output_shapes': shapes(raw[0]),
                           'adapter_stats': deepcopy(adapter.last_stats), 'raw_candidates': len(pred['scores']),
                           'selected_at_confidence_0_75': len(selected), 'pasted_mask_crops': len(masks),
                           'fast_benchmark_endpoint_max_abs_difference': difference,
                           'read_rgb_ms': (t1-t0)*1000, 'preprocess_ms': (t2-t1)*1000,
                           'input_transfer_ms': (t3-t2)*1000, 'model_call_ms': (t4-t3)*1000,
                           'output_decode_ms': (t5-t4)*1000, 'final_score_filter_ms': (t6-t5)*1000,
                           'native_selected_va_paste_ms': (t7-t6)*1000, 'read_to_masks_ms': (t7-t0)*1000}
                    if measured:
                        def ms(name):
                            return timers.rows.get(name, {}).get('inclusive_ms', 0.)
                        row['residual_ms'] = {
                            'model_other': row['model_call_ms']-sum(ms(s) for s in ('backbone_fpn', 'rpn', 'roi_total')),
                            'roi_other': ms('roi_total')-ms('roi_predictions')-ms('final_nms'),
                            'roi_predictions_other': ms('roi_predictions')-sum(ms(s) for s in ('pooled_roi_features', 'mask_predictions', 'gated_classifier_if_active', 'classification_evidence', 'occlusion_verifier_if_active'))}
                        assert min(row['residual_ms'].values()) >= -.01
                        row['outer_accounting_error_ms'] = row['read_to_masks_ms']-sum(row[key] for key in (
                            'read_rgb_ms', 'preprocess_ms', 'input_transfer_ms', 'model_call_ms', 'output_decode_ms', 'final_score_filter_ms', 'native_selected_va_paste_ms'))
                        assert abs(row['outer_accounting_error_ms']) < .01
                    del raw, pred, inputs, mask, masks, pixels, rgb
                    return row

                with torch.inference_mode():
                    for index in range(args.warmup):
                        infer(samples[index % len(samples)], False)
                        print(f'FAST_PROFILE warmup {index+1}/{args.warmup}', flush=True)
                    torch.cuda.reset_peak_memory_stats()
                    rows = []
                    for index, sample in enumerate(samples):
                        rows.append(infer(sample, True))
                        status.update(phase='profiling', frames=rows, completed=index+1)
                        write(out/'profile_status.json', status)
                        print('FAST_PROFILE_FRAME '+json.dumps(rows[-1], ensure_ascii=False), flush=True)
                timers.enabled = False
        state = model.state_dict(keep_vars=True)
        assert all(state[key]._version == value for key, value in versions.items())
        assert roi.roi_chunk_size == baseline['roi_chunk_size']
        assert '_strong_fast_adapter_active' not in roi.__dict__
        assert not any(key in roi.__dict__ for key in ('_roi_predictions', '_decoded_boxes', '_final_nms', '_mask_predictions'))
        assert source_inventory() == inventory and sha(REPO/'deployment/strong_fast.py') == adapter_sha
        assert sha(__file__) == script_sha and sha(REPO/'scripts/profile_strong_stages.py') == status['timer_helper_sha256']
        assert sha(checkpoint) == cfg['base_checkpoint_sha256']
        assert sha(baseline_path) == status['baseline_profile_sha256'] and sha(benchmark_path) == status['reference_benchmark_sha256']
        summary = {name: {'mean_inclusive_ms': statistics.mean(row['stages'].get(name, {}).get('inclusive_ms', 0.) for row in rows),
                          'calls_per_image': [row['stages'].get(name, {}).get('calls', 0) for row in rows]}
                   for name in sorted({name for row in rows for name in row['stages']})}
        status.update(phase='complete', frames=rows, stage_summary=summary,
                      residual_mean_ms={key: statistics.mean(row['residual_ms'][key] for row in rows) for key in rows[0]['residual_ms']},
                      outer_mean_ms={key: statistics.mean(row[key] for row in rows) for key in rows[0] if key.endswith('_ms') and isinstance(rows[0][key], (int, float))},
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      source_before_after_identical=True, checkpoint_before_after_identical=True,
                      state_tensor_versions_unchanged=True, temporary_wrappers_and_adapter_restored=True,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        # Every row compares one inclusive module against that same module;
        # parent and child rows must never be summed together.
        comparison = {name: {'original_fp32_ms': baseline['stage_summary'].get(name, {}).get('mean_inclusive_ms'),
                             'fast_fp16_ms': value['mean_inclusive_ms']}
                      for name, value in summary.items()}
        status['baseline_stage_comparison'] = comparison
        write(out/'profile_results.json', status)
        write(out/'profile_status.json', status)
        lines = ['# 原 Base 优化前后模块计时', '',
                 f"原 FP32/ROI8 与候选 FP16/ROI64/pre-NMS .35；相同{len(samples)}张固定原图，候选预热{args.warmup}次、原诊断预热{baseline['warmup_frames']}次。",
                 '逐阶段同步诊断，包含同步开销；各行有包含关系，不可把父子阶段相加；全验证集精度另行验收。', '',
                 '| 模块 | 原 FP32 ms | 候选 FP16 ms |', '|---|---:|---:|']
        for name, value in comparison.items():
            before = f"{value['original_fp32_ms']:.3f}" if value['original_fp32_ms'] is not None else '—'
            lines.append(f"| {name} | {before} | {value['fast_fp16_ms']:.3f} |")
        lines.extend(['', '候选最终框和分数与此前同配置 Base 部署 benchmark 核验，最大差值见 JSON。',
                      '最终可见/完整掩膜仍由原网络真实预测；没有禁用 amodal refinement，没有训练。'])
        (out/'profile_comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('FAST_PROFILE_RESULT '+json.dumps({'stages': summary, 'residual_mean_ms': status['residual_mean_ms'], 'outer_mean_ms': status['outer_mean_ms']}, ensure_ascii=False), flush=True)
    except BaseException as error:
        status.update(phase='failed', error=str(error), traceback=traceback.format_exc())
        write(out/'profile_status.json', status)
        raise
    finally:
        model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print('FAST_PROFILE_GPU_RELEASED process ending', flush=True)


if __name__ == '__main__':
    main()
