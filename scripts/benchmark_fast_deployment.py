"""Actual source-frame deployment latency for unchanged original strong weights.

Run after compatibility checks. This measures speed, not detection accuracy.
RGB is decoded before timing. The local case times a real 512-square centre
crop, not a resized full image, and is not a D405 camera experiment.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

from reproduce_strong_baseline import REPO, sha, source_inventory, write

sys.path.insert(0, str(REPO))

CASE_SPECS = {
    'base1024': {'models': ['base'], 'short_side': 1024, 'max_size': 1280,
                 'crop_edge': None, 'views_per_model': 1, 'pre_nms_threshold': .35,
                 'final_threshold': .75, 'native_va_masks': True},
    'small1024': {'models': ['small'], 'short_side': 1024, 'max_size': 1280,
                  'crop_edge': None, 'views_per_model': 1, 'pre_nms_threshold': .35,
                  'final_threshold': .75, 'native_va_masks': True},
    'small512_crop': {'models': ['small'], 'short_side': 512, 'max_size': 512,
                      'crop_edge': 512, 'views_per_model': 1, 'pre_nms_threshold': .70,
                      'final_threshold': .70, 'native_va_masks': True},
    'four_view1024': {'models': ['base', 'small'], 'short_side': 1024, 'max_size': 1280,
                      'crop_edge': None, 'views_per_model': 2, 'pre_nms_threshold': .35,
                      'final_threshold': .75, 'native_va_masks': False},
}


def center_crop_box(width, height, edge=512):
    if min(width, height) < edge or edge < 1:
        raise ValueError('Input must contain the complete fixed-size centre crop')
    left, top = (width-edge)//2, (height-edge)//2
    return left, top, left+edge, top+edge


def latency_summary(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError('Need measured frames')
    position = (len(ordered)-1)*.95
    lower, upper = int(position), min(int(position)+1, len(ordered)-1)
    p95 = ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)
    mean = statistics.mean(ordered)
    return {'n_source_frames': len(ordered), 'mean_ms': mean,
            'median_ms': statistics.median(ordered), 'p95_ms': p95,
            'min_ms': ordered[0], 'max_ms': ordered[-1], 'source_frames_per_second': 1000/mean}


def markdown_report(status):
    lines = ['# 原权重部署速度实测', '',
             f"状态：{status['phase']}。精度模式：{status['precision']}；ROI chunk：{status['roi_chunk_size']}；ROI 卷积继承 autocast。", '',
             '每次计时处理一张源图；四视图的每个模型以 batch=2 共享骨干，两个模型串行执行。FPS 不乘以视图数。', '',
             '输入从已解码 CPU RGB 开始；初始化、磁盘读取、权重校验、GT、绘图、相机和机器人通信均不计时。', '',
             'Small 512 为原图固定中心 512×512 裁剪，裁剪在计时内，仅作局部视野速度模拟，不代表 D405 实拍精度。', '',
             '单模型终点包括阈值筛选和原生坐标 V/A 掩膜贴回；四视图终点为原方案融合后筛选的框和分数，不包含融合掩膜。', '',
             '| 模式 | 源图数 | 平均 ms | 中位 ms | P95 ms | 源图 FPS |',
             '|---|---:|---:|---:|---:|---:|']
    for name, result in status['cases'].items():
        if result.get('phase') == 'complete':
            summary = result['summary']
            lines.append(f"| {name} | {summary['n_source_frames']} | {summary['mean_ms']:.2f} | {summary['median_ms']:.2f} | {summary['p95_ms']:.2f} | {summary['source_frames_per_second']:.3f} |")
        else:
            lines.append(f"| {name} | {result.get('phase', 'pending')} | — | — | — | — |")
    lines.extend(['', f"GPU 常驻权重：{', '.join(status['resident_models'])}；没有 CPU offload。", '',
                  '主测仅在整帧起点、终点同步 CUDA。阶段细分来自单独的同步诊断，不能将异步主测的中间 CPU 时间戳当作纯 GPU 算子耗时。', '',
                  '原始候选、pre-NMS 阈值筛选数量与最终分数筛选数量在 JSON 中记录；pre-NMS 筛选耗时包含在模型调用内。', '',
                  '这里只报告固定 8 张图片的一轮速度，样本量不足以刻画长期 P95；完整 P@R90 验收由独立全验证集评估完成。'])
    for name, result in status['cases'].items():
        if result.get('error'):
            lines.extend(['', f"{name} 失败：{result['error']}"])
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', '--cases', nargs='+', choices=tuple(CASE_SPECS),
                        default=list(CASE_SPECS), help='Select cases without running the others')
    parser.add_argument('--precision', choices=('fp16', 'bf16', 'fp32'), default='fp16')
    parser.add_argument('--roi-chunk', type=int, default=64)
    parser.add_argument('--warmup', type=int, choices=(3, 4, 5), default=4)
    parser.add_argument('--diagnostic-count', type=int, choices=range(4), default=3)
    args = parser.parse_args()
    assert args.roi_chunk > 0 and len(set(args.variants)) == len(args.variants)
    cfgpath, sampled, out = args.config.resolve(), args.sample_dir.resolve(), args.output.resolve()
    assert out.drive.upper() == 'F:', 'Output must remain on F drive'
    cfg = json.loads(cfgpath.read_text(encoding='utf-8-sig'))
    root = (cfgpath.parent/cfg['path_base']).resolve()
    paths = {key: (root/value).resolve() for key, value in cfg['paths'].items()}
    manifest_path = sampled/'sample_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    samples = manifest['samples']
    assert len(samples) == 8 and len({sample['image_id'] for sample in samples}) == 8
    assert all((sample['width'], sample['height']) == (1300, 1300) for sample in samples)
    out.mkdir(parents=True, exist_ok=False)
    inventory, adapter_sha, script_sha = source_inventory(), sha(REPO/'deployment/strong_fast.py'), sha(__file__)
    needed = [name for name in ('base', 'small') if any(name in CASE_SPECS[case]['models'] for case in args.variants)]
    status = {'schema': 'strong_fast_deployment_benchmark_v1', 'phase': 'initializing',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'precision': args.precision,
              'roi_precision': 'autocast', 'roi_chunk_size': args.roi_chunk, 'skip_dead_amodal': False,
              'sample_manifest_sha256': sha(manifest_path), 'samples': samples,
              'config_sha256': sha(cfgpath), 'source_sha256': inventory, 'adapter_sha256': adapter_sha,
              'script_sha256': script_sha, 'platform': platform.platform(), 'python': sys.version,
              'cpu_threads': cfg['cpu_threads'], 'warmup_source_frames_per_case': args.warmup,
              'diagnostic_source_frames_per_case': args.diagnostic_count, 'main_source_frames_per_case': 8,
              'resident_models': needed, 'cpu_offload': False, 'models': {}, 'cases': {},
              'input_start': 'Predecoded CPU RGB 1300x1300; local centre crop occurs inside timer',
              'primary_timing': 'Only whole-source-frame start/end synchronize CUDA; includes CPU, transfer, forward, decode, score filter and endpoint work',
              'diagnostic_timing': 'Separate pass synchronizes stages; includes additional synchronization overhead',
              'pre_nms_cutoff_timing': 'Included inside model_call_ms; adapter_stats reports before/after counts',
              'fps_definition': '1000 / mean source-frame latency, never multiplied by view count',
              'full_VAL': False, 'accuracy_claim': None,
              'excluded': ['weight/model initialization and SHA checks', 'disk RGB decode', 'GT', 'plotting', 'camera capture', 'robot I/O']}
    write(out/'benchmark_status.json', status)
    models, versions = {}, {}
    torch = None

    def snapshot():
        try:
            return subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu', '--format=csv,noheader'], text=True).strip()
        except Exception as error:
            return str(error)

    try:
        import torch
        from PIL import Image
        import flip_tta
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
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Requested BF16 is not supported by this GPU/runtime')
        dtype = torch.float16 if args.precision == 'fp16' else torch.bfloat16
        status.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
                      gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                      tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                      tf32_cudnn=torch.backends.cudnn.allow_tf32, gpu_before=snapshot())
        rgbs = []
        for sample in samples:
            with Image.open(sample['rgb_path']) as image:
                rgbs.append(image.convert('RGB'))
            assert rgbs[-1].size == (1300, 1300)
        # Every needed model stays on the GPU through all requested cases.
        # OOM is reported; no fallback offload or hidden candidate reduction.
        for name in needed:
            checkpoint = paths[name+'_checkpoint']
            assert checkpoint.drive.upper() == 'F:' and sha(checkpoint) == cfg[name+'_checkpoint_sha256']
            print('FAST_BENCH loading original '+name, flush=True)
            payload = load_checkpoint(checkpoint)
            model = build_od_amodal_detector(weights=None, **payload['constructor_config'])
            restored = _strict_restore(model, payload['model'])
            assert model.constructor_config == payload['constructor_config']
            status['models'][name] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': cfg[name+'_checkpoint_sha256'],
                                    'constructor_config': payload['constructor_config'], 'strict_restore': restored,
                                    'use_amodal_refinement': model.roi_heads.use_amodal_refinement}
            if name == 'base':
                assert model.roi_heads.use_amodal_refinement is True, 'Do not bypass the genuine Base amodal refinement'
            del payload
            gc.collect()
            model.requires_grad_(False).eval().to('cuda')
            versions[name] = {key: value._version for key, value in model.state_dict(keep_vars=True).items()}
            models[name] = model
        model = None
        status['loaded_cuda_allocated_bytes'] = torch.cuda.memory_allocated()
        status['loaded_cuda_reserved_bytes'] = torch.cuda.memory_reserved()
        write(out/'benchmark_status.json', status)

        def sync():
            torch.cuda.synchronize()
            return time.perf_counter()

        for case_name in args.variants:
            spec = deepcopy(CASE_SPECS[case_name])
            result = {'phase': 'running', 'spec': spec, 'resident_models': needed,
                      'source_frames_per_iteration': 1,
                      'total_views_per_source_frame': len(spec['models'])*spec['views_per_model'],
                      'shared_backbone_batch_per_model': spec['views_per_model'],
                      'endpoint': 'native V/A binary crops and filtered boxes/scores' if spec['native_va_masks'] else 'original fused, filtered native boxes/scores; no fused masks',
                      'local_camera_simulation': bool(spec['crop_edge']), 'frames': [], 'diagnostic_frames': []}
            status['cases'][case_name] = result
            status.update(phase='measuring', current_case=case_name)
            write(out/'benchmark_status.json', status)
            print('FAST_BENCH_START '+case_name, flush=True)
            try:
                with ExitStack() as stack:
                    adapters = {name: stack.enter_context(FastDeploymentAdapter(models[name],
                                pre_nms_threshold=spec['pre_nms_threshold'], roi_chunk_size=args.roi_chunk,
                                skip_dead_amodal=False, roi_precision='autocast')) for name in spec['models']}

                    def infer(rgb, image_id, diagnostic=False):
                        mark = sync if diagnostic else time.perf_counter
                        t0 = sync()
                        crop_box = center_crop_box(rgb.width, rgb.height, spec['crop_edge']) if spec['crop_edge'] else None
                        current_rgb = rgb.crop(crop_box) if crop_box else rgb
                        pixels = legacy_rgb_tensor(current_rgb, spec['short_side'], spec['max_size'])
                        hw, input_hw = (current_rgb.height, current_rgb.width), tuple(pixels.shape[-2:])
                        t1 = mark()
                        inputs = pixels[None].to('cuda')
                        if spec['views_per_model'] == 2:
                            inputs = torch.cat((inputs, inputs.flip(-1)), 0)
                        pixel_mask = torch.ones((len(inputs), *input_hw), dtype=torch.bool, device='cuda')
                        t2 = mark()
                        model_seconds, decode_seconds, layers, candidate_counts = 0., 0., [], []
                        for name in spec['models']:
                            before_model = mark()
                            with torch.autocast('cuda', enabled=args.precision != 'fp32', dtype=dtype):
                                raw = adapters[name].predict_views(inputs, pixel_mask)
                            after_model = mark()
                            model_seconds += after_model-before_model
                            for view, output in enumerate(raw):
                                prediction = _validate_prediction([output], input_hw, hw, mask_chunk_size=8)
                                if view:
                                    old = prediction['boxes'].clone()
                                    prediction['boxes'][:, 0] = hw[1]-old[:, 2]
                                    prediction['boxes'][:, 2] = hw[1]-old[:, 0]
                                candidate_counts.append({'model': name, 'view': 'hflip' if view else 'original', 'count': len(prediction['scores'])})
                                if spec['views_per_model'] == 2:
                                    layers.append([{'image_id': image_id, 'gt': {'boxes': []},
                                                    'pred': {'boxes': prediction['boxes'].tolist(), 'scores': prediction['scores']}}])
                                    # Four-view fusion is boxes/scores only. Do not claim
                                    # view-local masks have become fused native masks.
                                    del prediction
                            del raw
                            after_decode = mark()
                            decode_seconds += after_decode-after_model
                        t3 = mark()
                        if layers:
                            fused = flip_tta.fuse_records(layers, **cfg['fusion'])[0]['pred']
                            scores, boxes = fused['scores'], fused['boxes']
                        else:
                            scores, boxes = prediction['scores'], prediction['boxes']
                        t4 = mark()
                        selected = [index for index, score in enumerate(scores) if score >= spec['final_threshold']]
                        selected_scores = [float(scores[index]) for index in selected]
                        selected_boxes = [boxes[index].tolist() if isinstance(boxes[index], torch.Tensor) else boxes[index] for index in selected]
                        t5 = mark()
                        masks = ([paste_roi_mask(prediction[branch][index], prediction['boxes'][index], hw, mask_threshold=.5)
                                  for index in selected for branch in ('visible', 'amodal')]
                                 if spec['native_va_masks'] else [])
                        t6 = sync()
                        # Stats/JSON serialization are outside the measured endpoint.
                        row = {'image_id': image_id, 'diagnostic': diagnostic, 'crop_box_in_original': list(crop_box) if crop_box else None,
                               'native_output_hw': list(hw), 'input_hw': list(input_hw),
                               'preprocess_and_optional_crop_ms': (t1-t0)*1000,
                               'input_transfer_and_optional_flip_ms': (t2-t1)*1000,
                               'model_call_ms': model_seconds*1000, 'output_decode_and_flipback_ms': decode_seconds*1000,
                               'model_decode_host_other_ms': ((t3-t2)-model_seconds-decode_seconds)*1000,
                               'fusion_ms': (t4-t3)*1000, 'final_score_filter_ms': (t5-t4)*1000,
                               'native_mask_projection_ms': (t6-t5)*1000 if spec['native_va_masks'] else 0.,
                               'endpoint_sync_ms': 0. if spec['native_va_masks'] else (t6-t5)*1000,
                               'box_ready_ms': (t5-t0)*1000, 'end_to_end_ms': (t6-t0)*1000,
                               'raw_candidates_by_view': candidate_counts, 'selected_count': len(selected),
                               'native_mask_crops': len(masks),
                               'adapter_stats': {name: deepcopy(adapter.last_stats) for name, adapter in adapters.items()},
                               'endpoint_boxes': selected_boxes, 'endpoint_scores': selected_scores}
                        outer_sum = sum(row[key] for key in ('preprocess_and_optional_crop_ms', 'input_transfer_and_optional_flip_ms',
                                        'model_call_ms', 'output_decode_and_flipback_ms', 'model_decode_host_other_ms',
                                        'fusion_ms', 'final_score_filter_ms', 'native_mask_projection_ms', 'endpoint_sync_ms'))
                        row['accounting_error_ms'] = row['end_to_end_ms']-outer_sum
                        assert abs(row['accounting_error_ms']) < .01
                        if not layers:
                            del prediction
                        del masks, pixels, inputs, pixel_mask
                        return row

                    with torch.inference_mode():
                        for index in range(args.warmup):
                            infer(rgbs[index % len(rgbs)], samples[index % len(samples)]['image_id'])
                        for index in range(args.diagnostic_count):
                            result['diagnostic_frames'].append(infer(rgbs[index], samples[index]['image_id'], diagnostic=True))
                        torch.cuda.reset_peak_memory_stats()
                        for index, rgb in enumerate(rgbs):
                            row = infer(rgb, samples[index]['image_id'])
                            result['frames'].append(row)
                            status['completed_current_case'] = len(result['frames'])
                            write(out/'benchmark_status.json', status)
                            print(f"FAST_BENCH {case_name} {index+1}/8 {row['end_to_end_ms']:.2f}ms selected={row['selected_count']}", flush=True)
                    result.update(phase='complete', summary=latency_summary([row['end_to_end_ms'] for row in result['frames']]),
                                  box_ready_summary=latency_summary([row['box_ready_ms'] for row in result['frames']]),
                                  diagnostic_stage_mean_ms={key: statistics.mean(row[key] for row in result['diagnostic_frames'])
                                      for key in result['diagnostic_frames'][0] if key.endswith('_ms')} if result['diagnostic_frames'] else {},
                                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), gpu_snapshot=snapshot())
                print('FAST_BENCH_RESULT '+case_name+' '+json.dumps(result['summary']), flush=True)
            except Exception as error:
                result.update(phase='failed', error=str(error), traceback=traceback.format_exc())
                print('FAST_BENCH_FAIL '+case_name+' '+str(error), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            write(out/(case_name+'_timing.json'), result)
            write(out/'benchmark_status.json', status)
            (out/'benchmark_summary.md').write_text(markdown_report(status), encoding='utf-8')

        for name, loaded in models.items():
            state = loaded.state_dict(keep_vars=True)
            assert all(state[key]._version == version for key, version in versions[name].items())
            assert '_strong_fast_adapter_active' not in loaded.roi_heads.__dict__
            assert not any(key in loaded.roi_heads.__dict__ for key in ('_roi_predictions', '_decoded_boxes', '_final_nms', '_mask_predictions'))
            assert sha(paths[name+'_checkpoint']) == cfg[name+'_checkpoint_sha256']
        assert source_inventory() == inventory and sha(REPO/'deployment/strong_fast.py') == adapter_sha
        assert sha(__file__) == script_sha and sha(manifest_path) == status['sample_manifest_sha256']
        failed = any(result['phase'] != 'complete' for result in status['cases'].values())
        status.update(phase='complete_with_failures' if failed else 'complete',
                      source_before_after_identical=True, checkpoint_sha256_before_after_identical=True,
                      model_tensor_versions_unchanged=True, temporary_adapters_restored=True,
                      completed_utc=datetime.now(timezone.utc).isoformat(), gpu_after=snapshot())
        write(out/'benchmark_results.json', status)
        write(out/'benchmark_status.json', status)
        (out/'benchmark_summary.md').write_text(markdown_report(status), encoding='utf-8')
        print('FAST_BENCH_COMPLETE '+json.dumps({name: {'phase': result['phase'], 'summary': result.get('summary')} for name, result in status['cases'].items()}), flush=True)
        return 1 if failed else 0
    except BaseException as error:
        status.update(phase='failed', error=str(error), traceback=traceback.format_exc())
        write(out/'benchmark_status.json', status)
        (out/'benchmark_summary.md').write_text(markdown_report(status), encoding='utf-8')
        raise
    finally:
        models.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print('FAST_BENCH_GPU_RELEASED process ending', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
