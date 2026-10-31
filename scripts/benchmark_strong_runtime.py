"""Synchronized batch-one latency of the unchanged original strong models.

Starts with decoded CPU RGB. Excludes load, disk, GT, rendering and robot I/O.
Single-view endpoint includes native V/A binary masks; four-view endpoint is
the original box/score fusion, which does not fuse masks.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

from reproduce_strong_baseline import sha, write, source_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    assert args.rounds >= 1
    cfgpath, sampled = args.config.resolve(), args.sample_dir.resolve()
    cfg = json.loads(cfgpath.read_text(encoding='utf-8-sig'))
    root = (cfgpath.parent / cfg['path_base']).resolve()
    paths = {key: (root / value).resolve() for key, value in cfg['paths'].items()}
    manifest = json.loads((sampled / 'sample_manifest.json').read_text(encoding='utf-8-sig'))
    out = args.output.resolve()
    assert out.drive.upper() == 'F:'
    out.mkdir(parents=True, exist_ok=False)
    inventory, script_sha = source_inventory(), sha(__file__)
    started = time.perf_counter()
    status = {'phase': 'initializing', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'batch_size': 1, 'FP32': True, 'confidence': .75, 'mask_threshold': .5,
              'cpu_threads': cfg['cpu_threads'], 'short_side': cfg['short_side'], 'max_size': cfg['max_size'],
              'warmup_per_model': 10, 'single_view_rounds': args.rounds,
              'input_start': 'Decoded 1300x1300 CPU RGB; no camera or disk I/O in timer',
              'single_endpoint': 'CPU native visible/amodal binary crops for score >= .75',
              'four_view_endpoint': 'Original fused native boxes and scores; no fused masks',
              'excluded': ['model load/checksum', 'disk I/O', 'GT', 'plotting/display', 'camera capture', 'robot communication'],
              'timing': 'Primary: whole-frame synchronized wall clock. Separate diagnostic pass synchronizes every stage.',
              'source_sha256': inventory, 'script_sha256': script_sha,
              'manifest_sha256': sha(sampled / 'sample_manifest.json'), 'platform': platform.platform(),
              'models': {}, 'stages': {}}
    def gpu_snapshot():
        try:
            return subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu', '--format=csv,noheader'], text=True).strip()
        except Exception as e:
            return str(e)
    status['gpu_before'] = gpu_snapshot()
    write(out / 'benchmark_status.json', status)
    try:
        import numpy as np
        import torch
        from PIL import Image
        torch.set_num_threads(cfg['cpu_threads'])
        torch.set_num_interop_threads(1)
        import flip_tta
        from apple_seg.od_data import legacy_rgb_tensor
        from apple_seg.od_training import load_checkpoint
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.od_inference import _strict_restore
        from apple_seg.od_evaluate import _validate_prediction
        from apple_seg.od_roi import paste_roi_mask
        samples = manifest['samples']
        rgbs = []
        for sample in samples:
            with Image.open(sample['rgb_path']) as image:
                rgbs.append(image.convert('RGB'))
            assert rgbs[-1].size == (sample['width'], sample['height'])
        references = {name: {r['image_id']: r['pred'] for r in json.loads((sampled / name / 'original_predictions.json').read_text(encoding='utf-8'))}
                      for name in ('base', 'small')}
        fused_reference = {r['image_id']: r['pred'] for r in json.loads((sampled / 'fused_predictions.json').read_text(encoding='utf-8'))}
        status.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                      tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
        def sync():
            torch.cuda.synchronize()
            return time.perf_counter()
        def stats(values):
            a = np.asarray(values, dtype=float)
            return {'n': len(a), 'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
                    'p95_ms': float(np.percentile(a, 95)), 'min_ms': float(a.min()), 'max_ms': float(a.max())}
        def identity(pred, reference):
            differences = []
            for key in ('boxes', 'scores'):
                arr = pred[key].numpy() if isinstance(pred[key], torch.Tensor) else np.asarray(pred[key])
                ref = np.asarray(reference[key])
                if key == 'boxes':
                    arr, ref = arr.reshape(-1, 4), ref.reshape(-1, 4)
                assert arr.shape == ref.shape
                differences.append(float(np.max(np.abs(arr - ref), initial=0)))
            assert max(differences) <= 1e-4, f'Unchanged model output drifted: {differences}'
            return max(differences)
        def single(model, rgb, diagnostic=False):
            mark = sync if diagnostic else time.perf_counter
            t0 = sync()
            pixels = legacy_rgb_tensor(rgb, cfg['short_side'], cfg['max_size'])
            t1 = mark()
            input_hw, hw = tuple(pixels.shape[-2:]), (rgb.height, rgb.width)
            gpu_pixels = pixels[None].to('cuda')
            pixel_mask = torch.ones((1, *input_hw), dtype=torch.bool, device='cuda')
            t2 = mark()
            raw = model(gpu_pixels, pixel_mask=pixel_mask)
            t3 = mark()
            pred = _validate_prediction(raw, input_hw, hw, mask_chunk_size=8)
            t4 = mark()
            selected = [i for i, score in enumerate(pred['scores']) if score >= .75]
            masks = [paste_roi_mask(pred[branch][i], pred['boxes'][i], hw, mask_threshold=.5)
                     for i in selected for branch in ('visible', 'amodal')]
            t5 = sync()
            measurements = dict(preprocess_ms=(t1-t0)*1000, input_transfer_ms=(t2-t1)*1000,
                                model_call_ms=(t3-t2)*1000, output_decode_ms=(t4-t3)*1000,
                                native_masks_ms=(t5-t4)*1000, box_ready_ms=(t4-t0)*1000,
                                end_to_end_ms=(t5-t0)*1000, diagnostic=diagnostic,
                                candidates=len(pred['scores']), selected=len(selected))
            del raw, gpu_pixels, pixel_mask, pixels, masks
            return measurements, pred
        models = {}
        for name in ('base', 'small'):
            checkpoint = paths[name + '_checkpoint']
            assert sha(checkpoint) == cfg[name + '_checkpoint_sha256']
            print(f'BENCH loading original {name}', flush=True)
            payload = load_checkpoint(checkpoint)
            model = build_od_amodal_detector(weights=None, **payload['constructor_config'])
            restored = _strict_restore(model, payload['model'])
            assert model.constructor_config == payload['constructor_config']
            del payload
            model.requires_grad_(False).eval().to('cuda')
            models[name] = model
            status['models'][name] = {'checkpoint_sha256': cfg[name+'_checkpoint_sha256'], 'strict_restore': restored}
            rows, diagnostics = [], []
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                for i in range(10):
                    measurement, pred = single(model, rgbs[i % len(rgbs)])
                    del pred
                torch.cuda.reset_peak_memory_stats()
                for round_index in range(args.rounds + 1):
                    for i, rgb in enumerate(rgbs):
                        measurement, pred = single(model, rgb, diagnostic=round_index == 0)
                        measurement.update(image_id=samples[i]['image_id'], round=round_index+1,
                                           reference_max_abs_difference=identity(pred, references[name][samples[i]['image_id']]))
                        (diagnostics if round_index == 0 else rows).append(measurement)
                        del pred
                        if (i + 1) % 4 == 0:
                            status.update(phase='measuring', current=name, completed=len(rows))
                            write(out / 'benchmark_status.json', status)
                            print(f'BENCH {name} diagnostic={len(diagnostics)} measured={len(rows)}/{len(rgbs)*args.rounds}', flush=True)
            stage = {'frames': rows, 'diagnostic_frames': diagnostics,
                     'summary': {key: stats([r[key] for r in rows]) for key in ('box_ready_ms', 'end_to_end_ms')},
                     'diagnostic_summary': {key: stats([r[key] for r in diagnostics]) for key in diagnostics[0] if key.endswith('_ms')},
                     'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'gpu_snapshot': gpu_snapshot(),
                     'resident_models': [name]}
            stage['effective_fps'] = 1000 / stage['summary']['end_to_end_ms']['mean_ms']
            stage['box_ready_fps'] = 1000 / stage['summary']['box_ready_ms']['mean_ms']
            stage['model_call_fps'] = 1000 / stage['diagnostic_summary']['model_call_ms']['mean_ms']
            status['stages'][name] = stage
            write(out / (name + '_timing.json'), stage)
            write(out / 'benchmark_status.json', status)
            print(f'BENCH_RESULT {name} ' + json.dumps(stage['summary']['end_to_end_ms']) + f' FPS={stage["effective_fps"]:.3f}', flush=True)
            model.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
        # Both weights remain resident for serial four-view measurement.
        for model in models.values():
            model.to('cuda')
        def four(rgb, image_id, diagnostic=False):
            mark = sync if diagnostic else time.perf_counter
            t0 = sync()
            pixels = legacy_rgb_tensor(rgb, cfg['short_side'], cfg['max_size'])
            t1 = mark()
            input_hw, hw = tuple(pixels.shape[-2:]), (rgb.height, rgb.width)
            gpu_pixels = pixels[None].to('cuda')
            flipped = gpu_pixels.flip(-1).contiguous()
            pixel_mask = torch.ones((1, *input_hw), dtype=torch.bool, device='cuda')
            t2 = mark()
            layers, model_seconds, decode_seconds = [], 0., 0.
            for name in ('base', 'small'):
                for view, tensor in (('original', gpu_pixels), ('hflip', flipped)):
                    before = mark()
                    raw = models[name](tensor, pixel_mask=pixel_mask)
                    after_model = mark()
                    pred = _validate_prediction(raw, input_hw, hw, mask_chunk_size=8)
                    boxes = pred['boxes']
                    if view == 'hflip':
                        old = boxes.clone()
                        boxes[:, 0], boxes[:, 2] = hw[1] - old[:, 2], hw[1] - old[:, 0]
                    layers.append([{'image_id': image_id, 'gt': {'boxes': []},
                                    'pred': {'boxes': boxes.tolist(), 'scores': pred['scores']}}])
                    after_decode = mark()
                    model_seconds += after_model-before
                    decode_seconds += after_decode-after_model
                    del raw, pred, boxes
            t3 = mark()
            fused = flip_tta.fuse_records(layers, **cfg['fusion'])[0]['pred']
            t4 = sync()
            result = dict(image_id=image_id, preprocess_ms=(t1-t0)*1000,
                          input_transfer_and_flip_ms=(t2-t1)*1000, model_call_ms=model_seconds*1000,
                          output_decode_ms=decode_seconds*1000, four_views_ms=(t3-t2)*1000,
                          fusion_ms=(t4-t3)*1000, end_to_end_ms=(t4-t0)*1000, diagnostic=diagnostic,
                          selected=sum(s >= .75 for s in fused['scores']))
            return result, fused
        rows, diagnostics = [], []
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            # Separate warm-up after the deployment residency change.
            for i in range(2):
                four(rgbs[i], samples[i]['image_id'])
            torch.cuda.reset_peak_memory_stats()
            for diagnostic in (True, False):
                for i, rgb in enumerate(rgbs):
                    measurement, pred = four(rgb, samples[i]['image_id'], diagnostic=diagnostic)
                    measurement['reference_max_abs_difference'] = identity(pred, fused_reference[samples[i]['image_id']])
                    (diagnostics if diagnostic else rows).append(measurement)
                    status.update(phase='measuring', current='four_view', completed=len(rows))
                    write(out / 'benchmark_status.json', status)
                    print(f'BENCH four_view diagnostic={len(diagnostics)} measured={len(rows)}/{len(rgbs)}', flush=True)
        stage = {'frames': rows, 'diagnostic_frames': diagnostics,
                 'summary': {'end_to_end_ms': stats([r['end_to_end_ms'] for r in rows])},
                 'diagnostic_summary': {key: stats([r[key] for r in diagnostics]) for key in diagnostics[0] if key.endswith('_ms')},
                 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'gpu_snapshot': gpu_snapshot(),
                 'resident_models': ['base','small'], 'mask_fusion': False, 'four_view_warmup_frames': 2}
        stage['effective_fps'] = 1000 / stage['summary']['end_to_end_ms']['mean_ms']
        stage['box_ready_fps'] = stage['effective_fps']
        stage['model_call_fps'] = 1000 / stage['diagnostic_summary']['model_call_ms']['mean_ms']
        status['stages']['four_view'] = stage
        write(out / 'four_view_timing.json', stage)
        assert source_inventory() == inventory and sha(__file__) == script_sha
        status.update(phase='complete', gpu_after=gpu_snapshot(), source_before_after_identical=True,
                      elapsed_seconds=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'benchmark_status.json', status)
        print('BENCH_COMPLETE ' + json.dumps({name: {'fps': s['effective_fps'], 'end_to_end': s['summary']['end_to_end_ms']}
                                             for name, s in status['stages'].items()}), flush=True)
    except BaseException as error:
        status.update(phase='failed', error=str(error), traceback=traceback.format_exc())
        write(out / 'benchmark_status.json', status)
        raise


if __name__ == '__main__':
    main()
