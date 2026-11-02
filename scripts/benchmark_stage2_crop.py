"""Bounded original-Small 512 centre-crop latency simulation; no tracking claim."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from evaluate_fast_deployment import REPO, now, sha, write
from benchmark_premask import latency_summary, center_crop_box

SMALL_SHA = "e0c1f659c2509d7a01d617e12cb83c673768b440ea5b4b96cb4f763e4207e7e3"


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/strong_baseline_4060.json')
    parser.add_argument('--preprocess-device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--channels-last', action='store_true')
    parser.add_argument('--cudnn-benchmark', action='store_true')
    parser.add_argument('--roi-align-precision', choices=('fp32', 'fp16'), default='fp32')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.warmup not in (3, 4, 5) or args.rounds not in (2, 3, 4, 5):
        raise ValueError('Use approved 3-5 warmup, 2-5 rounds protocol')
    out = args.output.resolve()
    if out.drive.upper() != 'F:':
        raise ValueError('All outputs must stay on F:')
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    root = (args.config.resolve().parent / config['path_base']).resolve()
    paths = {key: (root / value).resolve() for key, value in config['paths'].items()}
    sample_file = REPO / 'configs/premask_audit_samples_20261102.json'
    samples = json.loads(sample_file.read_text(encoding='utf-8-sig'))['samples'][:8]
    sources = [Path(__file__), args.config.resolve(), sample_file,
               REPO / 'deployment/stage2.py', REPO / 'deployment/strong_fast.py',
               REPO / 'deployment/pre_mask.py', REPO / 'deployment/candidate_policy.py',
               REPO / 'strong_runtime/src/apple_seg/od_data.py',
               REPO / 'strong_runtime/src/apple_seg/od_model.py',
               REPO / 'strong_runtime/src/apple_seg/od_roi.py',
               REPO / 'strong_runtime/src/apple_seg/od_evaluate.py']
    receipts = {str(p): sha(p) for p in sources}
    status = {'schema': 'stage2_original_small_center_crop_speed_v1', 'phase': 'initializing',
              'created_utc': now(), 'training': False, 'full_VAL': False,
              'D405_tracking_accuracy_claimed': False, 'model': 'original_small',
              'crop': 'native RGB centre 512x512; no full-frame resize',
              'precision': 'fp16 autocast, FP32 original parameter values',
              'roi_align_precision': args.roi_align_precision,
              'preprocess_device': args.preprocess_device, 'channels_last': args.channels_last,
              'cudnn_benchmark': args.cudnn_benchmark, 'warmup': args.warmup,
              'rounds': args.rounds, 'source_frames': 8 * args.rounds,
              'pre_nms_threshold': .70, 'final_threshold': .70,
              'timing_scope': 'Predecoded RGB -> timed crop, preprocess, H2D, inference, native output validation and final V/A projection; excludes disk, loading, GT and drawing',
              'source_receipts': receipts, 'samples': [], 'cases': {}}
    write(out / 'benchmark_status.json', status)
    model = None
    try:
        sys.path[:0] = [str(REPO), str(REPO / 'strong_runtime/src')]
        import torch
        from PIL import Image
        from apple_seg.od_training import load_checkpoint
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.od_inference import _strict_restore
        from apple_seg.od_evaluate import _validate_prediction
        from apple_seg.od_roi import paste_roi_mask
        from deployment.stage2 import Stage2DeploymentAdapter, prepare_pixels
        torch.set_num_threads(config['cpu_threads'])
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        status['runtime'] = {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                             'cudnn_benchmark': torch.backends.cudnn.benchmark,
                             'cudnn_deterministic': torch.backends.cudnn.deterministic,
                             'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                             'matmul_tf32': False, 'cudnn_tf32': True}
        rgbs = []
        for sample in samples:
            path = paths['papple_root'] / 'images/val' / sample['image_id']
            receipts[str(path)] = sha(path)
            with Image.open(path) as image:
                rgb = image.convert('RGB')
            center_crop_box(rgb.width, rgb.height, 512)
            rgbs.append(rgb)
            status['samples'].append({'image_id': sample['image_id'], 'RGB_path': str(path),
                                      'native_hw': [rgb.height, rgb.width], 'sha256': sha(path)})
        checkpoint = paths['small_checkpoint']
        if checkpoint.drive.upper() != 'F:' or sha(checkpoint) != SMALL_SHA:
            raise ValueError('Pinned original Small checkpoint differs')
        receipts[str(checkpoint)] = SMALL_SHA
        payload = load_checkpoint(checkpoint)
        model = build_od_amodal_detector(weights=None, **payload['constructor_config'])
        status['strict_restore'] = _strict_restore(model, payload['model'])
        del payload
        model.requires_grad_(False).eval()
        original_digest = state_digest(model)
        model.to('cuda')
        if args.channels_last:
            model.to(memory_format=torch.channels_last)
        if state_digest(model) != original_digest:
            raise RuntimeError('Parameter values changed during device/layout conversion')
        for name, policy in [('crop512_nms080_k64', {'score_threshold': None, 'box_nms_iou': .8, 'topk': 64}),
                             ('crop512_nms075_k16', {'score_threshold': None, 'box_nms_iou': .75, 'topk': 16})]:
            case = {'phase': 'running', 'candidate_policy': policy, 'frames': []}
            status['cases'][name] = case
            status.update(phase='measuring', current_case=name)
            write(out / 'benchmark_status.json', status)
            with Stage2DeploymentAdapter(model, candidate_policy=policy, pre_nms_threshold=.70,
                    roi_chunk_size=64, roi_precision='autocast', roi_align_precision=args.roi_align_precision,
                    capture_candidates=False, profile_stages=False, profile_pools=False) as adapter, torch.inference_mode():
                def infer(index):
                    rgb = rgbs[index]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    crop_box = center_crop_box(rgb.width, rgb.height, 512)
                    crop = rgb.crop(crop_box)
                    pixels = prepare_pixels(crop, 512, 512, 'cuda',
                        preprocess_device=args.preprocess_device, channels_last=args.channels_last)
                    if tuple(pixels.shape) != (1, 3, 512, 512) or crop.size != (512, 512):
                        raise RuntimeError('Crop geometry changed')
                    pixel_mask = torch.ones((1, 512, 512), dtype=torch.bool, device='cuda')
                    with torch.autocast('cuda', dtype=torch.float16):
                        raw = adapter.predict_views(pixels, pixel_mask)
                    prediction = _validate_prediction(raw, (512, 512), (512, 512), mask_chunk_size=8)
                    selected = [i for i, score in enumerate(prediction['scores']) if score >= .70]
                    masks = [paste_roi_mask(prediction[branch][i], prediction['boxes'][i], (512, 512), mask_threshold=.5)
                             for i in selected for branch in ('visible', 'amodal')]
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    return {'image_id': samples[index]['image_id'], 'crop_box_in_original': list(crop_box),
                            'input_hw': [512, 512], 'native_output_hw': [512, 512],
                            'end_to_end_ms': elapsed, 'selected_count': len(selected),
                            'native_VA_mask_crops': len(masks), 'adapter_stats': deepcopy(adapter.last_stats),
                            'endpoint_boxes': prediction['boxes'][selected].tolist(),
                            'endpoint_scores': [prediction['scores'][i] for i in selected]}
                case['first_frame'] = infer(0)
                for i in range(args.warmup):
                    infer(i % 8)
                torch.cuda.reset_peak_memory_stats()
                for repeat in range(args.rounds):
                    for i in range(8):
                        row = infer(i)
                        row['round'] = repeat + 1
                        case['frames'].append(row)
                    write(out / 'benchmark_status.json', status)
            if state_digest(model) != original_digest:
                raise RuntimeError('Original model parameter values changed')
            case.update(phase='complete', summary=latency_summary([r['end_to_end_ms'] for r in case['frames']]),
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            write(out / (name + '.json'), case)
            write(out / 'benchmark_status.json', status)
            print('STAGE2_CROP_COMPLETE ' + name + ' ' + json.dumps(case['summary']), flush=True)
        for path, digest in receipts.items():
            if sha(path) != digest:
                raise RuntimeError('Input/source/weight changed: ' + path)
        status.update(phase='complete', completed_utc=now(), source_before_after_identical=True,
                      original_model_values_unchanged=True, checkpoint_sha256=SMALL_SHA)
        write(out / 'benchmark_results.json', status)
        write(out / 'benchmark_status.json', status)
    except BaseException as error:
        status.update(phase='failed', error=str(error), traceback=traceback.format_exc(), updated_utc=now())
        write(out / 'benchmark_status.json', status)
        raise
    finally:
        del model
        gc.collect()


if __name__ == '__main__':
    main()
