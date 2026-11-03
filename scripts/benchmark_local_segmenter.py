"""Bounded COCO-pretrained segmentation readiness/latency check; not apple accuracy validation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
EXPECTED_SHA = '3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=REPO / 'weights/local_rgbd/yolo26s-seg.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'F:' or Path(sys.executable).drive.upper() != 'F:':
        raise ValueError('Environment and outputs must be on F:')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'benchmark_results.json').exists():
        raise FileExistsError('Completed benchmark exists; choose a new output directory.')
    weight_sha = sha(args.weights)
    if weight_sha != EXPECTED_SHA:
        raise ValueError('Official checkpoint hash mismatch')
    status = {
        'schema': 'local_rgbd_segmentation_readiness_v1', 'phase': 'initializing',
        'created_utc': now(), 'pid': os.getpid(), 'training': False,
        'checkpoint_role': 'official COCO pretrained, not adapted to local apples',
        'apple_accuracy_validated': False, 'D405_tracking_validated': False,
        'full_segmentation_plus_geometry_20fps_validated': False,
        'weights': str(args.weights.resolve()), 'weights_sha256': weight_sha,
        'script_sha256': sha(__file__), 'warmup': args.warmup, 'rounds': args.rounds,
        'timing_scope': 'Predecoded original BGR -> letterbox/preprocess/H2D/model/native visible masks/boxes + CPU delivery. Excludes disk/model loading/drawing/depth/geometry/association/capture age.',
        'input_scope': 'Entire original field of view, letterboxed at stated network size; NOT a 512 centre crop or D405 video.',
        'cases': [], 'samples': [],
    }
    write(output / 'benchmark_status.json', status)
    lock = output.parent / 'gpu_benchmark.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no silent CPU fallback')
        status['runtime'] = {
            'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'ultralytics': importlib.metadata.version('ultralytics'),
            'numpy': np.__version__, 'opencv': cv2.__version__,
            'gpu': torch.cuda.get_device_name(0), 'cpu_threads': 4,
            'cudnn_benchmark': False, 'precision': 'FP16',
            'settings_dir': os.environ.get('YOLO_CONFIG_DIR'),
        }
        config_path = REPO / 'configs/strong_baseline_4060.json'
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        data_root = (config_path.parent / config['path_base'] / config['paths']['papple_root']).resolve()
        manifest = REPO / 'configs/premask_audit_samples_20261102.json'
        samples = json.loads(manifest.read_text(encoding='utf-8-sig'))['samples'][:8]
        status['sample_manifest_sha256'] = sha(manifest)
        images = []
        for sample in samples:
            path = data_root / 'images/val' / sample['image_id']
            pixels = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError(f'Unreadable image: {path}')
            images.append(pixels)
            status['samples'].append({'image_id': sample['image_id'], 'path': str(path),
                                      'sha256': sha(path), 'native_hw': list(pixels.shape[:2])})
        model = YOLO(str(args.weights), task='segment')
        apple_classes = [int(k) for k, v in model.names.items() if v == 'apple']
        if len(apple_classes) != 1:
            raise ValueError('Expected one COCO apple class')
        cases = [
            ('full512_apple025', 512, 0.25, apple_classes),
            ('full640_apple025', 640, 0.25, apple_classes),
            ('full640_all005_stress', 640, 0.05, None),
        ]
        for name, size, confidence, classes in cases:
            status.update(phase='measuring', current_case=name, updated_utc=now())
            case = {'name': name, 'imgsz': size, 'confidence': confidence, 'classes': classes,
                    'retina_masks': True, 'max_det': 64, 'rect': False, 'frames': []}
            status['cases'].append(case)
            write(output / 'benchmark_status.json', status)

            def infer(index):
                torch.cuda.synchronize()
                started = time.perf_counter()
                results = model.predict(source=images[index], imgsz=size, conf=confidence,
                    classes=classes, max_det=64, device=0, quantize=16, verbose=False,
                    save=False, retina_masks=True, rect=False)
                result = results[0]
                boxes = result.boxes.data.detach().cpu().numpy()
                masks = (result.masks.data.detach().cpu().numpy() if result.masks is not None
                         else np.empty((0, *images[index].shape[:2]), dtype=np.uint8))
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                if masks.shape[0] != len(boxes) or (len(masks) and tuple(masks.shape[1:]) != tuple(images[index].shape[:2])):
                    raise ValueError('Native instance masks/boxes mismatch')
                if not np.isfinite(boxes).all():
                    raise ValueError('Nonfinite boxes/scores')
                row = {'image_id': samples[index]['image_id'], 'end_to_end_ms': elapsed,
                       'instances': len(boxes), 'mask_hw': list(masks.shape[1:]),
                       'ultralytics_speed_ms': {k: float(v) for k, v in result.speed.items()}}
                return row, boxes, masks

            with torch.inference_mode():
                case['cold_frame'] = infer(0)[0]
                actual_dtype = next(model.model.parameters()).dtype
                actual_device = next(model.model.parameters()).device
                case['precision_receipt'] = {
                    'backend_fp16': bool(model.predictor.model.fp16),
                    'parameter_dtype': str(actual_dtype), 'device': str(actual_device),
                    'predictor_quantize': model.predictor.args.quantize,
                }
                if actual_dtype != torch.float16 or actual_device.type != 'cuda' or not model.predictor.model.fp16:
                    raise RuntimeError('Requested FP16 CUDA was not actually used')
                for i in range(args.warmup):
                    infer(i % len(images))
                torch.cuda.reset_peak_memory_stats()
                for repeat in range(args.rounds):
                    for index in range(len(images)):
                        row, boxes, masks = infer(index)
                        row['round'] = repeat + 1
                        case['frames'].append(row)
                        if repeat == 0 and name == 'full640_apple025':
                            prediction_dir = output / 'coco_predictions'
                            prediction_dir.mkdir(exist_ok=True)
                            np.savez_compressed(prediction_dir / (Path(samples[index]['image_id']).stem + '.npz'),
                                                boxes=boxes, visible=masks.astype(np.uint8))
                    write(output / 'benchmark_status.json', status)
            times = np.array([row['end_to_end_ms'] for row in case['frames']], dtype=float)
            counts = np.array([row['instances'] for row in case['frames']])
            case['summary'] = {
                'frames': len(times), 'mean_ms': float(times.mean()),
                'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
                'max_ms': float(times.max()), 'fps_from_mean': float(1000 / times.mean()),
                'instances_mean': float(counts.mean()), 'instances_min': int(counts.min()),
                'instances_max': int(counts.max()), 'frames_with_no_instances': int((counts == 0).sum()),
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                'network_path_p95_within_30ms': bool(np.percentile(times, 95) <= 30),
                'network_path_p95_within_50ms': bool(np.percentile(times, 95) <= 50),
            }
            case['phase'] = 'complete'
            write(output / (name + '.json'), case)
            print(json.dumps({'completed': name, **case['summary']}, ensure_ascii=False), flush=True)
        if sha(args.weights) != weight_sha:
            raise RuntimeError('Checkpoint was modified')
        status.update(phase='complete', updated_utc=now(), weights_unchanged=True)
        write(output / 'benchmark_results.json', status)
        write(output / 'benchmark_status.json', status)
    except BaseException:
        status.update(phase='failed', updated_utc=now(), error=traceback.format_exc())
        write(output / 'benchmark_status.json', status)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
