"""Fresh original Base/Small four-view inference on a precommitted random VAL sample.

Visualization only: no training, calibration, model selection or baseline gate.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import gc
import json
import random
import sys
import time
import traceback

from reproduce_strong_baseline import REPO, RUNTIME, sha, write, source_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--count', type=int, default=8)
    args = parser.parse_args()
    cfgpath = args.config.resolve()
    cfg = json.loads(cfgpath.read_text(encoding='utf-8-sig'))
    root = (cfgpath.parent / cfg['path_base']).resolve()
    paths = {k: (root / v).resolve() for k, v in cfg['paths'].items()}
    out = args.output.resolve()
    assert out.drive.upper() == 'F:', 'Outputs must remain on F drive'
    assert 1 <= args.count <= 814
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inventory = source_inventory()
    script_sha = sha(__file__)

    # Choose solely by the frozen full-VAL inventory, before loading any weights.
    provenance = json.loads(paths['provenance'].read_text(encoding='utf-8-sig'))
    frozen_images = provenance['image_metadata']
    assert len(frozen_images) == 814
    indices = random.Random(args.seed).sample(range(len(frozen_images)), args.count)
    manifest = {
        'schema': 'strong_random_visualization_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed, 'population': 814, 'sampling': 'uniform_without_replacement',
        'confidence': 0.75, 'iou_threshold': 0.5,
        'model_label': '原 Base＋Small 四视图',
        'samples': [{
            'index': i, 'image_id': frozen_images[i]['image_id'],
            'rgb_path': str(paths['papple_root'] / 'images' / 'val' / frozen_images[i]['image_id']),
            'width': frozen_images[i]['width'], 'height': frozen_images[i]['height']
        } for i in indices],
        'purpose': 'Qualitative random sample only, not a replacement for full-VAL metrics',
        'config_sha256': sha(cfgpath), 'script_sha256': script_sha,
        'provenance_sha256': sha(paths['provenance']),
    }
    write(out / 'sample_manifest.json', manifest)
    manifest_sha = sha(out / 'sample_manifest.json')
    print('SAMPLE_COMMITTED ' + json.dumps(manifest['samples'], ensure_ascii=False), flush=True)
    status = {'phase': 'initializing', 'from_fresh_weight_inference': True,
              'sample_manifest_sha256': manifest_sha, 'models': {}, 'source_sha256': inventory}
    write(out / 'inference_status.json', status)
    try:
        import torch
        torch.set_num_threads(cfg['cpu_threads'])
        torch.set_num_interop_threads(1)
        import flip_tta
        from apple_seg.evaluate import load_papple_contract, _audit_target, _audit_rgb
        from apple_seg.data import PAppleDataset
        from apple_seg.od_data import ODInputDataset
        from apple_seg.od_training import load_checkpoint
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.od_inference import _strict_restore
        from apple_seg.od_evaluate import _validate_prediction
        contract = load_papple_contract(paths['provenance'])
        assert contract['images'] == frozen_images
        native = PAppleDataset(paths['papple_root'], 'val', 32, augment=False)
        dataset = ODInputDataset(native, short_sides=(cfg['short_side'],), max_size=cfg['max_size'])
        assert len(dataset) == 814
        assert native.annotation_sha256 == contract['provenance']['binding']['current_annotation_sha256']
        status.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), FP32=True,
                      cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        layers = []
        for model_name in ('base', 'small'):
            checkpoint = paths[model_name + '_checkpoint']
            digest = sha(checkpoint)
            assert digest == cfg[model_name + '_checkpoint_sha256']
            print(f'LOADING original {model_name}: {checkpoint}', flush=True)
            payload = load_checkpoint(checkpoint)
            model = build_od_amodal_detector(weights=None, **payload['constructor_config'])
            restored = _strict_restore(model, payload['model'])
            assert model.constructor_config == payload['constructor_config']
            model_meta = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                          'strict_restore': restored, 'weights_are_ema': payload.get('weights_are_ema', False)}
            del payload
            model.requires_grad_(False).eval().to('cuda')
            torch.cuda.reset_peak_memory_stats()
            status.update(phase='inference', current_model=model_name, completed_current_model=0)
            status['models'][model_name] = model_meta
            write(out / 'inference_status.json', status)
            records = {'original': [], 'hflip': []}
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                for n, sample in enumerate(manifest['samples'], 1):
                    frozen = contract['images'][sample['index']]
                    item = dataset[sample['index']]
                    target = item['target']
                    truth = _audit_target(target, frozen, contract['truth'][frozen['image_id']])
                    rgb_path = Path(target['source']['rgb_path']).resolve()
                    assert rgb_path == Path(sample['rgb_path'])
                    rgb_sha = _audit_rgb(rgb_path, frozen)
                    pixels = item['pixel_values']
                    input_hw, hw = tuple(pixels.shape[-2:]), tuple(target['original_hw'])
                    for view in ('original', 'hflip'):
                        view_pixels = pixels if view == 'original' else pixels.flip(-1).contiguous()
                        raw = model(view_pixels[None].to('cuda'),
                                    pixel_mask=torch.ones((1, *input_hw), device='cuda', dtype=torch.bool))
                        prediction = _validate_prediction(raw, input_hw, hw, mask_chunk_size=8)
                        boxes = prediction['boxes']
                        if view == 'hflip':
                            b = boxes.clone()
                            boxes[:, 0] = hw[1] - b[:, 2]
                            boxes[:, 2] = hw[1] - b[:, 0]
                            del b
                        records[view].append({'image_id': frozen['image_id'],
                                              'gt': {'boxes': truth['boxes']},
                                              'pred': {'boxes': boxes.tolist(), 'scores': prediction['scores']}})
                        del raw, prediction, view_pixels, boxes
                    _audit_target(target, frozen, contract['truth'][frozen['image_id']])
                    assert _audit_rgb(rgb_path, frozen) == rgb_sha
                    status.update(completed_current_model=n, elapsed_seconds=time.monotonic() - started,
                                  updated_utc=datetime.now(timezone.utc).isoformat())
                    write(out / 'inference_status.json', status)
                    print(f'INFER {model_name} {n}/{args.count} both views: {sample["image_id"]}', flush=True)
                    del item, target, truth, pixels
            for view in ('original', 'hflip'):
                write(out / model_name / (view + '_predictions.json'), records[view])
                layers.append(records[view])
            model_meta.update(phase='complete', images=args.count,
                              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            del model, records
            gc.collect()
            torch.cuda.empty_cache()
        assert source_inventory() == inventory
        assert sha(__file__) == script_sha and sha(out / 'sample_manifest.json') == manifest_sha
        fused = flip_tta.fuse_records(layers, **cfg['fusion'])
        assert [r['image_id'] for r in fused] == [r['image_id'] for r in manifest['samples']]
        fixed = flip_tta.metrics.evaluate_at_threshold(fused, 0.75, 0.5)
        write(out / 'fused_predictions.json', fused)
        status.update(phase='complete', images=args.count, fusion=cfg['fusion'], metrics=fixed,
                      elapsed_seconds=time.monotonic() - started, source_before_after_identical=True,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'inference_status.json', status)
        print('COMPLETE ' + json.dumps(fixed, ensure_ascii=False), flush=True)
    except BaseException as error:
        status.update(phase='failed', error=str(error), traceback=traceback.format_exc())
        write(out / 'inference_status.json', status)
        raise


if __name__ == '__main__':
    main()
