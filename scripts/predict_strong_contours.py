"""Export native visible/amodal masks from the original strong model pair.

Uses an already fixed sample manifest. Each model runs its original image view;
no mask fusion, GT-guided selection, retraining or geometric shape fitting.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
import traceback

from reproduce_strong_baseline import REPO, sha, write, source_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfgpath = args.config.resolve()
    cfg = json.loads(cfgpath.read_text(encoding='utf-8-sig'))
    root = (cfgpath.parent / cfg['path_base']).resolve()
    paths = {key: (root / value).resolve() for key, value in cfg['paths'].items()}
    source_manifest = args.manifest.resolve()
    manifest = json.loads(source_manifest.read_text(encoding='utf-8-sig'))
    samples = manifest['samples']
    assert samples and len({s['image_id'] for s in samples}) == len(samples)
    out = args.output.resolve()
    assert out.drive.upper() == 'F:'
    out.mkdir(parents=True, exist_ok=False)
    manifest.update(model_label='原 Base／Small 原图单视图分割',
                    parent_manifest_path=str(source_manifest), parent_manifest_sha256=sha(source_manifest),
                    mask_threshold=0.5, confidence=0.75, resampled=False)
    write(out / 'sample_manifest.json', manifest)
    inventory = source_inventory()
    script_sha, manifest_sha = sha(__file__), sha(out / 'sample_manifest.json')
    started = time.monotonic()
    status = {'phase': 'initializing', 'from_fresh_weight_inference': True,
              'mask_fusion': False, 'confidence': .75, 'mask_threshold': .5,
              'models': {}, 'source_sha256': inventory, 'script_sha256': script_sha,
              'sample_manifest_sha256': manifest_sha}
    data = {'confidence': .75, 'mask_threshold': .5, 'models': {},
            'view': 'original', 'mask_fusion': False, 'GT_used_for_prediction_or_selection': False}
    write(out / 'inference_status.json', status)
    try:
        import torch
        import numpy as np
        from PIL import Image
        torch.set_num_threads(cfg['cpu_threads'])
        torch.set_num_interop_threads(1)
        from apple_seg.evaluate import load_papple_contract, _audit_target, _audit_rgb
        from apple_seg.data import PAppleDataset
        from apple_seg.od_data import ODInputDataset
        from apple_seg.od_training import load_checkpoint
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.od_inference import _strict_restore
        from apple_seg.od_evaluate import _validate_prediction
        from apple_seg.od_roi import paste_roi_mask
        contract = load_papple_contract(paths['provenance'])
        native = PAppleDataset(paths['papple_root'], 'val', 32, augment=False)
        dataset = ODInputDataset(native, short_sides=(cfg['short_side'],), max_size=cfg['max_size'])
        assert len(dataset) == 814
        assert native.annotation_sha256 == contract['provenance']['binding']['current_annotation_sha256']
        for sample in samples:
            frozen = contract['images'][sample['index']]
            assert frozen['image_id'] == sample['image_id']
            assert [frozen['width'], frozen['height']] == [sample['width'], sample['height']]
        status.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, FP32=True)
        for name in ('base', 'small'):
            checkpoint = paths[name + '_checkpoint']
            digest = sha(checkpoint)
            assert digest == cfg[name + '_checkpoint_sha256']
            print(f'CONTOURS loading original {name}', flush=True)
            payload = load_checkpoint(checkpoint)
            model = build_od_amodal_detector(weights=None, **payload['constructor_config'])
            restored = _strict_restore(model, payload['model'])
            assert model.constructor_config == payload['constructor_config']
            status['models'][name] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                                     'strict_restore': restored,
                                     'weights_are_ema': payload.get('weights_are_ema', False)}
            del payload
            model.requires_grad_(False).eval().to('cuda')
            torch.cuda.reset_peak_memory_stats()
            records = []
            data['models'][name] = records
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                for position, sample in enumerate(samples, 1):
                    frozen = contract['images'][sample['index']]
                    item = dataset[sample['index']]
                    target, pixels = item['target'], item['pixel_values']
                    _audit_target(target, frozen, contract['truth'][sample['image_id']])
                    rgb_path = Path(target['source']['rgb_path']).resolve()
                    assert rgb_path == Path(sample['rgb_path']).resolve()
                    rgb_sha = _audit_rgb(rgb_path, frozen)
                    input_hw, hw = tuple(pixels.shape[-2:]), tuple(target['original_hw'])
                    raw = model(pixels[None].to('cuda'),
                                pixel_mask=torch.ones((1, *input_hw), device='cuda', dtype=torch.bool))
                    pred = _validate_prediction(raw, input_hw, hw, mask_chunk_size=8)
                    del raw
                    # Retain every raw ROI probability; display selection is score-only.
                    directory = out / 'native_masks' / name / f'{position:02d}_{Path(sample["image_id"]).stem}'
                    directory.mkdir(parents=True)
                    archive = directory / 'compact_roi_probabilities.npz'
                    np.savez_compressed(archive, boxes=pred['boxes'].numpy(), scores=np.asarray(pred['scores']),
                                        visible=pred['visible'].numpy(), amodal=pred['amodal'].numpy())
                    row = {'image_id': sample['image_id'], 'rgb_path': str(rgb_path),
                           'width': hw[1], 'height': hw[0], 'detections': [],
                           'compact_roi_probabilities': archive.relative_to(out).as_posix(),
                           'compact_sha256': sha(archive), 'raw_candidate_count': len(pred['scores'])}
                    selected = sorted((i for i, score in enumerate(pred['scores']) if score >= .75),
                                      key=lambda i: (-pred['scores'][i], i))
                    for index in selected:
                        box = pred['boxes'][index]
                        detection = {'prediction_index': index, 'score': pred['scores'][index], 'box': box.tolist()}
                        for branch in ('visible', 'amodal'):
                            crop, window = paste_roi_mask(pred[branch][index], box, hw, mask_threshold=.5)
                            mask_path = directory / f'instance_{index:04d}_{branch}.png'
                            if crop.numel():
                                Image.fromarray(crop.numpy().astype(np.uint8) * 255).save(mask_path)
                            detection[branch + '_mask'] = {
                                'path': mask_path.relative_to(out).as_posix() if crop.numel() else None,
                                'window_xyxy': list(window), 'foreground_pixels': int(crop.sum()),
                                'sha256': sha(mask_path) if crop.numel() else None}
                        row['detections'].append(detection)
                    records.append(row)
                    assert _audit_rgb(rgb_path, frozen) == rgb_sha
                    _audit_target(target, frozen, contract['truth'][sample['image_id']])
                    status.update(phase='inference', current_model=name, completed_current_model=position,
                                  elapsed_seconds=time.monotonic() - started,
                                  updated_utc=datetime.now(timezone.utc).isoformat())
                    write(out / 'contour_predictions.json', data)
                    write(out / 'inference_status.json', status)
                    print(f'CONTOURS {name} {position}/{len(samples)} {sample["image_id"]} detections={len(selected)}', flush=True)
                    del pred, pixels, item, target
            status['models'][name].update(phase='complete', images=len(samples),
                                         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                         peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            del model
            gc.collect()
            torch.cuda.empty_cache()
        assert source_inventory() == inventory and sha(__file__) == script_sha
        assert sha(source_manifest) == manifest['parent_manifest_sha256']
        assert sha(out / 'sample_manifest.json') == manifest_sha
        status.update(phase='complete', images=len(samples), source_before_after_identical=True,
                      elapsed_seconds=time.monotonic() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'inference_status.json', status)
        print('CONTOURS_COMPLETE', flush=True)
    except BaseException as error:
        status.update(phase='failed', error=str(error), traceback=traceback.format_exc())
        write(out / 'inference_status.json', status)
        raise


if __name__ == '__main__':
    main()
