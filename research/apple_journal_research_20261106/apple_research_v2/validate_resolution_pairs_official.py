"""Separate official development AP, SHA-bound to an existing completed paired run.

The frozen paired script and partial official output are never modified. Each
640/960 validation is saved separately before the aggregate. No timing or robot
FPS claim is made. check-only imports no torch and writes nothing.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SIZES = (640, 960)
PAIRED_SOURCE_SHA = '317f72febb7ab154fa759fa3281ab9057321b6b78fb04faf6045a5d7a4c939bc'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def checked_path(value, exists=True):
    path = Path(value).resolve()
    require(path.drive.upper() == 'F:', f'F-drive path required: {path}')
    for part in path.parts:
        require(not {'test', 'tests', 'testing'} & set(re.split(r'[^a-z0-9]+', part.casefold())), f'Test path forbidden: {path}')
    require(not exists or path.is_file(), f'Missing input file: {path}')
    return path


def names_dict(value):
    result = dict(enumerate(value)) if isinstance(value, list) else {int(k): v for k, v in value.items()}
    require(sorted(result) == list(range(len(result))) and bool(result), 'Invalid class mapping')
    return result


def preflight(args):
    summary_path = checked_path(args.paired_summary)
    require(summary_path.is_relative_to(ROOT), 'Paired summary outside research root')
    require(sha(summary_path) == args.expected_summary_sha256.lower(), 'Paired summary SHA mismatch')
    paired = read(summary_path)
    require(paired.get('schema') == 'resolution_pairs_v1' and paired.get('phase') == 'complete', 'Completed paired run required')
    require(paired.get('source_inputs_unchanged') is True and paired.get('training_performed') is False, 'Invalid paired provenance')
    require(paired['protocol']['sizes'] == list(SIZES), 'Expected 640/960 paired run')
    preflight_path = checked_path(summary_path.parent / 'preflight.json')
    require(sha(preflight_path) == paired['preflight_sha256'], 'Paired preflight SHA mismatch')
    prior = read(preflight_path)
    require(prior['schema'] == 'resolution_pairs_preflight_v1' and prior['check_only'] is False, 'Invalid paired preflight')
    source_path = checked_path(Path(__file__).with_name('evaluate_resolution_pairs.py'))
    require(prior['source_sha256'] == PAIRED_SOURCE_SHA == sha(source_path), 'Frozen paired source differs')
    require(paired['kind'] == prior['kind'] and paired['images'] == prior['images'], 'Pair/preflight dataset mismatch')
    require(paired['kind'] in ('stage', 'ssda'), 'Unsupported paired task')
    names = names_dict(paired['class_names'])
    require(names == names_dict(prior['class_names']), 'Pair/preflight classes differ')
    weights, data = checked_path(paired['weights']), checked_path(prior['data'])
    require(weights == checked_path(prior['weights']), 'Pair/preflight weights differ')
    bindings = {str(checked_path(path)): digest for path, digest in prior['source_paths_sha256'].items()}
    require(bindings.get(str(weights)) == paired['weights_sha256'], 'Weight not bound to paired input hashes')
    require(str(data) in bindings, 'Dataset YAML absent from paired input hashes')
    bindings.update({str(summary_path): args.expected_summary_sha256.lower(), str(preflight_path): paired['preflight_sha256'], str(source_path): PAIRED_SOURCE_SHA, str(Path(__file__).resolve()): sha(__file__)})
    verify_bindings(bindings)
    output = checked_path(args.output, exists=False)
    require(output.is_relative_to(ROOT) and output != ROOT and not output.exists(), 'Fresh output under research root required')
    require(re.fullmatch(r'cuda:\d+', args.device), 'One CUDA device required; CPU tests call only pure helpers')
    support = [int(prior['class_support'].get(names[i], 0)) for i in names]
    require(sum(support) == prior['GT_instances'], 'Inconsistent paired GT support')
    return paired, prior, names, support, weights, data, output, bindings


def verify_bindings(bindings):
    for path, digest in bindings.items():
        require(sha(path) == digest, f'Bound input changed: {path}')


def validate_one(model, *, data, output, size, device, max_det, expected_images, expected_support):
    """Use actual callback validator arguments, never an assumed model.validator."""
    require(size in SIZES, 'Unsupported resolution')
    capture = {}

    def on_start(validator):
        require('validator' not in capture, 'Duplicate on_val_start callback')
        args = validator.args
        require(Path(args.data).resolve() == Path(data).resolve() and args.split == 'val', 'Validator dataset/split mismatch')
        require(args.imgsz == size and args.batch == 1 and args.rect is True, 'Validator geometry/batch differs')
        require(args.quantize in (None, 32), 'Validator unexpectedly uses non-FP32 inference')
        require(args.conf == .001 and args.iou == .7 and args.max_det == max_det, 'Validator thresholds differ')
        capture.update(validator=validator, actual_config={
            'imgsz': args.imgsz, 'batch': args.batch, 'effective_rect': args.rect,
            'conf': args.conf, 'requested_iou': args.iou, 'max_det': args.max_det,
            'requested_quantize': 32, 'effective_quantize': args.quantize,
            'precision': 'FP32', 'split': args.split, 'device': str(validator.device),
            'save_dir': str(validator.save_dir)})

    def on_end(validator):
        require(capture.get('validator') is validator, 'Validator callback identity mismatch')
        require('seen' not in capture, 'Duplicate on_val_end callback')
        require(int(validator.seen) == expected_images, 'Official validation image coverage differs')
        support = [int(v) for v in validator.metrics.nt_per_class]
        require(support == expected_support, 'Official validation GT class support differs')
        capture.update(seen=int(validator.seen), class_support=support,
                       end2end_head=bool(validator.end2end), ended=True)

    model.add_callback('on_val_start', on_start)
    model.add_callback('on_val_end', on_end)
    try:
        metrics = model.val(data=str(data), split='val', imgsz=size, batch=1,
            rect=True, device=device, quantize=32, conf=.001, iou=.7, max_det=max_det,
            plots=False, save_json=False, save_txt=False, cache=False,
            project=str(output / 'official_val'), name=str(size), exist_ok=False,
            workers=0, verbose=False)
    finally:
        for event, callback in (('on_val_start', on_start), ('on_val_end', on_end)):
            model.callbacks[event] = [item for item in model.callbacks[event] if item is not callback]
    require(capture.get('ended') is True, 'Validation callbacks did not complete')
    require(metrics is capture['validator'].metrics, 'Returned metrics differ from actual callback validator')
    values = {str(key): float(value) for key, value in metrics.results_dict.items()}
    require(values and all(math.isfinite(v) for v in values.values()), 'Nonfinite or empty validation metrics')
    return {'resolution': size, 'metrics': values, 'config': capture['actual_config'],
            'seen': capture['seen'], 'class_support': capture['class_support'],
            'end2end_head': capture['end2end_head'], 'callback_capture_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-summary', type=Path, required=True)
    parser.add_argument('--expected-summary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    paired, prior, names, support, weights, data, output, bindings = preflight(args)
    if args.check_only:
        print(json.dumps({'phase': 'preflight_passed', 'kind': paired['kind'], 'images': paired['images'], 'class_support': support, 'GPU_used': False, 'files_written': False, 'torch_imported': 'torch' in sys.modules}, ensure_ascii=False))
        return
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'resolution_pairs_official_validation_v1', 'phase': 'running',
        'paired_summary': str(args.paired_summary.resolve()), 'paired_summary_sha256': args.expected_summary_sha256.lower(),
        'paired_source_sha256': PAIRED_SOURCE_SHA, 'kind': paired['kind'], 'weights': str(weights),
        'weights_sha256': paired['weights_sha256'], 'class_names': names, 'semantics': paired['semantics'],
        'scope': 'Separate official development AP; not fixed-.25 paired matching and not a timing comparison',
        'precision_recall_scope': 'Official confidence-sweep metrics; P/R are not the fixed-.25 paired counts',
        'independent_fruit_test': False, 'test_read': False, 'training_performed': False,
        'robot_FPS': None, 'resolutions': {}, 'source_paths_sha256': bindings}
    try:
        from evaluate_resolution_pairs import configure_runtime
        configure_runtime(output)
        import torch
        import ultralytics
        from ultralytics import YOLO
        require(ultralytics.__version__ == '8.4.115', 'Audited Ultralytics 8.4.115 required')
        package = Path(ultralytics.__file__).resolve().parent
        for relative in ('engine/model.py', 'engine/validator.py', 'models/yolo/detect/val.py', 'models/yolo/segment/val.py'):
            source = package / relative
            bindings[str(source)] = sha(source)
        require(torch.cuda.is_available(), 'CUDA unavailable')
        torch.cuda.set_device(torch.device(args.device))
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('highest')
        model = YOLO(str(weights))
        require(model.task == ('segment' if paired['kind'] == 'stage' else 'detect'), 'Checkpoint task differs')
        require(names_dict(model.names) == names, 'Checkpoint classes differ')
        model.model.float().eval()
        report['environment'] = {'torch': str(torch.__version__), 'ultralytics': ultralytics.__version__, 'device': args.device}
        save(output / 'preflight.json', report)
        for size in SIZES:
            result = validate_one(model, data=data, output=output, size=size, device=args.device,
                max_det=paired['protocol']['max_det'], expected_images=paired['images'], expected_support=support)
            verify_bindings(bindings)
            result['weights_sha256'] = paired['weights_sha256']
            result['paired_summary_sha256'] = args.expected_summary_sha256.lower()
            save(output / f'official_{size}.json', result)
            report['resolutions'][str(size)] = result
            print(json.dumps({'phase': 'resolution_complete', 'resolution': size, 'report': str(output / f'official_{size}.json')}, ensure_ascii=False), flush=True)
        report.update(phase='complete', source_inputs_unchanged=True, completed_utc=datetime.now(timezone.utc).isoformat())
        save(output / 'official_validation.json', report)
        print(json.dumps({'phase': 'complete', 'report': str(output / 'official_validation.json')}, ensure_ascii=False))
    except BaseException as error:
        report.update(phase='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        save(output / 'failure.json', report)
        raise


if __name__ == '__main__':
    main()