"""Does the published evaluation resolution cost us the occluded apples?

Every published number in this project - including the 89.13/89.16 parent - was
measured with ``validation_short_side = 800`` on 1300x1300 source frames. A heavily
occluded apple is a small visible patch, so 800 px throws away 40% of the linear
resolution the annotation actually has.

This runs one checkpoint over the same deterministic VAL subset at several
resolutions and reports precision/recall plus heavy-occlusion recall. It is a
diagnostic on a subset, not a publishable full-VAL point.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tmp'))

from apple_seg import od_training as training  # noqa: E402
from apple_seg.data import PAppleDataset  # noqa: E402
from apple_seg.od_data import ODInputDataset  # noqa: E402
from apple_seg.od_model import build_od_amodal_detector  # noqa: E402
from stage2_crop import iou_matrix, mask_boxes  # noqa: E402


def load_model(config_path, checkpoint_path, device):
    """Rebuild from the checkpoint's own constructor config, so the probe can
    never disagree with the weights it is measuring."""
    config = training.load_od_config(Path(config_path))
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    options = dict(payload['constructor_config'])
    options.pop('classification_highres', None) if 'classification_highres' in options and not any(
        name.startswith('roi_heads.highres') for name in payload['model']) else None
    model = build_od_amodal_detector(weights=None, **options)
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()
    return model, config


def run(model, dataset, indices, device, short_side, max_size, threshold):
    dataset.short_sides = (short_side,)
    dataset.max_size = max_size
    model.eval()
    tp = fp = total = 0
    hits = {'heavy': 0, 'other': 0}
    totals = {'heavy': 0, 'other': 0}
    for index in indices:
        item = dataset[index]
        pixels = item['pixel_values'].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(pixels)[0]
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        target = item['target']
        # The model returns input-space boxes; the published protocol rescales
        # them to the original frame before matching.
        original_h, original_w = target['original_hw']
        input_h, input_w = output['image_hw']
        boxes = boxes * np.array([original_w / input_w, original_h / input_h] * 2)
        gt = mask_boxes(target['amodal_masks']).numpy()
        visible = target['visible_masks']
        amodal = target['amodal_masks']
        ratios = ((visible.sum(dim=(-1, -2)) / amodal.sum(dim=(-1, -2)).clamp_min(1)).clamp(0, 1).numpy()
                  if amodal is not None else np.ones(len(gt)))
        keep = scores >= threshold
        order = np.argsort(-scores[keep])
        matrix = iou_matrix(boxes[keep][order], gt)
        used, matched_gt = set(), set()
        for row in range(matrix.shape[0]):
            candidates = [(matrix[row, j], j) for j in range(matrix.shape[1])
                          if j not in used and matrix[row, j] >= .5]
            if candidates:
                best = max(candidates)[1]
                used.add(best)
                matched_gt.add(best)
        tp += len(matched_gt)
        fp += matrix.shape[0] - len(matched_gt)
        total += len(gt)
        for j in range(len(gt)):
            label = 'heavy' if float(ratios[j]) < .4 else 'other'
            totals[label] += 1
            hits[label] += int(j in matched_gt)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total, 1)
    return dict(short_side=short_side, max_size=max_size, threshold=threshold, tp=tp, fp=fp,
                fn=total - tp, precision=precision, recall=recall, score=min(precision, recall),
                heavy_recall=hits['heavy'] / max(totals['heavy'], 1),
                other_recall=hits['other'] / max(totals['other'], 1),
                heavy_gt=totals['heavy'], other_gt=totals['other'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', type=int, default=150)
    parser.add_argument('--sides', type=int, nargs='+', default=[800, 1024])
    parser.add_argument('--threshold', type=float, default=.4187745451927185)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default=str(ROOT / 'reports/stage2_resolution_probe.json'))
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _ = load_model(args.config, args.checkpoint, device)
    dataset = ODInputDataset(PAppleDataset(ROOT / 'assets/papple', 'val', 32),
                             short_sides=(800,), max_size=1024)
    step = max(1, len(dataset) // args.images)
    indices = list(range(0, len(dataset), step))[:args.images]
    print(f'{len(indices)} images, checkpoint {Path(args.checkpoint).name}', flush=True)
    results = []
    for side in args.sides:
        row = run(model, dataset, indices, device, side, max(1024, int(round(side * 1.25))),
                  args.threshold)
        results.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)
    Path(args.out).write_text(json.dumps(dict(config=args.config, checkpoint=args.checkpoint,
                                              images=len(indices), results=results), indent=1),
                              encoding='utf-8')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
