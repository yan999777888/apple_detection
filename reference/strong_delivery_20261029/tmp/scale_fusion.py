"""Fuse the same detector's predictions at several evaluation scales.

The resolution probe showed that the evaluation scale is the variable that moves
heavy-occlusion recall, so averaging the scale dimension is the cheap next step:
one checkpoint, two or three input sizes, boxes merged by greedy clustering.

Fusion rule (deterministic, no GT): boxes from every scale are pooled, ordered by
score, and a box joins an existing cluster when its IoU with the cluster's
representative is >= ``--iou``. The cluster's final box is the score-weighted
mean and its score is the maximum member score. No soft weighting by rank, so
the result stays a plain score that the usual global threshold sweep can use.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp'))
sys.path.insert(0, str(ROOT / 'src'))

import stage2_crop as s2  # noqa: E402
from stage2_analysis import sweep  # noqa: E402


def load(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return {row['image_id']: row for row in payload}


def fuse(entries, iou_threshold):
    boxes = np.concatenate([np.asarray(entry['pred']['boxes'], dtype='float64').reshape(-1, 4)
                            for entry in entries])
    scores = np.concatenate([np.asarray(entry['pred']['scores'], dtype='float64') for entry in entries])
    if not len(boxes):
        return boxes, scores
    order = np.argsort(-scores, kind='stable')
    boxes, scores = boxes[order], scores[order]
    clusters = []          # list of dict(box, score, members[list of (box, score)])
    taken = np.zeros(len(boxes), dtype=bool)
    for index in range(len(boxes)):
        if taken[index]:
            continue
        cluster = [(boxes[index], scores[index])]
        taken[index] = True
        matrix = s2.iou_matrix(boxes[index + 1:], boxes[index][None])[:, 0] if index + 1 < len(boxes) else np.zeros(0)
        for offset, value in enumerate(matrix):
            candidate = index + 1 + offset
            if not taken[candidate] and value >= iou_threshold:
                cluster.append((boxes[candidate], scores[candidate]))
                taken[candidate] = True
        clusters.append(cluster)
    fused_boxes, fused_scores = [], []
    for cluster in clusters:
        weight = np.asarray([item[1] for item in cluster])
        coordinates = np.asarray([item[0] for item in cluster])
        fused_boxes.append((coordinates * weight[:, None]).sum(0) / weight.sum())
        fused_scores.append(float(weight.max()))
    return np.asarray(fused_boxes), np.asarray(fused_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', nargs='+', required=True,
                        help='raw_predictions.json of each scale, in order')
    parser.add_argument('--iou', type=float, default=.55)
    parser.add_argument('--out', default=str(ROOT / 'reports/scale_fusion_report.json'))
    args = parser.parse_args()
    layers = [load(path) for path in args.predictions]
    names = sorted(set(layers[0]))
    missing = [name for name in names if any(name not in layer for layer in layers)]
    if missing:
        raise SystemExit(f'{len(missing)} images are missing from at least one scale')
    records, singles = [], []
    for name in names:
        entries = [layer[name] for layer in layers]
        gt = np.asarray(entries[0]['gt']['boxes'], dtype='float64').reshape(-1, 4)
        boxes, scores = fuse(entries, args.iou)
        records.append(dict(image=name, boxes=boxes, gt=gt, fused=scores))
        for label, entry in zip(('first', 'second', 'third'), entries):
            singles.append((label, dict(image=name, gt=gt,
                                        boxes=np.asarray(entry['pred']['boxes'], dtype='float64').reshape(-1, 4),
                                        score=np.asarray(entry['pred']['scores'], dtype='float64'))))
    pool = np.concatenate([record['fused'] for record in records])
    thresholds = np.unique(np.quantile(pool, np.linspace(.3, .99995, 700)))
    fused_best = sweep(records, 'fused', thresholds)
    report = dict(iou=args.iou, scales=[str(Path(p).parent.name) for p in args.predictions],
                  fused=fused_best)
    print(f"{'arm':10s} {'thr':>7s} {'P':>7s} {'R':>7s} {'min':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    for label in ('first', 'second', 'third'):
        rows = [row for tag, row in singles if tag == label]
        if not rows:
            continue
        own = np.concatenate([row['score'] for row in rows])
        own_thresholds = np.unique(np.quantile(own, np.linspace(.3, .99995, 700)))
        best = sweep(rows, 'score', own_thresholds)
        report[label] = best
        print(f"{label:10s} {best['threshold']:7.4f} {best['precision']*100:6.2f}% {best['recall']*100:6.2f}% "
              f"{best['score']*100:6.2f}% {best['tp']:5d} {best['fp']:5d} {best['fn']:5d}")
    print(f"{'fused':10s} {fused_best['threshold']:7.4f} {fused_best['precision']*100:6.2f}% "
          f"{fused_best['recall']*100:6.2f}% {fused_best['score']*100:6.2f}% "
          f"{fused_best['tp']:5d} {fused_best['fp']:5d} {fused_best['fn']:5d}")
    Path(args.out).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
