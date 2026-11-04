"""Offline what-if analysis over the saved per-box scores (no GPU, no re-decoding).

The first stage-2 read was unambiguous: replacing the detector score with a pure
crop-classifier score loses 25 points of min(P,R) even though its IoU ranking AUC
is only slightly lower (0.9476 vs 0.9593). That gap is a localisation blind spot:
a box 40% too large still crops to a convincing apple, while the detector score
carries box-regression evidence. So the useful question is not "which score
wins" but "which combination keeps the detector's ordering and still deletes the
foliage boxes".

Rules tested here, all with one global threshold sweep on the full VAL:

  base                    rank by base score (the published behaviour)
  stage2                  rank by the crop classifier only
  product                 rank by base * stage2
  gate_base               rank by base, drop boxes with stage2 < t
  gate_both               rank by base, drop boxes outside either gate
  stage2_then_nms         rank by stage2 after greedy IoU NMS at 0.5
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
from stage2_analysis import detail, sweep  # noqa: E402


def load_scores(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return [dict(image=name, boxes=np.asarray(row['boxes'], dtype='float64').reshape(-1, 4),
                 base=np.asarray(row['base'], dtype='float64'),
                 stage2=np.asarray(row['stage2'], dtype='float64'),
                 gt=np.asarray(row['gt'], dtype='float64').reshape(-1, 4))
            for name, row in payload.items()]


def nms(boxes, scores, threshold=.5):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        best = order[0]
        keep.append(best)
        if len(order) == 1:
            break
        rest = order[1:]
        matrix = s2.iou_matrix(boxes[rest], boxes[best][None])
        order = rest[(matrix[:, 0] <= threshold)]
    return np.asarray(keep, dtype='int64')


def variants(records):
    out = {}
    for name in ('base', 'stage2'):
        out[name] = [dict(image=r['image'], boxes=r['boxes'], gt=r['gt'], score=r[name]) for r in records]
    out['product'] = [dict(image=r['image'], boxes=r['boxes'], gt=r['gt'],
                           score=np.clip(r['base'], 1e-9, 1) * np.clip(r['stage2'], 1e-9, 1))
                      for r in records]
    for floor in (.2, .4, .6, .8):
        out[f'gate_base_{floor:.1f}'] = [
            dict(image=r['image'], boxes=r['boxes'][r['stage2'] >= floor],
                 gt=r['gt'], score=r['base'][r['stage2'] >= floor]) for r in records]
    out['stage2_then_nms'] = []
    for r in records:
        keep = nms(r['boxes'], r['stage2']) if len(r['boxes']) else np.zeros(0, dtype='int64')
        out['stage2_then_nms'].append(dict(image=r['image'], boxes=r['boxes'][keep],
                                           gt=r['gt'], score=r['stage2'][keep]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', required=True)
    parser.add_argument('--out', default=str(ROOT / 'reports/stage2_combo_report.json'))
    args = parser.parse_args()
    records = load_scores(args.scores)
    report = {}
    for name, rows in variants(records).items():
        pool = np.concatenate([row['score'] for row in rows if len(row['score'])]) if rows else np.zeros(1)
        if not len(pool):
            continue
        thresholds = np.unique(np.quantile(pool, np.linspace(.3, .99995, 700)))
        best = sweep(rows, 'score', thresholds)
        best['boxes_kept_pool'] = int(len(pool))
        report[name] = best
    order = sorted(report.items(), key=lambda kv: -kv[1]['score'])
    print(f"{'rule':22s} {'thr':>7s} {'P':>7s} {'R':>7s} {'min':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    for name, row in order:
        print(f"{name:22s} {row['threshold']:7.4f} {row['precision']*100:6.2f}% {row['recall']*100:6.2f}% "
              f"{row['score']*100:6.2f}% {row['tp']:5d} {row['fp']:5d} {row['fn']:5d}")
    Path(args.out).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
