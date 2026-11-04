"""Two-dimensional operating point: base threshold plus a crop-classifier gate.

Replacing the score failed, and so did a one-dimensional gate, because filtering
the pool makes the one-dimensional sweep lower the base threshold and buy the
false positives straight back. The honest formulation is a two-dimensional
operating point: select boxes with ``base >= b`` AND ``stage2 >= t``, then rank
by base. Both numbers are chosen on the full VAL, which is the same protocol the
project already uses for its single global threshold - one scalar becomes two,
and that is stated rather than hidden.

For a fixed gate the surviving boxes keep their base order, so the greedy
one-to-one matching is again a prefix of that order and every threshold can be
read in O(log n).
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


def load(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return [dict(image=name, boxes=np.asarray(row['boxes'], dtype='float64').reshape(-1, 4),
                 base=np.asarray(row['base'], dtype='float64'),
                 stage2=np.asarray(row['stage2'], dtype='float64'),
                 gt=np.asarray(row['gt'], dtype='float64').reshape(-1, 4))
            for name, row in payload.items()]


def prefix_for(record, gate):
    keep = np.ones(len(record['base']), dtype=bool) if gate is None else record['stage2'] >= gate
    scores = record['base'][keep]
    boxes = record['boxes'][keep]
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    matrix = s2.iou_matrix(boxes[order], record['gt']) if len(boxes) else np.zeros((0, len(record['gt'])))
    used, prefix = set(), np.zeros(len(order) + 1, dtype='int64')
    for row in range(matrix.shape[0]):
        candidates = [(matrix[row, j], j) for j in range(matrix.shape[1])
                      if j not in used and matrix[row, j] >= .5]
        if candidates:
            used.add(max(candidates)[1])
            prefix[row + 1] = prefix[row] + 1
        else:
            prefix[row + 1] = prefix[row]
    return sorted_scores, prefix, len(record['gt'])


def sweep(prepared, thresholds):
    best = None
    for threshold in thresholds:
        tp = fp = total = 0
        for sorted_scores, prefix, gt_count in prepared:
            k = int(np.searchsorted(-sorted_scores, -threshold, side='left'))
            tp += int(prefix[k])
            fp += k - int(prefix[k])
            total += gt_count
        precision, recall = tp / max(tp + fp, 1), tp / max(total, 1)
        score = min(precision, recall)
        if best is None or score > best['score']:
            best = dict(threshold=float(threshold), precision=precision, recall=recall, score=score,
                        tp=tp, fp=fp, fn=total - tp)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', required=True)
    parser.add_argument('--out', default=str(ROOT / 'reports/stage2_gate_report.json'))
    parser.add_argument('--gates', type=int, default=16)
    args = parser.parse_args()
    records = load(args.scores)
    pool = np.concatenate([r['base'] for r in records])
    base_thresholds = np.unique(np.quantile(pool, np.linspace(.55, .99995, 260)))
    gates = [None] + [float(v) for v in np.quantile(np.concatenate([r['stage2'] for r in records]),
                                                    np.linspace(.5, .995, args.gates))]
    best, table = None, []
    for gate in gates:
        prepared = [prefix_for(record, gate) for record in records]
        row = dict(sweep(prepared, base_thresholds), gate=gate)
        table.append(row)
        if best is None or row['score'] > best['score']:
            best = row
    print(f"{'gate':>7s} {'base_thr':>9s} {'P':>7s} {'R':>7s} {'min':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    for row in table:
        gate = 'none' if row['gate'] is None else f"{row['gate']:.3f}"
        print(f"{gate:>7s} {row['threshold']:9.4f} {row['precision']*100:6.2f}% {row['recall']*100:6.2f}% "
              f"{row['score']*100:6.2f}% {row['tp']:5d} {row['fp']:5d} {row['fn']:5d}")
    print(f"\nbest: gate={best['gate']} base={best['threshold']:.4f} min(P,R)={best['score']*100:.2f}%")

    published = 0.4187745451927185
    fp_scores = []
    for record in records:
        matrix = (s2.iou_matrix(record['boxes'], record['gt']) if len(record['gt'])
                  else np.zeros((len(record['boxes']), 1)))
        best_iou = matrix.max(1) if len(record['gt']) else np.zeros(len(record['boxes']))
        selected = record['base'] >= published
        zero_overlap = selected & (best_iou <= 0 if len(record['gt']) else np.ones(len(record['boxes']), bool))
        fp_scores.extend(record['stage2'][zero_overlap].tolist())
    fp_scores = np.asarray(fp_scores)
    if len(fp_scores):
        print(f"\nzero-overlap FP at the published threshold: n={len(fp_scores)} "
              f"stage2 median {np.median(fp_scores):.3f} q25 {np.quantile(fp_scores,.25):.3f} "
              f"q75 {np.quantile(fp_scores,.75):.3f}")
        for cut in (.2, .4, .6, .8):
            print(f"  stage2 < {cut:.1f}: {int((fp_scores < cut).sum())} of {len(fp_scores)}")
    Path(args.out).write_text(json.dumps(dict(best=best, table=table), indent=1), encoding='utf-8')
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
