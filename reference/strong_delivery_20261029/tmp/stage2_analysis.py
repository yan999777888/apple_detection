"""Full-VAL report for the second-stage rescoring, including strata and FP types.

Reads the per-box scores written by ``stage2_cli.py eval`` plus the validation
directory's own ``evaluation_metadata.json`` (which carries the official
occlusion stratum of every GT), so the report uses the same 814 / 3449 / 13 / 14
universe as every other number in this project.

Nothing here trains or selects on VAL beyond the single global threshold sweep
that the repository's own protocol has always used.
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

STRATA = {'heavy_visible_lt_0.40': 'heavy', 'moderate_visible_0.40_to_0.70': 'moderate',
          'light_visible_0.70_to_0.90': 'light', 'mostly_visible_0.90_to_1.00': 'mostly'}


def load(strata_path, scores_path):
    meta = json.loads(Path(strata_path).read_text(encoding='utf-8'))
    strata = {(row['image_id'], record['gt_index']): record['occlusion_stratum']
              for row in meta['ground_truth_identity'] for record in row['GT']}
    scores = json.loads(Path(scores_path).read_text(encoding='utf-8'))
    records = []
    for name, payload in scores.items():
        records.append(dict(image=name, boxes=np.asarray(payload['boxes'], dtype='float64').reshape(-1, 4),
                            base=np.asarray(payload['base'], dtype='float64'),
                            stage2=np.asarray(payload['stage2'], dtype='float64'),
                            gt=np.asarray(payload['gt'], dtype='float64').reshape(-1, 4)))
    return records, strata


def sweep(records, key, thresholds):
    best = None
    prepared = []
    for record in records:
        scores = np.asarray(record[key], dtype='float64')
        order = np.array(sorted(range(len(scores)), key=lambda i: (-scores[i], i)), dtype='int64')
        matrix = s2.iou_matrix(record['boxes'][order] if len(order) else np.zeros((0, 4)), record['gt'])
        used, prefix = set(), np.zeros(len(order) + 1, dtype='int64')
        for row in range(len(order)):
            candidates = [(matrix[row, j], j) for j in range(matrix.shape[1])
                          if j not in used and matrix[row, j] >= .5]
            if candidates:
                used.add(max(candidates)[1])
                prefix[row + 1] = prefix[row] + 1
            else:
                prefix[row + 1] = prefix[row]
        prepared.append((np.sort(scores)[::-1], prefix, len(record['gt'])))
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


def detail(records, strata, key, threshold):
    per_stratum = {}
    fp_types = {'background': 0, 'partial': 0, 'duplicate': 0, 'empty_gt': 0}
    fn_types = {'correct_box_below_threshold': 0, 'near_miss': 0, 'no_candidate': 0}
    for record in records:
        scores = np.asarray(record[key], dtype='float64')
        order = [i for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                 if scores[i] >= threshold]
        matrix = s2.iou_matrix(record['boxes'][order] if order else np.zeros((0, 4)), record['gt'])
        used, matched = set(), set()
        for row, index in enumerate(order):
            candidates = [(matrix[row, j], j) for j in range(matrix.shape[1])
                          if j not in used and matrix[row, j] >= .5]
            if candidates:
                best = max(candidates)[1]
                used.add(best)
                matched.add(row)
                name = STRATA.get(strata.get((record['image'], best), ''), 'unknown')
                row_stats = per_stratum.setdefault(name, dict(gt=0, tp=0))
                row_stats['tp'] += 1
            elif not matrix.shape[1]:
                fp_types['empty_gt'] += 1
            else:
                best_iou = float(matrix[row].max())
                fp_types['duplicate' if best_iou >= .5 else
                         ('background' if best_iou <= 0 else 'partial')] += 1
        for row in range(len(order)):
            if row not in matched and matrix.shape[1] and matrix[row].max() >= .5:
                fp_types['duplicate'] += 1
        full = s2.iou_matrix(record['boxes'], record['gt'])
        for j in range(len(record['gt'])):
            name = STRATA.get(strata.get((record['image'], j), ''), 'unknown')
            per_stratum.setdefault(name, dict(gt=0, tp=0))['gt'] += 1
            if j in used:
                continue
            column = full[:, j] if full.shape[0] else np.zeros(0)
            best_value = float(column.max()) if len(column) else 0.
            fn_types['correct_box_below_threshold' if best_value >= .5 else
                     ('near_miss' if best_value >= .25 else 'no_candidate')] += 1
    for name, row in per_stratum.items():
        row['recall'] = row['tp'] / max(row['gt'], 1)
    return dict(per_stratum=per_stratum, fp_types=fp_types, fn_types=fn_types)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', required=True)
    parser.add_argument('--metadata', required=True, help='validation dir evaluation_metadata.json')
    parser.add_argument('--out', default=str(ROOT / 'reports/stage2_full_val_report.json'))
    parser.add_argument('--reference', type=float, nargs=2, action='append', default=None,
                        metavar=('P', 'R'))
    args = parser.parse_args()
    records, strata = load(args.metadata, args.scores)
    # Every arm gets its own global threshold sweep on the full VAL, which is
    # the protocol this project has always used; no arm is handicapped by
    # another arm's score scale.
    def grid(key):
        pool = np.concatenate([np.asarray(r[key]) for r in records]) if records else np.zeros(1)
        return np.unique(np.quantile(pool, np.linspace(.3, .99995, 900)))

    thresholds = {key: grid(key) for key in ('base', 'stage2', 'fused')}
    report = dict(images=len(records), gt=int(sum(len(r['gt']) for r in records)),
                  candidate_boxes=int(sum(len(r['base']) for r in records)))
    for key in ('base', 'stage2'):
        report[key] = sweep(records, key, thresholds[key])
        report[key]['detail'] = detail(records, strata, key, report[key]['threshold'])
    fused = [dict(boxes=r['boxes'], gt=r['gt'],
                  fused=np.sqrt(np.clip(r['base'], 1e-9, 1) * np.clip(r['stage2'], 1e-9, 1)))
             for r in records]
    report['fused'] = sweep(fused, 'fused', thresholds['fused'])
    report['fused']['detail'] = detail(fused, strata, 'fused', report['fused']['threshold'])
    Path(args.out).write_text(json.dumps(report, indent=1), encoding='utf-8')

    print(f"images {report['images']}  GT {report['gt']}  candidates {report['candidate_boxes']}")
    header = f"{'arm':8s} {'thr':>6s} {'P':>7s} {'R':>7s} {'min':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}"
    print(header)
    for key in ('base', 'stage2', 'fused'):
        row = report[key]
        print(f"{key:8s} {row['threshold']:6.3f} {row['precision']*100:6.2f}% {row['recall']*100:6.2f}% "
              f"{row['score']*100:6.2f}% {row['tp']:5d} {row['fp']:5d} {row['fn']:5d}")
    for key in ('base', 'stage2'):
        row = report[key]['detail']
        print(f"\n[{key}] strata recall:")
        for name, stats in sorted(row['per_stratum'].items()):
            print(f"  {name:9s} GT {stats['gt']:5d} TP {stats['tp']:5d} R {stats['recall']:.3f}")
        print(f"[{key}] FP types {row['fp_types']}  FN types {row['fn_types']}")
    print('wrote', args.out)


if __name__ == '__main__':
    main()
