"""Where exactly does the crop classifier disagree with the detector?

Uses only the saved per-box scores plus the validation directory's own
``evaluation_metadata.json`` occlusion strata, so it costs nothing and cannot
touch training data.
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


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype='float64')
    ranks[order] = np.arange(1, len(scores) + 1)
    positive = labels >= .5
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', required=True)
    parser.add_argument('--metadata', required=True)
    args = parser.parse_args()
    scores = json.loads(Path(args.scores).read_text(encoding='utf-8'))
    meta = json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    strata = {(row['image_id'], record['gt_index']): STRATA.get(record['occlusion_stratum'], 'unknown')
              for row in meta['ground_truth_identity'] for record in row['GT']}

    per_stratum = {}
    background = {'base': [], 'stage2': []}
    for name, payload in scores.items():
        boxes = np.asarray(payload['boxes'], dtype='float64').reshape(-1, 4)
        if not len(boxes):
            continue
        gt = np.asarray(payload['gt'], dtype='float64').reshape(-1, 4)
        matrix = s2.iou_matrix(boxes, gt) if len(gt) else np.zeros((len(boxes), 1))
        best = matrix.max(1) if len(gt) else np.zeros(len(boxes))
        best_index = matrix.argmax(1) if len(gt) else np.zeros(len(boxes), dtype='int64')
        base = np.asarray(payload['base'], dtype='float64')
        stage2 = np.asarray(payload['stage2'], dtype='float64')
        matched = best >= .5
        for box in np.nonzero(matched)[0]:
            label = strata.get((name, int(best_index[box])), 'unknown')
            row = per_stratum.setdefault(label, {'base': [], 'stage2': [], 'n': 0})
            row['base'].append(base[box])
            row['stage2'].append(stage2[box])
            row['n'] += 1
        for box in np.nonzero(best <= 0)[0]:
            background['base'].append(base[box])
            background['stage2'].append(stage2[box])

    print('median score of boxes matching a GT at IoU>=0.5, by occlusion stratum')
    print(f"{'stratum':9s} {'n':>6s} {'base q25':>9s} {'base med':>9s} {'s2 q25':>8s} {'s2 med':>8s} {'s2 q75':>8s}")
    for label, row in sorted(per_stratum.items()):
        b, s = np.asarray(row['base']), np.asarray(row['stage2'])
        print(f'{label:9s} {row["n"]:6d} {np.quantile(b,.25):9.3f} {np.median(b):9.3f} '
              f'{np.quantile(s,.25):8.3f} {np.median(s):8.3f} {np.quantile(s,.75):8.3f}')
    for key in ('base', 'stage2'):
        values = np.asarray(background[key])
        print(f'background (IoU=0) n={len(values)} {key}: median {np.median(values):.3f} '
              f'q25 {np.quantile(values,.25):.3f} q75 {np.quantile(values,.75):.3f}')
    print()
    for label, row in sorted(per_stratum.items()):
        s = np.asarray(row['stage2'])
        b = np.asarray(row['base'])
        combined = np.concatenate([s, np.asarray(background['stage2'])])
        labels = np.concatenate([np.ones(len(s)), np.zeros(len(background['stage2']))])
        combined_base = np.concatenate([b, np.asarray(background['base'])])
        print(f'{label:9s} vs background: AUC stage2 {auc(combined, labels):.4f}  '
              f'AUC base {auc(combined_base, labels):.4f}')


if __name__ == '__main__':
    main()
