"""Score-separation diagnostics for any completed OD validation point (read-only).

Reports, at the point's own calibrated global threshold: TP/FP/FN, the occlusion
strata recall, the score distribution of every error class, and the arithmetic
distance to the 95/95 requirement. It reads published artifacts only and never
changes GT, thresholds or model outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STRATA = {"heavy_visible_lt_0.40": "heavy", "moderate_visible_0.40_to_0.70": "moderate",
          "light_visible_0.70_to_0.90": "light", "mostly_visible_0.90_to_1.00": "mostly",
          "annotation_visible_area_exceeds_amodal": "anomaly"}


def iou_matrix(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return dict(n=int(len(values)), q10=float(np.quantile(values, .1)),
                q50=float(np.quantile(values, .5)), q90=float(np.quantile(values, .9)),
                max=float(values.max()))


def _mask_block(summary):
    block = {}
    for name, key in (("visible", "visible_instance_mask_pr_iou50"),
                      ("amodal", "amodal_instance_mask_pr_iou50")):
        section = summary.get(key)
        if isinstance(section, dict):
            block.update({name + '_' + field: section.get(field) for field in ("tp", "fp", "fn")})
    return block


def diagnose(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))
    confidence = summary["confidence"]
    meta = json.loads((directory / "evaluation_metadata.json").read_text(encoding="utf-8-sig"))
    strata = {(image["image_id"], row["gt_index"]): row["occlusion_stratum"]
              for image in meta["ground_truth_identity"] for row in image["GT"]}
    raw = json.loads((directory / "raw_predictions.json").read_text(encoding="utf-8"))
    tp_scores, fn_correct, fp_background, fp_partial, fp_duplicate = [], [], [], [], []
    per_stratum = {}
    for image in raw:
        image_id = image["image_id"]
        gt_boxes = image["gt"]["boxes"]
        boxes = image["pred"]["boxes"]
        scores = image["pred"]["scores"]
        selected = [i for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                    if scores[i] >= confidence]
        iou = iou_matrix(boxes, gt_boxes)
        used, tp_boxes = set(), set()
        for index in selected:
            candidates = [(iou[index, j], j) for j in range(iou.shape[1])
                          if j not in used and iou[index, j] >= .5]
            if candidates:
                used.add(max(candidates)[1])
                tp_boxes.add(index)
                tp_scores.append(scores[index])
        for index in selected:
            if index in tp_boxes:
                continue
            if not gt_boxes:
                fp_background.append(scores[index])
                continue
            best = float(iou[index].max())
            if best >= .5:
                fp_duplicate.append(scores[index])
            elif best <= 0:
                fp_background.append(scores[index])
            else:
                fp_partial.append(scores[index])
        for j in range(len(gt_boxes)):
            key = STRATA.get(strata.get((image_id, j), ""), "unknown")
            row = per_stratum.setdefault(key, dict(GT=0, TP=0))
            row["GT"] += 1
            if j in used:
                row["TP"] += 1
                continue
            if iou.shape[0]:
                best = iou[:, j].max()
                if best >= .5:
                    fn_correct.append(float(scores[int(np.argmax(iou[:, j]))]))
    for row in per_stratum.values():
        row["recall"] = row["TP"] / row["GT"] if row["GT"] else None
    block = summary["summary"]
    required_tp = 3277
    return dict(directory=str(directory), confidence=confidence,
                box={key: block[key] for key in ("tp", "fp", "fn", "precision", "recall",
                                                 "f1", "goal_95_met", "gt_count")},
                mask=_mask_block(summary),
                strata=per_stratum,
                scores=dict(TP=quantiles(tp_scores), FN_correct_box=quantiles(fn_correct),
                            FP_background=quantiles(fp_background), FP_partial=quantiles(fp_partial),
                            FP_duplicate=quantiles(fp_duplicate)),
                budget=dict(required_tp=required_tp, required_fp_max=required_tp // 19,
                            tp_gap=required_tp - block["tp"],
                            fp_excess=block["fp"] - required_tp // 19,
                            fp_over_tp=block["fp"] / max(block["tp"], 1),
                            allowed_fp_over_tp=1 / 19))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Run directory containing validation/')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    points = []
    for directory in sorted(args.run.glob('validation/update_*')):
        if (directory / 'summary.json').is_file() and (directory / 'evaluation_metadata.json').is_file():
            summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8-sig'))
            if summary.get('phase') == 'complete':
                points.append(diagnose(directory))
    if not points:
        raise SystemExit('No completed validation point under ' + str(args.run))
    for point in points:
        box = point['box']
        print(f"== {Path(point['directory']).name} conf={point['confidence']:.4f} "
              f"P={box['precision']:.4f} R={box['recall']:.4f} TP/FP/FN={box['tp']}/{box['fp']}/{box['fn']} "
              f"95={box['goal_95_met']}")
        for name, row in sorted(point['strata'].items()):
            print(f"   {name:9s} GT={row['GT']:5d} TP={row['TP']:5d} R={row['recall']:.3f}")
        for name, values in point['scores'].items():
            if values:
                print(f"   score {name:16s} n={values['n']:4d} q50={values['q50']:.4f} q90={values['q90']:.4f}")
        budget = point['budget']
        print(f"   budget TP +{budget['tp_gap']} FP -{budget['fp_excess']} "
              f"(fp/tp={budget['fp_over_tp']:.4f} vs allowed {budget['allowed_fp_over_tp']:.4f})")
    payload = json.dumps(points, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(payload + '\n', encoding='utf-8', newline='\n')
        print('wrote', args.out)


if __name__ == '__main__':
    main()
