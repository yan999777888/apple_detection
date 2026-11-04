"""Graft Step 2's locally better boxes onto Step 1's better score ranking.

Both predictions are already two-view fused products of the same evaluation
protocol, so this probe changes nothing but the geometry of the candidates that
already exist in the better-ranked arm. A Step 2 box is only allowed to move a
Step 1 box, never to add one: the candidate set, its scores and therefore the
global threshold stay bit-identical to the 90.90% baseline.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))

import flip_tta  # noqa: E402


def graft(primary, secondary, match_iou, weight):
    """Replace matched primary boxes by a weighted average with secondary boxes."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0,1]")
    names = [record["image_id"] for record in primary]
    if [record["image_id"] for record in secondary] != names:
        raise ValueError("Prediction layers differ in image identity or order")
    fused, matched_total, candidate_total = [], 0, 0
    for first, second in zip(primary, secondary):
        if first["gt"] != second["gt"]:
            raise ValueError("Prediction layers differ in GT identity or values")
        boxes = np.asarray(first["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        other = np.asarray(second["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        candidate_total += len(boxes)
        result = boxes.copy()
        if len(boxes) and len(other):
            overlaps = flip_tta.metrics.box_iou(boxes, other)
            order = np.argsort(-overlaps, axis=None, kind="stable")
            used_rows, used_columns = set(), set()
            for flat in order:
                row, column = divmod(int(flat), overlaps.shape[1])
                if overlaps[row, column] < match_iou:
                    break
                if row in used_rows or column in used_columns:
                    continue
                used_rows.add(row)
                used_columns.add(column)
                result[row] = (1 - weight) * boxes[row] + weight * other[column]
            matched_total += len(used_rows)
        fused.append({"image_id": first["image_id"], "gt": first["gt"],
                      "pred": {"boxes": result.tolist(),
                               "scores": list(first["pred"]["scores"])}})
    return fused, matched_total, candidate_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, help="Better score ranking (Step 1)")
    parser.add_argument("--secondary", required=True, help="Better boxes (Step 2 GIoU)")
    parser.add_argument("--match-ious", nargs="+", type=float, default=[.55])
    parser.add_argument("--weights", nargs="+", type=float, default=[.4])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    primary = flip_tta.load_records(args.primary)
    secondary = flip_tta.load_records(args.secondary)
    report = {"schema": "apple_score_box_graft_v1", "images": len(primary),
              "baseline": flip_tta.evaluate(primary), "arms": []}
    print("baseline", {k: round(report["baseline"][k], 6) for k in
                       ("precision", "recall", "min_precision_recall")},
          report["baseline"]["tp"], report["baseline"]["fp"], report["baseline"]["fn"], flush=True)
    for match_iou in args.match_ious:
        for weight in args.weights:
            fused, matched, total = graft(primary, secondary, match_iou, weight)
            row = dict(match_iou=match_iou, weight=weight, matched=matched, candidates=total,
                       **flip_tta.evaluate(fused))
            report["arms"].append(row)
            print(f"iou>={match_iou:.2f} w={weight:.2f} matched={matched}/{total} "
                  f"thr={row['threshold']:.6f} P={row['precision'] * 100:.2f} "
                  f"R={row['recall'] * 100:.2f} TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}",
                  flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
