"""Reconcile the FP composition at one threshold and apply the audited splits.

Two corrections are computed, and they answer different questions:

* padding-excluded precision drops boxes that lie in the camera's black fill,
  a region that cannot contain fruit by construction, so it is a measurement
  fix rather than a model change;
* annotation-corrected precision additionally counts boxes the manual audit
  judged to be unlabelled apples.

Neither replaces the raw protocol number; they bound what the detector actually
got wrong.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))
import flip_tta  # noqa: E402
from flip_tta import metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--audit", required=True,
                        help="JSON with per-rank verdicts: apple / padding / other")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = flip_tta.load_records(args.predictions)
    total_fp = duplicate = partial = background = empty_gt = tp = 0
    background_rows = []
    for record in records:
        gt = np.asarray(record["gt"]["boxes"], dtype=np.float64).reshape(-1, 4)
        boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        keep = scores >= args.threshold
        boxes, scores = boxes[keep], scores[keep]
        if not len(boxes):
            continue
        matched = metrics.match_image(gt, boxes, scores, 0.5, args.threshold)
        verdict = dict(zip(matched["pred_indices"], matched["gt_indices"]))
        for position in range(len(boxes)):
            if verdict.get(position, -1) >= 0:
                tp += 1
                continue
            total_fp += 1
            if not len(gt):
                empty_gt += 1
                background += 1
                background_rows.append({"image_id": record["image_id"],
                                        "box": boxes[position].tolist(),
                                        "score": float(scores[position])})
                continue
            best = float(metrics.box_iou(boxes[position:position + 1], gt)[0].max())
            if best >= 0.5:
                duplicate += 1
            elif best >= 0.1:
                partial += 1
            else:
                background += 1
                background_rows.append({"image_id": record["image_id"],
                                        "box": boxes[position].tolist(),
                                        "score": float(scores[position])})

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    verdicts = [row["verdict"] for row in audit]
    counts = {name: verdicts.count(name) for name in ("apple", "padding", "other")}
    inspected = len(verdicts)
    scale = background / inspected if inspected else 0.0
    estimated = {name: value * scale for name, value in counts.items()}

    report = {
        "threshold": args.threshold,
        "true_positives": tp,
        "false_positives": total_fp,
        "composition": {"duplicate_iou_ge_0.5": duplicate, "partial_0.1_to_0.5": partial,
                        "background": background, "of_which_empty_gt_image": empty_gt},
        "raw": {"precision": round(tp / (tp + total_fp), 6), "recall": round(tp / 3449, 6)},
        "audit": {"inspected": inspected, "verdicts": counts,
                  "share": {name: round(value / inspected, 4) for name, value in counts.items()},
                  "estimated_background_splits": {name: round(value, 1)
                                                  for name, value in estimated.items()}},
        "padding_excluded": None,
        "annotation_corrected": None,
    }
    padding_excluded_fp = total_fp - estimated["padding"]
    report["padding_excluded"] = {
        "fp": round(padding_excluded_fp, 1),
        "precision": round(tp / (tp + padding_excluded_fp), 6),
        "recall": round(tp / 3449, 6)}
    corrected_tp = tp + estimated["apple"]
    corrected_fp = total_fp - estimated["padding"] - estimated["apple"]
    report["annotation_corrected"] = {
        "tp": round(corrected_tp, 1), "fp": round(corrected_fp, 1),
        "precision": round(corrected_tp / (corrected_tp + corrected_fp), 6),
        "recall": round(corrected_tp / 3449, 6)}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
