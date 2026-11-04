"""Quantify how many background false positives sit in the black image padding.

PApple frames are 1300x1300 with black fill outside the sensor area. A box whose
area is mostly near-black cannot contain an apple, so rejecting it is a
deterministic rule rather than a tuned threshold: the fraction is measured on VAL
but the rule itself is fixed in advance at 0.60.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))
import flip_tta  # noqa: E402
from flip_tta import metrics  # noqa: E402

BLACK_LEVEL = 20
RULE_FRACTION = 0.60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = flip_tta.load_records(args.predictions)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    sources = {row["image_id"]: row["source"] for row in metadata["ground_truth_identity"]}

    rows = []
    for record in records:
        gt = np.asarray(record["gt"]["boxes"], dtype=np.float64).reshape(-1, 4)
        boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        keep = scores >= args.threshold
        boxes, scores = boxes[keep], scores[keep]
        if not len(boxes):
            continue
        matched = metrics.match_image(gt, boxes, scores, 0.5, args.threshold)
        matched_gt = dict(zip(matched["pred_indices"], matched["gt_indices"]))
        source = sources[record["image_id"]]
        image = np.asarray(Image.open(source["rgb_path"]).convert("RGB"))
        crop = source.get("crop_xyxy")
        if crop:
            image = image[crop[1]:crop[3], crop[0]:crop[2]]
        black = (image.max(axis=2) < BLACK_LEVEL)
        height, width = black.shape
        for position in range(len(boxes)):
            x1, y1, x2, y2 = boxes[position]
            x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
            x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
            if x2 <= x1 or y2 <= y1:
                fraction = 1.0
            else:
                fraction = float(black[y1:y2, x1:x2].mean())
            label = "tp" if matched_gt.get(position, -1) >= 0 else "fp"
            if label == "fp" and len(gt):
                best = float(metrics.box_iou(boxes[position:position + 1], gt)[0].max())
            else:
                best = 1.0 if label == "tp" else 0.0
            rows.append({"label": label, "black_fraction": fraction, "score": float(scores[position]),
                         "max_iou": best})

    background = [row for row in rows if row["label"] == "fp" and row["max_iou"] < 0.1]
    true_positive = [row for row in rows if row["label"] == "tp"]
    flagged = [row for row in background if row["black_fraction"] >= RULE_FRACTION]
    report = {
        "threshold": args.threshold,
        "rule_fraction": RULE_FRACTION,
        "true_positives": len(true_positive),
        "tp_flagged_by_rule": sum(row["black_fraction"] >= RULE_FRACTION for row in true_positive),
        "background_fp": len(background),
        "background_fp_flagged": len(flagged),
        "background_fp_black_fraction_percentiles": {
            str(p): round(float(np.percentile([row["black_fraction"] for row in background], p)), 3)
            for p in (10, 50, 90)} if background else {},
        "tp_black_fraction_percentiles": {
            str(p): round(float(np.percentile([row["black_fraction"] for row in true_positive], p)), 3)
            for p in (50, 90, 99)} if true_positive else {},
    }
    # What the operating point becomes once those boxes are removed.
    remaining_fp = len(background) - len(flagged)
    kept_tp = len(true_positive) - report["tp_flagged_by_rule"]
    report["projected"] = {"tp": kept_tp, "fp": remaining_fp,
                           "precision": round(kept_tp / (kept_tp + remaining_fp), 6),
                           "recall": round(kept_tp / 3449, 6)}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
