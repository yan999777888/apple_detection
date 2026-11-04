"""Audit the precision ceiling at Recall >= 0.90 on an existing fused prediction.

The deliverable is one number: the best Precision reachable with a single global
threshold that still keeps Recall >= 0.90. If that number is already >= 0.96 no
training is needed; if it is not, the gap in FP count says how much has to be
removed. FP composition at the chosen operating point is reported so the next
step can target a specific error type rather than guessing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))
import flip_tta  # noqa: E402
from flip_tta import metrics  # noqa: E402

RECALL_TARGET = 0.90
PRECISION_TARGET = 0.96


def apply_nms(records, iou_threshold):
    """One-sided cleanup: a second box on an already-confident target is pure FP."""
    filtered = []
    for record in records:
        boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        if len(boxes):
            keep = nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(),
                       iou_threshold).numpy()
        else:
            keep = np.zeros(0, dtype=np.int64)
        filtered.append({"image_id": record["image_id"], "gt": record["gt"],
                         "pred": {"boxes": boxes[keep].tolist(), "scores": scores[keep].tolist()}})
    return filtered


def counts(records, confidence):
    return metrics.evaluate_at_threshold(records, confidence, 0.5)


def sweep(records, lows, highs):
    rows = []
    for value in sorted(set(lows + highs)):
        row = counts(records, value)
        rows.append(dict(threshold=value, precision=row["precision"], recall=row["recall"],
                         tp=row["tp"], fp=row["fp"], fn=row["fn"]))
    return rows


def best_at_recall(records, rows, target):
    eligible = [row for row in rows if row["recall"] >= target]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["precision"], row["recall"]))


def refine(records, centre, span, target, steps=40):
    grid = [centre + span * (index / steps - 0.5) for index in range(steps + 1)]
    rows = sweep(records, grid, [])
    return best_at_recall(records, rows, target), rows


def fp_composition(records, confidence, iou_threshold=0.5):
    """Split FPs and FNs by what they actually overlap, at one threshold."""
    buckets = {"background_maxiou_lt_0.1": 0, "partial_0.1_to_0.5": 0,
               "duplicate_iou_ge_0.5": 0, "empty_gt_image": 0}
    fn_types = {"below_threshold_iou50": 0, "no_candidate_iou50": 0}
    for record in records:
        gt = np.asarray(record["gt"]["boxes"], dtype=np.float64).reshape(-1, 4)
        boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        matched = metrics.match_image(gt, boxes, scores, iou_threshold, confidence)
        for pred_index in range(len(matched["pred_indices"])):
            if matched["gt_indices"][pred_index] >= 0:
                continue
            if not len(gt):
                buckets["empty_gt_image"] += 1
                continue
            best = float(metrics.box_iou(boxes[pred_index:pred_index + 1], gt)[0].max())
            if best >= iou_threshold:
                buckets["duplicate_iou_ge_0.5"] += 1
            elif best >= 0.1:
                buckets["partial_0.1_to_0.5"] += 1
            else:
                buckets["background_maxiou_lt_0.1"] += 1
        if len(gt):
            used_gt = {int(index) for index in matched["gt_indices"] if index >= 0}
            for gt_index in range(len(gt)):
                if gt_index in used_gt:
                    continue
                if len(boxes):
                    best = float(metrics.box_iou(boxes, gt[gt_index:gt_index + 1]).max())
                else:
                    best = 0.0
                key = "below_threshold_iou50" if best >= iou_threshold else "no_candidate_iou50"
                fn_types[key] += 1
    return buckets, fn_types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="One or more fused prediction files; each is audited separately")
    parser.add_argument("--out", required=True)
    parser.add_argument("--nms-ious", type=float, nargs="+", default=[],
                        help="Optional post-fusion NMS thresholds to audit as extra arms")
    args = parser.parse_args()

    report = {"recall_target": RECALL_TARGET, "precision_target": PRECISION_TARGET, "runs": []}
    variants = []
    for path in args.predictions:
        base = flip_tta.load_records(path)
        variants.append((path, None, base))
        for value in args.nms_ious:
            variants.append((path, value, apply_nms(base, value)))
    for path, nms_iou, records in variants:
        total_gt = sum(len(record["gt"]["boxes"]) for record in records)
        coarse = [0.30 + 0.005 * index for index in range(130)]
        rows = sweep(records, coarse, [])
        balanced = max(rows, key=lambda row: (min(row["precision"], row["recall"]),
                                             row["precision"] + row["recall"]))
        near = best_at_recall(records, rows, RECALL_TARGET)
        entry = {"predictions": path, "nms_iou": nms_iou, "images": len(records), "gt": total_gt,
                 "balanced": balanced}
        if near is not None:
            narrowed, _ = refine(records, near["threshold"], 0.02, RECALL_TARGET)
            chosen = narrowed if narrowed["precision"] > near["precision"] else near
            buckets, fn_types = fp_composition(records, chosen["threshold"])
            entry["at_recall_target"] = {**chosen, "fp_composition": buckets, "fn_composition": fn_types,
                                         "max_fp_for_precision_target":
                                             int(chosen["tp"] * (1 - PRECISION_TARGET) / PRECISION_TARGET),
                                         "fp_excess": chosen["fp"] - int(chosen["tp"] * (1 - PRECISION_TARGET) / PRECISION_TARGET)}
        else:
            entry["at_recall_target"] = None
        # How much of each class sits between the balanced threshold and the recall target.
        if near is not None and near["threshold"] > balanced["threshold"]:
            dropped = counts(records, (near["threshold"] + balanced["threshold"]) / 2)
            entry["between_balanced_and_target"] = dropped
        report["runs"].append(entry)
        print(json.dumps({k: entry[k] for k in ("predictions", "nms_iou", "balanced", "at_recall_target")},
                         ensure_ascii=False, indent=1), flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
