"""Numpy-only detection calibration and paired-mask diagnostics.

Record schema: {image_id, gt: {boxes, masks?, visible_masks?},
                pred: {boxes, scores, masks?, visible_masks?}}.
Boxes are XYXY with exclusive right/bottom coordinates. Masks can be N,H,W
or N,1,H,W and are thresholded at 0.5. This module does NOT claim mask AP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _boxes(value, ground_truth=False):
    array = np.asarray(value, dtype=np.float64)
    if not array.size:
        return array.reshape(0, 4)
    if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
        raise ValueError("boxes must be finite N x 4 arrays")
    if ground_truth and np.any(array[:, 2:] <= array[:, :2]):
        raise ValueError("Ground-truth boxes must have positive width and height")
    return array


def box_iou(boxes1, boxes2):
    """Pairwise IoU; zero/negative-area predictions have zero IoU."""
    a, b = _boxes(boxes1), _boxes(boxes2)
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                    - np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _normalize_record(record):
    gt, pred = record["gt"], record["pred"]
    gt_boxes = _boxes(gt["boxes"], ground_truth=True)
    pred_boxes = _boxes(pred["boxes"])
    scores = np.asarray(pred["scores"], dtype=np.float64)
    if scores.shape != (len(pred_boxes),) or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite vector with one value per prediction")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("scores must be probabilities in [0, 1]")
    return gt_boxes, pred_boxes, scores


def match_image(gt_boxes, pred_boxes, scores, iou_threshold=0.5, confidence=0.0):
    """Greedy, descending-confidence, one-to-one matching with stable ties.

    Equal-score predictions retain input order. Equal-IoU GTs retain GT order.
    Zero-area predictions are included as FP when their scores pass threshold.
    """
    if not 0 < float(iou_threshold) <= 1:
        raise ValueError("iou_threshold must be in (0,1]")
    if not np.isfinite(confidence):
        raise ValueError("confidence must be finite")
    gt_boxes = _boxes(gt_boxes, ground_truth=True)
    pred_boxes = _boxes(pred_boxes)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(pred_boxes),) or not np.isfinite(scores).all():
        raise ValueError("scores shape/value error")
    order = np.argsort(-scores, kind="stable")
    order = order[scores[order] >= confidence]
    matched = np.full(len(order), -1, dtype=np.int64)
    matched_ious = np.zeros(len(order), dtype=np.float64)
    used = np.zeros(len(gt_boxes), dtype=bool)
    overlaps = box_iou(pred_boxes[order], gt_boxes)
    for row in range(len(order)):
        if not len(gt_boxes):
            break
        candidates = np.where(~used, overlaps[row], -1.0)
        gt_index = int(np.argmax(candidates))
        if candidates[gt_index] >= iou_threshold:
            used[gt_index] = True
            matched[row] = gt_index
            matched_ious[row] = candidates[gt_index]
    return {"pred_indices": order, "gt_indices": matched,
            "matched_ious": matched_ious, "tp": matched >= 0,
            "unmatched_gt_indices": np.flatnonzero(~used)}


def _counts(tp, fp, gt_count, confidence, iou_threshold):
    tp, fp, gt_count = int(tp), int(fp), int(gt_count)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"confidence": float(confidence), "iou_threshold": float(iou_threshold),
            "precision": precision, "recall": recall, "f1": f1,
            "min_precision_recall": min(precision, recall),
            "tp": tp, "fp": fp, "fn": gt_count - tp, "gt_count": gt_count,
            "prediction_count": tp + fp,
            "goal_95_met": gt_count > 0 and precision >= 0.95 and recall >= 0.95}


def _mask(array, index, count):
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3 or array.shape[0] != count:
        raise ValueError("Masks must be N,H,W or N,1,H,W, aligned with boxes")
    chosen = array[index]
    if not np.isfinite(chosen).all():
        raise ValueError("Masks must contain finite values")
    return chosen >= 0.5


def _mask_iou(left, right):
    if left.shape != right.shape:
        raise ValueError("GT and predicted masks must be in the same original image coordinates")
    union = np.count_nonzero(left | right)
    return np.count_nonzero(left & right) / union if union else 0.0


def _mask_summary(values, total_tp):
    return {"pairs_evaluated": len(values), "box_true_positives": int(total_tp),
            "mean_iou": float(np.mean(values)) if values else None,
            "median_iou": float(np.median(values)) if values else None,
            "fraction_iou_ge_0_5": float(np.mean(np.asarray(values) >= 0.5)) if values else None,
            "interpretation": "Mask IoU only for matched box true positives; not mask AP or overall mask recall."}


def evaluate_at_threshold(records, confidence, iou_threshold=0.5, include_per_image=False):
    """Evaluate one fixed setting; no threshold selection or test-set tuning."""
    tp = fp = gt_count = 0
    visible_ious, amodal_ious, per_image = [], [], []
    empty_gt_images = empty_gt_predictions = 0
    records = list(records)
    for image_index, record in enumerate(records):
        gt_boxes, pred_boxes, scores = _normalize_record(record)
        match = match_image(gt_boxes, pred_boxes, scores, iou_threshold, confidence)
        image_tp = int(match["tp"].sum())
        image_fp = len(match["tp"]) - image_tp
        tp += image_tp
        fp += image_fp
        gt_count += len(gt_boxes)
        if not len(gt_boxes):
            empty_gt_images += 1
            empty_gt_predictions += image_fp
        for p_index, g_index in zip(match["pred_indices"], match["gt_indices"]):
            if g_index < 0:
                continue
            for key, destination in (("masks", amodal_ious), ("visible_masks", visible_ious)):
                if key in record["gt"] and key in record["pred"]:
                    gt_mask = _mask(record["gt"][key], g_index, len(gt_boxes))
                    pred_mask = _mask(record["pred"][key], p_index, len(pred_boxes))
                    destination.append(_mask_iou(gt_mask, pred_mask))
        if include_per_image:
            entry = _counts(image_tp, image_fp, len(gt_boxes), confidence, iou_threshold)
            entry["image_id"] = record.get("image_id", image_index)
            entry["matches"] = [{"prediction_index": int(p), "gt_index": int(g), "box_iou": float(iou)}
                                for p, g, iou in zip(match["pred_indices"], match["gt_indices"],
                                                     match["matched_ious"]) if g >= 0]
            entry["unmatched_gt_indices"] = match["unmatched_gt_indices"].tolist()
            per_image.append(entry)
    summary = _counts(tp, fp, gt_count, confidence, iou_threshold)
    summary.update({"images": len(records), "empty_gt_images": empty_gt_images,
                    "false_positives_on_empty_gt_images": empty_gt_predictions,
                    "amodal_mask_iou_on_box_tp": _mask_summary(amodal_ious, tp),
                    "visible_mask_iou_on_box_tp": _mask_summary(visible_ious, tp),
                    "matching": "Descending score, stable input-order ties, one-to-one IoU >= threshold",
                    "ap_computed": False})
    if include_per_image:
        summary["per_image"] = per_image
    return summary


def calibrate_threshold(records, iou_threshold=0.5, return_curve=False):
    """Select on VALIDATION only: maximize min(P,R), then F1, then confidence.

    All distinct observed score thresholds are evaluated. A tied score group is
    included/excluded as a whole, never split to invent an unattainable threshold.
    IoUs are matched once per image; adding lower-score predictions cannot alter
    higher-score assignments. Evaluation is limited to supplied raw predictions.
    """
    records = list(records)
    all_scores, all_tp = [], []
    gt_count = 0
    for record in records:
        gt_boxes, pred_boxes, scores = _normalize_record(record)
        match = match_image(gt_boxes, pred_boxes, scores, iou_threshold, 0.0)
        all_scores.extend(scores[match["pred_indices"]].tolist())
        all_tp.extend(match["tp"].tolist())
        gt_count += len(gt_boxes)
    if not all_scores:
        chosen = 1.0
        curve = [_counts(0, 0, gt_count, chosen, iou_threshold)]
    else:
        scores = np.asarray(all_scores, dtype=np.float64)
        flags = np.asarray(all_tp, dtype=np.int64)
        order = np.argsort(-scores, kind="stable")
        scores, flags = scores[order], flags[order]
        cumulative_tp = np.cumsum(flags)
        ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
        curve = [_counts(cumulative_tp[i], i + 1 - cumulative_tp[i], gt_count,
                         scores[i], iou_threshold) for i in ends]
        best = max(curve, key=lambda point: (point["min_precision_recall"], point["f1"], point["confidence"]))
        chosen = best["confidence"]
    result = {"confidence": float(chosen), "iou_threshold": float(iou_threshold),
              "candidate_thresholds": len(curve),
              "selection_rule": "validation max(min(P,R)); tie max(F1); tie higher confidence",
              "summary": evaluate_at_threshold(records, chosen, iou_threshold),
              "scope": "Only supplied predictions; raw score cutoff/NMS/top-k can limit attainable recall."}
    if return_curve:
        result["curve"] = curve
    return result


def self_test():
    """Hand-built CPU-only cases protecting evaluation semantics."""
    records = [
        {"image_id": "apple", "gt": {"boxes": [[0, 0, 2, 2]]},
         "pred": {"boxes": [[0, 0, 2, 2], [0, 0, 2, 2], [1, 1, 1, 2]],
                  "scores": [0.8, 0.8, 0.7]}},
        {"image_id": "negative", "gt": {"boxes": []},
         "pred": {"boxes": [[0, 0, 3, 3]], "scores": [0.8]}},
        {"image_id": "miss", "gt": {"boxes": [[3, 3, 4, 4]]},
         "pred": {"boxes": [], "scores": []}},
    ]
    fixed = evaluate_at_threshold(records, 0.7)
    assert (fixed["tp"], fixed["fp"], fixed["fn"]) == (1, 3, 1)
    assert fixed["false_positives_on_empty_gt_images"] == 1
    chosen = calibrate_threshold(records, return_curve=True)
    assert chosen["confidence"] == 0.8
    assert chosen["summary"]["prediction_count"] == 3  # All tied predictions retained.
    assert (chosen["summary"]["tp"], chosen["summary"]["fp"]) == (1, 2)
    assert len(chosen["curve"]) == 2
    assert box_iou([[0, 0, 0, 2]], [[0, 0, 2, 2]])[0, 0] == 0
    mask = np.ones((1, 2, 2), dtype=np.uint8)
    predicted = mask.copy()
    predicted[0, 0, 0] = 0
    masked = [{"gt": {"boxes": [[0, 0, 2, 2]], "masks": mask, "visible_masks": mask},
               "pred": {"boxes": [[0, 0, 2, 2]], "scores": [0.9],
                        "masks": mask, "visible_masks": predicted}}]
    stats = evaluate_at_threshold(masked, 0.5)
    assert stats["amodal_mask_iou_on_box_tp"]["mean_iou"] == 1.0
    assert stats["visible_mask_iou_on_box_tp"]["mean_iou"] == 0.75
    empty = calibrate_threshold([{"gt": {"boxes": []}, "pred": {"boxes": [], "scores": []}}])
    assert empty["summary"]["goal_95_met"] is False
    assert empty["summary"]["precision"] == 0
    # Greedy matching chooses the next unused GT when the best has been assigned.
    match = match_image([[0, 0, 2, 2], [0, 0, 3, 2]],
                        [[0, 0, 2, 2], [0, 0, 2, 2]], [0.9, 0.8])
    assert match["gt_indices"].tolist() == [0, 1]
    return {"passed": True, "checks": ["empty-GT FP", "duplicates", "whole score ties",
             "zero-area FP", "paired mask IoU", "empty data", "one-to-one assignment"]}


def main():
    parser = argparse.ArgumentParser(description="Numpy-only P/R calibration; no fake mask AP")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", help="JSON list of records, normally boxes/scores only")
    parser.add_argument("--output")
    parser.add_argument("--confidence", type=float)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        if not args.input:
            parser.error("--input or --self-test is required")
        records = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
        result = (evaluate_at_threshold(records, args.confidence) if args.confidence is not None
                  else calibrate_threshold(records, return_curve=True))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
