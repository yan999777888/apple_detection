"""Exact error/stratum comparison for two raw prediction files."""
import argparse
import json
from pathlib import Path

import numpy as np

import flip_tta


def strata_map(metadata_path):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return {(row["image_id"], gt["gt_index"]): gt["occlusion_stratum"]
            for row in metadata["ground_truth_identity"] for gt in row["GT"]}


def summarize(records, strata):
    calibrated = flip_tta.metrics.calibrate_threshold(records, .5, return_curve=False)
    threshold = calibrated["confidence"]
    groups = {}
    matched_keys = set()
    fn_types = {"below_threshold_iou50": 0, "no_candidate_iou50": 0}
    fp_types = {"background": 0, "partial": 0, "duplicate": 0, "empty_gt": 0}
    for record in records:
        gt = np.asarray(record["gt"]["boxes"], dtype=np.float64).reshape(-1, 4)
        boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        match = flip_tta.metrics.match_image(gt, boxes, scores, .5, threshold)
        for gt_index in range(len(gt)):
            name = strata.get((record["image_id"], gt_index), "unknown")
            groups.setdefault(name, {"gt": 0, "tp": 0})["gt"] += 1
        selected_overlaps = flip_tta.metrics.box_iou(boxes[match["pred_indices"]], gt)
        for row, (pred_index, gt_index) in enumerate(zip(match["pred_indices"], match["gt_indices"])):
            if gt_index >= 0:
                key = (record["image_id"], int(gt_index))
                matched_keys.add(key)
                name = strata.get(key, "unknown")
                groups[name]["tp"] += 1
            elif not len(gt):
                fp_types["empty_gt"] += 1
            else:
                best_iou = float(selected_overlaps[row].max())
                fp_types["duplicate" if best_iou >= .5 else
                         ("partial" if best_iou > 0 else "background")] += 1
        used = set(int(index) for index in match["gt_indices"] if index >= 0)
        overlaps = flip_tta.metrics.box_iou(boxes, gt)
        for gt_index in range(len(gt)):
            if gt_index in used:
                continue
            best_iou = float(overlaps[:, gt_index].max()) if len(boxes) else 0.0
            fn_types["below_threshold_iou50" if best_iou >= .5 else "no_candidate_iou50"] += 1
    for row in groups.values():
        row["recall"] = row["tp"] / max(row["gt"], 1)
    summary = calibrated["summary"]
    return {"threshold": threshold, "precision": summary["precision"], "recall": summary["recall"],
            "min_precision_recall": summary["min_precision_recall"], "tp": summary["tp"],
            "fp": summary["fp"], "fn": summary["fn"], "groups": groups,
            "fp_types": fp_types, "fn_types": fn_types}, matched_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    strata = strata_map(args.metadata)
    candidate, candidate_keys = summarize(flip_tta.load_records(args.candidate), strata)
    baseline, baseline_keys = summarize(flip_tta.load_records(args.baseline), strata)
    report = {"candidate": candidate, "baseline": baseline,
              "rescued_gt": len(candidate_keys - baseline_keys), "lost_gt": len(baseline_keys - candidate_keys),
              "net_tp_change": len(candidate_keys) - len(baseline_keys)}
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
