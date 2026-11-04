"""Use crop views only to propose boxes absent from the stronger full views."""
import argparse
import json
from pathlib import Path

import numpy as np

import flip_tta


def add_unmatched(primary, auxiliary, exclusion_iou, score_scale):
    if len(primary) != len(auxiliary):
        raise ValueError("Primary and auxiliary image scopes differ")
    result = []
    for main, extra in zip(primary, auxiliary):
        main_boxes = np.asarray(main["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        extra_boxes = np.asarray(extra["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        extra_scores = np.asarray(extra["pred"]["scores"], dtype=np.float64)
        overlaps = (flip_tta.metrics.box_iou(extra_boxes, main_boxes)
                    if len(extra_boxes) and len(main_boxes) else np.zeros((len(extra_boxes), len(main_boxes))))
        novel = overlaps.max(axis=1) < exclusion_iou if len(main_boxes) else np.ones(len(extra_boxes), bool)
        boxes = np.concatenate((main_boxes, extra_boxes[novel]))
        scores = np.concatenate((np.asarray(main["pred"]["scores"], dtype=np.float64),
                                 extra_scores[novel] * score_scale))
        result.append({"image_id": main["image_id"], "gt": main["gt"],
                       "pred": {"boxes": boxes.tolist(), "scores": scores.tolist()}})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", nargs="+", required=True)
    parser.add_argument("--auxiliary", nargs="+", required=True)
    parser.add_argument("--fuse-iou", type=float, default=.35)
    parser.add_argument("--exclusion-ious", type=float, nargs="+", default=[.4, .5, .6])
    parser.add_argument("--score-scales", type=float, nargs="+", default=[1., .85, .7])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main_layers = [flip_tta.load_records(path) for path in args.primary]
    aux_layers = [flip_tta.load_records(path) for path in args.auxiliary]
    primary = flip_tta.fuse_records(main_layers, args.fuse_iou, "mean", "score_weighted")
    auxiliary = flip_tta.fuse_records(aux_layers, args.fuse_iou, "mean", "score_weighted")
    report = {"primary": flip_tta.evaluate(primary), "primary_files": args.primary,
              "auxiliary": flip_tta.evaluate(auxiliary), "arms": []}
    best = None
    for exclusion in args.exclusion_ious:
        for scale in args.score_scales:
            records = add_unmatched(primary, auxiliary, exclusion, scale)
            row = {"exclusion_iou": exclusion, "score_scale": scale, **flip_tta.evaluate(records)}
            report["arms"].append(row)
            print(f"excl={exclusion:.2f} scale={scale:.2f} thr={row['threshold']:.4f} "
                  f"P={row['precision'] * 100:.2f} R={row['recall'] * 100:.2f} "
                  f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}", flush=True)
            key = (row["min_precision_recall"], row["precision"] + row["recall"], row["threshold"])
            if best is None or key > best[0]:
                best = (key, row, records)
    report["best"] = best[1]
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("BEST", json.dumps(best[1]), "\nwrote", args.out)


if __name__ == "__main__":
    main()
