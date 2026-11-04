"""Post-fusion NMS sweep with exact validation matching."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms

import flip_tta


def apply_nms(records, iou_threshold):
    result = []
    for record in records:
        boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        keep = nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(),
                   iou_threshold).numpy()
        result.append({"image_id": record["image_id"], "gt": record["gt"],
                       "pred": {"boxes": boxes[keep].tolist(), "scores": scores[keep].tolist()}})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ious", type=float, nargs="+", default=[.3, .4, .5, .55, .6])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = flip_tta.load_records(args.predictions)
    report = {"baseline": flip_tta.evaluate(records), "arms": []}
    best = None
    for threshold in args.ious:
        filtered = apply_nms(records, threshold)
        row = {"nms_iou": threshold, **flip_tta.evaluate(filtered)}
        report["arms"].append(row)
        print(f"NMS={threshold:.2f} thr={row['threshold']:.4f} P={row['precision'] * 100:.2f} "
              f"R={row['recall'] * 100:.2f} TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}", flush=True)
        key = (row["min_precision_recall"], row["precision"] + row["recall"], row["threshold"])
        if best is None or key > best[0]:
            best = (key, row)
    report["best"] = best[1]
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("BEST", json.dumps(best[1]), "\nwrote", args.out)


if __name__ == "__main__":
    main()
