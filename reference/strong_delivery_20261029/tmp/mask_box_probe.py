"""Probe whether the trained masks can localize an apple better than the box head.

The evaluator scores boxes at IoU 0.5. The visible mask V is trained on the same
visible apple that the GT box encloses, so its bounding box is an independent
estimate of that extent. This probe never runs the network: it reads the cached
56x56 ROI probabilities for the published 1024-scale single view and asks
whether V-geometry, A-geometry or a blend recovers more IoU>=0.5 boxes than the
regressed box, on exactly the same images and scores.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))

import flip_tta  # noqa: E402

SIDE = 56


def mask_bounds(probability, threshold):
    """Return per-ROI fractional XYXY bounds of the thresholded mask."""
    binary = probability > threshold
    rows = binary.any(axis=1)
    cols = binary.any(axis=2)
    present = rows.any(axis=1)
    x1 = rows.argmax(axis=1)
    x2 = SIDE - rows[:, ::-1].argmax(axis=1)
    y1 = cols.argmax(axis=1)
    y2 = SIDE - cols[:, ::-1].argmax(axis=1)
    bounds = np.stack([x1, y1, x2, y2], axis=1).astype(np.float64) / SIDE
    bounds[~present] = 0.
    return bounds, present


def to_image(bounds, boxes):
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    return np.stack([boxes[:, 0] + bounds[:, 0] * width,
                     boxes[:, 1] + bounds[:, 1] * height,
                     boxes[:, 0] + bounds[:, 2] * width,
                     boxes[:, 1] + bounds[:, 3] * height], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--mask-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[.5])
    parser.add_argument("--blends", nargs="+", type=float, default=[.25, .5])
    args = parser.parse_args()

    arms = [("box", 0., 0.)]
    for threshold in args.thresholds:
        arms.append(("v", threshold, 0.))
        arms.append(("a", threshold, 0.))
        for blend in args.blends:
            arms.append(("blend", threshold, blend))

    records = flip_tta.load_records(args.predictions)
    per_arm = {arm: [] for arm in arms}
    used = 0
    for index, record in enumerate(records):
        path = os.path.join(args.mask_cache, f"roi_{index:06d}.npz")
        if not os.path.exists(path):
            continue
        boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        with np.load(path) as payload:
            masks = {key: payload[key] for key in ("visible", "amodal")}
        if masks["visible"].shape[0] != len(boxes):
            raise ValueError("Mask cache and prediction count differ")
        der = {}
        for kind in ("visible", "amodal"):
            for threshold in args.thresholds:
                bounds, present = mask_bounds(masks[kind], threshold)
                der[(kind, threshold)] = (to_image(bounds, boxes), present)
        for arm in arms:
            kind, threshold, blend = arm
            if kind == "box":
                fused = boxes.copy()
            elif kind in ("v", "a"):
                key = ("visible" if kind == "v" else "amodal", threshold)
                derived, present = der[key]
                fused = boxes.copy()
                fused[present] = derived[present]
            else:
                derived, present = der[("visible", threshold)]
                fused = boxes.copy()
                fused[present] = (1 - blend) * boxes[present] + blend * derived[present]
            per_arm[arm].append({"image_id": record["image_id"], "gt": record["gt"],
                                 "pred": {"boxes": fused.tolist(),
                                          "scores": list(record["pred"]["scores"])}})
        used += 1

    print(f"cached images={used}/{len(records)}", flush=True)
    report = {"images": used, "arms": []}
    for arm in arms:
        kind, threshold, blend = arm
        summary = flip_tta.evaluate(per_arm[arm])
        row = dict(kind=kind, mask_threshold=threshold, blend=blend,
                   **{key: summary[key] for key in ("threshold", "precision", "recall",
                                                    "min_precision_recall", "tp", "fp", "fn")})
        report["arms"].append(row)
        print(f"{kind:5s} t={threshold:.2f} blend={blend:.2f} thr={row['threshold']:.4f} "
              f"P={row['precision'] * 100:.2f} R={row['recall'] * 100:.2f} "
              f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}", flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
