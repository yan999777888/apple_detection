"""Compare box NMS with cluster-weighted box voting on real pre-NMS candidates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms

import flip_tta


def box_nms(records, iou):
    result = []
    for record in records:
        boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
        keep = nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), iou).numpy()
        result.append({"image_id": record["image_id"], "gt": record["gt"],
                       "pred": {"boxes": boxes[keep].tolist(), "scores": scores[keep].tolist()}})
    return result


def vote(record, cluster_iou, weight_power, score_mode, post_nms):
    boxes = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(record["pred"]["scores"], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    taken = np.zeros(len(order), dtype=bool)
    out_boxes, out_scores = [], []
    for position, index in enumerate(order):
        if taken[index]:
            continue
        members = [index]
        taken[index] = True
        overlaps = flip_tta.metrics.box_iou(boxes[index:index + 1], boxes[order[position + 1:]])[0]
        for offset, overlap in enumerate(overlaps):
            candidate = order[position + 1 + offset]
            if not taken[candidate] and overlap >= cluster_iou:
                members.append(candidate)
                taken[candidate] = True
        member_scores = scores[np.asarray(members)]
        weights = np.power(member_scores, weight_power)
        weights = weights / weights.sum()
        out_boxes.append((boxes[np.asarray(members)] * weights[:, None]).sum(axis=0))
        out_scores.append(member_scores.max() if score_mode == "max" else member_scores.mean())
    result = np.asarray(out_boxes).reshape(-1, 4)
    result_scores = np.asarray(out_scores)
    if post_nms:
        keep = nms(torch.from_numpy(result).float(), torch.from_numpy(result_scores).float(),
                   post_nms).numpy()
        result, result_scores = result[keep], result_scores[keep]
    return {"image_id": record["image_id"], "gt": record["gt"],
            "pred": {"boxes": result.tolist(), "scores": result_scores.tolist()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--published", required=True)
    parser.add_argument("--cluster-ious", type=float, nargs="+", default=[.45, .55, .65])
    parser.add_argument("--weight-powers", type=float, nargs="+", default=[1., 2.])
    parser.add_argument("--post-nms", type=float, nargs="+", default=[0, .5])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    candidates = flip_tta.load_records(args.candidates)
    published = flip_tta.load_records(args.published)
    report = {"images": len(candidates), "published": flip_tta.evaluate(published),
              "box_nms": [dict(iou=value, **flip_tta.evaluate(box_nms(candidates, value)))
                          for value in (.4, .5, .6)],
              "vote_arms": []}
    best = None
    for cluster_iou in args.cluster_ious:
        for power in args.weight_powers:
            for post in args.post_nms:
                for score_mode in ("max", "mean"):
                    records = [vote(row, cluster_iou, power, score_mode, post) for row in candidates]
                    row = {"cluster_iou": cluster_iou, "weight_power": power,
                           "score_mode": score_mode, "post_nms": post, **flip_tta.evaluate(records)}
                    report["vote_arms"].append(row)
                    print(f"cluster={cluster_iou:.2f} power={power:.0f} score={score_mode:4s} "
                          f"post={post:.2f} P={row['precision'] * 100:.2f} "
                          f"R={row['recall'] * 100:.2f} TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}",
                          flush=True)
                    key = (row["min_precision_recall"], row["precision"] + row["recall"])
                    if best is None or key > best[0]:
                        best = (key, row)
    report["best_vote"] = best[1]
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("BEST", json.dumps(best[1]), "\nwrote", args.out)


if __name__ == "__main__":
    main()
