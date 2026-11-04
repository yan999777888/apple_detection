"""Twelve-feature geometric / cross-view re-ranker for the two-view TTA pool.

The published pipeline fuses two views with a fixed rule (mean score, score
weighted box). This probe keeps the identical candidate pool and boxes, then
replaces the score with a supervised logistic model over features that the
fixed rule never looks at: asymmetric neighbour containment, mask geometry
self-consistency, cross-view agreement and image context.

Fitting never touches validation labels of the images it is scored on: features
are built from predictions only, labels are built per image, and the model is
fitted on one half of VAL and scored on the other half in both directions.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))

import flip_tta  # noqa: E402

SIDE = 56
CLUSTER_IOU = 0.35


def area(boxes):
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def build_image(record, flipped, flipped_boxes, masks, threshold=.5):
    """Return fused boxes/scores and their feature rows for one image."""
    boxes_a = np.asarray(record["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
    scores_a = np.asarray(record["pred"]["scores"], dtype=np.float64)
    boxes_b = np.asarray(flipped_boxes, dtype=np.float64).reshape(-1, 4)
    scores_b = np.asarray(flipped["pred"]["scores"], dtype=np.float64)
    boxes = np.concatenate([boxes_a, boxes_b])
    scores = np.concatenate([scores_a, scores_b])
    views = np.concatenate([np.zeros(len(boxes_a), dtype=np.int64),
                            np.ones(len(boxes_b), dtype=np.int64)])
    if not len(boxes):
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 13))
    order = np.argsort(-scores, kind="stable")
    boxes, scores, views = boxes[order], scores[order], views[order]
    local = np.concatenate([np.arange(len(boxes_a)), np.arange(len(boxes_b))])[order]

    taken = np.zeros(len(boxes), dtype=bool)
    clusters = []
    for index in range(len(boxes)):
        if taken[index]:
            continue
        members = [index]
        used = {int(views[index])}
        taken[index] = True
        overlaps = flip_tta.metrics.box_iou(boxes[index:index + 1], boxes[index + 1:])[0]
        for offset, overlap in enumerate(overlaps):
            candidate = index + 1 + offset
            if not taken[candidate] and int(views[candidate]) not in used and overlap >= CLUSTER_IOU:
                members.append(candidate)
                used.add(int(views[candidate]))
                taken[candidate] = True
        clusters.append(members)

    fused_boxes = np.empty((len(clusters), 4))
    fused_scores = np.empty(len(clusters))
    features, has_mask = [], []
    for position, members in enumerate(clusters):
        member_boxes = boxes[members]
        member_scores = scores[members]
        weights = member_scores / member_scores.sum()
        fused_boxes[position] = (member_boxes * weights[:, None]).sum(axis=0)
        fused_scores[position] = member_scores.mean()
        agreement = (flip_tta.metrics.box_iou(member_boxes[:1], member_boxes[1:])[0, 0]
                     if len(members) > 1 else 0.)
        delta = float(member_scores.max() - member_scores.min()) if len(members) > 1 else 0.
        # Mask evidence comes from the unflipped member only; its cache rows are
        # aligned with the unflipped prediction order.
        primary = next((m for m in members if views[m] == 0), None)
        fill = ratio = log_area = components = box_agreement = np.nan
        if primary is not None and masks is not None:
            row = int(local[primary])
            visible, amodal = masks
            if row < len(visible):
                visible_area = float((visible[row] > .5).sum()) / (SIDE * SIDE)
                amodal_area = float((amodal[row] > .5).sum()) / (SIDE * SIDE)
                box_area = max(float(area(fused_boxes[position:position + 1])[0]), 1e-6)
                # ROI covers exactly the fused box, so the mask pixel fraction is
                # already the area fraction of that box: a filled apple is ~pi/4.
                fill = amodal_area
                ratio = visible_area / amodal_area if amodal_area > 0 else 0.
                log_area = float(np.log(visible_area * box_area + 1.))
                binary = amodal[row] > .5
                if binary.any():
                    rows = np.where(binary.any(axis=1))[0]
                    cols = np.where(binary.any(axis=0))[0]
                    width = fused_boxes[position, 2] - fused_boxes[position, 0]
                    height = fused_boxes[position, 3] - fused_boxes[position, 1]
                    box = np.array([[fused_boxes[position, 0] + rows[0] / SIDE * width,
                                     fused_boxes[position, 1] + cols[0] / SIDE * height,
                                     fused_boxes[position, 0] + (rows[-1] + 1) / SIDE * width,
                                     fused_boxes[position, 1] + (cols[-1] + 1) / SIDE * height]])
                    box_agreement = float(flip_tta.metrics.box_iou(fused_boxes[position:position + 1], box)[0, 0])
                components = float(ndimage.label(visible[row] > .5)[1])
        has_mask.append(0. if primary is None or masks is None else 1.)
        features.append([len(members), agreement, delta, member_scores.max(), member_scores.min(),
                         fill, ratio, log_area, components, box_agreement, has_mask[-1]])
    features = np.asarray(features, dtype=np.float64)
    features = np.where(np.isfinite(features), features, 0.)

    sizes = np.sqrt(np.clip(area(fused_boxes), 1e-6, None))
    overlaps = flip_tta.metrics.box_iou(fused_boxes, fused_boxes)
    containment = overlaps * area(fused_boxes)[None, :] / np.clip(area(fused_boxes)[:, None], 1e-6, None)
    coverage = overlaps * area(fused_boxes)[None, :] / np.clip(area(fused_boxes)[None, :], 1e-6, None)
    np.fill_diagonal(containment, 0.)
    np.fill_diagonal(coverage, 0.)
    higher = fused_scores[None, :] > fused_scores[:, None]
    masked_in = np.where(higher, containment, 0.).max(axis=1) if len(fused_scores) else np.zeros(0)
    masked_out = np.where(higher, coverage, 0.).max(axis=1) if len(fused_scores) else np.zeros(0)
    with np.errstate(divide="ignore"):
        centre = np.stack([(fused_boxes[:, 0] + fused_boxes[:, 2]) / 2,
                           (fused_boxes[:, 1] + fused_boxes[:, 3]) / 2], axis=1)
        distance = np.sqrt(((centre[:, None, :] - centre[None, :, :]) ** 2).sum(-1))
    distance = np.where(higher, distance, np.inf).min(axis=1) / sizes
    high_count = float((fused_scores > .65).sum())
    relative = fused_scores / (fused_scores.max() + 1e-9)
    extra = np.stack([masked_in, masked_out, np.clip(distance, 0, 50),
                      high_count * np.ones(len(fused_scores)), relative], axis=1)
    return fused_boxes, fused_scores, np.concatenate([features, extra], axis=1)


def labels_for(boxes, gt, iou=.5):
    if not len(boxes):
        return np.zeros(0)
    if not len(gt):
        return np.zeros(len(boxes))
    return (flip_tta.metrics.box_iou(boxes, gt).max(axis=1) >= iou).astype(np.float64)


def fit_logistic(features, labels, l2=1., iterations=30):
    """Standardized L2 logistic regression solved by IRLS (Newton steps).

    Only the ranking matters because the score is thresholded globally, so the
    fit is deliberately plain: no sampling, no class weights, 16 coefficients.
    """
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.
    design = np.concatenate([(features - mean) / scale, np.ones((len(features), 1))], axis=1)
    weights = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * l2
    penalty[-1, -1] = 0.
    for _ in range(iterations):
        probability = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
        curvature = np.clip(probability * (1 - probability), 1e-6, None)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        gradient = design.T @ (probability - labels) + penalty @ weights
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.abs(step).max()) < 1e-8:
            break
    return {"mean": mean, "scale": scale, "weights": weights}


def apply_logistic(model, features):
    design = np.concatenate([(features - model["mean"]) / model["scale"],
                             np.ones((len(features), 1))], axis=1)
    logits = design @ model["weights"]
    return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--flipped", required=True)
    parser.add_argument("--mask-cache", default="")
    parser.add_argument("--metadata", default="", help="Full-VAL metadata with original_hw per image")
    parser.add_argument("--out", required=True)
    parser.add_argument("--l2", type=float, nargs="+", default=[1.0, 10.0])
    args = parser.parse_args()

    base = flip_tta.load_records(args.base)
    flipped = flip_tta.load_records(args.flipped)
    images = []
    for index, (record, other) in enumerate(zip(base, flipped)):
        # The flip inference pass already maps boxes back into the original
        # image frame, so the two views are directly comparable here.
        boxes_b = np.asarray(other["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
        masks = None
        if args.mask_cache:
            path = os.path.join(args.mask_cache, f"roi_{index:06d}.npz")
            if os.path.exists(path):
                with np.load(path) as payload:
                    masks = (payload["visible"], payload["amodal"])
        fused_boxes, fused_scores, features = build_image(record, other, boxes_b, masks)
        images.append({"image_id": record["image_id"], "gt": record["gt"],
                       "boxes": fused_boxes, "scores": fused_scores, "features": features,
                       "labels": labels_for(fused_boxes, np.asarray(record["gt"]["boxes"], float).reshape(-1, 4))})
        if (index + 1) % 200 == 0:
            print(f"features {index + 1}/{len(base)}", flush=True)

    report = {"schema": "apple_two_view_rerank_v1", "images": len(images),
              "baseline": flip_tta.evaluate([{"image_id": row["image_id"], "gt": row["gt"],
                                              "pred": {"boxes": row["boxes"].tolist(),
                                                       "scores": row["scores"].tolist()}}
                                             for row in images]), "arms": []}
    print("fused baseline", round(report["baseline"]["precision"], 6),
          round(report["baseline"]["recall"], 6), report["baseline"]["tp"], flush=True)

    parity = np.array([index % 2 for index in range(len(images))])
    for l2 in args.l2:
        fold_results = []
        for fold in (0, 1):
            train = [row for row, p in zip(images, parity) if p != fold]
            test = [row for row, p in zip(images, parity) if p == fold]
            features = np.concatenate([row["features"] for row in train])
            labels = np.concatenate([row["labels"] for row in train])
            model = fit_logistic(np.clip(features, -1e6, 1e6), labels, l2=l2)
            scored = []
            for row in test:
                if not len(row["boxes"]):
                    continue
                probability = apply_logistic(model, np.clip(row["features"], -1e6, 1e6))
                scored.append({"image_id": row["image_id"], "gt": row["gt"],
                               "pred": {"boxes": row["boxes"].tolist(),
                                        "scores": probability.tolist()}})
            summary = flip_tta.evaluate(scored)
            reference = flip_tta.evaluate([{"image_id": row["image_id"], "gt": row["gt"],
                                            "pred": {"boxes": row["boxes"].tolist(),
                                                     "scores": row["scores"].tolist()}}
                                           for row in test if len(row["boxes"])])
            fold_results.append({"rerank": summary, "fused_reference": reference})
            print(f"l2={l2:5.1f} fold={fold} rerank P={summary['precision'] * 100:.2f} "
                  f"R={summary['recall'] * 100:.2f} TP/FP/FN={summary['tp']}/{summary['fp']}/{summary['fn']} "
                  f"| same-half fused P={reference['precision'] * 100:.2f} R={reference['recall'] * 100:.2f} "
                  f"TP/FP/FN={reference['tp']}/{reference['fp']}/{reference['fn']}", flush=True)
        mean = float(np.mean([row["rerank"]["min_precision_recall"] for row in fold_results]))
        report["arms"].append({"l2": l2, "folds": fold_results, "mean_min_pr": mean})
        print(f"l2={l2:5.1f} held-out mean min(P,R) = {mean * 100:.2f}", flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
