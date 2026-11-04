"""TRAIN quality labels aligned with original-resolution amodal-box recovery.

This module is separate from the live quality experiment. PApple labels use
one shared assignment that first maximizes the number of box IoU >= .5
pairs, then maximizes summed box IoU. Every GT is assigned, even if its best
available query is below .5. MinneApple retains the original V-only matcher.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from .evaluate import POSTPROCESS_PATH, _module
from .quality import _box, _box_iou, quality_targets


def _aligned_box_assignment(overlaps):
    """Lexicographic cardinality/IoU assignment in fixed query and GT order.

    One extra qualifying edge outweighs all possible total-IoU differences:
    there are N selected pairs, each with IoU in [0, 1], and its bonus is N+1.
    SciPy receives fixed ascending query/GT order, with no random or floating
    tie perturbation; exact objective ties use its deterministic solver order.
    """
    overlaps = np.asarray(overlaps, dtype=np.float64)
    if overlaps.ndim != 2 or overlaps.shape[0] != 100 or overlaps.shape[1] > 100:
        raise ValueError("Expected box IoU[100,N] with N <= 100")
    if not np.isfinite(overlaps).all() or np.any((overlaps < 0) | (overlaps > 1)):
        raise ValueError("Box IoU must be finite in [0,1]")
    count = overlaps.shape[1]
    if not count:
        return [], []
    utility = (count + 1) * (overlaps >= .5) + overlaps
    queries, targets = linear_sum_assignment(-utility)
    if len(queries) != count or set(targets.tolist()) != set(range(count)):
        raise RuntimeError("Aligned assignment failed to retain every GT")
    return queries.tolist(), targets.tolist()


@torch.no_grad()
def aligned_quality_targets(original, target, *, criterion, generator=None):
    """Return the original quality-target schema, without filtering TRAIN queries.

    The criterion retains responsibility for annotation validity, finite logits,
    query capacity, and original CPU GT validation. Unknown amodal annotations
    take the unchanged V-only path, so A affects neither matching nor labels.
    """
    _, visible, amodal, _, count, known_amodal = criterion._validate(original, [target])
    if not known_amodal:
        return quality_targets(original, target, criterion=criterion, generator=generator)

    result = torch.zeros((1, 100, 3), dtype=torch.float32)
    valid = torch.ones(3, dtype=torch.bool)
    if not count:
        return {"quality_targets": result, "quality_valid": valid, "GT_count": 0,
                "query_indices": [], "target_indices": []}

    post = _module(POSTPROCESS_PATH, "_apple_aligned_quality_target_postprocess")
    original_hw = tuple(target["original_hw"])
    prediction_boxes = []
    # Keep only 100 small boxes; never retain all original-resolution masks.
    for start in range(0, 100, 8):
        queries = list(range(start, min(start + 8, 100)))
        masks = post._binary_chunk(amodal, queries, original_hw, torch, F.interpolate)
        prediction_boxes.extend(_box(mask) for mask in masks)
        del masks
    truth_boxes = [_box(mask) for mask in target["amodal_masks"]]
    overlaps = np.asarray([[_box_iou(predicted, truth) for truth in truth_boxes]
                           for predicted in prediction_boxes], dtype=np.float64)
    queries, targets = _aligned_box_assignment(overlaps)

    for prediction, key, column in ((visible, "visible_masks", 1), (amodal, "amodal_masks", 2)):
        for start in range(0, len(queries), 8):
            chosen_queries, chosen_targets = queries[start:start + 8], targets[start:start + 8]
            masks = post._binary_chunk(prediction, chosen_queries, original_hw, torch, F.interpolate)
            for local, (query, truth_index) in enumerate(zip(chosen_queries, chosen_targets, strict=True)):
                predicted, truth = masks[local], target[key][truth_index].bool()
                union = torch.logical_or(predicted, truth).sum().item()
                intersection = torch.logical_and(predicted, truth).sum().item()
                result[0, query, column] = intersection / union if union else 0.
                if column == 2:
                    result[0, query, 0] = overlaps[query, truth_index]
            del masks
    return {"quality_targets": result, "quality_valid": valid, "GT_count": count,
            "query_indices": queries, "target_indices": targets}
