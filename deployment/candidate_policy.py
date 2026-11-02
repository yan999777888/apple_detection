"""Deterministic mask-before candidate selection, without ground truth.

The public function accepts floating tensors and returns original int64
indices on the same device. Selection uses float32 scores and, for box NMS,
float32 decoded XYXY boxes. It does not mutate boxes or model probabilities.
No selection rule here is claimed to preserve final detector accuracy.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
import struct

import torch
from torchvision.ops import nms


_MAX_EXACT_FLOAT32_INTEGER = 2**24


def _probability(value, name, *, float32=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be None or a finite number in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be None or a finite number in [0, 1]")
    # Use the same scalar representation as comparison with model FP32 scores.
    return struct.unpack("<f", struct.pack("<f", value))[0] if float32 else value


def policy_description(*, score_threshold=None, box_nms_iou=None, topk=None):
    """Validate options and expose their effective numerical semantics."""
    score = _probability(score_threshold, "score_threshold", float32=True)
    iou = _probability(box_nms_iou, "box_nms_iou")
    if topk is not None and (isinstance(topk, bool) or not isinstance(topk, Integral) or topk < 1):
        raise ValueError("topk must be None or a positive integer")
    topk = int(topk) if topk is not None else None
    return {
        "score_threshold": score,
        "score_threshold_requested": float(score_threshold) if score_threshold is not None else None,
        "box_nms_iou": iou,
        "topk": topk,
        "enabled": any(value is not None for value in (score, iou, topk)),
        "order": "score_filter_then_stable_descending_score_then_box_nms_then_topk",
        "all_disabled_order": "original_input_order",
        "score_comparison": ">=",
        "score_dtype": "float32",
        "box_nms_dtype": "float32",
        "box_convention": "continuous_xyxy_no_plus_one",
        "nms_suppression": "IoU > box_nms_iou",
        "tie_order": "original_input_index",
        "nms_tie_strategy": "strictly_decreasing_exact_float32_integer_ranks_after_stable_score_sort",
        "nms_rank_limit": _MAX_EXACT_FLOAT32_INTEGER,
        "gt_used": False,
    }


def _inputs(decoded_boxes, initial_scores):
    if not isinstance(decoded_boxes, torch.Tensor) or not isinstance(initial_scores, torch.Tensor):
        raise TypeError("decoded_boxes and initial_scores must be torch tensors")
    if decoded_boxes.ndim != 2 or decoded_boxes.shape[1] != 4:
        raise ValueError("decoded_boxes must have shape [N,4]")
    if initial_scores.ndim != 1 or len(initial_scores) != len(decoded_boxes):
        raise ValueError("initial_scores must have shape [N] matching boxes")
    if decoded_boxes.device != initial_scores.device:
        raise ValueError("boxes and initial scores must share a device")
    if not decoded_boxes.is_floating_point() or not initial_scores.is_floating_point():
        raise ValueError("boxes and initial scores must be floating tensors")
    boxes, scores = decoded_boxes.float(), initial_scores.float()
    if not bool(torch.isfinite(boxes).all() & torch.isfinite(scores).all()):
        raise ValueError("boxes and scores must be finite after conversion to float32")
    if not bool(((scores >= 0) & (scores <= 1)).all()):
        raise ValueError("initial apple scores must be probabilities in [0,1]")
    if not bool((boxes[:, 2:] >= boxes[:, :2]).all()):
        raise ValueError("decoded boxes must have nonnegative width and height")
    return boxes, scores


def select_candidates(decoded_boxes: torch.Tensor, initial_scores: torch.Tensor, *,
                      score_threshold=None, box_nms_iou=None, topk=None) -> torch.Tensor:
    """Select original indices; each optional rule can be enabled alone.

    With every option disabled, the original input order is retained. With any
    option enabled, ties in float32 initial scores retain original input order.
    Box NMS is applied to decoded boxes before the top-K limit. It uses unique
    rank scores internally, so equal model probabilities cannot leave the NMS
    survivor order to backend-dependent tie handling. This does not promise
    identical CPU/GPU IoU rounding at a suppression boundary.

    Zero-area boxes are legal and are not automatically removed; invalid or
    non-finite inputs are rejected. Empty [0,4]/[0] inputs are legal.
    """
    policy = policy_description(score_threshold=score_threshold,
                                box_nms_iou=box_nms_iou, topk=topk)
    boxes, scores = _inputs(decoded_boxes, initial_scores)
    indices = torch.arange(len(scores), device=scores.device, dtype=torch.int64)
    if not policy["enabled"] or not len(indices):
        return indices
    if policy["score_threshold"] is not None:
        indices = indices[scores >= policy["score_threshold"]]
    if not len(indices):
        return indices
    order = torch.argsort(scores[indices], descending=True, stable=True)
    indices = indices[order]
    if policy["box_nms_iou"] is not None:
        if len(indices) > _MAX_EXACT_FLOAT32_INTEGER:
            raise ValueError("NMS rank count exceeds exact float32 integer range")
        rank_scores = torch.arange(len(indices), 0, -1, device=scores.device, dtype=torch.float32)
        selected = nms(boxes[indices], rank_scores, policy["box_nms_iou"])
        indices = indices[selected]
    if policy["topk"] is not None:
        indices = indices[:policy["topk"]]
    return indices
