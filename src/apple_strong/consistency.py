"""GT-anchored, stop-gradient original/flip consistency in a common frame.

This module never matches candidates, repairs GT or changes detector topology.
The caller must select at most one reliable proposal per known GT ID. Decoded
boxes and ROI support boxes are distinct: masks must retain the support on which
their logits were predicted. Comparing raw ROI cells across different supports
would be geometrically wrong, even after horizontally reversing the cells.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch.nn import functional as F


def horizontal_flip_boxes(boxes: torch.Tensor, width: float) -> torch.Tensor:
    """Return exclusive-extent XYXY boxes mirrored about the supplied width."""
    if (not isinstance(boxes, torch.Tensor) or not boxes.is_floating_point()
            or boxes.ndim != 2 or boxes.shape[1] != 4):
        raise ValueError("boxes must be floating[N,4] XYXY")
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0:
        raise ValueError("width must be positive and finite")
    return torch.stack((width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]), dim=1)


def _positive_boxes(boxes, count, name):
    if (not isinstance(boxes, torch.Tensor) or not boxes.is_floating_point()
            or tuple(boxes.shape) != (count, 4)
            or not bool(torch.isfinite(boxes).all())
            or bool((boxes[:, 2:] <= boxes[:, :2]).any())):
        raise ValueError(name + " must contain finite positive floating[N,4] XYXY extents")


def _validate_view(view, known, name):
    if not isinstance(view, Mapping):
        raise TypeError(name + " must be a mapping")
    ids = view.get("instance_ids")
    if (not isinstance(ids, (list, tuple)) or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError(name + " requires unique nonempty string GT instance_ids")
    if not set(ids) <= known:
        raise ValueError(name + " contains an ID not present in the supplied genuine GT mapping")
    n = len(ids)
    _positive_boxes(view.get("boxes"), n, name + ".boxes")
    _positive_boxes(view.get("roi_boxes"), n, name + ".roi_boxes")
    device = view["boxes"].device
    if view["roi_boxes"].device != device:
        raise ValueError(name + " box and ROI support devices differ")
    for key in ("visible_logits", "amodal_logits"):
        value = view.get(key)
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.ndim not in (3, 4) or value.shape[0] != n
                or value.ndim == 4 and value.shape[1] != 1
                or min(value.shape[-2:]) < 1 or value.device != device
                or not bool(torch.isfinite(value).all())):
            raise ValueError(name + "." + key + " must be finite floating[N,H,W] or [N,1,H,W]")
    return list(ids)


def _select(view, indices, *, flipped, scale, device, stop_gradient):
    def get(key):
        value = view[key].detach() if stop_gradient else view[key]
        return value.index_select(0, torch.tensor(indices, dtype=torch.int64, device=value.device)).to(device=device, dtype=torch.float32)
    boxes, supports = get("boxes") / scale, get("roi_boxes") / scale
    result = {"boxes": horizontal_flip_boxes(boxes, 1.) if flipped else boxes,
              "roi_boxes": horizontal_flip_boxes(supports, 1.) if flipped else supports}
    _positive_boxes(result["boxes"], len(indices), "normalized decoded boxes")
    _positive_boxes(result["roi_boxes"], len(indices), "normalized ROI supports")
    for branch in ("visible", "amodal"):
        logits = get(branch + "_logits")
        logits = logits[:, None] if logits.ndim == 3 else logits
        result[branch] = (logits.flip(-1) if flipped else logits).sigmoid()
    return result


def _canonical_grid(gt_boxes, size):
    axis = (torch.arange(size, dtype=torch.float32, device=gt_boxes.device) + .5) / size
    x = gt_boxes[:, 0, None] + (gt_boxes[:, 2] - gt_boxes[:, 0])[:, None] * axis
    y = gt_boxes[:, 1, None] + (gt_boxes[:, 3] - gt_boxes[:, 1])[:, None] * axis
    return torch.stack((x[:, None, :].expand(-1, size, -1), y[:, :, None].expand(-1, -1, size)), -1)


def _sample_roi(probability, supports, canonical):
    lower, extent = supports[:, None, None, :2], (supports[:, 2:] - supports[:, :2])[:, None, None]
    relative = (canonical - lower) / extent
    valid = ((relative >= 0) & (relative < 1)).all(-1).detach()
    sampled = F.grid_sample(probability, relative * 2 - 1, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
    return sampled, valid


def gt_anchored_consistency(
    moving: Mapping,
    reference: Mapping,
    gt_boxes: Mapping[str, torch.Tensor | Sequence[float]],
    *,
    image_hw: tuple[int, int],
    moving_flipped: bool = True,
    reference_flipped: bool = False,
    grid_size: int = 28,
    box_weight: float = 1.,
    visible_weight: float = 1.,
    amodal_weight: float = 1.,
    huber_beta: float = .05,
    coordinate_space: str = "pixels",
) -> dict[str, torch.Tensor]:
    """Compare shared genuine GT IDs after explicit flip-back and ROI reprojection.

    Views contain unique ``instance_ids``, decoded ``boxes``, true ``roi_boxes``
    supporting ``visible_logits``/``amodal_logits``. All coordinates use either
    input-image pixels or normalized image extents, as declared by
    ``coordinate_space``; GT boxes are in the unflipped common frame. The two
    views must use the same spatial resize. ``image_hw`` is that common frame,
    not automatically the native RGB size.

    Reference tensors and GT are always detached. Mask probabilities are
    sampled on a fixed canonical GT crop and compared only where both true ROI
    supports cover the sampled image location. Non-overlapping supports acquire
    zero mask consistency but still receive decoded-box Huber consistency.
    Each mask term averages valid crop points per instance, then instances.
    Existing GT losses remain the caller's responsibility and are never gated.

    Returned values are already multiplied by branch weights; sum once. Empty
    or disjoint ID inventories return moving-connected FP32 zeros. No classes,
    guesses, spatial exclusions, auxiliary heads or full-image masks are added.
    """
    if (not isinstance(image_hw, (tuple, list)) or len(image_hw) != 2
            or any(type(v) is not int or v < 1 for v in image_hw)):
        raise ValueError("image_hw must contain positive integer height,width")
    if type(moving_flipped) is not bool or type(reference_flipped) is not bool:
        raise ValueError("view flip flags must be explicit booleans")
    if type(grid_size) is not int or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")
    if coordinate_space not in ("pixels", "normalized"):
        raise ValueError("coordinate_space must be pixels or normalized")
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
           for w in (box_weight, visible_weight, amodal_weight)):
        raise ValueError("consistency weights must be finite nonnegative numbers")
    if isinstance(huber_beta, bool) or not isinstance(huber_beta, (int, float)) or not math.isfinite(huber_beta) or huber_beta <= 0:
        raise ValueError("huber_beta must be finite and positive")
    if not isinstance(gt_boxes, Mapping) or any(not isinstance(i, str) or not i for i in gt_boxes):
        raise ValueError("gt_boxes must map genuine nonempty string instance IDs to canonical boxes")
    known = set(gt_boxes)
    moving_ids = _validate_view(moving, known, "moving")
    reference_ids = _validate_view(reference, known, "reference")
    device = moving["boxes"].device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA consistency supported")
    with torch.autocast(device_type=device.type, enabled=False):
        zero = sum(moving[k].float().sum() * 0 for k in ("boxes", "roi_boxes", "visible_logits", "amodal_logits"))
        names = ("loss_consistency_box", "loss_consistency_visible", "loss_consistency_amodal")
        pairs = [i for i in moving_ids if i in set(reference_ids)]
        if not pairs:
            return dict.fromkeys(names, zero)
        height, width = image_hw
        scale = torch.tensor([width, height, width, height] if coordinate_space == "pixels" else [1.] * 4,
                             dtype=torch.float32, device=device)
        gt = torch.stack([torch.as_tensor(gt_boxes[i], device=device, dtype=torch.float32).detach() for i in pairs])
        _positive_boxes(gt, len(pairs), "GT canonical boxes")
        gt = gt / scale
        if bool((gt < 0).any()) or bool((gt > 1).any()):
            raise ValueError("GT canonical boxes must lie inside the declared image frame; no silent clipping")
        a = _select(moving, [moving_ids.index(i) for i in pairs], flipped=moving_flipped,
                    scale=scale, device=device, stop_gradient=False)
        b = _select(reference, [reference_ids.index(i) for i in pairs], flipped=reference_flipped,
                    scale=scale, device=device, stop_gradient=True)
        grid = _canonical_grid(gt, grid_size)
        losses = {names[0]: box_weight * F.smooth_l1_loss(a["boxes"], b["boxes"], beta=huber_beta)}
        for branch, weight in (("visible", visible_weight), ("amodal", amodal_weight)):
            predicted, valid_a = _sample_roi(a[branch], a["roi_boxes"], grid)
            target, valid_b = _sample_roi(b[branch], b["roi_boxes"], grid)
            valid = (valid_a & valid_b).float()
            point_loss = (predicted - target).square() * valid
            losses["loss_consistency_" + branch] = weight * (point_loss.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1.)).mean()
        return losses
