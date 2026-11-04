"""Context ROI fusion and learned box refinements, independent of GT inference."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _image_shape(image_shape):
    if (not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2
            or any(type(value) is not int or value < 1 for value in image_shape)):
        raise ValueError("image_shape must be positive integer [H,W]")
    height, width = image_shape
    try:
        finite = all(math.isfinite(float(value)) and value <= torch.finfo(torch.float32).max
                     for value in (height, width))
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("image_shape must fit finite FP32 coordinates")
    return height, width


def _boxes(boxes, name="boxes"):
    if (not isinstance(boxes, torch.Tensor) or not boxes.is_floating_point()
            or boxes.ndim != 2 or boxes.shape[1] != 4):
        raise ValueError(f"{name} must be floating [N,4] XYXY")
    result = boxes.float()
    extent = result[:, 2:] - result[:, :2]
    if (not bool(torch.isfinite(result).all()) or not bool(torch.isfinite(extent).all())
            or not bool((extent > 0).all())):
        raise ValueError(f"{name} must have finite positive FP32 extents")
    return result


def _clip_predicted_boxes(boxes, image_shape):
    """Clip predictions without dropping ROIs or repairing already valid boxes.

    Only a collapsed extent after clipping is replaced with a representable
    positive extent at the boundary. GT never passes through this function.
    """
    height, width = _image_shape(image_shape)
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("Decoded box predictions must be finite")
    epsilon = torch.finfo(torch.float32).eps

    def axis(first, last, limit):
        first, last = first.clamp(0, limit), last.clamp(0, limit)
        collapsed = last <= first
        minimum = max(1e-4, 4 * epsilon * limit)
        repaired_first = first.clamp(max=limit - minimum)
        repaired_last = torch.maximum(last, repaired_first + minimum).clamp(max=limit)
        return (torch.where(collapsed, repaired_first, first),
                torch.where(collapsed, repaired_last, last))

    x1, x2 = axis(boxes[:, 0], boxes[:, 2], width)
    y1, y2 = axis(boxes[:, 1], boxes[:, 3], height)
    result = torch.stack((x1, y1, x2, y2), dim=1)
    if not bool((result[:, 2:] > result[:, :2]).all()):
        raise ValueError("Cannot represent positive clipped FP32 box extents")
    return result


def expand_context_boxes(boxes, image_shape, factor=2.2):
    """Expand each predicted ROI about its center, retaining its row identity.

    Returns FP32 half-open XYXY coordinates in [0,W] x [0,H]. Neither the
    original boxes nor annotation tensors are changed.
    """
    _image_shape(image_shape)
    try:
        valid_factor = (type(factor) in (int, float) and math.isfinite(factor)
                        and 1 <= factor <= torch.finfo(torch.float32).max)
    except OverflowError:
        valid_factor = False
    if not valid_factor:
        raise ValueError("Context expansion factor must be finite FP32 and at least1")
    boxes = _boxes(boxes)
    growth = (boxes[:, 2:] - boxes[:, :2]) * ((factor - 1) * .5)
    expanded = torch.cat((boxes[:, :2] - growth, boxes[:, 2:] + growth), dim=1)
    return _clip_predicted_boxes(expanded, image_shape)


class ContextROIFusion(nn.Module):
    """Learn a residual from native and expanded-context ROI features.

    Both inputs must already use the same pooled spatial shape. The result
    stays in native ROI coordinates; a zero projection starts as identity.
    """

    def __init__(self, channels=256):
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        self.context_projection = nn.Conv2d(2 * channels, channels, 1)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)

    def forward(self, native, context):
        if (not isinstance(native, torch.Tensor) or not isinstance(context, torch.Tensor)
                or not native.is_floating_point() or not context.is_floating_point()
                or native.ndim != 4 or native.shape != context.shape
                or native.shape[1] != self.channels or min(native.shape[-2:]) < 1
                or native.device != context.device or native.dtype != context.dtype):
            raise ValueError("Native/context features must share floating [N,C,H,W], device and dtype")
        if not bool(torch.isfinite(native).all() and torch.isfinite(context).all()):
            raise ValueError("Native/context features must be finite")
        return native + self.context_projection(torch.cat((native, context), dim=1))


class ContextCascadeBoxHead(nn.Module):
    """Independent learned residual box heads; stage matching is caller-owned."""

    def __init__(self, channels=256, stages=3):
        super().__init__()
        if type(channels) is not int or channels < 1 or type(stages) is not int or stages < 1:
            raise ValueError("channels and stages must be positive integers")
        self.channels, self.num_stages = channels, stages
        self.box_coder_weights = (10., 10., 5., 5.)
        self.box_xform_clip = math.log(1000. / 16.)
        self.stage_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(channels, 256), nn.ReLU(), nn.Linear(256, 4))
            for _ in range(stages)])
        for head in self.stage_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward_stage(self, pooled, boxes, image_shape, stage):
        if type(stage) is not int or not 0 <= stage < self.num_stages:
            raise ValueError("stage must index an existing cascade head")
        _image_shape(image_shape)
        boxes = _boxes(boxes)
        if (not isinstance(pooled, torch.Tensor) or not pooled.is_floating_point()
                or pooled.ndim != 4 or pooled.shape[0] != len(boxes)
                or pooled.shape[1] != self.channels or min(pooled.shape[-2:]) < 1
                or pooled.device != boxes.device):
            raise ValueError("Pooled features must be floating [N,C,H,W] on the box device")
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("Pooled features must be finite")
        delta = self.stage_heads[stage](pooled.mean((-1, -2))).float()
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Cascade regression deltas must be finite")
        widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        wx, wy, ww, wh = self.box_coder_weights
        dx, dy = delta[:, 0] / wx, delta[:, 1] / wy
        dw = (delta[:, 2] / ww).clamp(max=self.box_xform_clip)
        dh = (delta[:, 3] / wh).clamp(max=self.box_xform_clip)
        shift_x, shift_y = dx * widths, dy * heights
        grow_x, grow_y = .5 * widths * torch.expm1(dw), .5 * heights * torch.expm1(dh)
        # Equivalent to center + exp(size) BoxCoder decoding, expressed as a
        # residual so zero deltas preserve valid FP32 boxes exactly.
        refined = torch.stack((boxes[:, 0] + shift_x - grow_x,
                               boxes[:, 1] + shift_y - grow_y,
                               boxes[:, 2] + shift_x + grow_x,
                               boxes[:, 3] + shift_y + grow_y), dim=1)
        return _clip_predicted_boxes(refined, image_shape)


def matched_box_refinement_loss(refined, GTboxes, image_shape):
    """Aligned GIoU plus mean SmoothL1 on image-normalized coordinates.

    The caller supplies only matched genuine known-A positives. No GT is
    accepted by either inference head. Fixed GT must be valid as supplied;
    this loss never clips, repairs or reassigns it. SmoothL1 beta is1/9.
    """
    height, width = _image_shape(image_shape)
    refined = _boxes(refined, "refined")
    if isinstance(GTboxes, torch.Tensor) and GTboxes.requires_grad:
        raise ValueError("GTboxes must be fixed annotation coordinates")
    target = _boxes(GTboxes, "GTboxes")
    if refined.shape != target.shape or refined.device != target.device:
        raise ValueError("Refined and matched GT boxes must share shape and device")
    limits = target.new_tensor([width, height, width, height])
    if not bool(((target >= 0) & (target <= limits)).all()):
        raise ValueError("GTboxes must lie inside the original image bounds")
    if not len(refined):
        return refined.sum() * 0
    # GIoU is invariant to this common per-axis scaling. Normalizing before
    # area products also avoids overflowing otherwise finite pixel extents.
    predicted, truth = refined / limits, target / limits
    intersection_extent = (torch.minimum(predicted[:, 2:], truth[:, 2:])
                           - torch.maximum(predicted[:, :2], truth[:, :2])).clamp_min(0)
    intersection = intersection_extent.prod(dim=1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).prod(dim=1)
    target_area = (truth[:, 2:] - truth[:, :2]).prod(dim=1)
    union = predicted_area + target_area - intersection
    enclosure_extent = (torch.maximum(predicted[:, 2:], truth[:, 2:])
                         - torch.minimum(predicted[:, :2], truth[:, :2]))
    enclosure = enclosure_extent.prod(dim=1)
    giou = intersection / union - (enclosure - union) / enclosure
    result = (1 - giou).mean() + F.smooth_l1_loss(predicted, truth, beta=1 / 9, reduction="mean")
    if not bool(torch.isfinite(result)):
        raise ValueError("Refinement loss must remain finite")
    return result
