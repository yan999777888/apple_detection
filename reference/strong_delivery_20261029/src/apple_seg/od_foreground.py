"""Predicted global V evidence; original amodal/hidden GT is never read here.

The global output uses the padded P2 grid at stride four. Auxiliary labels are
an explicitly coarse grid: union original CPU visible masks, then adaptive-max
pool directly to ceil(resized, unpadded image H/W divided by four). This does
not resize, repair, exclude, or overwrite any original instance annotation.
"""
from collections.abc import Mapping
import math

import torch
from torch import nn
import torch.nn.functional as F


P2_STRIDE = 4


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _floating(value, name, *, ndim=None):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or (ndim is not None and value.ndim != ndim)):
        raise ValueError(f"{name} must be a floating tensor with {ndim} dimensions")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _image_shape(value):
    if (not isinstance(value, (tuple, list, torch.Size)) or len(value) != 2
            or any(type(item) is not int or item < 1 for item in value)):
        raise ValueError("image_shapes must contain positive integer (H,W) pairs")
    return tuple(value)


class GlobalVisibleForegroundHead(nn.Module):
    """P2/P3/P4 1x1 projections, mean at P2, and one 3x3 heatmap readout.

    ``features`` is the ordinary FPN mapping with keys '0', '1', and '2'.
    The returned [B,1,H_P2,W_P2] logits include padding; inference takes no GT.
    """
    def __init__(self, channels=256, hidden=64):
        super().__init__()
        _positive_int(channels, "channels")
        _positive_int(hidden, "hidden")
        self.channels = channels
        self.projections = nn.ModuleDict({key: nn.Conv2d(channels, hidden, 1)
                                          for key in ("0", "1", "2")})
        self.readout = nn.Conv2d(hidden, 1, 3, padding=1)

    def forward(self, features):
        if not isinstance(features, Mapping) or any(key not in features for key in self.projections):
            raise ValueError("Global foreground requires FPN keys '0', '1', '2'")
        first = features["0"]
        _floating(first, "P2", ndim=4)
        if first.shape[1] != self.channels or min(first.shape[-2:]) < 1:
            raise ValueError("FPN levels must have shape [B,channels,H,W] with positive H/W")
        size, previous = first.shape[-2:], first.shape[-2:]
        projected = []
        for key, projection in self.projections.items():
            value = features[key]
            _floating(value, f"FPN {key}", ndim=4)
            if (value.shape[:2] != first.shape[:2] or min(value.shape[-2:]) < 1
                    or value.device != first.device or value.dtype != first.dtype
                    or any(current > upper for current, upper in zip(value.shape[-2:], previous))):
                raise ValueError("FPN levels must share batch/channels/device/dtype and decrease in size")
            previous = value.shape[-2:]
            value = projection(value)
            if value.shape[-2:] != size:
                value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            projected.append(value)
        result = self.readout(torch.stack(projected).mean(0))
        _floating(result, "foreground logits", ndim=4)
        return result


def _visible_union_on_cpu(target):
    if not isinstance(target, Mapping):
        raise ValueError("Each foreground target must be a mapping")
    masks = target.get("visible_masks")
    if (not isinstance(masks, torch.Tensor) or masks.ndim != 3
            or masks.device.type != "cpu" or masks.requires_grad
            or min(masks.shape[-2:]) < 1 or masks.is_complex()):
        raise ValueError("Original visible_masks[N,H,W] must be fixed real CPU annotations")
    original = target.get("original_hw")
    if original is not None and _image_shape(original) != tuple(masks.shape[-2:]):
        raise ValueError("Original visible mask shape and original_hw differ")
    union = torch.zeros(masks.shape[-2:], dtype=torch.bool, device="cpu")
    # One original-size temporary at a time; original masks never reach device.
    for mask in masks:
        if not bool(torch.isfinite(mask).all()) or bool(((mask != 0) & (mask != 1)).any()):
            raise ValueError("Original visible masks must be finite binary annotations")
        union |= mask != 0
    return union


def visible_foreground_losses(logits, targets, image_shapes, *, focal_gamma=2.,
                              focal_alpha=.25):
    """Unweighted focal BCE and Dice on valid stride-4 cells only.

    ``image_shapes`` are resized input-pixel (H,W), before padding. CPU pooling
    maps the *full native* V union to ceil(H/4),ceil(W/4); partial boundary cells
    are valid. Everything right/below that rectangle has no loss/gradient.
    Empty instance sets are genuine all-background images. Only pooled coarse
    labels are transferred to the logits device; no A/H field is accessed.
    """
    _floating(logits, "foreground logits", ndim=4)
    if logits.shape[1] != 1 or min(logits.shape[-2:]) < 1:
        raise ValueError("Foreground logits must have shape [B,1,H_P2,W_P2]")
    if (not isinstance(targets, (list, tuple)) or not isinstance(image_shapes, (list, tuple))
            or len(targets) != len(logits) or len(image_shapes) != len(logits)):
        raise ValueError("One original visible target and image shape are required per image")
    if (isinstance(focal_gamma, bool) or not isinstance(focal_gamma, (float, int))
            or not math.isfinite(focal_gamma) or focal_gamma < 0
            or isinstance(focal_alpha, bool) or not isinstance(focal_alpha, (float, int))
            or not math.isfinite(focal_alpha) or not 0 <= focal_alpha <= 1):
        raise ValueError("Focal gamma must be nonnegative and alpha must be in [0,1]")
    focal, dice = [], []
    for index, (target, shape) in enumerate(zip(targets, image_shapes)):
        height, width = _image_shape(shape)
        valid_h, valid_w = (height + P2_STRIDE - 1) // P2_STRIDE, (width + P2_STRIDE - 1) // P2_STRIDE
        if valid_h > logits.shape[-2] or valid_w > logits.shape[-1]:
            raise ValueError("P2 logits do not cover the unpadded image")
        union = _visible_union_on_cpu(target)
        truth = F.adaptive_max_pool2d(union[None, None].float(), (valid_h, valid_w))
        truth = truth.to(device=logits.device, dtype=torch.float32)
        prediction = logits[index:index + 1, :, :valid_h, :valid_w].float()
        bce = F.binary_cross_entropy_with_logits(prediction, truth, reduction="none")
        probability = prediction.sigmoid()
        p_t = truth * probability + (1 - truth) * (1 - probability)
        alpha_t = truth * focal_alpha + (1 - truth) * (1 - focal_alpha)
        focal.append((alpha_t * (1 - p_t).clamp_min(0).pow(focal_gamma) * bce).mean())
        dice.append(1 - (2 * (probability * truth).sum() + 1)
                    / (probability.sum() + truth.sum() + 1))
    if not focal:
        zero = logits.float().sum() * 0
        return dict(loss_foreground_focal=zero, loss_foreground_dice=zero)
    return dict(loss_foreground_focal=torch.stack(focal).mean(),
                loss_foreground_dice=torch.stack(dice).mean())


class _ExactResidualAdd(torch.autograd.Function):
    """Addition with signed-zero identity; its derivative is still addition."""
    @staticmethod
    def forward(ctx, base, residual):
        return torch.where(residual == 0, base, base + residual)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


class ForegroundPrototypeFusion(nn.Module):
    """Pure predicted ROI evidence added through a zero-initialized readout.

    Inputs refer to the same pixel-coordinate ROI: pooled native ROI features,
    predicted V logits, and ROIAlign(P2 semantic logits, spatial_scale=1/4).
    V/semantic grids may differ and are resized to the feature grid. Evidence
    is prototype cosine, semantic mean, semantic maximum, and V/semantic soft
    overlap. There is no hard multiplication of base logits or power of cosine.
    Gradients remain available to predicted inputs; stage freezing is external.
    """
    def __init__(self, channels=256):
        super().__init__()
        _positive_int(channels, "channels")
        self.channels = channels
        self.prototype = nn.Parameter(torch.empty(channels))
        nn.init.normal_(self.prototype, std=.02)
        self.readout = nn.Linear(4, 2)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, pooled_features, visible_logits, semantic_roi_logits, base_logits):
        _floating(pooled_features, "pooled_features", ndim=4)
        if pooled_features.shape[1] != self.channels or min(pooled_features.shape[-2:]) < 1:
            raise ValueError("Pooled features must have shape [N,channels,H,W]")
        count, device, size = len(pooled_features), pooled_features.device, pooled_features.shape[-2:]
        for value, name in ((visible_logits, "visible_logits"),
                            (semantic_roi_logits, "semantic_roi_logits")):
            _floating(value, name, ndim=4)
            if (value.shape[:2] != (count, 1) or min(value.shape[-2:]) < 1
                    or value.device != device):
                raise ValueError("Predicted ROI masks must share N/device and have one channel")
        _floating(base_logits, "base_logits", ndim=2)
        if base_logits.shape != (count, 2) or base_logits.device != device:
            raise ValueError("Base logits must share the ROI device and have shape [N,2]")
        _floating(self.prototype, "prototype", ndim=1)
        probabilities = []
        for value in (visible_logits, semantic_roi_logits):
            value = value.float().sigmoid()
            if value.shape[-2:] != size:
                value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            probabilities.append(value)
        visible, semantic = probabilities
        feature = (pooled_features.float() * visible).sum((-1, -2)) / visible.sum((-1, -2)).clamp_min(1e-6)
        cosine = (F.normalize(feature, dim=1, eps=1e-6)
                  * F.normalize(self.prototype.float(), dim=0, eps=1e-6)).sum(1).clamp(-1, 1)
        semantic_mean = semantic.mean((-1, -2))[:, 0]
        semantic_max = semantic.amax((-1, -2))[:, 0]
        overlap = (2 * (visible * semantic).sum((-1, -2))
                   / (visible.sum((-1, -2)) + semantic.sum((-1, -2))).clamp_min(1e-6))[:, 0]
        evidence = torch.stack((cosine, semantic_mean, semantic_max, overlap), 1)
        residual = self.readout(evidence.to(dtype=self.readout.weight.dtype)).to(dtype=base_logits.dtype)
        result = _ExactResidualAdd.apply(base_logits, residual)
        _floating(result, "fused logits", ndim=2)
        return result
