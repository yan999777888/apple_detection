"""Differentiable predicted-A mask geometry; inference never accepts annotations."""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn

from .od_context import _boxes, _clip_predicted_boxes, _image_shape


def _bounded_boxes(boxes, image_shape, name):
    height, width = _image_shape(image_shape)
    result = _boxes(boxes, name)
    limits = result.new_tensor([width, height, width, height])
    if not bool(((result >= 0) & (result <= limits)).all()):
        raise ValueError(f"{name} must already lie inside image bounds")
    return result


def _axis_edges(occupancy, fallback_first, fallback_last):
    """Expected half-open first/last occupied bin, including empty probability."""
    length = occupancy.shape[1]
    fourth = occupancy.pow(4)
    occupied = fourth / (fourth + (1 - occupancy).pow(4))
    absent = 1 - occupied
    one = torch.ones_like(absent[:, :1])
    before = torch.cat((one, absent.cumprod(dim=1)[:, :-1]), dim=1)
    after = torch.cat((absent.flip(1).cumprod(dim=1).flip(1)[:, 1:], one), dim=1)
    bins = torch.arange(length, device=occupancy.device, dtype=torch.float32)
    empty = absent.prod(dim=1)
    first = (occupied * before * (bins / length)).sum(dim=1) + empty * fallback_first
    last = (occupied * after * ((bins + 1) / length)).sum(dim=1) + empty * fallback_last
    return first, last


def soft_mask_boxes(probabilities, support_boxes, fallback_boxes, image_shape):
    """Turn probabilities in their support ROI into FP32 input-coordinate boxes.

    Row/column occupancy is an amax, sharpened by a symmetric fourth power.
    Exclusive empty products give first/last occupancy probabilities. Their
    expectation uses half-open bin boundaries, with the probability of an
    empty axis assigned to that axis's normalized fallback bounds. Supports
    may extend beyond the base box; no predicted box is confined to the base.
    """
    _image_shape(image_shape)
    if (not isinstance(probabilities, torch.Tensor) or not probabilities.is_floating_point()
            or probabilities.ndim != 4 or probabilities.shape[1] != 1
            or min(probabilities.shape[-2:]) < 1):
        raise ValueError("probabilities must be floating [N,1,H,W] with nonempty grid axes")
    probabilities = probabilities.float()
    if (not bool(torch.isfinite(probabilities).all())
            or not bool(((probabilities >= 0) & (probabilities <= 1)).all())):
        raise ValueError("Mask probabilities must be finite and inside [0,1]")
    support = _bounded_boxes(support_boxes, image_shape, "support_boxes")
    fallback = _bounded_boxes(fallback_boxes, image_shape, "fallback_boxes")
    if (len(probabilities) != len(support) or support.shape != fallback.shape
            or probabilities.device != support.device or support.device != fallback.device):
        raise ValueError("Probabilities, support and fallback rows/devices must align")
    if not len(support):
        return fallback.clone()
    origin, extent = support[:, :2], support[:, 2:] - support[:, :2]
    normalized_fallback = (fallback - torch.cat((origin, origin), dim=1)) / torch.cat((extent, extent), dim=1)
    horizontal = probabilities[:, 0].amax(dim=1)
    vertical = probabilities[:, 0].amax(dim=2)
    x1, x2 = _axis_edges(horizontal, normalized_fallback[:, 0], normalized_fallback[:, 2])
    y1, y2 = _axis_edges(vertical, normalized_fallback[:, 1], normalized_fallback[:, 3])
    normalized = torch.stack((x1, y1, x2, y2), dim=1)
    predicted = torch.cat((origin, origin), dim=1) + normalized * torch.cat((extent, extent), dim=1)
    predicted = _clip_predicted_boxes(predicted, image_shape)
    # Avoid round-trip coordinate rounding, including signed-zero changes.
    all_empty = (probabilities == 0).flatten(1).all(dim=1, keepdim=True)
    all_full = (probabilities == 1).flatten(1).all(dim=1, keepdim=True)
    return torch.where(all_empty, fallback, torch.where(all_full, support, predicted))


class GeometryAHead(nn.Module):
    """Independent, value-identical copy of the actual twelve-state A tower."""

    def __init__(self, mask_head, mask_predictor, apple_index):
        super().__init__()
        if not isinstance(mask_head, nn.Module) or not isinstance(mask_predictor, nn.Module):
            raise ValueError("Geometry A head requires real source mask modules")
        convs = [layer for layer in mask_head.modules() if isinstance(layer, nn.Conv2d)]
        deconv = getattr(mask_predictor, "conv5_mask", None)
        predictor = getattr(mask_predictor, "mask_fcn_logits", None)
        if (len(convs) != 4 or any(layer.in_channels != 256 or layer.out_channels != 256 for layer in convs)
                or not isinstance(deconv, nn.ConvTranspose2d) or deconv.in_channels != 256
                or deconv.out_channels != 256 or deconv.kernel_size != (2, 2)
                or deconv.stride != (2, 2) or deconv.padding != (0, 0)
                or not isinstance(predictor, nn.Conv2d) or predictor.in_channels != 256
                or predictor.kernel_size != (1, 1)
                or type(apple_index) is not int or not 0 <= apple_index < predictor.out_channels):
            raise ValueError("Geometry A head requires the genuine 256-channel 28-to-56 A tower and apple row")
        self.mask_head, self.mask_predictor = deepcopy(mask_head), deepcopy(mask_predictor)
        self.apple_index = apple_index
        original = {"mask_head." + key: value for key, value in mask_head.state_dict().items()}
        original.update({"mask_predictor." + key: value for key, value in mask_predictor.state_dict().items()})
        copied = self.state_dict()
        keys_equal = original.keys() == copied.keys() and len(copied) == 12
        equal = keys_equal and all(torch.equal(value, copied[key]) for key, value in original.items())
        independent = keys_equal and all(not value.numel() or value.untyped_storage().data_ptr()
            != copied[key].untyped_storage().data_ptr() for key, value in original.items())
        if not equal or not independent:
            raise ValueError("Geometry A copy must preserve exactly twelve finite, storage-independent states")
        if not all(bool(torch.isfinite(value).all()) for value in copied.values()):
            raise ValueError("Geometry A source states must be finite")
        self.initialization_provenance = dict(initialization="deepcopy_actual_A_base_head_and_predictor",
            copied_tensor_count=12, strict_key_shapes_equal=keys_equal, all_source_tensors_equal=equal,
            all_copied_tensor_storages_independent=independent, apple_index=apple_index,
            optimizer_or_RNG_restored=False)

    def forward(self, pooled):
        if (not isinstance(pooled, torch.Tensor) or not pooled.is_floating_point() or pooled.ndim != 4
                or tuple(pooled.shape[1:]) != (256, 28, 28)
                or not bool(torch.isfinite(pooled).all())):
            raise ValueError("Geometry A pooled features must be finite floating [N,256,28,28]")
        logits = self.mask_predictor(self.mask_head(pooled))[:, self.apple_index:self.apple_index + 1]
        if tuple(logits.shape) != (len(pooled), 1, 56, 56) or not bool(torch.isfinite(logits).all()):
            raise ValueError("Geometry A logits must be finite [N,1,56,56]")
        return logits


class _ExactResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, residual):
        return torch.where(residual == 0, base, base + residual)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


class MaskGeometryRefiner(nn.Module):
    """Signed learned BoxCoder interpolation; zero starts as the exact base box."""

    def __init__(self):
        super().__init__()
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.box_coder_weights = (10., 10., 5., 5.)
        self.box_xform_clip = math.log(1000. / 16.)

    def blend(self, mask_boxes, base_boxes, image_shape):
        predicted = _bounded_boxes(mask_boxes, image_shape, "mask_boxes")
        base = _bounded_boxes(base_boxes, image_shape, "base_boxes")
        if predicted.shape != base.shape or predicted.device != base.device or base.device != self.mix_logit.device:
            raise ValueError("Mask and base boxes must share rows and the refiner device")
        if not bool(torch.isfinite(self.mix_logit)):
            raise ValueError("Geometry mixture must be finite")
        widths, heights = base[:, 2] - base[:, 0], base[:, 3] - base[:, 1]
        pw, ph = predicted[:, 2] - predicted[:, 0], predicted[:, 3] - predicted[:, 1]
        center = base[:, :2] + .5 * torch.stack((widths, heights), dim=1)
        pcenter = predicted[:, :2] + .5 * torch.stack((pw, ph), dim=1)
        wx, wy, ww, wh = self.box_coder_weights
        encoded = torch.stack((wx * ((pcenter[:, 0] - center[:, 0]) / widths),
                               wy * ((pcenter[:, 1] - center[:, 1]) / heights),
                               ww * (pw.log() - widths.log()), wh * (ph.log() - heights.log())), dim=1)
        delta = self.mix_logit.float().tanh() * encoded
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Geometry BoxCoder residual must be finite")
        dx, dy = delta[:, 0] / wx * widths, delta[:, 1] / wy * heights
        gx = .5 * widths * torch.expm1((delta[:, 2] / ww).clamp(max=self.box_xform_clip))
        gy = .5 * heights * torch.expm1((delta[:, 3] / wh).clamp(max=self.box_xform_clip))
        residual = torch.stack((dx - gx, dy - gy, dx + gx, dy + gy), dim=1)
        return _clip_predicted_boxes(_ExactResidualAdd.apply(base, residual), image_shape)

    def forward(self, probabilities, support_boxes, base_boxes, image_shape):
        mask_boxes = soft_mask_boxes(probabilities, support_boxes, base_boxes, image_shape)
        return self.blend(mask_boxes, base_boxes, image_shape), mask_boxes
