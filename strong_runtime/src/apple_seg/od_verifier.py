"""Amodal evidence verifier for predicted apple instances.

The measured VAL failure mode is a score inversion, not a box problem: heavily
occluded apples receive a near-zero apple probability while background patches
receive a high one. The existing gated classifier only sees mask-pooled means,
which throws the spatial layout away and never pools appearance under the
predicted amodal mask.

This module adds one additive, zero-initialized logit correction built from
  * the ROI appearance feature map,
  * the model's own predicted visible/hidden/amodal probability maps, and
  * a small set of predicted-mask geometry descriptors.
Every mask here is a model prediction that is detached, so no ground truth ever
enters the forward pass and a visible-only source cannot acquire A/H edges.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

SCALAR_DIM = 12
DEFAULT_MARGIN = .3
# TRAIN sampling boundary: the VAL heavy-occlusion stratum is defined at 0.40.
HEAVY_VISIBLE_RATIO = .4


def _probability(logits, name):
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.ndim != 4 or logits.shape[1] != 1 or min(logits.shape[-2:]) < 2):
        raise ValueError(f"{name} must be floating [N,1,H,W] predicted mask logits")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{name} must be finite")
    return logits.float().sigmoid().detach()


def mask_geometry_descriptors(visible, amodal, *, eps=1e-6):
    """Twelve deterministic descriptors of the predicted V/A masks on the ROI grid.

    Six describe the apple evidence (fill, V/A area ratio, V-to-A centroid shift,
    peak, V/A agreement) and six describe how well the box fits that evidence
    (A centroid offset from the ROI centre, and the A probability mass touching
    each of the four ROI borders). The box-fit half is what lets the scorer tell
    a tight detection from a duplicate or loosely shifted box of the same apple.
    """
    if visible.shape != amodal.shape or visible.ndim != 4 or visible.shape[1] != 1:
        raise ValueError("Visible and amodal probabilities must share [N,1,H,W]")
    # Evidence, never a training edge: descriptors are always detached.
    visible, amodal = visible.detach()[:, 0], amodal.detach()[:, 0]
    height, width = amodal.shape[-2:]
    total_a = amodal.sum((-1, -2)).clamp_min(eps)
    total_v = visible.sum((-1, -2))
    fill = amodal.mean((-1, -2))
    ratio = total_v / total_a
    grid_y = torch.linspace(0., 1., height, device=amodal.device, dtype=amodal.dtype)[None, :, None]
    grid_x = torch.linspace(0., 1., width, device=amodal.device, dtype=amodal.dtype)[None, None, :]
    center_v = torch.stack((((visible * grid_x).sum((-1, -2)) / total_v.clamp_min(eps)),
                            ((visible * grid_y).sum((-1, -2)) / total_v.clamp_min(eps))), -1)
    center_a = torch.stack((((amodal * grid_x).sum((-1, -2)) / total_a),
                            ((amodal * grid_y).sum((-1, -2)) / total_a)), -1)
    offset = center_v - center_a
    fit = center_a - .5
    top, bottom = amodal[:, 0, :].sum(-1), amodal[:, -1, :].sum(-1)
    left, right = amodal[:, :, 0].sum(-1), amodal[:, :, -1].sum(-1)
    border = torch.stack((top, bottom, left, right), -1) / total_a[:, None]
    binary_v, binary_a = (visible > .5).float(), (amodal > .5).float()
    intersection = (binary_v * binary_a).sum((-1, -2))
    union = ((binary_v + binary_a) > 0).float().sum((-1, -2))
    agreement = intersection / union.clamp_min(eps)
    peak = amodal.amax((-1, -2))
    return torch.stack((fill, ratio, offset[:, 0], offset[:, 1], fit[:, 0], fit[:, 1],
                        border[:, 0], border[:, 1], border[:, 2], border[:, 3],
                        agreement, peak), -1)


def verifier_ranking_loss(logits, labels, *, margin=DEFAULT_MARGIN, hardest=3):
    """Hinge ranking of every positive against its hardest negatives.

    ``logits`` is the full two-class output (base + gate + verifier). The
    measured failure mode is a score inversion, and a plain cross entropy on a
    handful of positives cannot express "this apple must outrank that
    background patch".

    Only the ``hardest`` negatives of each positive enter the mean, which keeps
    the term O(margin) instead of proportional to the negative count; an
    all-pairs sum would outweigh the cross entropy by two orders of magnitude
    on the first steps. Weighting is uniform over pairs, so the gradient with
    respect to any class-prior shift still cancels exactly: the ranking term
    shapes the order and the ordinary cross entropy sets the operating level.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Ranking requires two-class [N,2] logits")
    if not isinstance(margin, float) or not 0 < margin < 10:
        raise ValueError("Ranking margin must be a float in (0,10)")
    if not isinstance(hardest, int) or isinstance(hardest, bool) or not 1 <= hardest <= 64:
        raise ValueError("hardest must be an integer in [1,64]")
    flat = torch.cat(labels) if isinstance(labels, (list, tuple)) else labels
    if not isinstance(flat, torch.Tensor) or flat.ndim != 1 or len(flat) != len(logits):
        raise ValueError("Labels must be [N] and align with the logits")
    margin_logit = logits[:, 1] - logits[:, 0]
    positive, negative = flat >= 1, flat == 0
    if not bool(positive.any()) or not bool(negative.any()):
        return None
    gap = margin_logit[positive][:, None] - margin_logit[negative][None, :]
    count = min(hardest, int(negative.sum()))
    hardest_gap = gap.topk(count, dim=1, largest=False).values
    return F.relu(margin - hardest_gap).mean()


class AmodalEvidenceVerifier(nn.Module):
    """Zero-initialized additive apple logit from predicted V/H/A evidence."""

    def __init__(self, in_channels=256, hidden_channels=64, *, pool_size=4, eps=1e-6):
        super().__init__()
        for name, value in (("in_channels", in_channels), ("hidden_channels", hidden_channels),
                            ("pool_size", pool_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(eps, float) or not 0 < eps < 1e-3:
            raise ValueError("eps must be a small positive float")
        self.configuration = dict(in_channels=in_channels, hidden_channels=hidden_channels,
                                  pool_size=pool_size, scalar_dim=SCALAR_DIM, eps=eps)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 3, hidden_channels, 3, padding=1), nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(pool_size))
        self.scalar_encoder = nn.Sequential(nn.Linear(SCALAR_DIM, 32), nn.GELU())
        self.readout = nn.Linear(hidden_channels * pool_size * pool_size + 32, 2)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, roi_features, visible_logits, hidden_logits, amodal_logits,
                base_class_logits=None, *, targets=None):
        if targets is not None:
            raise ValueError("GT never enters the verifier")
        in_channels = self.configuration["in_channels"]
        if (not isinstance(roi_features, torch.Tensor) or not roi_features.is_floating_point()
                or roi_features.ndim != 4 or roi_features.shape[1] != in_channels):
            raise ValueError(f"ROI features must be floating [N,{in_channels},H,W]")
        visible = _probability(visible_logits, "visible_logits")
        hidden = _probability(hidden_logits, "hidden_logits")
        amodal = _probability(amodal_logits, "amodal_logits")
        spatial = visible.shape[-2:]
        if hidden.shape[-2:] != spatial or amodal.shape[-2:] != spatial:
            raise ValueError("Predicted V/H/A masks must share one ROI grid")
        if roi_features.shape[0] != visible.shape[0]:
            raise ValueError("ROI features and predicted masks must describe the same instances")
        # Predicted masks live on the doubled mask grid; align them to the ROI
        # feature grid so the appearance features and mask evidence share pixels.
        target_hw = tuple(roi_features.shape[-2:])
        if spatial != target_hw:
            visible = F.interpolate(visible, size=target_hw, mode="bilinear", align_corners=False)
            hidden = F.interpolate(hidden, size=target_hw, mode="bilinear", align_corners=False)
            amodal = F.interpolate(amodal, size=target_hw, mode="bilinear", align_corners=False)
        dtype = self.readout.weight.dtype
        evidence = torch.cat((roi_features.float(), visible, hidden, amodal), 1)
        pooled = self.encoder(evidence).flatten(1)
        scalars = self.scalar_encoder(mask_geometry_descriptors(visible, amodal,
                                                                eps=self.configuration["eps"]))
        correction = self.readout(torch.cat((pooled, scalars), 1).to(dtype=dtype))
        if base_class_logits is None:
            return correction
        if (not isinstance(base_class_logits, torch.Tensor)
                or base_class_logits.shape != correction.shape
                or base_class_logits.device != correction.device):
            raise ValueError("Base class logits must share the verifier's [N,2] shape and device")
        return base_class_logits + correction.to(dtype=base_class_logits.dtype)
