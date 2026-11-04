"""ROI components for an experimental visible/hidden/amodal detector.

H denotes this instance's hidden surface, A & ~V, rather than a semantic mask
of every occluder in the ROI. No containment, circularization, or inference-GT
projection is performed. These components make no measured accuracy claim.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
import torch.nn.functional as F


MASK_KEYS = ("visible_logits", "hidden_logits", "amodal_logits")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _probabilities(value, name, *, chunk_size=8):
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise ValueError(f"{name} must be a probability or binary tensor")
    if value.dtype != torch.bool and not value.is_floating_point():
        raise ValueError(f"{name} must be bool or floating probabilities")
    if value.dtype == torch.bool:
        return
    for start in range(0, len(value), chunk_size):
        chunk = value[start:start + chunk_size]
        if not bool(torch.isfinite(chunk).all()) or bool(((chunk < 0) | (chunk > 1)).any()):
            raise ValueError(f"{name} must contain finite probabilities in [0,1]")


class OcclusionDecoupledROIMaskHead(nn.Module):
    """28x28 ROI features -> independent V/H logits -> conditioned A at 56x56.

    The V tower matches the four-convolution torchvision Mask R-CNN layout.
    H can share V features for the legacy ablation or use its own cloned tower
    and deconvolution; independent H losses then have no V-parameter path.
    A starts exactly at V through a zero-initialized *unrestricted* residual;
    later A can exclude V pixels when the original annotations require it.
    Optional weak x/y/radius features are coordinates, not a circle target.
    """
    def __init__(self, in_channels=256, hidden_channels=128, *, weak_ellipse_features=False,
                 hidden_tower_mode="shared_visible", zero_initialize_hidden_predictor=False):
        super().__init__()
        _positive_integer(in_channels, "in_channels")
        _positive_integer(hidden_channels, "hidden_channels")
        if type(hidden_tower_mode) is not str or hidden_tower_mode not in ("shared_visible", "independent"):
            raise ValueError("hidden_tower_mode must be shared_visible or independent")
        if type(zero_initialize_hidden_predictor) is not bool:
            raise ValueError("zero_initialize_hidden_predictor must be an explicit bool")
        self.hidden_tower_mode = hidden_tower_mode
        self.zero_initialize_hidden_predictor = zero_initialize_hidden_predictor
        self.configuration = dict(in_channels=in_channels, hidden_channels=hidden_channels,
                                  weak_ellipse_features=bool(weak_ellipse_features),
                                  hidden_tower_mode=hidden_tower_mode,
                                  zero_initialize_hidden_predictor=zero_initialize_hidden_predictor,
                                  input_size=28, output_size=56)
        layers = []
        for index in range(4):
            layers.extend((nn.Conv2d(in_channels if index == 0 else hidden_channels,
                                     hidden_channels, 3, padding=1), nn.ReLU(inplace=False)))
        self.visible_tower = nn.Sequential(*layers)
        self.visible_upsample = nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, stride=2)
        self.visible_predictor = nn.Conv2d(hidden_channels, 1, 1)
        self.hidden_predictor = nn.Conv2d(hidden_channels, 1, 1)
        condition_channels = hidden_channels + 2 + (3 if weak_ellipse_features else 0)
        layers = []
        for index, dilation in enumerate((1, 2, 4, 1)):
            layers.extend((nn.Conv2d(condition_channels if index == 0 else hidden_channels,
                                     hidden_channels, 3, padding=dilation, dilation=dilation),
                           nn.ReLU(inplace=False)))
        self.amodal_tower = nn.Sequential(*layers)
        self.amodal_predictor = nn.Conv2d(hidden_channels, 1, 1)
        # Always allocate and initialize these modules, even in the shared
        # ablation. Mode/zero flags must not change the Torch RNG trajectory.
        self.hidden_tower = deepcopy(self.visible_tower)
        self.hidden_upsample = deepcopy(self.visible_upsample)
        self.amodal_base_head = None
        self.amodal_base_predictor = None
        self.amodal_base_apple_index = None
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.amodal_predictor.weight)
        nn.init.zeros_(self.amodal_predictor.bias)
        # The H readout has already consumed its ordinary random initialization
        # above. Optional zeroing performs no additional random draws.
        if self.zero_initialize_hidden_predictor:
            nn.init.zeros_(self.hidden_predictor.weight)
            nn.init.zeros_(self.hidden_predictor.bias)
        self._copy_hidden_features_from_visible()
        self.set_hidden_feature_training(True)
        axis = (torch.arange(56, dtype=torch.float32) + .5) * (2. / 56) - 1
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("ellipse_coordinates", .1 * torch.stack((xx, yy, xx.square() + yy.square()))[None],
                             persistent=False)

    def forward(self, roi_features: torch.Tensor, *, targets=None) -> dict[str, torch.Tensor]:
        if targets is not None:
            raise ValueError("GT never enters ROI mask inference")
        if (not isinstance(roi_features, torch.Tensor) or not roi_features.is_floating_point()
                or roi_features.ndim != 4 or tuple(roi_features.shape[1:]) !=
                (self.configuration["in_channels"], 28, 28)):
            raise ValueError("Expected floating ROI features[N,in_channels,28,28]")
        features = F.relu(self.visible_upsample(self.visible_tower(roi_features)))
        visible = self.visible_predictor(features)
        hidden_features = (features if self.hidden_tower_mode == "shared_visible" else
                           F.relu(self.hidden_upsample(self.hidden_tower(roi_features))))
        hidden = self.hidden_predictor(hidden_features)
        conditions = [features, visible.sigmoid(), hidden.sigmoid()]
        if self.configuration["weak_ellipse_features"]:
            conditions.append(self.ellipse_coordinates.to(dtype=features.dtype).expand(len(features), -1, -1, -1))
        if self.amodal_base_head is None:
            amodal_base = visible
        else:
            amodal_base = self.amodal_base_predictor(self.amodal_base_head(roi_features))[
                :, self.amodal_base_apple_index:self.amodal_base_apple_index + 1]
        amodal = amodal_base + self.amodal_predictor(self.amodal_tower(torch.cat(conditions, 1)))
        return dict(visible_logits=visible, hidden_logits=hidden, amodal_logits=amodal)

    @torch.no_grad()
    def _copy_hidden_features_from_visible(self):
        """Copy feature values without sharing storage or altering the readout."""
        self.hidden_tower.load_state_dict(self.visible_tower.state_dict(), strict=True)
        self.hidden_upsample.load_state_dict(self.visible_upsample.state_dict(), strict=True)

    def set_hidden_feature_training(self, joint):
        """Unused shared-mode H modules stay frozen, including after stage changes."""
        enabled = joint and self.hidden_tower_mode == "independent"
        self.hidden_tower.requires_grad_(enabled)
        self.hidden_upsample.requires_grad_(enabled)

    @torch.no_grad()
    def initialize_visible_from_maskrcnn(self, mask_head, mask_predictor, *, apple_index=53):
        """Copy genuine V weights and clone its feature tower/deconv for H.

        Use hidden_channels=256 for the standard COCO torchvision checkpoint.
        Shape checks complete before any parameter is copied. H's independent
        readout is never copied from a visible or amodal class predictor.
        """
        if any(not isinstance(module, (nn.Sequential, nn.Conv2d, nn.ReLU)) for module in mask_head.modules()):
            raise ValueError("Visible warm start requires the standard convolution/ReLU-only source tower")
        source = [m for m in mask_head.modules() if isinstance(m, nn.Conv2d)]
        destination = [m for m in self.visible_tower.modules() if isinstance(m, nn.Conv2d)]
        if len(source) != len(destination):
            raise ValueError("Source and destination visible convolution counts differ")
        deconvolution = getattr(mask_predictor, "conv5_mask", None)
        predictor = getattr(mask_predictor, "mask_fcn_logits", None)
        if not isinstance(deconvolution, nn.ConvTranspose2d) or not isinstance(predictor, nn.Conv2d):
            raise ValueError("Expected torchvision MaskRCNNPredictor submodules")
        if not isinstance(apple_index, int) or not 0 <= apple_index < predictor.out_channels:
            raise ValueError("apple_index is outside the source predictor")
        pairs = list(zip(destination, source)) + [(self.visible_upsample, deconvolution)]
        for target, original in pairs:
            if (target.weight.shape != original.weight.shape or
                    (target.bias is None) != (original.bias is None) or
                    (target.bias is not None and target.bias.shape != original.bias.shape)):
                raise ValueError("Source/destination visible tensor shapes differ")
        if (self.visible_predictor.weight.shape != predictor.weight[apple_index:apple_index + 1].shape
                or predictor.bias is None):
            raise ValueError("Source apple predictor shape differs")
        for target, original in pairs:
            target.weight.copy_(original.weight)
            if target.bias is not None:
                target.bias.copy_(original.bias)
        self.visible_predictor.weight.copy_(predictor.weight[apple_index:apple_index + 1])
        self.visible_predictor.bias.copy_(predictor.bias[apple_index:apple_index + 1])
        self._copy_hidden_features_from_visible()
        return dict(visible_convolutions=len(destination), apple_index=apple_index,
                    hidden_tower_copied_from_visible=True,
                    hidden_predictor_or_amodal_weights_copied=False)

    def initialize_amodal_base_from_maskrcnn(self, mask_head, mask_predictor, *, apple_index=1):
        """Retain an actual trained legacy A tower as the residual's base.

        These are independent ``amodal_*`` modules, so V-only losses never
        update them. Deep copies preserve all real source layers and weights.
        The full predictor is retained for exact checkpoint compatibility and
        only its genuine apple output row is used.
        """
        if not isinstance(mask_head, nn.Module) or not isinstance(mask_predictor, nn.Module):
            raise ValueError("Amodal base requires actual torch mask modules")
        deconvolution = getattr(mask_predictor, "conv5_mask", None)
        predictor = getattr(mask_predictor, "mask_fcn_logits", None)
        convolutions = [m for m in mask_head.modules() if isinstance(m, nn.Conv2d)]
        if (not convolutions or convolutions[0].in_channels != self.configuration["in_channels"]
                or not isinstance(deconvolution, nn.ConvTranspose2d)
                or not isinstance(predictor, nn.Conv2d)
                or convolutions[-1].out_channels != deconvolution.in_channels
                or deconvolution.out_channels != predictor.in_channels
                or deconvolution.kernel_size != (2, 2) or deconvolution.stride != (2, 2)
                or deconvolution.padding != (0, 0)):
            raise ValueError("Amodal base must be a compatible 28-to-56 Mask R-CNN tower")
        if not isinstance(apple_index, int) or not 0 <= apple_index < predictor.out_channels:
            raise ValueError("apple_index is outside the amodal base predictor")
        self.amodal_base_head = deepcopy(mask_head)
        self.amodal_base_predictor = deepcopy(mask_predictor)
        self.amodal_base_apple_index = apple_index
        self.configuration["amodal_base"] = dict(apple_index=apple_index, source="actual_legacy_A_tower")
        return dict(amodal_base_copied=True, apple_index=apple_index,
                    visible_or_hidden_weights_copied=False)


def _mask_probability_for_features(logits, features, *, detach):
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or logits.ndim != 4
            or tuple(logits.shape[:2]) != (len(features), 1) or min(logits.shape[-2:]) < 1
            or logits.device != features.device):
        raise ValueError("Mask logits must be floating[N,1,H,W] on the feature device")
    logits = logits.detach() if detach else logits
    probability = logits.float().sigmoid()
    if probability.shape[-2:] != features.shape[-2:]:
        probability = F.interpolate(probability, size=features.shape[-2:], mode="bilinear", align_corners=False)
    return probability


def predicted_visible_pool(roi_features, visible_logits, *, threshold=.3, detach_mask=True, eps=1e-6):
    """Mass-normalized PV*I(PV>.3) pool; an empty gate yields finite zeros."""
    if not 0 <= threshold < 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("A probability threshold and positive finite eps are required")
    probability = _mask_probability_for_features(visible_logits, roi_features, detach=detach_mask)
    weights = probability * (probability > threshold)
    mass = weights.sum((-1, -2))
    pooled = (roi_features.float() * weights).sum((-1, -2)) / mass.clamp_min(eps)
    return pooled, mass


class PredictedVisibleGatedClassifier(nn.Module):
    """Predict a correction from pure-visible/hidden pools and predicted V/A.

    H and A are always detached, including PApple classification. Thus a Minne
    V-only class loss cannot invent H/A supervision. Visible masks are detached
    by default too; ROI appearance features still receive class gradients.
    """
    def __init__(self, in_channels=256, hidden_channels=256, num_classes=2, *,
                 visible_threshold=.3, detach_visible=True, eps=1e-6):
        super().__init__()
        for name, value in (("in_channels", in_channels), ("hidden_channels", hidden_channels),
                            ("num_classes", num_classes)):
            _positive_integer(value, name)
        if not 0 <= visible_threshold < 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("A probability threshold and positive finite eps are required")
        self.configuration = dict(in_channels=in_channels, hidden_channels=hidden_channels,
                                  num_classes=num_classes, visible_threshold=visible_threshold,
                                  detach_visible=bool(detach_visible), eps=eps)
        self.encoder = nn.Sequential(nn.Linear(2 * in_channels + 1, hidden_channels),
                                     nn.LayerNorm(hidden_channels), nn.GELU())
        self.readout = nn.Linear(hidden_channels, num_classes)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, roi_features, visible_logits, hidden_logits, amodal_logits,
                base_class_logits=None, *, targets=None):
        if targets is not None:
            raise ValueError("GT never enters gated classifier inference")
        if (not isinstance(roi_features, torch.Tensor) or not roi_features.is_floating_point()
                or roi_features.ndim != 4 or roi_features.shape[1] != self.configuration["in_channels"]
                or min(roi_features.shape[-2:]) < 1):
            raise ValueError("ROI features must be floating[N,in_channels,H,W]")
        eps = self.configuration["eps"]
        visible, _ = predicted_visible_pool(roi_features, visible_logits,
                                            threshold=self.configuration["visible_threshold"],
                                            detach_mask=self.configuration["detach_visible"], eps=eps)
        pv = _mask_probability_for_features(visible_logits, roi_features,
                                           detach=self.configuration["detach_visible"])
        ph = _mask_probability_for_features(hidden_logits, roi_features, detach=True)
        pa = _mask_probability_for_features(amodal_logits, roi_features, detach=True)
        hidden = (roi_features.float() * ph).sum((-1, -2)) / ph.sum((-1, -2)).clamp_min(eps)
        # Ratios above one remain valid evidence; never project V into A.
        ratio = pv.sum((-1, -2)) / (pa.sum((-1, -2)) + eps)
        evidence = torch.cat((visible, hidden, ratio), -1).to(dtype=self.readout.weight.dtype)
        correction = self.readout(self.encoder(evidence))
        if base_class_logits is None:
            return correction
        if (not isinstance(base_class_logits, torch.Tensor) or base_class_logits.shape != correction.shape
                or base_class_logits.device != correction.device):
            raise ValueError("Base class logits must have the classifier's [N,num_classes] shape/device")
        return base_class_logits + correction.to(dtype=base_class_logits.dtype)


def _target_masks(value, count, spatial, device, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must contain genuine projected annotation masks")
    if value.ndim == 3:
        value = value[:, None]
    if tuple(value.shape) != (count, 1, *spatial):
        raise ValueError(f"{name} has the wrong projected mask shape")
    if value.requires_grad or not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must be fixed finite annotation probabilities in [0,1]")
    return value.to(device=device, dtype=torch.float32)


def _bce_dice(logits, truth):
    if not len(logits):
        zero = logits.float().sum() * 0
        return zero, zero
    logits, truth = logits.float(), truth.float()
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none").flatten(1).mean(1).mean()
    probability = logits.sigmoid().flatten(1)
    truth = truth.flatten(1)
    dice = 1 - (2 * (probability * truth).sum(1) + 1) / (probability.sum(1) + truth.sum(1) + 1)
    return bce, dice.mean()


def od_roi_mask_losses(predictions, visible_targets, amodal_targets=None, *, amodal_valid=None,
                       hidden_targets=None, bce_weight=1., dice_weight=1., hidden_weight=1., amodal_weight=1.):
    """Losses for positive ROIs; unknown A/H has no loss-graph edge.

    V targets contain every ROI. A/H contain only ``amodal_valid`` true ROIs in
    that same order. Derive H=A&~V on the ORIGINAL binary CPU annotations before
    ROIAlign, then supply its projected ``hidden_targets``; projected fractional
    A*(1-V) is not that same annotation. Binary fixture targets can omit H and
    use the exact Boolean difference here. No GT is repaired or constrained.
    Returned scalar losses are already weighted and should be summed once.
    """
    if not isinstance(predictions, dict) or any(key not in predictions for key in MASK_KEYS):
        raise ValueError("Predictions must contain visible_logits, hidden_logits, amodal_logits")
    visible, hidden, amodal = (predictions[key] for key in MASK_KEYS)
    if (visible.ndim != 4 or visible.shape[1] != 1 or hidden.shape != visible.shape
            or amodal.shape != visible.shape or min(visible.shape[-2:]) < 1):
        raise ValueError("All mask predictions must share [N,1,H,W]")
    count, spatial, device = len(visible), visible.shape[-2:], visible.device
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x < 0
           for x in (bce_weight, dice_weight, hidden_weight, amodal_weight)):
        raise ValueError("Loss weights must be finite and nonnegative")
    truth_v = _target_masks(visible_targets, count, spatial, device, "visible_targets")
    if amodal_valid is None:
        valid = torch.full((count,), amodal_targets is not None, dtype=torch.bool, device=device)
    elif (not isinstance(amodal_valid, torch.Tensor) or amodal_valid.dtype != torch.bool
          or tuple(amodal_valid.shape) != (count,)):
        raise ValueError("amodal_valid must be explicit bool[N]")
    else:
        valid = amodal_valid.to(device=device)
    bce, dice = _bce_dice(visible, truth_v)
    losses = dict(loss_visible_bce=bce_weight * bce, loss_visible_dice=dice_weight * dice)
    known_count = int(valid.sum())
    if not known_count:
        if amodal_targets is not None and len(amodal_targets):
            raise ValueError("Unknown amodal ROIs cannot carry amodal targets")
        if hidden_targets is not None and len(hidden_targets):
            raise ValueError("Unknown amodal ROIs cannot carry hidden targets")
        return losses
    truth_a = _target_masks(amodal_targets, known_count, spatial, device, "amodal_targets")
    if hidden_targets is None:
        selected_v = truth_v[valid]
        if bool(((truth_a != 0) & (truth_a != 1)).any()) or bool(((selected_v != 0) & (selected_v != 1)).any()):
            raise ValueError("Fractional ROI targets require H projected from original A&~V annotations")
        truth_h = ((truth_a > 0) & ~(selected_v > 0)).float()
    else:
        truth_h = _target_masks(hidden_targets, known_count, spatial, device, "hidden_targets")
    for name, logits, truth, branch_weight in (("hidden", hidden[valid], truth_h, hidden_weight),
                                               ("amodal", amodal[valid], truth_a, amodal_weight)):
        bce, dice = _bce_dice(logits, truth)
        losses[f"loss_{name}_bce"] = branch_weight * bce_weight * bce
        losses[f"loss_{name}_dice"] = branch_weight * dice_weight * dice
    return losses


def _window(box, image_shape):
    height, width = image_shape
    return (max(0, min(width, math.floor(box[0]))), max(0, min(height, math.floor(box[1]))),
            max(0, min(width, math.ceil(box[2]))), max(0, min(height, math.ceil(box[3]))))


def _tiles(window, max_pixels):
    left, top, right, bottom = window
    if right <= left or bottom <= top:
        return
    # Bound both dimensions, including a very wide image's single row.
    for x1 in range(left, right, max_pixels):
        x2 = min(right, x1 + max_pixels)
        rows = max(1, max_pixels // (x2 - x1))
        for y1 in range(top, bottom, rows):
            yield (x1, y1, x2, min(bottom, y1 + rows))


def _sample_roi_window(masks, boxes, window):
    """Sample [K,C,R,R] probabilities into common image pixel centers."""
    left, top, right, bottom = window
    dtype, device = torch.float32, masks.device
    xx = torch.arange(left, right, dtype=dtype, device=device) + .5
    yy = torch.arange(top, bottom, dtype=dtype, device=device) + .5
    x = (xx[None] - boxes[:, 0, None]) / (boxes[:, 2] - boxes[:, 0])[:, None] * 2 - 1
    y = (yy[None] - boxes[:, 1, None]) / (boxes[:, 3] - boxes[:, 1])[:, None] * 2 - 1
    grid = torch.stack((x[:, None].expand(-1, len(yy), -1),
                        y[:, :, None].expand(-1, -1, len(xx))), -1)
    return F.grid_sample(masks.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _validate_boxes(boxes, count, device, image_shape):
    if (not isinstance(boxes, torch.Tensor) or not boxes.is_floating_point()
            or tuple(boxes.shape) != (count, 4) or boxes.device != device
            or not bool(torch.isfinite(boxes).all())
            or bool((boxes[:, 2:] <= boxes[:, :2]).any())):
        raise ValueError("ROI-local masks require finite positive XYXY boxes[N,4] on the mask device")
    if (not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in image_shape)):
        raise ValueError("image_shape must be positive integer [H,W]")


@torch.no_grad()
def paste_roi_mask_tile(roi_mask, box, tile_xyxy, *, threshold=.5):
    """Return a bounded binary image-coordinate tile using the NMS convention.

    The caller supplies an image-clipped integer XYXY tile and its desired
    memory bound. ``box`` can be rescaled from input to original coordinates;
    ROI probabilities remain unchanged. No full-frame mask is constructed.
    """
    _probabilities(roi_mask, "roi_mask")
    if roi_mask.ndim != 2 or min(roi_mask.shape) < 1:
        raise ValueError("roi_mask must have positive [Hroi,Wroi] dimensions")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must lie in (0,1]")
    if (not isinstance(tile_xyxy, (tuple, list)) or len(tile_xyxy) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in tile_xyxy)
            or tile_xyxy[2] < tile_xyxy[0] or tile_xyxy[3] < tile_xyxy[1]):
        raise ValueError("tile_xyxy must be ordered integer [left,top,right,bottom]")
    box = torch.as_tensor(box, device=roi_mask.device, dtype=torch.float32).reshape(1, 4)
    if not bool(torch.isfinite(box).all()) or bool((box[:, 2:] <= box[:, :2]).any()):
        raise ValueError("box must contain finite positive XYXY extents")
    left, top, right, bottom = tile_xyxy
    if right == left or bottom == top:
        return torch.empty((bottom - top, right - left), dtype=torch.bool, device=roi_mask.device)
    return _sample_roi_window(roi_mask[None, None], box, tile_xyxy)[0, 0] >= threshold


@torch.no_grad()
def paste_roi_mask(mask, box, image_shape, *, mask_threshold=.5, paste_chunk_pixels=65536):
    """Return one binary crop and its integer XYXY window, never N*H*W masks.

    Uses the same align_corners=False image-pixel-center convention as ROI NMS.
    Probability zero padding stays background because mask_threshold is >0.
    """
    _probabilities(mask, "mask")
    if mask.ndim != 2:
        raise ValueError("mask must have shape [Hroi,Wroi]")
    _positive_integer(paste_chunk_pixels, "paste_chunk_pixels")
    if not 0 < mask_threshold <= 1:
        raise ValueError("mask_threshold must lie in (0,1]")
    box = torch.as_tensor(box, device=mask.device, dtype=torch.float32).reshape(1, 4)
    _validate_boxes(box, 1, mask.device, image_shape)
    window = _window(box[0].cpu().tolist(), image_shape)
    left, top, right, bottom = window
    result = torch.zeros((bottom - top, right - left), dtype=torch.bool, device=mask.device)
    for tile in _tiles(window, paste_chunk_pixels):
        x1, y1, x2, y2 = tile
        result[y1 - top:y2 - top, x1 - left:x2 - left] = (
            _sample_roi_window(mask[None, None], box, tile)[0, 0] >= mask_threshold)
    return result, window


@torch.no_grad()
def dual_mask_nms(amodal_masks, visible_masks, scores, *, amodal_iou_threshold=.5,
                  visible_iou_threshold=.45, mask_threshold=.5, mask_chunk_size=8,
                  boxes=None, image_shape=None, paste_chunk_pixels=65536):
    """Suppress only when BOTH A-IoU>.5 and V-IoU>.45, with stable ties.

    Inputs are probabilities/binary masks [N,H,W], not logits. With no boxes,
    grids already share image coordinates. With boxes, grids are ROI local and
    overlap/area is evaluated after exact image-pixel-center pasting, in bounded
    tiles. Original-resolution N-mask tensors and N*N*H*W products never exist.
    Empty/empty masks have IoU zero; no GT or exclusive visible map is used.
    """
    _positive_integer(mask_chunk_size, "mask_chunk_size")
    _probabilities(amodal_masks, "amodal_masks", chunk_size=mask_chunk_size)
    _probabilities(visible_masks, "visible_masks", chunk_size=mask_chunk_size)
    if (amodal_masks.ndim != 3 or visible_masks.shape != amodal_masks.shape
            or visible_masks.device != amodal_masks.device or min(amodal_masks.shape[-2:]) < 1):
        raise ValueError("A/V masks must share [N,H,W] and device")
    count, device = len(amodal_masks), amodal_masks.device
    if (not isinstance(scores, torch.Tensor) or not scores.is_floating_point()
            or tuple(scores.shape) != (count,) or scores.device != device
            or not bool(torch.isfinite(scores).all())):
        raise ValueError("scores must be finite floating[N] on the mask device")
    for value in (amodal_iou_threshold, visible_iou_threshold):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("IoU thresholds must lie in [0,1]")
    if not 0 < mask_threshold <= 1:
        raise ValueError("mask_threshold must lie in (0,1]")
    _positive_integer(paste_chunk_pixels, "paste_chunk_pixels")
    order = torch.argsort(scores, descending=True, stable=True).cpu().tolist()
    windows = None
    if boxes is not None:
        _validate_boxes(boxes, count, device, image_shape)
        boxes = boxes.detach().float()
        windows = [_window(box, image_shape) for box in boxes.cpu().tolist()]
    elif image_shape is not None:
        raise ValueError("image_shape is only used with ROI-local boxes")
    areas = torch.zeros((count, 2), dtype=torch.float64, device=device)
    for start in range(0, count, mask_chunk_size):
        for index in range(start, min(start + mask_chunk_size, count)):
            if windows is None:
                h, w = amodal_masks.shape[-2:]
                for x1, y1, x2, y2 in _tiles((0, 0, w, h), paste_chunk_pixels):
                    areas[index, 0] += (amodal_masks[index, y1:y2, x1:x2] >= mask_threshold).sum()
                    areas[index, 1] += (visible_masks[index, y1:y2, x1:x2] >= mask_threshold).sum()
            else:
                source = torch.stack((amodal_masks[index], visible_masks[index]))[None]
                for tile in _tiles(windows[index], paste_chunk_pixels):
                    areas[index] += (_sample_roi_window(source, boxes[index:index + 1], tile)
                                     >= mask_threshold).sum((0, 2, 3))
    area_values = areas.cpu().tolist()
    kept, suppressed = [], set()
    for position, index in enumerate(order):
        if index in suppressed:
            continue
        kept.append(index)
        remaining = [other for other in order[position + 1:] if other not in suppressed]
        for start in range(0, len(remaining), mask_chunk_size):
            for other in remaining[start:start + mask_chunk_size]:
                overlap = None
                if windows is not None:
                    a, b = windows[index], windows[other]
                    overlap = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
                    if overlap[2] <= overlap[0] or overlap[3] <= overlap[1]:
                        continue
                # An area-ratio/overlap upper bound can rule out suppression
                # without rendering either mask pair or synchronizing per pixel.
                possible = True
                for branch, threshold in enumerate((amodal_iou_threshold, visible_iou_threshold)):
                    max_intersection = min(area_values[index][branch], area_values[other][branch])
                    if overlap is not None:
                        max_intersection = min(max_intersection,
                                               (overlap[2] - overlap[0]) * (overlap[3] - overlap[1]))
                    min_union = area_values[index][branch] + area_values[other][branch] - max_intersection
                    upper_iou = max_intersection / min_union if min_union > 0 else 0.
                    if upper_iou <= threshold:
                        possible = False
                        break
                if not possible:
                    continue
                intersections = torch.zeros(2, dtype=torch.float64, device=device)
                if windows is None:
                    h, w = amodal_masks.shape[-2:]
                    for x1, y1, x2, y2 in _tiles((0, 0, w, h), paste_chunk_pixels):
                        for branch, masks in enumerate((amodal_masks, visible_masks)):
                            intersections[branch] += ((masks[index, y1:y2, x1:x2] >= mask_threshold)
                                                       & (masks[other, y1:y2, x1:x2] >= mask_threshold)).sum()
                else:
                    source = torch.stack((torch.stack((amodal_masks[index], visible_masks[index])),
                                          torch.stack((amodal_masks[other], visible_masks[other]))))
                    pair_boxes = boxes[[index, other]]
                    for tile in _tiles(overlap, paste_chunk_pixels):
                        binary = _sample_roi_window(source, pair_boxes, tile) >= mask_threshold
                        intersections += (binary[0] & binary[1]).sum((1, 2))
                union = areas[index] + areas[other] - intersections
                ious = torch.where(union > 0, intersections / union.clamp_min(1), torch.zeros_like(union))
                if float(ious[0]) > amodal_iou_threshold and float(ious[1]) > visible_iou_threshold:
                    suppressed.add(other)
    return torch.tensor(kept, dtype=torch.int64, device=device)
