"""High resolution, two-stage visible/hidden/amodal apple detector.

The data module already normalizes and resizes RGB. This model deliberately
bypasses torchvision's image transform: original, CPU instance masks are kept
at their annotated size and only small projected ROI targets reach the device.
Inference never receives targets. Final NMS and ablations are explicit config.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign, roi_align
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops.misc import FrozenBatchNorm2d

from .od_verifier import (AmodalEvidenceVerifier, DEFAULT_MARGIN, HEAVY_VISIBLE_RATIO,
                        verifier_ranking_loss)
from .od_roi import (OcclusionDecoupledROIMaskHead,
                     PredictedVisibleGatedClassifier, dual_mask_nms,
                     od_roi_mask_losses)


def varifocal_classification_loss(logits, quality, labels, *, alpha=.75, gamma=2.0,
                                  sample_weights=None, quality_margin=0.0, quality_gamma=1.0,
                                  negative_weight=1.0):
    """Binary varifocal loss with continuous IoU quality targets.

    Positive candidates use the proposal-to-matched-GT IoU as their target and
    are not down-weighted by a focal factor. Negative candidates retain the
    focal term so high-scoring background proposals receive a strong penalty.
    Labels below zero are ignored by the caller and must already be filtered.

    ``quality_margin``/``quality_gamma`` optionally sharpen the positive target:
    ``q <- ((q - margin)/(1 - margin))_+ ** quality_gamma``. With the defaults the
    target is bit-identical to the historical objective. Sharpening is the
    precision-oriented knob: a 0.50-IoU box stops being rewarded like a 0.85 one,
    so the classifier stops spending score on near-miss localisations.
    ``negative_weight`` scales only the background term, letting hard negatives be
    punished harder without changing the alpha/gamma shape that was already tuned.
    """
    if (logits.ndim != 2 or logits.shape[1] != 2 or quality.ndim != 1
            or labels.ndim != 1 or len(logits) != len(quality) or len(labels) != len(quality)):
        raise ValueError("VFL expects logits[N,2], quality[N] and labels[N]")
    if not 0 <= alpha <= 1 or gamma < 0:
        raise ValueError("VFL alpha/gamma must be finite and nonnegative")
    if not 0 <= quality_margin < 1 or quality_gamma <= 0 or negative_weight <= 0:
        raise ValueError("VFL quality margin/gamma and negative weight must be valid")
    if quality_margin or quality_gamma != 1.0:
        # One-sided sharpening: boxes at or below the margin lose their positive
        # reward entirely, while boxes near 1.0 keep almost all of it.
        quality = ((quality - quality_margin) / (1 - quality_margin)).clamp(0., 1.) ** quality_gamma
    positive = labels > 0
    negative = labels == 0
    if bool(((quality < 0) | (quality > 1)).any()) or not bool(torch.isfinite(quality).all()):
        raise ValueError("VFL quality targets must be finite probabilities")
    if sample_weights is None:
        sample_weights = logits.new_ones(len(labels))
    if (sample_weights.shape != labels.shape or not bool(torch.isfinite(sample_weights).all())
            or bool((sample_weights < 0).any())):
        raise ValueError("VFL sample weights must be finite nonnegative[N]")
    # Inference takes softmax over the two logits, so optimize the exact
    # foreground-vs-background log-odds. Using logits[:,1] alone would leave the
    # background logit unconstrained and disagree with the deployed probability.
    foreground = logits[:, 1] - logits[:, 0]
    probability = foreground.sigmoid()
    positive_loss = -quality * (quality * F.logsigmoid(foreground)
                                + (1 - quality) * F.logsigmoid(-foreground))
    negative_loss = alpha * negative_weight * probability.pow(gamma) * (-F.logsigmoid(-foreground))
    loss = torch.where(positive, positive_loss,
                       torch.where(negative, negative_loss, torch.zeros_like(foreground)))
    loss = loss * sample_weights
    active_positive = positive & (sample_weights > 0)
    active_negative = negative & (sample_weights > 0)
    normalizer = quality[active_positive].sum().clamp_min(1.) if bool(active_positive.any()) else \
        active_negative.sum().clamp_min(1).to(foreground.dtype)
    return loss.sum() / normalizer


def _mask_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Exact exclusive xyxy boxes, including one-pixel instances, on CPU."""
    boxes = torch.empty((len(masks), 4), dtype=torch.float32)
    for index, mask in enumerate(masks):
        rows, columns = torch.where(mask != 0)
        if rows.numel() == 0:
            raise ValueError(f"Empty annotated instance {index}; no GT may be discarded")
        boxes[index] = torch.tensor([columns.min(), rows.min(),
                                     columns.max() + 1, rows.max() + 1])
    return boxes


def prepare_detection_target(target: Mapping, input_hw: tuple[int, int],
                             device: torch.device | str, *, precise_boxes=False,
                             include_visible_boxes=False) -> dict:
    """Make device boxes/labels without moving or truncating source masks.

    Amodal annotations, when genuinely present, define PApple detection boxes.
    Visible-only sources use visible boxes and remain without amodal targets.
    Caller dictionaries and tensors are never changed.
    """
    visible, amodal = target.get("visible_masks"), target.get("amodal_masks")
    if (not isinstance(visible, torch.Tensor) or visible.device.type != "cpu"
            or visible.ndim != 3):
        raise ValueError("Original visible_masks[N,H,W] must stay on CPU")
    original_hw = tuple(int(value) for value in target.get("original_hw", ()))
    if original_hw != tuple(visible.shape[-2:]) or min(original_hw) < 1:
        raise ValueError("Original mask size and original_hw differ")
    labels = target.get("labels")
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 1
            or len(labels) != len(visible) or not bool((labels == 0).all())):
        raise ValueError("Source labels must contain one apple=0 per instance")
    if len(target.get("instance_ids", ())) != len(visible):
        raise ValueError("Preserve one source instance ID per original GT")
    validity = target.get("annotation_validity", {})
    if validity.get("visible") is not True:
        raise ValueError("Genuine visible annotations are required")
    if amodal is None:
        if validity.get("amodal") is not False:
            raise ValueError("Visible-only targets must declare amodal=False")
    elif (not isinstance(amodal, torch.Tensor) or amodal.device.type != "cpu"
          or amodal.shape != visible.shape or validity.get("amodal") is not True):
        raise ValueError("Only genuine aligned original CPU amodal masks are supported")
    boxes = _mask_boxes(visible if amodal is None else amodal)
    height, width = original_hw
    input_height, input_width = input_hw
    if precise_boxes:
        # Avoid FP32(scale)*edge overflowing an exact image boundary. This is
        # coordinate arithmetic for the new cascade, never a GT clamp/repair.
        boxes = (boxes.double() * torch.tensor(
            [input_width / width, input_height / height] * 2, dtype=torch.float64)).float()
    else:
        boxes *= torch.tensor([input_width / width, input_height / height] * 2)
    result = dict(target)
    result["boxes"] = boxes.to(device=device)
    if include_visible_boxes:
        # An independent genuine V frame for the extra proposal branch; no
        # projection/intersection with A, even for original V>A annotations.
        visible_boxes = (_mask_boxes(visible).double() * torch.tensor(
            [input_width / width, input_height / height] * 2, dtype=torch.float64)).float()
        result["visible_boxes"] = visible_boxes.to(device=device)
    # Torchvision reserves class zero for background; this translation is local.
    result["labels"] = torch.ones(len(visible), dtype=torch.int64, device=device)
    result["input_hw"] = tuple(input_hw)
    result["proposal_box_supervision"] = "visible" if amodal is None else "genuine_amodal"
    result["final_amodal_box_supervision"] = amodal is not None
    if amodal is not None:
        denominator = amodal.sum(dim=(-1, -2)).clamp_min(1)
        result["visible_ratio"] = (visible.sum(dim=(-1, -2)) / denominator).clamp(0, 1)
    else:
        result["visible_ratio"] = None
    return result


def project_original_masks(target: Mapping, proposals: torch.Tensor,
                           matched_idxs: torch.Tensor, output_size: int,
                           *, branch: str = "visible", chunk_size: int = 8) -> torch.Tensor:
    """Project original CPU masks in bounded chunks using torchvision ROIAlign.

    Proposals use resized input coordinates. They are mapped back to original
    coordinates before projection. Hidden supervision is the original binary
    difference A & ~V, formed BEFORE interpolation, never synthetic completion.
    Only [K,output_size,output_size] is transferred to the proposal device.
    """
    if branch not in ("visible", "amodal", "hidden"):
        raise ValueError("Unknown mask branch")
    if chunk_size < 1 or output_size < 1:
        raise ValueError("Projection sizes must be positive")
    if len(proposals) != len(matched_idxs):
        raise ValueError("Proposals and matched GT indices must align")
    if branch != "visible" and target.get("amodal_masks") is None:
        raise ValueError("A/H projection requires genuine amodal annotations")
    original_h, original_w = target["original_hw"]
    input_h, input_w = target["input_hw"]
    boxes = proposals.detach().to(device="cpu", dtype=torch.float32).clone()
    boxes *= torch.tensor([original_w / input_w, original_h / input_h] * 2)
    indices = matched_idxs.detach().to(device="cpu", dtype=torch.int64)
    projected = []
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        selected = indices[start:stop]
        if branch == "visible":
            masks = target["visible_masks"].index_select(0, selected)
        else:
            masks = target["amodal_masks"].index_select(0, selected)
            if branch == "hidden":
                visible = target["visible_masks"].index_select(0, selected)
                masks = masks.bool() & ~visible.bool()
        rois = torch.cat((torch.arange(stop - start, dtype=torch.float32)[:, None],
                          boxes[start:stop]), dim=1)
        small = roi_align(masks[:, None].float(), rois,
                          output_size=(output_size, output_size),
                          spatial_scale=1.0, sampling_ratio=2, aligned=False)[:, 0]
        # Binary annotation interpolation is mathematically in [0,1], but the
        # float32 CPU kernel can round a convex sum to 1 + one ULP. Bound only
        # that numerical projection; source masks are never changed. Genuine
        # invalid targets and nonfinite results still fail before the loss.
        tolerance = 8 * torch.finfo(small.dtype).eps
        if (small.requires_grad or not bool(torch.isfinite(small).all())
                or bool(((small < -tolerance) | (small > 1 + tolerance)).any())):
            raise ValueError(f"ROIAlign {branch} annotation projection exceeds finite roundoff bounds")
        small = small.clamp(0, 1)
        projected.append(small.to(device=proposals.device, dtype=proposals.dtype))
    return (torch.cat(projected) if projected else
            proposals.new_empty((0, output_size, output_size)))


class ODRoIHeads(RoIHeads):
    """Torchvision proposal assignment/box coder plus jointly gated ROI heads."""

    def __init__(self, source: RoIHeads, *, mask_roi_size=28,
                 mask_hidden_channels=256, roi_chunk_size=16,
                 mask_projection_chunk_size=8, score_thresh=.001,
                 detections_per_img=300, pre_mask_nms_topk=600,
                 checkpoint_roi=True, occlusion_weight=2.0,
                 focal_gamma=0.0, use_visible_gate=True,
                 use_amodal_refinement=True, final_nms="dual_mask",
                 hidden_tower_mode="shared_visible", zero_initialize_hidden_predictor=False,
                 unknown_amodal_classifier_policy="legacy_box_iou", occlusion_verifier=False,
                 classification_highres=False):
        if (type(unknown_amodal_classifier_policy) is not str or
                unknown_amodal_classifier_policy not in ("legacy_box_iou", "ignore_overlapping_visible")):
            raise ValueError("unknown_amodal_classifier_policy must be legacy_box_iou or ignore_overlapping_visible")
        super().__init__(source.box_roi_pool, source.box_head, source.box_predictor,
                         source.proposal_matcher.high_threshold,
                         source.proposal_matcher.low_threshold,
                         source.fg_bg_sampler.batch_size_per_image,
                         source.fg_bg_sampler.positive_fraction,
                         source.box_coder.weights, score_thresh, .5,
                         detections_per_img)
        self.mask_roi_pool = MultiScaleRoIAlign(["0", "1", "2", "3"],
                                                mask_roi_size, 2)
        if mask_roi_size != 28:
            raise ValueError("The high-resolution OD mask head requires ROIAlign size 28")
        self.od_mask_head = OcclusionDecoupledROIMaskHead(
            256, hidden_channels=mask_hidden_channels, hidden_tower_mode=hidden_tower_mode,
            zero_initialize_hidden_predictor=zero_initialize_hidden_predictor)
        self.gated_classifier = PredictedVisibleGatedClassifier(256)
        if type(occlusion_verifier) is not bool:
            raise ValueError("occlusion_verifier must be an explicit boolean")
        self.occlusion_verifier_enabled = occlusion_verifier
        # Zero-initialized readout: enabling it cannot change any initial output.
        if type(classification_highres) is not bool:
            raise ValueError("classification_highres must be an explicit boolean")
        # P2-only (stride 4) pooling for the classification evidence: the mask
        # head keeps its P2-P5 pool, but a heavily occluded apple is a small
        # visible patch that the coarse pool smears away. Adds no parameters.
        self.classification_highres = classification_highres
        self.highres_roi_pool = (MultiScaleRoIAlign(["0"], mask_roi_size, 2)
                                 if classification_highres else None)
        self.occlusion_verifier = AmodalEvidenceVerifier(256) if occlusion_verifier else None
        self.verifier_margin = DEFAULT_MARGIN
        # Recipe-level switch: the measured pilot showed the hardest-negative
        # ranking inflating near-duplicate scores, so it is off by default and
        # must be requested explicitly by a training recipe.
        self.verifier_ranking = True
        self.roi_chunk_size = int(roi_chunk_size)
        self.mask_projection_chunk_size = int(mask_projection_chunk_size)
        self.pre_mask_nms_topk = int(pre_mask_nms_topk)
        self.checkpoint_roi = bool(checkpoint_roi)
        self.occlusion_weight = float(occlusion_weight)
        self.focal_gamma = float(focal_gamma)
        self.classifier_only = False
        # Sampling/loss choices are training recipes, not tensor topology or
        # inference options. Defaults reproduce the historical cross entropy.
        self.classifier_negative_sampling = "random"
        self.classification_loss_type = "cross_entropy"
        self.vfl_alpha = .75
        self.vfl_gamma = 2.0
        # Precision-oriented VFL knobs; defaults reproduce the historical target.
        self.vfl_quality_margin = 0.0
        self.vfl_quality_gamma = 1.0
        self.vfl_negative_weight = 1.0
        # Localization recipe weight. Zero reproduces the historical Smooth L1
        # objective bit for bit, so the default never changes a published run.
        self.box_giou_weight = 0.0
        if type(use_visible_gate) is not bool or type(use_amodal_refinement) is not bool:
            raise ValueError("Ablation switches must be explicit booleans")
        if final_nms not in ("dual_mask", "box"):
            raise ValueError("final_nms must be dual_mask or box")
        self.use_visible_gate = use_visible_gate
        self.use_amodal_refinement = use_amodal_refinement
        self.final_nms = final_nms
        self.unknown_amodal_classifier_policy = unknown_amodal_classifier_policy
        self.configuration = dict(mask_roi_size=mask_roi_size,
                                  mask_hidden_channels=mask_hidden_channels,
                                  roi_chunk_size=roi_chunk_size,
                                  mask_projection_chunk_size=mask_projection_chunk_size,
                                  score_thresh=score_thresh,
                                  detections_per_img=detections_per_img,
                                  pre_mask_nms_topk=pre_mask_nms_topk,
                                  checkpoint_roi=checkpoint_roi,
                                  occlusion_weight=occlusion_weight,
                                  focal_gamma=focal_gamma,
                                  use_visible_gate=use_visible_gate,
                                  use_amodal_refinement=use_amodal_refinement,
                                  final_nms=final_nms,
                                  hidden_tower_mode=hidden_tower_mode,
                                  zero_initialize_hidden_predictor=zero_initialize_hidden_predictor,
                                  unknown_amodal_classifier_policy=unknown_amodal_classifier_policy,
                                  **({"occlusion_verifier": True} if occlusion_verifier else {}),
                                  **({"classification_highres": True} if classification_highres else {}))
        if min(self.roi_chunk_size, self.mask_projection_chunk_size,
               self.pre_mask_nms_topk, self.detections_per_img) < 1:
            raise ValueError("ROI, projection and prediction limits must be positive")

    def _mask_predictions(self, pooled):
        prediction = self.od_mask_head(pooled)
        if not self.use_amodal_refinement:
            head = self.od_mask_head
            if head.amodal_base_head is None or head.amodal_base_predictor is None:
                raise ValueError("Disabling A refinement requires the genuine legacy A-base topology")
            # Retain the independently trained legacy A output, never V. The
            # discarded conditioned A branch has no loss edge and grad=None;
            # genuine A losses still train the retained A-base modules.
            index = head.amodal_base_apple_index
            prediction["amodal_logits"] = head.amodal_base_predictor(head.amodal_base_head(pooled))[
                :, index:index + 1]
        return prediction

    def _roi_predictions(self, features, proposals, image_shapes, base_logits):
        # Batching is one image by contract. ROIAlign is chunked as well as its
        # tower, avoiding materialization of hundreds of [256,28,28] regions.
        if len(proposals) != 1:
            raise ValueError("OD detector requires one image per batch")
        count = len(proposals[0])
        outputs = {key: [] for key in ("visible_logits", "hidden_logits", "amodal_logits")}
        classification = []
        for start in range(0, count, self.roi_chunk_size):
            stop = min(count, start + self.roi_chunk_size)
            chunk_boxes = [proposals[0][start:stop]]
            pooled = self._pooled_roi_features(features, chunk_boxes, image_shapes)
            if self.classifier_only:
                pooled = pooled.detach()
            if self.training and self.checkpoint_roi and torch.is_grad_enabled():
                prediction = checkpoint(self._mask_predictions, pooled, use_reentrant=False)
            else:
                prediction = self._mask_predictions(pooled)
            # Classification evidence may read a finer grid than the mask head.
            classifier_pooled = (self.highres_roi_pool(features, chunk_boxes, image_shapes)
                                 if self.highres_roi_pool is not None else pooled)
            if self.classifier_only and classifier_pooled is not pooled:
                classifier_pooled = classifier_pooled.detach()
            # Class supervision may train V, but must never create H/A gradients
            # for MinneApple, whose annotations contain no genuine A or H.
            if self.use_visible_gate:
                gated = self.gated_classifier(
                    classifier_pooled, prediction["visible_logits"],
                    prediction["hidden_logits"].detach(),
                    prediction["amodal_logits"].detach(),
                    base_logits[start:stop].detach() if self.classifier_only else base_logits[start:stop])
            else:
                gated = base_logits[start:stop]
            gated = self._fuse_classification_evidence(
                features, chunk_boxes, image_shapes, classifier_pooled,
                prediction["visible_logits"], gated)
            if self.occlusion_verifier is not None:
                # Predicted-mask evidence only; every mask is detached inside.
                gated = self.occlusion_verifier(
                    classifier_pooled, prediction["visible_logits"], prediction["hidden_logits"],
                    prediction["amodal_logits"], gated)
            classification.append(gated)
            for key in outputs:
                outputs[key].append(prediction[key])
        if count == 0:
            side = int(self.mask_roi_pool.output_size[0]) * 2
            empty = base_logits.new_empty((0, 1, side, side))
            return {key: empty for key in outputs}, base_logits.new_empty((0, 2))
        return ({key: torch.cat(value) for key, value in outputs.items()},
                torch.cat(classification))

    def _pooled_roi_features(self, features, proposals, image_shapes):
        return self.mask_roi_pool(features, proposals, image_shapes)

    def _fuse_classification_evidence(self, features, proposals, image_shapes, pooled, visible_logits, logits):
        return logits

    def _verifier_losses(self, logits, labels):
        """Optional ranking term of the verifier; never emitted when it is off."""
        if self.occlusion_verifier is None or not self.verifier_ranking:
            return {}
        value = verifier_ranking_loss(logits, labels, margin=self.verifier_margin)
        return {} if value is None else {"loss_verifier_rank": value}

    def _candidate_visible_ratio(self, matched, target):
        """Per-candidate GT visible ratio; background and unknown A count as visible."""
        if self.occlusion_verifier is None or target.get("visible_ratio") is None:
            return None
        device = matched.device
        ratio = target["visible_ratio"].to(device=device, dtype=torch.float32)
        index = matched.clamp_min(0)
        return torch.where(matched >= 0, ratio[index], torch.ones_like(ratio[index]))

    def _quality_targets(self, labels, matched_idxs, targets, proposals):
        """Proposal-to-matched-GT IoU for every sampled positive candidate."""
        quality = torch.zeros(len(torch.cat(labels)), dtype=torch.float32,
                              device=torch.cat(labels).device)
        offset = 0
        for label, matched, target, boxes in zip(labels, matched_idxs, targets, proposals):
            positive = torch.where(label > 0)[0]
            if len(positive):
                gt_indices = matched[positive].detach().clamp_min(0)
                overlaps = box_ops.box_iou(boxes[positive].detach(),
                                           target["boxes"][gt_indices])
                quality[offset + positive] = overlaps.diag().to(quality)
            offset += len(label)
        return quality

    def _ignored_negative_weights(self, labels, proposals, targets):
        """Zero only ROI-classification loss on audited missing-label boxes.

        The boxes come from an explicit TRAIN-only audit file, never from VAL.
        They do not create positives, do not alter boxes/masks and do not affect
        RPN; they solely stop background BCE from punishing those regions.
        """
        weights = torch.ones(len(torch.cat(labels)), dtype=torch.float32,
                             device=torch.cat(labels).device)
        offset = 0
        for label, boxes, target in zip(labels, proposals, targets):
            ignore = target.get("ignore_boxes")
            if ignore is not None and len(ignore) and len(boxes):
                overlaps = box_ops.box_iou(boxes.detach(), ignore.to(boxes))
                ignored = (label == 0) & (overlaps.max(dim=1).values >= .5)
                weights[offset + torch.where(ignored)[0]] = 0
            offset += len(label)
        return weights

    def _classification_loss(self, logits, labels, matched_idxs, targets, proposals=None):
        flat_labels = torch.cat(labels)
        sample_weights = (self._ignored_negative_weights(labels, proposals, targets)
                          if proposals is not None else logits.new_ones(len(flat_labels)))
        if self.classification_loss_type == "varifocal":
            if proposals is None:
                raise ValueError("Varifocal classification loss requires proposal boxes")
            quality = self._quality_targets(labels, matched_idxs, targets, proposals)
            return varifocal_classification_loss(
                logits, quality, flat_labels, alpha=self.vfl_alpha, gamma=self.vfl_gamma,
                sample_weights=sample_weights, quality_margin=self.vfl_quality_margin,
                quality_gamma=self.vfl_quality_gamma, negative_weight=self.vfl_negative_weight)
        if self.classification_loss_type != "cross_entropy":
            raise ValueError("Unknown classification loss type")
        cross_entropy = F.cross_entropy(logits, flat_labels, reduction="none")
        weights = sample_weights
        offset = 0
        for label, matched, target in zip(labels, matched_idxs, targets):
            if target["visible_ratio"] is not None:
                positive = torch.where(label > 0)[0]
                ratio = target["visible_ratio"][matched[positive].detach().cpu()]
                weights[offset + positive] *= 1 + self.occlusion_weight * (1 - ratio.to(weights)) ** 2
            offset += len(label)
        if self.focal_gamma:
            cross_entropy = cross_entropy * (1 - torch.exp(-cross_entropy)) ** self.focal_gamma
        return (cross_entropy * weights).sum() / weights.sum().clamp_min(1)

    def _giou_loss(self, deltas, targets, proposals):
        """Overlap-aware localization term on the already-pooled ROI features.

        Smooth L1 scores the four sides independently, so a displaced box can
        keep a small coordinate error while its IoU with the target stays under
        0.5 — exactly the error that shows up as one partial-overlap FP plus one
        FN. Both boxes are decoded with the same coder, so this adds no second
        ROIAlign and cannot desynchronise the V/H/A masks from the box.
        """
        predicted = self.box_coder.decode(deltas, [proposals])
        target = self.box_coder.decode(targets, [proposals])
        return (1 - torch.diagonal(box_ops.generalized_box_iou(predicted, target))).mean()

    def _final_nms(self, masks, boxes, scores, image_shape):
        if self.final_nms == "box":
            return box_ops.nms(boxes, scores, .5)
        return dual_mask_nms(masks["amodal_logits"][:, 0],
                             masks["visible_logits"][:, 0], scores,
                             boxes=boxes, image_shape=image_shape)

    def _decoded_boxes(self, box_regression, proposals, image_shape):
        boxes = self.box_coder.decode(box_regression, proposals).reshape(-1, 2, 4)[:, 1]
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        valid = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        return boxes[valid], valid

    def _decoded_boxes_all(self, box_regression, proposals, image_shape):
        boxes = self.box_coder.decode(box_regression, [proposals]).reshape(-1, 2, 4)[:, 1]
        return box_ops.clip_boxes_to_image(boxes, image_shape)

    def _classifier_supervision_labels(self, boxes, labels, target):
        """Ignore ambiguous V-overlapping negatives when genuine A is unknown.

        This loss-only rule never invents a positive or changes the original
        target. Positive-area intersection is strict: touching edges remain
        background. Known-A sources and the legacy policy return unchanged.
        """
        if (self.unknown_amodal_classifier_policy == "legacy_box_iou"
                or target["amodal_masks"] is not None):
            return labels
        visible_boxes = target["boxes"]
        if not len(boxes) or not len(visible_boxes):
            return labels
        left_top = torch.maximum(boxes[:, None, :2], visible_boxes[None, :, :2])
        right_bottom = torch.minimum(boxes[:, None, 2:], visible_boxes[None, :, 2:])
        overlaps = (right_bottom > left_top).all(-1).any(-1)
        return labels.masked_fill((labels == 0) & overlaps, -1)

    def _classifier_training(self, features, proposals, image_shapes, targets):
        # Calibration sees the same RPN -> box-regression regions as inference.
        # GT affects only labels/sampling; no GT box is appended or pooled.
        with torch.no_grad():
            box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
            base_logits, box_regression = self.box_predictor(box_features)
            boxes, valid = self._decoded_boxes(box_regression, proposals, image_shapes[0])
            base_logits = base_logits[valid]
            matched, labels = self.assign_targets_to_proposals(
                [boxes], [target["boxes"] for target in targets],
                [target["labels"] for target in targets])
            labels = [self._classifier_supervision_labels(boxes, labels[0], targets[0])]
            occlusion = self._candidate_visible_ratio(matched[0], targets[0])
            sampled = self._classifier_samples(features, boxes, image_shapes, base_logits, labels[0],
                                               occlusion=occlusion)
            boxes, base_logits = boxes[sampled], base_logits[sampled]
            labels, matched = [labels[0][sampled]], [matched[0][sampled]]
        if not len(sampled):
            # No mask/backbone/box edge is attached when every candidate is
            # ignored (or the decoded pool is empty). Backward stays legal.
            zero = sum(parameter.sum() * 0 for parameter in self.gated_classifier.parameters())
            return [], {"loss_classifier": zero}
        _, logits = self._roi_predictions(features, [boxes], image_shapes, base_logits)
        return [], {"loss_classifier": self._classification_loss(
                        logits, labels, matched, targets, [boxes]),
                    **self._verifier_losses(logits, labels)}

    @torch.no_grad()
    def _negative_classifier_losses(self, features, boxes, image_shapes, base_logits, negative_indices):
        """Rank genuine background using the current predicted-visible gate.

        Score only TRAIN negatives after the unknown-A ignore rule. Discard
        each chunk's masks immediately; the selected ROIs are recomputed with
        gradients by the ordinary classifier loss. No backbone/RPN rerun.
        """
        losses = []
        for start in range(0, len(negative_indices), self.roi_chunk_size):
            indices = negative_indices[start:start + self.roi_chunk_size]
            masks, logits = self._roi_predictions(features, [boxes[indices]], image_shapes, base_logits[indices])
            values = F.cross_entropy(logits.float(), torch.zeros(len(indices), dtype=torch.int64,
                                                                 device=logits.device), reduction="none")
            del masks, logits
            if not bool(torch.isfinite(values).all()):
                raise ValueError("Hard-negative gate losses must be finite")
            losses.append(values)
        return torch.cat(losses) if losses else base_logits.new_empty((0,), dtype=torch.float32)

    @torch.no_grad()
    def _stratified_positives(self, positive, occlusion, num_positive):
        """At least half of the positive quota from heavily occluded apples.

        Heavily occluded apples are 44% of VAL GT yet keep a flat 0.79 recall
        across 9216 joint updates, i.e. they never receive enough positive
        gradient. This is a TRAIN sampling rule: it reads original annotation
        ratios only and never touches VAL pixels or GT values.
        """
        ratio = occlusion[positive]
        heavy = positive[ratio < HEAVY_VISIBLE_RATIO]
        other = positive[ratio >= HEAVY_VISIBLE_RATIO]
        if not len(heavy):
            return positive[torch.randperm(len(positive), device=positive.device)[:num_positive]]
        quota = min(len(heavy), max(num_positive // 2, num_positive - len(other)))
        chosen_heavy = heavy[torch.randperm(len(heavy), device=positive.device)[:quota]]
        remaining = num_positive - len(chosen_heavy)
        pool = other
        if len(pool) < remaining:
            spare = heavy[~torch.isin(heavy, chosen_heavy)]
            pool = torch.cat((spare, pool))
        chosen_other = pool[torch.randperm(len(pool), device=positive.device)[:remaining]]
        return torch.cat((chosen_heavy, chosen_other))

    def _classifier_samples(self, features, boxes, image_shapes, base_logits, labels, *, occlusion=None):
        policy = self.classifier_negative_sampling
        if policy == "random":
            return self.subsample([labels])[0]
        if policy not in ("gate_ohem_half", "gate_ohem_half_heavy"):
            raise ValueError("Unknown classifier_negative_sampling policy")

        # Match torchvision's BalancedPositiveNegativeSampler draws exactly,
        # retaining the negative permutation that its output masks discard.
        # Keeping that order avoids biasing the random remainder to low ROI IDs.
        positive, negative = torch.where(labels >= 1)[0], torch.where(labels == 0)[0]
        sampler = self.fg_bg_sampler
        num_positive = min(len(positive), int(sampler.batch_size_per_image * sampler.positive_fraction))
        num_negative = min(len(negative), sampler.batch_size_per_image - num_positive)
        if policy == "gate_ohem_half_heavy" and occlusion is not None:
            positive = self._stratified_positives(positive, occlusion, num_positive)
        else:
            positive = positive[torch.randperm(len(positive), device=positive.device)[:num_positive]]
        random_negative = negative[torch.randperm(len(negative), device=negative.device)[:num_negative]]
        hard_count = num_negative // 2
        if hard_count and len(negative) > num_negative:
            losses = self._negative_classifier_losses(features, boxes, image_shapes, base_logits, negative)
            if losses.shape != negative.shape or not bool(torch.isfinite(losses).all()):
                raise ValueError("Hard-negative ranking must contain one finite loss per eligible negative")
            hard = negative[torch.argsort(losses, descending=True, stable=True)[:hard_count]]
            remainder = random_negative[~torch.isin(random_negative, hard)][:num_negative - hard_count]
            selected_negative = torch.cat((hard, remainder))
        else:
            selected_negative = random_negative
        return torch.cat((positive, selected_negative)).sort().values

    def _mask_losses(self, predictions, proposals, labels, matched_idxs, targets):
        positives, visible, amodal, hidden, known = [], [], [], [], []
        offset = 0
        side = predictions["visible_logits"].shape[-1]
        for boxes, label, matched, target in zip(proposals, labels, matched_idxs, targets):
            positive = torch.where(label > 0)[0]
            positives.append(positive + offset)
            indices, rois = matched[positive], boxes[positive]
            options = {"chunk_size": self.mask_projection_chunk_size}
            visible.append(project_original_masks(target, rois, indices, side, **options))
            has_amodal = target["amodal_masks"] is not None
            known.append(torch.full((len(positive),), has_amodal,
                                    device=boxes.device, dtype=torch.bool))
            if has_amodal:
                amodal.append(project_original_masks(target, rois, indices, side,
                                                      branch="amodal", **options))
                hidden.append(project_original_masks(target, rois, indices, side,
                                                      branch="hidden", **options))
            offset += len(label)
        selected = torch.cat(positives)
        small_predictions = {key: value[selected] for key, value in predictions.items()}
        return od_roi_mask_losses(
            small_predictions, torch.cat(visible),
            amodal_targets=torch.cat(amodal) if amodal else None,
            amodal_valid=torch.cat(known),
            hidden_targets=torch.cat(hidden) if hidden else None)

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training and self.classifier_only:
            return self._classifier_training(features, proposals, image_shapes, targets)
        if self.training:
            proposals, matched, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        base_logits, box_regression = self.box_predictor(box_features)
        if self.training:
            predictions, class_logits = self._roi_predictions(features, proposals, image_shapes, base_logits)
            # VFL quality/ignore tests use the final regressed box, not the raw
            # RPN proposal. Detach is implicit inside the quality computation.
            quality_boxes = [self._decoded_boxes_all(box_regression, proposals[0],
                                                      image_shapes[0])]
            losses = {"loss_classifier": self._classification_loss(
                          class_logits, labels, matched, targets, quality_boxes),
                      **self._verifier_losses(class_logits, labels)}
            if not self.classifier_only:
                flat_labels, flat_targets = torch.cat(labels), torch.cat(regression_targets)
                amodal_positives, offset = [], 0
                for label, target in zip(labels, targets):
                    if (target["amodal_masks"] is not None
                            and target["annotation_validity"]["amodal"] is True):
                        amodal_positives.append(torch.where(label > 0)[0] + offset)
                    offset += len(label)
                supervised = (torch.cat(amodal_positives) if amodal_positives else
                              flat_labels.new_empty(0))
                # A visible box is a valid proposal/assignment target, but is
                # not an amodal-box annotation. Do not attach even a zero loss
                # to bbox_pred for such sources: grad=None also avoids AdamW
                # updating these parameters through decoupled weight decay.
                if len(supervised):
                    deltas = box_regression.reshape(len(flat_labels), 2, 4)
                    normalizer = max(len(flat_labels), 1)
                    box_loss = F.smooth_l1_loss(
                        deltas[supervised, 1], flat_targets[supervised], beta=1 / 9,
                        reduction="sum") / normalizer
                    if self.box_giou_weight:
                        box_loss = box_loss + self.box_giou_weight * self._giou_loss(
                            deltas[supervised, 1], flat_targets[supervised],
                            proposals[0][supervised])
                    losses["loss_box_reg"] = box_loss
                losses.update(self._mask_losses(predictions, proposals, labels, matched, targets))
            return [], losses
        # Every RPN proposal reaches the configured classifier. There is no raw-box
        # classification threshold or final box NMS that could discard it first.
        boxes, valid = self._decoded_boxes(box_regression, proposals, image_shapes[0])
        base_logits = base_logits[valid]
        predictions, class_logits = self._roi_predictions(features, [boxes], image_shapes, base_logits)
        scores = class_logits.softmax(-1)[:, 1]
        keep = torch.where(scores > self.score_thresh)[0]
        order = torch.argsort(scores[keep], descending=True, stable=True)
        keep = keep[order[:self.pre_mask_nms_topk]]
        boxes, scores = boxes[keep], scores[keep]
        masks = {key: value[keep].sigmoid() for key, value in predictions.items()}
        selected = self._final_nms(masks, boxes, scores, image_shapes[0])[:self.detections_per_img]
        result = {"boxes": boxes[selected], "scores": scores[selected],
                  "labels": torch.zeros(len(selected), dtype=torch.int64, device=boxes.device),
                  "roi_visible_masks": masks["visible_logits"][selected],
                  "roi_hidden_masks": masks["hidden_logits"][selected],
                  "roi_amodal_masks": masks["amodal_logits"][selected],
                  "image_hw": tuple(image_shapes[0])}
        return [result], {}


class ODAmodalDetector(nn.Module):
    """R50-FPN P2-P6/RPN, 28→56 ROI heads, visible-gated classification.

    Training returns a loss dictionary. Evaluation returns one detection dict,
    with ROI-local probability masks and their input-coordinate boxes. Keeping
    masks local permits bounded full-resolution evaluation and dual-mask NMS.
    """

    def __init__(self, source: MaskRCNN, *, roi_head_type=ODRoIHeads,
                 visible_proposals=False, visible_proposal_limit=300, **roi_options):
        super().__init__()
        self.backbone, self.rpn = source.backbone, source.rpn
        self.roi_heads = roi_head_type(source.roi_heads, **roi_options)
        self.visible_proposals = visible_proposals
        self.visible_proposal_limit = visible_proposal_limit
        self.shallow_detail = False
        if visible_proposals:
            from .od_proposals import VisibleProposalBranch
            self.visible_proposal_branch = VisibleProposalBranch(
                self.rpn, roi_chunk_size=roi_options.get("roi_chunk_size", 8))
        self.training_stage = "joint"
        self.checkpoint_provenance = None
        self.model_config = dict(self.roi_heads.configuration)

    def set_training_stage(self, stage: str):
        if stage not in ("joint", "classifier"):
            raise ValueError("Training stage must be joint or classifier")
        if stage == "classifier" and not self.roi_heads.use_visible_gate:
            raise ValueError("Classifier-only calibration requires use_visible_gate=True")
        self.training_stage = stage
        self.requires_grad_(stage == "joint")
        self.roi_heads.od_mask_head.set_hidden_feature_training(stage == "joint")
        if stage == "classifier":
            self.roi_heads.gated_classifier.requires_grad_(True)
            if self.roi_heads.occlusion_verifier is not None:
                self.roi_heads.occlusion_verifier.requires_grad_(True)
        self.roi_heads.classifier_only = stage == "classifier"
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "training_stage", None) == "classifier":
            self.backbone.eval()
            self.rpn.eval()
            self.roi_heads.od_mask_head.eval()
            self.roi_heads.box_head.eval()
            self.roi_heads.box_predictor.eval()
            if self.visible_proposals:
                self.visible_proposal_branch.eval()
            if self.shallow_detail:
                self.shallow_detail_fusion.eval()
        return self

    def forward(self, pixel_values, pixel_mask=None, *, targets=None):
        if (not isinstance(pixel_values, torch.Tensor) or not pixel_values.is_floating_point()
                or pixel_values.ndim != 4 or tuple(pixel_values.shape[:2]) != (1, 3)
                or min(pixel_values.shape[-2:]) < 1):
            raise ValueError("Expected already-normalized floating RGB[1,3,H,W]")
        if pixel_mask is not None:
            if (tuple(pixel_mask.shape) != (1, *pixel_values.shape[-2:])
                    or not bool(pixel_mask.all())):
                raise ValueError("Only an unpadded all-one input pixel_mask is supported")
        if self.training != (targets is not None):
            raise ValueError("Training requires targets; inference must not receive GT")
        if targets is not None and len(targets) != 1:
            raise ValueError("One full, untruncated target is required")
        height, width = pixel_values.shape[-2:]
        image_shape = (height, width)
        prepared = (None if targets is None else
                    [prepare_detection_target(
                        targets[0], image_shape, pixel_values.device,
                        precise_boxes=self.roi_heads.configuration.get("context_cascade", False),
                        include_visible_boxes=self.visible_proposals)])
        padded_h, padded_w = ((height + 31) // 32 * 32, (width + 31) // 32 * 32)
        images = ImageList(F.pad(pixel_values, (0, padded_w - width, 0, padded_h - height)),
                           [image_shape])
        if self.shallow_detail:
            from .od_detail import forward_backbone_with_detail
            features = forward_backbone_with_detail(self.backbone, images.tensors,
                                                    self.shallow_detail_fusion)
        else:
            features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        rpn_targets = None if self.training_stage == "classifier" else prepared
        proposals, proposal_losses = self.rpn(images, features, rpn_targets)
        if self.visible_proposals:
            branch_targets = prepared if self.training and self.training_stage == "joint" else None
            extra, extra_losses = self.visible_proposal_branch(images, features, branch_targets)
            # Preserve the entire original A route, including its order. Only
            # predicted V->A proposals (not auxiliary GT V boxes) are appended.
            proposals = [torch.cat((base, added[:self.visible_proposal_limit]))
                         for base, added in zip(proposals, extra)]
            proposal_losses.update(extra_losses)
        detections, detector_losses = self.roi_heads(features, proposals, [image_shape], prepared)
        if self.training:
            return {**proposal_losses, **detector_losses}
        return detections


def _load_coco_or_binary_source(source: MaskRCNN, checkpoint_path: str | Path,
                                expected_sha256: str | None = None) -> dict:
    """Load compatible detection tensors only; no optimizer or RNG restoration."""
    path = Path(checkpoint_path)
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise ValueError("Detector checkpoint SHA256 differs")
    # Complete historical payloads include non-tensor training metadata. Only
    # an explicitly hash-verified file may use the unrestricted legacy loader.
    payload = torch.load(path, map_location="cpu", weights_only=expected_sha256 is None,
                         mmap=True)
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise ValueError("Expected a detector tensor state dictionary")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    classes = state.get("roi_heads.box_predictor.cls_score.weight")
    if not isinstance(classes, torch.Tensor) or classes.shape[0] not in (2, 91):
        raise ValueError("Only audited binary or official 91-class R50-FPN heads are supported")
    if classes.shape[0] == 2:
        source.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        source.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    legacy_dual = any(key.startswith("roi_heads.visible_mask_head.") for key in state)
    if legacy_dual:
        source.roi_heads.visible_mask_head = deepcopy(source.roi_heads.mask_head)
        source.roi_heads.visible_mask_predictor = deepcopy(source.roi_heads.mask_predictor)
    expected = source.state_dict()
    # Legacy dual-mask heads may add tensors. Backbone/RPN/box/mask originals
    # must still match exactly; added heads are recorded rather than executed.
    filtered = {key: state[key] for key in expected if key in state}
    missing = sorted(set(expected) - set(filtered))
    mismatch = [key for key, value in filtered.items()
                if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape]
    if missing or mismatch:
        raise ValueError(f"Incompatible detector checkpoint: missing={missing}, shape={mismatch}")
    source.load_state_dict(filtered, strict=True)
    exact = all(torch.equal(value, filtered[key]) for key, value in source.state_dict().items())
    if not exact:
        raise RuntimeError("Source detector tensor restoration is not exact")
    return {"checkpoint_path": str(path.absolute()), "checkpoint_sha256": actual_sha256,
            "source_classes": int(classes.shape[0]), "loaded_tensors": len(filtered),
            "strict_source_tensor_load": True, "all_source_tensors_equal": exact,
            "legacy_independent_visible_amodal": legacy_dual,
            "source_epoch": payload.get("epoch"),
            "source_model_config": deepcopy(payload.get("model_config")),
            "extra_source_keys": sorted(set(state) - set(expected)),
            "state_scope": "model_only", "optimizer_restored": False,
            "checkpoint_RNG_restored": False}


CONVNEXT_DEFAULTS = dict(convnext_small=(96, 192, 384, 768), convnext_base=(128, 256, 512, 1024),
                         convnext_large=(192, 384, 768, 1536))
SWIN_DEFAULTS = dict(swin_s=(96, 192, 384, 768), swin_b=(128, 256, 512, 1024))
DEFAULT_STOCHASTIC_DEPTH = 0.1
REPLACED_BACKBONES = tuple(CONVNEXT_DEFAULTS) + tuple(SWIN_DEFAULTS)


def _convnext_fpn_backbone(name="convnext_small", stochastic_depth=DEFAULT_STOCHASTIC_DEPTH):
    """ConvNeXt body with the same 256-channel P2-P6 FPN contract as R50.

    The body keeps its ImageNet initialization. Only the four stage interfaces
    feed a fresh FPN, so the RPN and every ROI head keep their tensor shapes.
    ``stochastic_depth`` is train-only (identity at eval), but it is recorded in
    the constructor so a chunked continuation rebuilds the identical topology.
    """
    from torchvision import models
    if name not in CONVNEXT_DEFAULTS:
        raise ValueError("Unsupported ConvNeXt variant: " + str(name))
    weights = {"convnext_small": models.ConvNeXt_Small_Weights.DEFAULT,
               "convnext_base": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
               "convnext_large": models.ConvNeXt_Large_Weights.IMAGENET1K_V1}[name]
    builder = getattr(models, name)
    body = builder(weights=weights, stochastic_depth_prob=float(stochastic_depth)).features
    return BackboneWithFPN(body, return_layers={"1": "0", "3": "1", "5": "2", "7": "3"},
                           in_channels_list=list(CONVNEXT_DEFAULTS[name]), out_channels=256,
                           extra_blocks=LastLevelMaxPool())


class _SwinFeatureChain(nn.Module):
    """Run the Swin feature chain and return four NCHW maps (strides 4/8/16/32).

    torchvision's ``features`` chain is NHWC and feeds each stage into the next,
    so a permute cannot be spliced into the chain itself; the collected outputs
    are converted only after every stage has consumed the native layout.
    """

    def __init__(self, body, indices=(1, 3, 5, 7)):
        super().__init__()
        self.body = body
        self.indices = tuple(indices)

    def forward(self, tensor):
        outputs = {}
        for index, module in enumerate(self.body.children()):
            tensor = module(tensor)
            if index in self.indices:
                key = str(self.indices.index(index))
                outputs[key] = tensor.permute(0, 3, 1, 2).contiguous()
            if index == self.indices[-1]:
                break
        return outputs


class _SwinFPNBackbone(nn.Module):
    """Swin body plus a fresh 256-channel FPN with the R50 output contract."""

    def __init__(self, body, in_channels_list, out_channels=256):
        super().__init__()
        self.body = _SwinFeatureChain(body)
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels, extra_blocks=LastLevelMaxPool())
        self.out_channels = out_channels

    def forward(self, tensor):
        return self.fpn(self.body(tensor))


def _swin_fpn_backbone(name="swin_s", stochastic_depth=DEFAULT_STOCHASTIC_DEPTH):
    """Swin body with the same 256-channel P2-P6 FPN contract as R50.

    Window attention is a genuinely different inductive bias from both ResNet
    convolutions and ConvNeXt depthwise convolutions, which is the point of
    adding it to the fusion pool rather than replacing an existing arm.
    """
    from torchvision import models
    if name not in SWIN_DEFAULTS:
        raise ValueError("Unsupported Swin variant: " + str(name))
    weights = models.Swin_S_Weights.DEFAULT if name == "swin_s" else models.Swin_B_Weights.DEFAULT
    body = getattr(models, name)(weights=weights).features
    if float(stochastic_depth) != 0.1:
        # torchvision hard-codes 0.1 for Swin, so the requested rate is applied
        # to the stochastic-depth modules after construction instead.
        from torchvision.ops.stochastic_depth import StochasticDepth
        for module in body.modules():
            if isinstance(module, StochasticDepth):
                module.p = float(stochastic_depth)
    return _SwinFPNBackbone(body, list(SWIN_DEFAULTS[name]))


def build_od_amodal_detector(*, weights=None, checkpoint_path=None, expected_sha256=None,
                             backbone_name="resnet50",
                             backbone_stochastic_depth=DEFAULT_STOCHASTIC_DEPTH,
                             rpn_pre_nms_top_n_train=2000,
                             rpn_pre_nms_top_n_test=1000,
                             rpn_post_nms_top_n_train=1000,
                             rpn_post_nms_top_n_test=1000,
                             box_batch_size_per_image=256,
                             box_positive_fraction=.5,
                             anchor_sizes=(32, 64, 128, 256, 512),
                             amodal_base=None, hidden_tower_mode="shared_visible",
                             zero_initialize_hidden_predictor=False,
                             context_cascade=False, context_expansion=2.2,
                             cascade_iou_thresholds=(.5, .6, .7),
                             visible_proposals=False, visible_proposal_limit=300,
                             shallow_detail=False,
                             mask_box_refinement=False, occlusion_verifier=False,
                             classification_highres=False,
                             unknown_amodal_classifier_policy="legacy_box_iou", **roi_options):
    """Build on CPU; ``weights=None`` never downloads pretrained tensors.

    A local official MaskRCNN COCO V1 checkpoint preserves its backbone, FPN,
    RPN, box features, apple/background predictors and compatible visible tower.
    COCO apple is category ID 53. Binary checkpoints use apple row one.
    """
    if weights is not None and checkpoint_path is not None:
        raise ValueError("Supply official weights or one local checkpoint")
    if backbone_name != "resnet50" and backbone_name not in REPLACED_BACKBONES:
        raise ValueError("backbone_name must be resnet50 or one of " + ", ".join(REPLACED_BACKBONES))
    if (isinstance(backbone_stochastic_depth, bool)
            or not isinstance(backbone_stochastic_depth, (int, float))
            or not 0 <= backbone_stochastic_depth <= .5):
        raise ValueError("backbone_stochastic_depth must be a number in [0,0.5]")
    if backbone_name != "resnet50" and (weights is not None or checkpoint_path is not None):
        raise ValueError("A replaced backbone cannot claim the R50 official checkpoint")
    if type(hidden_tower_mode) is not str or hidden_tower_mode not in ("shared_visible", "independent"):
        raise ValueError("hidden_tower_mode must be shared_visible or independent")
    if type(zero_initialize_hidden_predictor) is not bool:
        raise ValueError("zero_initialize_hidden_predictor must be an explicit bool")
    if type(context_cascade) is not bool:
        raise ValueError("context_cascade must be an explicit bool")
    if type(visible_proposals) is not bool:
        raise ValueError("visible_proposals must be an explicit bool")
    if type(shallow_detail) is not bool:
        raise ValueError("shallow_detail must be an explicit bool")
    if shallow_detail and not visible_proposals:
        raise ValueError("shallow_detail requires visible_proposals=True")
    if type(mask_box_refinement) is not bool:
        raise ValueError("mask_box_refinement must be an explicit bool")
    if type(occlusion_verifier) is not bool:
        raise ValueError("occlusion_verifier must be an explicit bool")
    # The verifier only needs the final boxes plus the head's own predicted
    # V/H/A maps, so it is allowed on the plain head as well: adding it to the
    # parent architecture is the only way to attribute its effect, since every
    # context/proposal/detail arm carries its own untrained sub-networks.
    if type(classification_highres) is not bool:
        raise ValueError("classification_highres must be an explicit bool")
    if mask_box_refinement and (not shallow_detail or amodal_base is not True):
        raise ValueError("mask_box_refinement requires shallow_detail=True and amodal_base=True")
    if type(visible_proposal_limit) is not int or not 1 <= visible_proposal_limit <= 1000:
        raise ValueError("visible_proposal_limit must be an integer in [1,1000]")
    if visible_proposals and not context_cascade:
        raise ValueError("visible_proposals requires context_cascade=True")
    if not visible_proposals and visible_proposal_limit != 300:
        raise ValueError("visible_proposal_limit requires visible_proposals=True")
    head_type = ODRoIHeads
    if context_cascade:
        from .od_context_model import ContextODRoIHeads
        head_type = ContextODRoIHeads
        if mask_box_refinement:
            from .od_mask_geometry_model import MaskGeometryODRoIHeads
            head_type = MaskGeometryODRoIHeads
        roi_options = dict(roi_options, context_expansion=context_expansion,
                           cascade_iou_thresholds=cascade_iou_thresholds,
                           occlusion_verifier=occlusion_verifier,
                           classification_highres=classification_highres)
    else:
        if context_expansion != 2.2 or tuple(cascade_iou_thresholds) != (.5, .6, .7):
            raise ValueError("Context options require context_cascade=True")
        roi_options = dict(roi_options, classification_highres=classification_highres,
                           occlusion_verifier=occlusion_verifier)
    if (type(unknown_amodal_classifier_policy) is not str or
            unknown_amodal_classifier_policy not in ("legacy_box_iou", "ignore_overlapping_visible")):
        raise ValueError("unknown_amodal_classifier_policy must be legacy_box_iou or ignore_overlapping_visible")
    with torch.device("cpu"):
        if weights is None:
            backbone = (resnet_fpn_backbone(backbone_name="resnet50", weights=None,
                                            norm_layer=FrozenBatchNorm2d, trainable_layers=5)
                        if backbone_name == "resnet50" else
                        _swin_fpn_backbone(backbone_name, backbone_stochastic_depth)
                        if backbone_name in SWIN_DEFAULTS else
                        _convnext_fpn_backbone(backbone_name, backbone_stochastic_depth))
            source = MaskRCNN(backbone, num_classes=91,
                             rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                             rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                             rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                             rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                             box_batch_size_per_image=box_batch_size_per_image,
                             box_positive_fraction=box_positive_fraction)
        else:
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            source = maskrcnn_resnet50_fpn(
                weights=weights, weights_backbone=None,
                rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                box_batch_size_per_image=box_batch_size_per_image,
                box_positive_fraction=box_positive_fraction,
                trainable_backbone_layers=5)
        provenance = (None if checkpoint_path is None else
                      _load_coco_or_binary_source(source, checkpoint_path, expected_sha256))
        original_predictor = source.roi_heads.box_predictor
        classes = original_predictor.cls_score.out_features
        apple = 53 if classes == 91 else 1
        predictor = FastRCNNPredictor(1024, 2)
        with torch.no_grad():
            for name in ("weight", "bias"):
                getattr(predictor.cls_score, name).copy_(
                    getattr(original_predictor.cls_score, name)[[0, apple]])
                getattr(predictor.bbox_pred, name).copy_(
                    getattr(original_predictor.bbox_pred, name).reshape(classes, 4, -1)[[0, apple]].reshape(
                        getattr(predictor.bbox_pred, name).shape))
        source.roi_heads.box_predictor = predictor
        if len(anchor_sizes) != 5 or any(int(size) < 1 for size in anchor_sizes):
            raise ValueError("One positive anchor size is required for each P2-P6 level")
        source.rpn.anchor_generator = AnchorGenerator(
            sizes=tuple((int(size),) for size in anchor_sizes),
            aspect_ratios=((.5, 1., 2.),) * 5)
        model = ODAmodalDetector(source, roi_head_type=head_type,
                                visible_proposals=visible_proposals, visible_proposal_limit=visible_proposal_limit,
                                hidden_tower_mode=hidden_tower_mode,
                                zero_initialize_hidden_predictor=zero_initialize_hidden_predictor,
                                unknown_amodal_classifier_policy=unknown_amodal_classifier_policy, **roi_options)
        if checkpoint_path is not None or weights is not None:
            visible_transfer = model.roi_heads.od_mask_head.initialize_visible_from_maskrcnn(
                getattr(source.roi_heads, "visible_mask_head", source.roi_heads.mask_head),
                getattr(source.roi_heads, "visible_mask_predictor", source.roi_heads.mask_predictor),
                apple_index=apple)
            if hasattr(source.roi_heads, "visible_mask_head") and amodal_base is not False:
                amodal_transfer = model.roi_heads.od_mask_head.initialize_amodal_base_from_maskrcnn(
                    source.roi_heads.mask_head, source.roi_heads.mask_predictor, apple_index=apple)
                provenance["mask_warm_start"] = "actual_legacy_visible_and_amodal_towers"
                provenance["amodal_refinement_initialization"] = "zero_residual_over_trained_amodal_base"
                provenance["amodal_transfer"] = amodal_transfer
            if provenance is not None:
                provenance["visible_transfer"] = visible_transfer
                shared_prefixes = ("backbone.", "rpn.", "roi_heads.box_head.",
                                   "roi_heads.box_predictor.")
                shared = {key: value for key, value in source.state_dict().items()
                          if key.startswith(shared_prefixes)}
                actual_state = model.state_dict()
                equal = all(torch.equal(value, actual_state[key]) for key, value in shared.items())
                if not equal:
                    raise RuntimeError("Transferred backbone/RPN/box tensors differ from source")
                provenance["transferred_shared_tensor_count"] = len(shared)
                provenance["transferred_shared_tensors_equal"] = equal
        if amodal_base and model.roi_heads.od_mask_head.amodal_base_head is None:
            # Topology-only reconstruction for a saved OD state needs no parent
            # checkpoint. Keep its binary A predictor shape identical to legacy.
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            binary_mask = MaskRCNNPredictor(256, 256, 2)
            with torch.no_grad():
                binary_mask.conv5_mask.load_state_dict(source.roi_heads.mask_predictor.conv5_mask.state_dict())
                binary_mask.mask_fcn_logits.weight.copy_(
                    source.roi_heads.mask_predictor.mask_fcn_logits.weight[[0, apple]])
                binary_mask.mask_fcn_logits.bias.copy_(
                    source.roi_heads.mask_predictor.mask_fcn_logits.bias[[0, apple]])
            model.roi_heads.od_mask_head.initialize_amodal_base_from_maskrcnn(
                source.roi_heads.mask_head, binary_mask, apple_index=1)
        for module in model.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.0
        if (not model.roi_heads.use_amodal_refinement
                and model.roi_heads.od_mask_head.amodal_base_head is None):
            raise ValueError("use_amodal_refinement=False requires the genuine legacy A base; never fall back to V")
        model.checkpoint_provenance = deepcopy(provenance) if provenance is not None else {
            "state_scope": "random_initialization" if weights is None else "official_coco_weights",
            "weights": None if weights is None else str(weights), "source_apple_row": apple,
            "optimizer_restored": False, "checkpoint_RNG_restored": False}
        architecture_name = (f"od_amodal_{backbone_name}_fpn_v1" if backbone_name != "resnet50" else
                             "od_dual_proposal_context_r50_fpn_v1" if visible_proposals else
                             "od_context_cascade_foreground_r50_fpn_v1" if context_cascade else
                             "od_amodal_r50_fpn_v1")
        model.model_config.update({"architecture": architecture_name,
                                   "normalized_input": True, "final_nms": model.roi_heads.final_nms,
                                   "use_visible_gate": model.roi_heads.use_visible_gate,
                                   "use_amodal_refinement": model.roi_heads.use_amodal_refinement,
                                   "hidden_tower_mode": hidden_tower_mode,
                                   "zero_initialize_hidden_predictor": zero_initialize_hidden_predictor,
                                   "unknown_amodal_classifier_policy": unknown_amodal_classifier_policy,
                                   "rpn_proposal_supervision": {
                                       "genuine_amodal_sources": "amodal_boxes",
                                       "visible_only_sources": "visible_boxes"},
                                   "final_amodal_box_supervision": "genuine_amodal_positives_only",
                                   "source_apple_row": apple,
                                   "anchor_sizes": list(anchor_sizes),
                                   "box_batch_size_per_image": box_batch_size_per_image,
                                   "box_positive_fraction": box_positive_fraction,
                                   "rpn_pre_nms_top_n_train": rpn_pre_nms_top_n_train,
                                   "rpn_pre_nms_top_n_test": rpn_pre_nms_top_n_test,
                                   "rpn_post_nms_top_n_train": rpn_post_nms_top_n_train,
                                   "rpn_post_nms_top_n_test": rpn_post_nms_top_n_test})
        model.constructor_config = {
            **model.roi_heads.configuration,
            "rpn_pre_nms_top_n_train": rpn_pre_nms_top_n_train,
            "rpn_pre_nms_top_n_test": rpn_pre_nms_top_n_test,
            "rpn_post_nms_top_n_train": rpn_post_nms_top_n_train,
            "rpn_post_nms_top_n_test": rpn_post_nms_top_n_test,
            "box_batch_size_per_image": box_batch_size_per_image,
            "box_positive_fraction": box_positive_fraction,
            "anchor_sizes": list(anchor_sizes),
            "amodal_base": model.roi_heads.od_mask_head.amodal_base_head is not None}
        if backbone_name != "resnet50":
            model.constructor_config["backbone_name"] = backbone_name
            if backbone_stochastic_depth != DEFAULT_STOCHASTIC_DEPTH:
                model.constructor_config["backbone_stochastic_depth"] = float(backbone_stochastic_depth)
        model.model_config["amodal_base"] = model.constructor_config["amodal_base"]
        if visible_proposals:
            model.constructor_config.update(visible_proposals=True, visible_proposal_limit=visible_proposal_limit)
            model.model_config.update(visible_proposals=True, visible_proposal_limit=visible_proposal_limit,
                visible_proposal_supervision="genuine_V_RPN_matching_then_genuine_A_conversion_only",
                original_amodal_proposals_preserved=True, auxiliary_GT_V_proposals_returned=False)
            model.checkpoint_provenance["visible_proposal_initialization"] = deepcopy(
                model.visible_proposal_branch.initialization_provenance)
        if shallow_detail:
            # Install only after every original parameter has been initialized.
            # The original backbone object and state paths stay intact.
            from .od_detail import ShallowDetailFusion
            with torch.random.fork_rng(devices=[]):
                model.shallow_detail_fusion = ShallowDetailFusion()
            model.shallow_detail = True
            model.constructor_config["shallow_detail"] = True
            model.model_config.update(shallow_detail=True,
                architecture="od_dual_proposal_context_detail_r50_fpn_v1")
            model.checkpoint_provenance["shallow_detail_initialization"] = {
                "source": "genuine_stem_relu_stride2", "destination": "P2",
                "zero_initialized_residual": True, "original_body_fpn_state_paths_preserved": True,
                "optimizer_restored": False, "checkpoint_RNG_restored": False}
        if mask_box_refinement:
            # Copy after the genuine parent A base has been restored; this
            # allocates independent geometry parameters without changing old
            # values or drawing a new random trajectory.
            with torch.random.fork_rng(devices=[]):
                model.roi_heads.initialize_mask_geometry()
            model.constructor_config['mask_box_refinement'] = True
            model.model_config.update(mask_box_refinement=True,
                architecture='od_mask_geometry_dual_detail_r50_fpn_v1')
            model.checkpoint_provenance['mask_geometry_initialization'] = deepcopy(
                model.roi_heads.mask_geometry_head.initialization_provenance)
        return model
