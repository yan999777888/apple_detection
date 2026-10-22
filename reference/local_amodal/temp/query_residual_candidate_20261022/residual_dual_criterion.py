"""Real-PApple V/A/box supervision with one assignment per decoder stage.

The unchanged baseline criterion supplies provenance/mask validation and point
sampling/loss methods. Only final and penultimate stages are supervised. Import
is standard-library-only; the explicit factory lazily imports Torch and SciPy.
All returned scalar losses are already weighted and must be summed exactly once.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "query_dual_mask_candidate_20261021"
FROZEN_CRITERION_SHA256 = "6ceec79c357b0e30864dfd47e0113b574ce7b65eecd24c40ae193bdf9e5b74ee"
RESIDUAL_CRITERION_CONFIGURATION = {
    "match_points": 12544, "loss_points": 12544,
    "oversample_ratio": 3.0, "importance_ratio": 0.75,
    "class_weight": 2.0, "mask_weight": 5.0, "dice_weight": 5.0,
    "no_object_weight": 0.1, "visible_weight": 0.5, "amodal_weight": 0.5,
    "bbox_weight": 5.0, "giou_weight": 2.0, "auxiliary_weight": 0.5,
    "auxiliary_stages": 1, "original_mask_size": [1300, 1300],
    "batch_size": 1, "external_padding": False,
    "box_target_format": "original_xyxy", "box_prediction_format": "normalized_cxcywh",
    "shared_assignment_per_stage": True,
}
FINAL_LOSS_KEYS = ("loss_class", "loss_visible_bce", "loss_visible_dice",
                   "loss_amodal_bce", "loss_amodal_dice", "loss_bbox", "loss_giou")


def residual_criterion_configuration():
    """Return independent plain metadata; no numerical import or execution."""
    return deepcopy(RESIDUAL_CRITERION_CONFIGURATION)


def _frozen_factory():
    path = (BASE / "query_dual_criterion.py").resolve()
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != FROZEN_CRITERION_SHA256:
            raise ValueError("Frozen dual-mask criterion source changed")
    cached = sys.modules.get("query_dual_criterion")
    if cached is not None and Path(getattr(cached, "__file__", "")).resolve() != path:
        raise ImportError("A shadow criterion would replace the pinned baseline")
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    module = importlib.import_module("query_dual_criterion")
    if Path(module.__file__).resolve() != path:
        raise ImportError("Imported criterion is not the pinned baseline")
    return module.build_query_dual_criterion


def build_residual_dual_criterion():
    """criterion(outputs, [target], generator=None)->14 weighted scalar losses.

    Targets retain the original CPU float32 xyxy boxes and binary V/A masks at
    1300x1300. One stage's shared Hungarian assignment supervises both masks and
    boxes. The penultimate stage gets its own shared assignment and half weight.
    """
    baseline = _frozen_factory()()
    config = residual_criterion_configuration()
    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment

    def cxcywh_to_xyxy(boxes):
        center, size = boxes[..., :2], boxes[..., 2:]
        return torch.cat((center - size/2, center + size/2), dim=-1)

    def xyxy_to_cxcywh(boxes):
        lower, upper = boxes[..., :2], boxes[..., 2:]
        return torch.cat(((lower+upper)/2, upper-lower), dim=-1)

    def pairwise_giou(boxes1, boxes2):
        # Boxes are unclipped for matching/loss; corrected predictions can cross
        # the frame edge. Union and enclosing area use true geometric lengths.
        area1 = (boxes1[:, 2:]-boxes1[:, :2]).clamp(min=0).prod(-1)
        area2 = (boxes2[:, 2:]-boxes2[:, :2]).clamp(min=0).prod(-1)
        intersection_lower = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        intersection_upper = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        intersection = (intersection_upper-intersection_lower).clamp(min=0).prod(-1)
        union = area1[:, None] + area2[None, :] - intersection
        enclosing_lower = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
        enclosing_upper = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        enclosing = (enclosing_upper-enclosing_lower).clamp(min=0).prod(-1)
        # eps (1e-7 in fp32) would corrupt IoU for valid tiny normalized boxes.
        # tiny only prevents a zero denominator while preserving such geometry.
        epsilon = torch.finfo(boxes1.dtype).tiny
        return intersection/union.clamp(min=epsilon) - (enclosing-union)/enclosing.clamp(min=epsilon)

    def sample_points(masks, coordinates):
        return F.grid_sample(masks[:, None].float(), coordinates[:, None].float()*2-1,
                             mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0, 0]

    def pairwise_mask_cost(prediction, truth):
        positive = F.binary_cross_entropy_with_logits(prediction, torch.ones_like(prediction), reduction="none")
        negative = F.binary_cross_entropy_with_logits(prediction, torch.zeros_like(prediction), reduction="none")
        bce = (positive @ truth.transpose(0, 1) + negative @ (1-truth).transpose(0, 1))/prediction.shape[-1]
        probability = prediction.sigmoid()
        dice = 1-(2*(probability @ truth.transpose(0, 1))+1)/(probability.sum(-1)[:, None]+truth.sum(-1)[None, :]+1)
        return bce, dice

    class ResidualDualCriterion(type(baseline)):
        def __init__(self):
            super().__init__()
            self.configuration = deepcopy(config)
            self.criterion_metadata = {
                **deepcopy(baseline.criterion_metadata),
                "schema": "shared_hungarian_real_dual_mask_residual_box_aux_criterion_v1",
                "frozen_mask_criterion_sha256": FROZEN_CRITERION_SHA256,
                "reused_frozen_methods": ["_validate", "_sample_target", "_branch_losses", "_uncertain_coordinates"],
                "same_assignment_for_visible_amodal_boxes": True,
                "final_layer_only": False, "auxiliary_losses": True,
                "auxiliary_decoder_stages": "penultimate_only", "box_target_origin": "original_target.boxes exact CPU float32",
                "prediction_box_loss_geometry": "unclipped normalized xyxy for GIoU; normalized cxcywh for L1",
                "source_GT_clipping_filtering_repair": False,
            }

        def _validate_stage(self, stage, targets):
            if not isinstance(stage, dict) or set(stage) != {
                    "class_queries_logits", "visible_masks_queries_logits", "amodal_masks_queries_logits", "amodal_boxes"}:
                raise ValueError("Each supervised stage must contain exactly class/V/A logits and amodal_boxes")
            mask_outputs = {key: stage[key] for key in (
                "class_queries_logits", "visible_masks_queries_logits", "amodal_masks_queries_logits")}
            classes, visible, amodal, target, count = super()._validate(mask_outputs, targets)
            if tuple(target["amodal_masks"].shape[-2:]) != (1300, 1300):
                raise ValueError("Preserve original paired TRAIN masks at1300x1300")
            for key in ("labels", "mask_valid", "visible_masks", "amodal_masks"):
                if target[key].device.type != "cpu":
                    raise ValueError("Original TRAIN target tensors must remain on CPU")
            truth_boxes = target.get("boxes")
            if (not isinstance(truth_boxes, torch.Tensor) or truth_boxes.dtype != torch.float32
                    or tuple(truth_boxes.shape) != (count, 4) or truth_boxes.device.type != "cpu"
                    or truth_boxes.requires_grad):
                raise TypeError("Original boxes must be fixed CPU float32[N,4] xyxy")
            if not bool(torch.isfinite(truth_boxes).all()):
                raise ValueError("Nonfinite source boxes must fail without repair")
            if count and not bool((truth_boxes[:, 2:] > truth_boxes[:, :2]).all()):
                raise ValueError("Original source boxes must have positive width/height; never filter instances")
            boxes = stage["amodal_boxes"]
            if (not isinstance(boxes, torch.Tensor) or not boxes.is_floating_point()
                    or tuple(boxes.shape) != (1, 100, 4) or boxes.device != classes.device):
                raise TypeError("amodal_boxes must be floating normalized cxcywh[1,100,4]")
            if (not bool(torch.isfinite(boxes).all()) or not bool((boxes[..., :2] >= 0).all())
                    or not bool((boxes[..., :2] <= 1).all())
                    or not bool((boxes[..., 2:] > 0).all()) or not bool((boxes[..., 2:] <= 1).all())):
                raise ValueError("Predicted centers/positive sizes must meet normalized residual-box contract")
            return classes, visible, amodal, boxes, target, count

        def _normalized_targets(self, target, device):
            height, width = target["amodal_masks"].shape[-2:]
            # Exact source coordinates are divided only; no clipping, filtering,
            # rasterized-box surrogate, mask repair, or GT containment operation.
            boxes = target["boxes"].to(device=device, dtype=torch.float32)
            xyxy = boxes/boxes.new_tensor((width, height, width, height))
            return xyxy, xyxy_to_cxcywh(xyxy)

        def _match_stage(self, classes, visible, amodal, boxes, target, count, truth_xyxy, truth_cxcywh, generator):
            device = classes.device
            if count == 0:
                empty = torch.empty(0, device=device, dtype=torch.int64)
                return empty, empty
            with torch.no_grad():
                coordinates = torch.rand((1, config["match_points"], 2), device=device, generator=generator)
                query_coordinates = coordinates.expand(100, -1, -1)
                target_coordinates = coordinates.expand(count, -1, -1)
                target_indices = torch.arange(count, device=device)
                cost = -config["class_weight"]*classes[0].float().softmax(-1)[:, 0:1].expand(-1, count)
                for prediction, mask_key, branch_weight in (
                        (visible[0], "visible_masks", config["visible_weight"]),
                        (amodal[0], "amodal_masks", config["amodal_weight"])):
                    prediction_points = sample_points(prediction.detach().float(), query_coordinates)
                    truth_points = self._sample_target(target[mask_key], target_indices, target_coordinates, device)
                    bce, dice = pairwise_mask_cost(prediction_points, truth_points)
                    cost = cost + branch_weight*(config["mask_weight"]*bce + config["dice_weight"]*dice)
                prediction_boxes = boxes[0].detach().float()
                cost = cost + config["bbox_weight"]*torch.cdist(prediction_boxes, truth_cxcywh, p=1)
                cost = cost - config["giou_weight"]*pairwise_giou(cxcywh_to_xyxy(prediction_boxes), truth_xyxy)
                if not bool(torch.isfinite(cost).all()):
                    raise ValueError("Nonfinite class/V/A/box shared Hungarian costs")
                rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
                if len(rows) != count or len(set(rows)) != count or set(columns) != set(range(count)):
                    raise RuntimeError("Hungarian assignment must preserve every original GT exactly once")
                return (torch.as_tensor(rows, device=device, dtype=torch.int64),
                        torch.as_tensor(columns, device=device, dtype=torch.int64))

        def _stage_losses(self, validated, generator):
            classes, visible, amodal, boxes, target, count = validated
            truth_xyxy, truth_cxcywh = self._normalized_targets(target, classes.device)
            query_indices, target_indices = self._match_stage(
                classes, visible, amodal, boxes, target, count, truth_xyxy, truth_cxcywh, generator)
            class_targets = torch.full((1, 100), 1, device=classes.device, dtype=torch.int64)
            class_targets[0, query_indices] = 0
            ce = F.cross_entropy(classes.float().transpose(1, 2), class_targets,
                                 weight=self.class_weights.to(device=classes.device, dtype=torch.float32), reduction="mean")
            v_bce, v_dice = self._branch_losses(visible, target["visible_masks"], query_indices, target_indices, generator)
            a_bce, a_dice = self._branch_losses(amodal, target["amodal_masks"], query_indices, target_indices, generator)
            if count:
                selected = boxes[0].float().index_select(0, query_indices)
                expected_cxcywh = truth_cxcywh.index_select(0, target_indices)
                expected_xyxy = truth_xyxy.index_select(0, target_indices)
                bbox = F.l1_loss(selected, expected_cxcywh, reduction="sum")/count
                giou = (1-pairwise_giou(cxcywh_to_xyxy(selected), expected_xyxy).diagonal()).sum()/count
            else:
                bbox = boxes.float().sum()*0
                giou = boxes.float().sum()*0
            return {
                "loss_class": config["class_weight"]*ce,
                "loss_visible_bce": config["visible_weight"]*config["mask_weight"]*v_bce,
                "loss_visible_dice": config["visible_weight"]*config["dice_weight"]*v_dice,
                "loss_amodal_bce": config["amodal_weight"]*config["mask_weight"]*a_bce,
                "loss_amodal_dice": config["amodal_weight"]*config["dice_weight"]*a_dice,
                "loss_bbox": config["bbox_weight"]*bbox,
                "loss_giou": config["giou_weight"]*giou,
            }

        def forward(self, outputs, targets, *, generator=None):
            if not isinstance(outputs, dict) or "auxiliary_logits" in outputs:
                raise TypeError("Expected the residual model's explicit final/auxiliary output dictionary")
            auxiliary = outputs.get("auxiliary_outputs")
            if not isinstance(auxiliary, list) or len(auxiliary) != 1:
                raise ValueError("Exactly one penultimate auxiliary stage is required")
            final = {key: value for key, value in outputs.items() if key != "auxiliary_outputs"}
            validated_final = self._validate_stage(final, targets)
            validated_auxiliary = self._validate_stage(auxiliary[0], targets)
            if (validated_final[0].device != validated_auxiliary[0].device
                    or validated_final[1].shape != validated_auxiliary[1].shape):
                raise ValueError("Final and penultimate stages must share image geometry and device")
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                raise NotImplementedError("This criterion requires single-process normalization")
            with torch.autocast(device_type=validated_final[0].device.type, enabled=False):
                losses = self._stage_losses(validated_final, generator)
                for key, value in self._stage_losses(validated_auxiliary, generator).items():
                    losses[key+"_aux"] = config["auxiliary_weight"]*value
                if (len(losses) != 14 or any(value.ndim != 0 or not bool(torch.isfinite(value))
                                              for value in losses.values())):
                    raise ValueError("Fourteen finite weighted scalar losses are required")
                return losses

    return ResidualDualCriterion()
