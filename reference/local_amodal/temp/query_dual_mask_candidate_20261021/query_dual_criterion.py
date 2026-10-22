"""Final-layer, real-PApple dual-mask criterion candidate; runtime UNVERIFIED.

Only standard-library imports occur at module scope. A future authorized call
to build_query_dual_criterion would import Torch and SciPy, never Transformers,
weights, datasets or a model. This module has no execution/training entry point.
The frozen cef593... wrapper and its intentionally unimplemented old loss stub
are unchanged; a future caller must pass wrapper outputs to this new criterion.
"""
from copy import deepcopy
import math
import re

TRAIN_ANNOTATION_SHA256 = {
    "visible": "423afe0e0fca7251c9e9da5b9ebae8b60e6b0f7240b9bcf39bc7ff8a6c0d9cab",
    "amodal": "4b3a739628a974ca42ff84d1e0f431cc98adc1591377552b6262454f41c71751",
}


def audit_criterion_config(*, match_points=12544, loss_points=12544,
                           oversample_ratio=3.0, importance_ratio=0.75,
                           class_weight=2.0, mask_weight=5.0, dice_weight=5.0,
                           no_object_weight=0.1, visible_weight=0.5, amodal_weight=0.5):
    """Pure metadata validation; does not import a numerical backend."""
    for name, value in (("match_points", match_points), ("loss_points", loss_points)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    values = dict(oversample_ratio=oversample_ratio, importance_ratio=importance_ratio,
                  class_weight=class_weight, mask_weight=mask_weight, dice_weight=dice_weight,
                  no_object_weight=no_object_weight, visible_weight=visible_weight, amodal_weight=amodal_weight)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in values.values()):
        raise ValueError("Criterion weights/ratios must be finite numbers")
    if oversample_ratio < 1 or not 0 <= importance_ratio <= 1:
        raise ValueError("oversample_ratio>=1 and importance_ratio in[0,1] are required")
    if any(values[k] <= 0 for k in ("class_weight", "mask_weight", "dice_weight", "no_object_weight",
                                   "visible_weight", "amodal_weight")):
        raise ValueError("Both branches and classification/mask losses must have positive weights")
    if not math.isclose(visible_weight+amodal_weight, 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("Branch weights must sum to1; no silent loss rescaling")
    return dict(match_points=match_points, loss_points=loss_points, **values)


def build_query_dual_criterion(**configuration):
    """Future lazy factory; currently permitted checks must not call this.

    criterion(outputs, [target], generator=None) -> dict[str, scalar Tensor].
    Each returned loss is already weighted; sum values exactly once.
    Only single-image, unpadded, final-layer, single-process supervision exists.
    """
    config = audit_criterion_config(**configuration)
    import torch
    from torch import nn
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment

    def sample_points(masks, coordinates):
        # masks[K,H,W], coordinates[K,P,2] normalized x/y in[0,1].
        # There is never a[Q,GT,H_original,W_original] tensor.
        return F.grid_sample(masks[:, None].float(), coordinates[:, None].float()*2-1,
                             mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0, 0]

    def pairwise_cost(predicted, truth):
        # QxP and NxP -> QxN; algebraically equivalent to pairwise BCE+Dice.
        positive = F.binary_cross_entropy_with_logits(predicted, torch.ones_like(predicted), reduction="none")
        negative = F.binary_cross_entropy_with_logits(predicted, torch.zeros_like(predicted), reduction="none")
        points = predicted.shape[-1]
        bce = (positive @ truth.transpose(0, 1) + negative @ (1-truth).transpose(0, 1))/points
        probability = predicted.sigmoid()
        numerator = 2*(probability @ truth.transpose(0, 1))+1
        denominator = probability.sum(-1)[:, None]+truth.sum(-1)[None, :]+1
        return bce, 1-numerator/denominator

    class FinalQueryDualCriterion(nn.Module):
        def __init__(self):
            super().__init__()
            self.configuration = deepcopy(config)
            self.register_buffer("class_weights", torch.tensor([1., config["no_object_weight"]], dtype=torch.float32))
            self.criterion_metadata = {
                "schema": "final_shared_hungarian_real_dual_mask_criterion_candidate_v1",
                "train_annotation_sha256": deepcopy(TRAIN_ANNOTATION_SHA256),
                "class_labels": {"apple": 0, "no_object": 1},
                "same_assignment_for_visible_and_amodal": True,
                "source_GT_subset_repair": False, "box_only_supervision": False,
                "final_layer_only": True, "auxiliary_losses": False,
                "batch_size": 1, "padding_support": False, "distributed_support": False,
                "source_identity_is_caller_assertion_not_tensor_origin_proof": True,
                "backend_execution_checked": False, "autograd_checked": False,
                "model_trainability_checked": False, "quality_improvement_verified": False,
            }

        def _validate(self, outputs, targets):
            if not isinstance(outputs, dict):
                raise TypeError("Expected the frozen wrapper output dictionary")
            if "auxiliary_logits" in outputs or "auxiliary_outputs" in outputs:
                raise NotImplementedError("Auxiliary layers are not silently ignored by this final-only baseline")
            tensors = [outputs.get(key) for key in ("class_queries_logits", "visible_masks_queries_logits",
                                                   "amodal_masks_queries_logits")]
            if any(not isinstance(x, torch.Tensor) or not x.is_floating_point() for x in tensors):
                raise TypeError("Predictions must be floating Torch tensors")
            classes, visible, amodal = tensors
            if tuple(classes.shape) != (1, 100, 2):
                raise ValueError("Class logits must be[1,100,2], apple0/no-object1")
            if visible.ndim != 4 or amodal.shape != visible.shape or tuple(visible.shape[:2]) != (1, 100):
                raise ValueError("V/A logits must share shape[1,100,h,w]")
            if min(visible.shape[-2:]) < 1 or any(x.device != classes.device for x in tensors):
                raise ValueError("Prediction shapes/device mismatch")
            if classes.device.type not in ("cpu", "cuda"):
                raise NotImplementedError("Only prospective CPU/CUDA runtimes are within this source audit")
            if any(not bool(torch.isfinite(x).all()) for x in tensors):
                raise ValueError("Nonfinite predictions must fail rather than be repaired into matching costs")
            if not isinstance(targets, (tuple, list)) or len(targets) != 1 or not isinstance(targets[0], dict):
                raise ValueError("Exactly one real paired TRAIN target is required")
            target = targets[0]
            source = target.get("source", {})
            if not isinstance(source, dict):
                raise TypeError("Target source must be an explicit provenance dictionary")
            if (source.get("dataset") != "PApple" or source.get("split") != "train" or
                    source.get("mask_origin") != "paired_instance_polygons"):
                raise ValueError("Only PApple TRAIN genuine paired polygons are permitted")
            if source.get("annotation_sha256") != TRAIN_ANNOTATION_SHA256:
                raise ValueError("Target provenance must identify the fixed genuine TRAIN annotation bytes")
            if not isinstance(source.get("filename"), str) or not re.fullmatch(r"_MG_[0-9]+_[0-9]+\.png", source["filename"]):
                raise ValueError("Source TRAIN filename assertion is required")
            labels, valid = target.get("labels"), target.get("mask_valid")
            if not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64 or labels.ndim != 1:
                raise TypeError("labels must be int64[N]")
            count = len(labels)
            if count > 100:
                raise ValueError("GT count exceeds inherited query capacity; never silently truncate GT")
            if labels.requires_grad or bool((labels != 0).any()):
                raise ValueError("GT apple label must be0; source label1 needs an explicit future adapter")
            if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or
                    tuple(valid.shape) != (count,) or not bool(valid.all())):
                raise ValueError("mask_valid must be bool[N], alltrue; box-only/fake-mask paths are unsupported")
            masks = [target.get(k) for k in ("visible_masks", "amodal_masks")]
            for mask in masks:
                if (not isinstance(mask, torch.Tensor) or mask.dtype not in (torch.uint8, torch.bool) or
                        mask.ndim != 3 or mask.shape[0] != count or min(mask.shape[-2:]) < 1):
                    raise TypeError("Real masks must be bool/uint8[N,H,W], including[0,H,W] for empty GT")
                if mask.requires_grad or bool(((mask != 0) & (mask != 1)).any()):
                    raise ValueError("GT masks must be fixed binary0/1, not soft/predicted masks")
                if count and not bool(mask.reshape(count, -1).any(dim=1).all()):
                    raise ValueError("Empty per-instance V/A is rejected, never filtered or made background")
            if masks[0].shape != masks[1].shape:
                raise ValueError("Paired real V/A target dimensions must agree")
            # Deliberately no V<=A test, clipping, union or containment penalty.
            return classes, visible, amodal, target, count

        def _sample_target(self, mask, target_indices, coordinates, device):
            chosen = mask.index_select(0, target_indices.to(device=mask.device))
            return sample_points(chosen.to(device=device, dtype=torch.float32), coordinates)

        def _match(self, classes, visible, amodal, target, count, generator):
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
                for predicted, mask_key, branch_weight in (
                        (visible[0], "visible_masks", config["visible_weight"]),
                        (amodal[0], "amodal_masks", config["amodal_weight"])):
                    prediction_points = sample_points(predicted.detach().float(), query_coordinates)
                    truth_points = self._sample_target(target[mask_key], target_indices, target_coordinates, device)
                    bce, dice = pairwise_cost(prediction_points, truth_points)
                    cost = cost+branch_weight*(config["mask_weight"]*bce+config["dice_weight"]*dice)
                if not bool(torch.isfinite(cost).all()):
                    raise ValueError("Nonfinite shared Hungarian costs")
                # SciPy sees only100xN detached scalar costs, never mask tensors.
                rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
                if len(rows) != count or len(set(rows)) != count or set(columns) != set(range(count)):
                    raise RuntimeError("Hungarian result failed to preserve every original GT exactly once")
                return (torch.as_tensor(rows, dtype=torch.int64, device=device),
                        torch.as_tensor(columns, dtype=torch.int64, device=device))

        def _uncertain_coordinates(self, logits, generator):
            with torch.no_grad():
                count = logits.shape[0]
                total = int(config["loss_points"]*config["oversample_ratio"])
                sampled = torch.rand((count, total, 2), device=logits.device, generator=generator)
                uncertainty = -sample_points(logits.detach().float(), sampled).abs()
                important = int(config["importance_ratio"]*config["loss_points"])
                if important:
                    indices = uncertainty.topk(important, dim=1).indices
                    focused = sampled.gather(1, indices[..., None].expand(-1, -1, 2))
                else:
                    focused = sampled[:, :0]
                random_count = config["loss_points"]-important
                random = torch.rand((count, random_count, 2), device=logits.device, generator=generator)
                return torch.cat((focused, random), dim=1)

        def _branch_losses(self, predicted, truth, query_indices, target_indices, generator):
            logits = predicted[0].index_select(0, query_indices).float()
            count = len(query_indices)
            if count == 0:
                zero = predicted.float().sum()*0
                return zero, zero
            coordinates = self._uncertain_coordinates(logits, generator)
            with torch.no_grad():
                target_points = self._sample_target(truth, target_indices, coordinates, logits.device)
            prediction_points = sample_points(logits, coordinates)
            bce = F.binary_cross_entropy_with_logits(prediction_points, target_points, reduction="none").mean(1).sum()/count
            probability = prediction_points.sigmoid()
            dice_per_instance = 1-(2*(probability*target_points).sum(1)+1)/(probability.sum(1)+target_points.sum(1)+1)
            return bce, dice_per_instance.sum()/count

        def forward(self, outputs, targets, *, generator=None):
            classes, visible, amodal, target, count = self._validate(outputs, targets)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                raise NotImplementedError("Distributed normalization is not implemented by this single-process baseline")
            # Outer AMP must not silently downcast matmul-based matching costs.
            with torch.autocast(device_type=classes.device.type, enabled=False):
                query_indices, target_indices = self._match(classes, visible, amodal, target, count, generator)
                class_targets = torch.full((1, 100), 1, device=classes.device, dtype=torch.int64)
                class_targets[0, query_indices] = 0
                ce = F.cross_entropy(classes.float().transpose(1, 2), class_targets,
                                     weight=self.class_weights.to(device=classes.device, dtype=torch.float32), reduction="mean")
                v_bce, v_dice = self._branch_losses(visible, target["visible_masks"], query_indices, target_indices, generator)
                a_bce, a_dice = self._branch_losses(amodal, target["amodal_masks"], query_indices, target_indices, generator)
                losses = {
                    "loss_class": config["class_weight"]*ce,
                    "loss_visible_bce": config["visible_weight"]*config["mask_weight"]*v_bce,
                    "loss_visible_dice": config["visible_weight"]*config["dice_weight"]*v_dice,
                    "loss_amodal_bce": config["amodal_weight"]*config["mask_weight"]*a_bce,
                    "loss_amodal_dice": config["amodal_weight"]*config["dice_weight"]*a_dice,
                }
                if any(not bool(torch.isfinite(value)) for value in losses.values()):
                    raise ValueError("Nonfinite final-layer dual-mask loss")
                return losses

    return FinalQueryDualCriterion()
