"""Annotation-aware, final-layer Hungarian supervision for apple masks.

PApple supplies real visible and amodal annotations; MinneApple supplies only
visible annotations. Unknown amodal supervision is absent from both matching
and the loss graph, rather than represented by an all-zero target. Dataset and
driver code own source-file identity checks; this module validates the target
contract without pinning local paths or annotation bytes.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def audit_criterion_config(
    *, match_points: int = 12544, loss_points: int = 12544,
    oversample_ratio: float = 3.0, importance_ratio: float = 0.75,
    class_weight: float = 2.0, mask_weight: float = 5.0,
    dice_weight: float = 5.0, no_object_weight: float = 0.1,
    target_sample_chunk_size: int = 8,
) -> dict[str, Any]:
    """Validate configuration without changing annotation-dependent weights."""
    integers = dict(match_points=match_points, loss_points=loss_points,
                    target_sample_chunk_size=target_sample_chunk_size)
    for name, value in integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    numbers = dict(oversample_ratio=oversample_ratio, importance_ratio=importance_ratio,
                   class_weight=class_weight, mask_weight=mask_weight,
                   dice_weight=dice_weight, no_object_weight=no_object_weight)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in numbers.values()):
        raise ValueError("Criterion weights and ratios must be finite numbers")
    if oversample_ratio < 1 or not 0 <= importance_ratio <= 1:
        raise ValueError("oversample_ratio >= 1 and importance_ratio in [0, 1] required")
    if any(numbers[key] <= 0 for key in ("class_weight", "mask_weight", "dice_weight", "no_object_weight")):
        raise ValueError("Classification, mask, Dice and no-object weights must be positive")
    return {**integers, **numbers}


def sample_cpu_target_points(
    masks: torch.Tensor,
    target_indices: torch.Tensor,
    coordinates: torch.Tensor,
    *, device: torch.device | str,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Sample CPU binary GT at normalized x/y points with bounded workspace.

    This matches bilinear ``grid_sample(..., align_corners=False,
    padding_mode='zeros')``. It gathers four CPU pixels per point directly from
    the original uint8/bool tensor; it never selects/converts a dense float mask
    batch or transfers an N*H*W target to the prediction device. Only sampled
    chunks, ultimately N*P values, are transferred. GT storage is read-only.
    """
    if not isinstance(masks, torch.Tensor) or masks.device.type != "cpu":
        raise ValueError("Dense GT masks must remain on CPU")
    if masks.dtype not in (torch.uint8, torch.bool) or masks.ndim != 3:
        raise TypeError("Dense GT masks must be CPU uint8/bool[N,H,W]")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if (not isinstance(target_indices, torch.Tensor) or target_indices.dtype != torch.int64
            or target_indices.ndim != 1):
        raise TypeError("target_indices must be int64[K]")
    if (not isinstance(coordinates, torch.Tensor) or not coordinates.is_floating_point()
            or coordinates.ndim != 3 or coordinates.shape[0] != len(target_indices)
            or coordinates.shape[-1] != 2):
        raise ValueError("Point coordinates must be floating[K,P,2]")
    height, width = masks.shape[-2:]
    if min(height, width) < 1:
        raise ValueError("GT spatial dimensions must be positive")
    destination = torch.device(device)
    result = torch.empty(coordinates.shape[:2], device=destination, dtype=torch.float32)
    indices = target_indices.detach().to(device="cpu")
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= len(masks)):
        raise ValueError("Target index outside the original GT inventory")
    with torch.no_grad():
        for start in range(0, len(indices), chunk_size):
            stop = min(start + chunk_size, len(indices))
            points = coordinates[start:stop].detach().to(device="cpu", dtype=torch.float32)
            if not bool(torch.isfinite(points).all()) or bool(((points < 0) | (points > 1)).any()):
                raise ValueError("Point coordinates must be finite in [0, 1]")
            # Reproduce the normalized-grid round trip used for prediction
            # sampling, including align_corners=False boundary interpolation.
            grid = points * 2 - 1
            x = ((grid[..., 0] + 1) / 2) * width - 0.5
            y = ((grid[..., 1] + 1) / 2) * height - 0.5
            x0, y0 = x.floor().long(), y.floor().long()
            wx, wy = x - x0, y - y0
            sampled = torch.zeros_like(x)
            chosen = indices[start:stop, None]
            for xi, yi, weight in (
                (x0, y0, (1 - wx) * (1 - wy)),
                (x0 + 1, y0, wx * (1 - wy)),
                (x0, y0 + 1, (1 - wx) * wy),
                (x0 + 1, y0 + 1, wx * wy),
            ):
                inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
                values = masks[chosen, yi.clamp(0, height - 1), xi.clamp(0, width - 1)].float()
                sampled.add_(values * inside * weight)
            result[start:stop].copy_(sampled.to(device=destination))
    return result


def _sample_prediction_points(masks: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(masks[:, None].float(), coordinates[:, None].float() * 2 - 1,
                         mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0, 0]


def _pairwise_mask_cost(predicted: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Q*P and N*P -> Q*N, without Q*N*P or any original-size mask product.
    positive = F.softplus(-predicted)
    negative = F.softplus(predicted)
    bce = (positive @ truth.T + negative @ (1 - truth).T) / predicted.shape[-1]
    probability = predicted.sigmoid()
    dice = 1 - (2 * (probability @ truth.T) + 1) / (
        probability.sum(-1)[:, None] + truth.sum(-1)[None, :] + 1)
    return bce, dice


class AnnotationAwareHungarianCriterion(nn.Module):
    """One image, 100 queries, one Hungarian assignment for known branches.

    ``forward(outputs, [target], generator=None)`` returns already-weighted
    scalar losses: sum ``losses.values()`` exactly once. Unknown A returns no A
    loss entries and has no path to the total loss. Known branches on empty GT
    return differentiable zero mask losses; classification still trains all
    queries as no-object. No distributed/padded/multi-image normalization is
    silently approximated. ``generator`` must match the prediction device.
    """
    def __init__(self, **configuration: Any) -> None:
        super().__init__()
        self.configuration = audit_criterion_config(**configuration)
        self.register_buffer("class_weights", torch.tensor(
            [1.0, self.configuration["no_object_weight"]], dtype=torch.float32))
        self.last_assignment: dict[str, list[int]] | None = None
        self.criterion_metadata = {
            "schema": "annotation_aware_shared_hungarian_dual_mask_v1",
            "class_labels": {"apple": 0, "no_object": 1},
            "branch_weights": {"visible_and_amodal_known": {"visible": 0.5, "amodal": 0.5},
                               "visible_only_known": {"visible": 1.0, "amodal": 0.0}},
            "same_assignment_for_known_branches": True, "unknown_amodal_has_loss_graph_edge": False,
            "final_layer_only": True, "auxiliary_losses": False, "direct_bbox_head": False,
            "GT_repair_or_subset": False, "visible_subset_of_amodal_required": False,
            "dense_GT_device": "cpu", "target_transfer": "chunked sampled N*P float32 only",
            "source_binding_owner": "dataset and driver", "batch_size": 1,
        }

    def _validate(self, outputs: dict, targets: list[dict] | tuple[dict, ...]):
        if not isinstance(outputs, dict):
            raise TypeError("outputs must be a prediction dictionary")
        if "auxiliary_logits" in outputs or "auxiliary_outputs" in outputs:
            raise NotImplementedError("This criterion is explicitly final-layer only")
        if "amodal_boxes" in outputs:
            raise NotImplementedError("Direct geometry-head supervision is outside this criterion")
        tensors = [outputs.get(key) for key in ("class_queries_logits", "visible_masks_queries_logits",
                                               "amodal_masks_queries_logits")]
        if any(not isinstance(value, torch.Tensor) or not value.is_floating_point() for value in tensors):
            raise TypeError("Prediction logits must be floating Torch tensors")
        classes, visible, amodal = tensors
        if tuple(classes.shape) != (1, 100, 2):
            raise ValueError("Class logits must be [1,100,2], apple=0/no-object=1")
        if (visible.ndim != 4 or amodal.shape != visible.shape or tuple(visible.shape[:2]) != (1, 100)
                or min(visible.shape[-2:]) < 1):
            raise ValueError("V/A logits must share shape [1,100,h,w]")
        if any(value.device != classes.device for value in tensors):
            raise ValueError("Prediction devices must agree")
        if classes.device.type not in ("cpu", "cuda"):
            raise NotImplementedError("Only CPU and CUDA prediction devices are supported")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("Nonfinite predictions must fail rather than be repaired")
        if not isinstance(targets, (tuple, list)) or len(targets) != 1 or not isinstance(targets[0], dict):
            raise ValueError("Exactly one unpadded annotation-aware target is required")
        target = targets[0]
        source, validity = target.get("source"), target.get("annotation_validity")
        if (not isinstance(source, dict) or source.get("dataset") not in ("PApple", "MinneApple")
                or source.get("split") not in ("train", "val")
                or not isinstance(source.get("image_id"), str) or not source["image_id"]):
            raise ValueError("Source dataset, train/val split, and nonempty image_id are required")
        if (not isinstance(validity, dict) or validity.get("visible") is not True
                or type(validity.get("amodal")) is not bool):
            raise ValueError("annotation_validity must explicitly declare visible=True and bool amodal")
        known_amodal = validity["amodal"]
        if known_amodal != (source["dataset"] == "PApple"):
            raise ValueError("PApple requires real V/A; MinneApple is visible-only with unknown A")
        labels = target.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64 or labels.ndim != 1:
            raise TypeError("labels must be int64[N]")
        count = len(labels)
        if count > 100:
            raise ValueError("GT count exceeds 100-query capacity; never truncate or exclude GT")
        if labels.requires_grad or bool((labels != 0).any()):
            raise ValueError("All genuine apple target labels must equal 0")
        ids = target.get("instance_ids")
        if (not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count
                or any(not isinstance(value, str) or not value for value in ids)):
            raise ValueError("instance_ids must be unique nonempty strings in original GT order")
        original_hw = target.get("original_hw")
        if (not isinstance(original_hw, (list, tuple)) or len(original_hw) != 2
                or any(type(value) is not int or value < 1 for value in original_hw)):
            raise ValueError("original_hw must identify positive original [H,W]")
        visible_truth, amodal_truth = target.get("visible_masks"), target.get("amodal_masks")
        if not known_amodal and amodal_truth is not None:
            raise ValueError("Unknown amodal annotation must be None, never fabricated zero masks")
        for name, mask in (("visible", visible_truth), ("amodal", amodal_truth)):
            if name == "amodal" and not known_amodal:
                continue
            if (not isinstance(mask, torch.Tensor) or mask.device.type != "cpu"
                    or mask.dtype not in (torch.uint8, torch.bool)
                    or tuple(mask.shape) != (count, *original_hw)):
                raise TypeError("Known GT masks must be original-size CPU uint8/bool[N,H,W]")
            if mask.requires_grad:
                raise ValueError("GT masks must be fixed annotations")
            # Scalar reductions and point gathers avoid dense float copies and
            # full-size temporary boolean masks, even during GT validation.
            for start in range(0, count, self.configuration["target_sample_chunk_size"]):
                chunk = mask[start:start + self.configuration["target_sample_chunk_size"]]
                if mask.dtype == torch.uint8 and int(chunk.max()) > 1:
                    raise ValueError("GT masks must contain only binary 0/1 values")
                if not bool(chunk.any(dim=(1, 2)).all()):
                    raise ValueError("An empty per-instance mask must be diagnosed, never dropped or repaired")
        return classes, visible, amodal, target, count, known_amodal

    def _target_points(self, masks, target_indices, coordinates, device):
        return sample_cpu_target_points(masks, target_indices, coordinates, device=device,
                                        chunk_size=self.configuration["target_sample_chunk_size"])

    def _match(self, classes, branches, target, count, generator):
        config, device = self.configuration, classes.device
        if count == 0:
            empty = torch.empty(0, device=device, dtype=torch.int64)
            return empty, empty
        with torch.no_grad():
            coordinates = torch.rand((1, config["match_points"], 2), device=device, generator=generator)
            query_coordinates = coordinates.expand(100, -1, -1)
            target_coordinates = coordinates.expand(count, -1, -1)
            target_indices = torch.arange(count, dtype=torch.int64, device="cpu")
            cost = -config["class_weight"] * classes[0].float().softmax(-1)[:, 0:1].expand(-1, count)
            for predicted, mask_key, weight in branches:
                predicted_points = _sample_prediction_points(predicted[0].detach(), query_coordinates)
                truth_points = self._target_points(target[mask_key], target_indices, target_coordinates, device)
                bce, dice = _pairwise_mask_cost(predicted_points, truth_points)
                cost = cost + weight * (config["mask_weight"] * bce + config["dice_weight"] * dice)
            if not bool(torch.isfinite(cost).all()):
                raise ValueError("Nonfinite shared Hungarian costs")
            # Fixed query/GT input order and SciPy's deterministic assignment;
            # no random tie perturbation and no per-branch rematching.
            rows, columns = linear_sum_assignment(cost.detach().double().cpu().numpy())
            if len(rows) != count or len(set(rows)) != count or set(columns) != set(range(count)):
                raise RuntimeError("Hungarian assignment did not preserve every original GT exactly once")
            return (torch.as_tensor(rows, dtype=torch.int64, device=device),
                    torch.as_tensor(columns, dtype=torch.int64, device=device))

    def _uncertain_coordinates(self, logits, generator):
        config = self.configuration
        with torch.no_grad():
            total = int(config["loss_points"] * config["oversample_ratio"])
            sampled = torch.rand((len(logits), total, 2), device=logits.device, generator=generator)
            uncertainty = -_sample_prediction_points(logits.detach(), sampled).abs()
            important = int(config["importance_ratio"] * config["loss_points"])
            if important:
                # Stable argsort defines exact-uncertainty ties by sample order.
                indices = torch.argsort(uncertainty, dim=1, descending=True, stable=True)[:, :important]
                focused = sampled.gather(1, indices[..., None].expand(-1, -1, 2))
            else:
                focused = sampled[:, :0]
            random = torch.rand((len(logits), config["loss_points"] - important, 2),
                                device=logits.device, generator=generator)
            return torch.cat((focused, random), dim=1)

    def _branch_losses(self, predicted, truth, query_indices, target_indices, generator):
        if not len(query_indices):
            zero = predicted.float().sum() * 0
            return zero, zero
        logits = predicted[0].index_select(0, query_indices).float()
        coordinates = self._uncertain_coordinates(logits, generator)
        truth_points = self._target_points(truth, target_indices, coordinates, logits.device)
        predicted_points = _sample_prediction_points(logits, coordinates)
        bce = F.binary_cross_entropy_with_logits(predicted_points, truth_points, reduction="none").mean(1).mean()
        probability = predicted_points.sigmoid()
        dice = 1 - (2 * (probability * truth_points).sum(1) + 1) / (
            probability.sum(1) + truth_points.sum(1) + 1)
        return bce, dice.mean()

    def forward(self, outputs, targets, *, generator=None):
        classes, visible, amodal, target, count, known_amodal = self._validate(outputs, targets)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise NotImplementedError("Distributed loss normalization is not implemented")
        config = self.configuration
        branches = [(visible, "visible_masks", 0.5 if known_amodal else 1.0)]
        if known_amodal:
            branches.append((amodal, "amodal_masks", 0.5))
        # Keep matching and loss math in float32 under an outer AMP context.
        with torch.autocast(device_type=classes.device.type, enabled=False):
            query_indices, target_indices = self._match(classes, branches, target, count, generator)
            self.last_assignment = {"query_indices": query_indices.detach().cpu().tolist(),
                                    "target_indices": target_indices.detach().cpu().tolist()}
            class_targets = torch.full((1, 100), 1, device=classes.device, dtype=torch.int64)
            class_targets[0, query_indices] = 0
            ce = F.cross_entropy(classes.float().transpose(1, 2), class_targets,
                                 weight=self.class_weights.to(device=classes.device, dtype=torch.float32))
            losses = {"loss_class": config["class_weight"] * ce}
            for predicted, mask_key, weight in branches:
                bce, dice = self._branch_losses(predicted, target[mask_key], query_indices, target_indices, generator)
                branch_name = "visible" if mask_key == "visible_masks" else "amodal"
                losses[f"loss_{branch_name}_bce"] = weight * config["mask_weight"] * bce
                losses[f"loss_{branch_name}_dice"] = weight * config["dice_weight"] * dice
            if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in losses.values()):
                raise ValueError("Nonfinite or nonscalar annotation-aware mask loss")
            return losses


def build_query_dual_criterion(**configuration: Any) -> AnnotationAwareHungarianCriterion:
    """Compatibility factory; source provenance is bound by dataset/driver."""
    return AnnotationAwareHungarianCriterion(**configuration)


def build_annotation_aware_criterion(**configuration: Any) -> AnnotationAwareHungarianCriterion:
    return AnnotationAwareHungarianCriterion(**configuration)
