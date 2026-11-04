"""Independent visible proposals and supervised visible-to-amodal geometry.

Only actual RPN candidates are returned. Appended GT V boxes are auxiliary
training examples, never proposals. Unknown A never executes the converter.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign, box_iou

from .od_context import (_boxes, _clip_predicted_boxes, _image_shape,
                         expand_context_boxes, matched_box_refinement_loss)


def _fixed_GT_boxes(boxes, image_shape, name):
    if isinstance(boxes, torch.Tensor) and boxes.requires_grad:
        raise ValueError(f"{name} must be fixed GT")
    result = _boxes(boxes, name)
    height, width = _image_shape(image_shape)
    limits = result.new_tensor([width, height, width, height])
    if not bool(((result >= 0) & (result <= limits)).all()):
        raise ValueError(f"{name} must remain inside image bounds; GT is never repaired")
    return result


class VisibleToAmodalConverter(nn.Module):
    """Learn FP32 BoxCoder residuals from two predicted-V context windows."""

    def __init__(self, roi_chunk_size=8):
        super().__init__()
        if type(roi_chunk_size) is not int or roi_chunk_size < 1:
            raise ValueError("roi_chunk_size must be a positive integer")
        self.roi_chunk_size = roi_chunk_size
        self.context22_pool = MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2)
        self.context4_pool = MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2)
        self.projection = nn.Sequential(nn.Conv2d(512, 64, 1), nn.ReLU())
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 256),
                                       nn.ReLU(), nn.Linear(256, 4))
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)
        self.box_coder_weights = (10., 10., 5., 5.)
        self.box_xform_clip = math.log(1000. / 16.)

    def _decode(self, deltas, boxes, image_shape):
        deltas, boxes = deltas.float(), _boxes(boxes)
        if deltas.shape != boxes.shape or not bool(torch.isfinite(deltas).all()):
            raise ValueError("Converter deltas must be finite [N,4]")
        widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        wx, wy, ww, wh = self.box_coder_weights
        shift_x, shift_y = deltas[:, 0] / wx * widths, deltas[:, 1] / wy * heights
        grow_x = .5 * widths * torch.expm1((deltas[:, 2] / ww).clamp(max=self.box_xform_clip))
        grow_y = .5 * heights * torch.expm1((deltas[:, 3] / wh).clamp(max=self.box_xform_clip))
        decoded = torch.stack((boxes[:, 0] + shift_x - grow_x, boxes[:, 1] + shift_y - grow_y,
                               boxes[:, 2] + shift_x + grow_x, boxes[:, 3] + shift_y + grow_y), dim=1)
        return _clip_predicted_boxes(decoded, image_shape)

    def forward(self, features, proposals, image_shapes):
        if (not isinstance(features, Mapping) or not isinstance(proposals, (tuple, list))
                or len(proposals) != len(image_shapes) or not proposals):
            raise ValueError("Converter requires matching proposal/image lists and FPN features")
        for shape in image_shapes:
            _image_shape(shape)
        for key in ("0", "1", "2", "3"):
            value = features.get(key)
            if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                    or value.ndim != 4 or value.shape[:2] != (len(proposals), 256)
                    or min(value.shape[-2:]) < 1 or not bool(torch.isfinite(value).all())):
                raise ValueError("Converter requires finite floating P2-P5 [B,256,H,W]")
        boxes = [_boxes(value) for value in proposals]
        if any(value.device != features["0"].device for value in boxes):
            raise ValueError("Converter boxes and FPN features must share a device")
        results = []
        for index, (regions, shape) in enumerate(zip(boxes, image_shapes)):
            chunks = []
            for start in range(0, len(regions), self.roi_chunk_size):
                current = regions[start:start + self.roi_chunk_size]
                # Preserve batch ROI indexing and the shared padded FPN scales.
                context22 = [value.new_empty((0, 4)) for value in boxes]
                context4 = [value.new_empty((0, 4)) for value in boxes]
                context22[index] = expand_context_boxes(current, shape, 2.2)
                context4[index] = expand_context_boxes(current, shape, 4.)
                pooled22 = self.context22_pool(features, context22, image_shapes)
                pooled4 = self.context4_pool(features, context4, image_shapes)
                pooled = self.projection(torch.cat((pooled22, pooled4), dim=1))
                if not bool(torch.isfinite(pooled).all()):
                    raise ValueError("Converter pooled features must remain finite")
                chunks.append(self._decode(self.regressor(pooled), current, shape))
            results.append(torch.cat(chunks) if chunks else regions.clone())
        return results


class VisibleProposalBranch(nn.Module):
    """Keep the existing A RPN separate; learn V RPN and same-ID V-to-A boxes."""

    def __init__(self, source_rpn, roi_chunk_size=8):
        super().__init__()
        if not isinstance(source_rpn, nn.Module):
            raise ValueError("source_rpn must be a module")
        self.rpn = deepcopy(source_rpn)
        original, copied = source_rpn.state_dict(), self.rpn.state_dict()
        keys_equal = original.keys() == copied.keys()
        equal = keys_equal and all(torch.equal(value, copied[key]) for key, value in original.items())
        independent = keys_equal and all(not value.numel() or
            value.untyped_storage().data_ptr() != copied[key].untyped_storage().data_ptr()
            for key, value in original.items())
        if not equal or not independent:
            raise RuntimeError("Visible RPN must be an equal, storage-independent copy")
        self.initialization_provenance = dict(initialization="deepcopy_existing_A_RPN",
            copied_tensor_count=len(copied), strict_key_shapes_equal=keys_equal,
            all_source_tensors_equal=equal, all_copied_tensor_storages_independent=independent,
            optimizer_or_RNG_restored=False)
        self.converter = VisibleToAmodalConverter(roi_chunk_size)

    def forward(self, images, features, targets=None):
        if self.training != (targets is not None):
            raise ValueError("Visible branch training requires targets; eval must not receive GT")
        shapes = list(images.image_sizes)
        if not shapes or (targets is not None and len(targets) != len(shapes)):
            raise ValueError("One target per image is required")
        visible_targets, known = [], []
        if targets is not None:
            for target, shape in zip(targets, shapes):
                validity = target.get("annotation_validity", {})
                if validity.get("visible") is not True or type(validity.get("amodal")) is not bool:
                    raise ValueError("Explicit visible/amodal annotation validity required")
                masks = target.get("amodal_masks")
                has_A = validity["amodal"]
                if has_A != (masks is not None):
                    raise ValueError("A validity and mask availability must agree")
                visible = _fixed_GT_boxes(target["visible_boxes"], shape, "visible_boxes")
                if visible.device != images.tensors.device:
                    raise ValueError("Prepared visible GT and input images must share a device")
                if has_A:
                    amodal = _fixed_GT_boxes(target["boxes"], shape, "amodal_boxes")
                    if (len(amodal) != len(visible) or not isinstance(masks, torch.Tensor)
                            or masks.ndim != 3 or masks.device.type != "cpu" or masks.requires_grad
                            or len(masks) != len(visible) or amodal.device != visible.device):
                        raise ValueError("Known V/A rows must retain the same IDs and fixed CPU masks")
                visible_targets.append(dict(boxes=visible))
                known.append(has_A)
        proposals, raw_losses = self.rpn(images, features, visible_targets if targets is not None else None)
        if len(proposals) != len(shapes):
            raise ValueError("Visible RPN changed image count")
        proposals = [_boxes(value) for value in proposals]
        names = {"loss_objectness": "loss_visible_rpn_objectness",
                 "loss_rpn_box_reg": "loss_visible_rpn_box_reg"}
        if set(raw_losses) != (set(names) if self.training else set()):
            raise ValueError("Unexpected visible RPN loss dictionary")
        losses = {names[key]: value for key, value in raw_losses.items()}
        if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in losses.values()):
            raise ValueError("Visible RPN losses must be finite scalars")
        if targets is None:
            converted = self.converter(features, proposals, shapes)
            return [value.detach() for value in converted], losses
        if not any(known):
            return [value.new_empty((0, 4)) for value in proposals], losses
        # Only the auxiliary loss sees appended genuine V boxes. Output slices
        # always end at the original RPN proposal count, including an empty RPN.
        augmented = [torch.cat((value, target["visible_boxes"].detach())) if has_A
                     else value.new_empty((0, 4))
                     for value, target, has_A in zip(proposals, targets, known)]
        converted = self.converter(features, augmented, shapes)
        extra, matched_predictions, matched_A = [], [], []
        for index, (value, refined, target, has_A, shape) in enumerate(zip(proposals, converted, targets, known, shapes)):
            extra.append(refined[:len(value)].detach() if has_A else value.new_empty((0, 4)))
            if not has_A or not len(target["visible_boxes"]) or not len(refined):
                continue
            with torch.no_grad():
                overlap, matching = box_iou(augmented[index].float(), target["visible_boxes"].float()).max(dim=1)
                positive = overlap >= .5
            if bool(positive.any()):
                matched_predictions.append((refined[positive], shape))
                matched_A.append(target["boxes"][matching[positive]].detach())
        if matched_predictions:
            # Average per-image matched losses; no dummy edge on empty/unknown A.
            losses["loss_visible_to_amodal_box"] = torch.stack([
                matched_box_refinement_loss(predicted, truth, shape)
                for (predicted, shape), truth in zip(matched_predictions, matched_A)]).mean()
        return extra, losses
