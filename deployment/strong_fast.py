"""Reversible inference adapter; does not load or alter checkpoint tensors.

FP32 is the reference fallback. Batch-2, larger ROI chunks, AMP, and a fusion
cutoff are candidates requiring actual output/VAL comparisons. The original
exact tiled dual-mask NMS is always called; no 56-grid replacement is used.
Instances are temporarily patched, so do not share this model concurrently.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
import math
from types import MethodType

import torch
from torch.nn import functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.ops.misc import Conv2dNormActivation


_MISSING = object()


def _float_features(features):
    return OrderedDict((key, value.float()) for key, value in features.items())


def _fp32(device):
    return torch.autocast(device_type=device.type, enabled=False)


def _dead_amodal_prediction(head, pooled):
    """Only for the legacy-A override; V/H and retained A are unchanged.

    The original mask head computes V, H, A-base+conditioned residual, then
    ODRoIHeads._mask_predictions overwrites A with another A-base evaluation.
    When refinement is explicitly false, neither the residual nor that first
    A-base value reaches the gated classifier/verifier/NMS. Eval-only use and
    stateless A-base modules are checked before this helper can be installed.
    """
    mask_head = head.od_mask_head
    features = F.relu(mask_head.visible_upsample(mask_head.visible_tower(pooled)))
    visible = mask_head.visible_predictor(features)
    hidden_features = (features if mask_head.hidden_tower_mode == "shared_visible"
                       else F.relu(mask_head.hidden_upsample(mask_head.hidden_tower(pooled))))
    hidden = mask_head.hidden_predictor(hidden_features)
    index = mask_head.amodal_base_apple_index
    amodal = mask_head.amodal_base_predictor(mask_head.amodal_base_head(pooled))[:, index:index + 1]
    return {"visible_logits": visible, "hidden_logits": hidden, "amodal_logits": amodal}


class FastDeploymentAdapter:
    """Use ``with adapter: outputs = adapter.predict_views(pixels, mask)``.

    Inputs are already-normalized Tensor[B,3,H,W], B in {1,2}, with an
    optional all-one Tensor[B,H,W]. Outputs are one original detection dict per
    view, in that view's resized input coordinates. No flip-back or fusion is
    performed here. Calling code controls autocast (FP16/BF16); with
    roi_precision='autocast', ROI convolutions inherit it. ROIAlign, decoding,
    final softmax/sigmoid inputs, exact NMS and returned floating tensors are
    FP32. This does not make AMP outputs numerically equal to FP32 outputs.

    pre_nms_threshold=None retains all original final-score candidates. A .75
    cutoff preserves the reference single-view >=.75 detections because it is
    applied after final V/H/A-dependent scores and original stable pre-NMS
    top-k; lower scores cannot suppress higher scores in descending greedy NMS.
    The original detections_per_img cap is untouched. This is not a guarantee
    for fusion, where low-score members can alter clustering/averaging; .35 or
    any other fusion cutoff requires complete validation.
    """

    def __init__(self, model, pre_nms_threshold=None, roi_chunk_size=None,
                 skip_dead_amodal=False, roi_precision="fp32"):
        if pre_nms_threshold is not None:
            if isinstance(pre_nms_threshold, bool) or not isinstance(pre_nms_threshold, (int, float)) or not math.isfinite(pre_nms_threshold) or not 0 <= pre_nms_threshold <= 1:
                raise ValueError("pre_nms_threshold must be None or a finite probability")
        if roi_chunk_size is not None and (type(roi_chunk_size) is not int or roi_chunk_size < 1):
            raise ValueError("roi_chunk_size must be a positive integer")
        if type(skip_dead_amodal) is not bool:
            raise ValueError("skip_dead_amodal must be boolean")
        if roi_precision not in ("fp32", "autocast"):
            raise ValueError("roi_precision must be fp32 or autocast")
        self.model = model
        self.pre_nms_threshold = pre_nms_threshold
        self.roi_chunk_size = roi_chunk_size
        self.skip_dead_amodal = skip_dead_amodal
        self.roi_precision = roi_precision
        self._patches = []
        self._active = False
        self.last_stats = {}

    def _validate_model(self):
        model, head = self.model, self.model.roi_heads
        if (type(model).__name__ != "ODAmodalDetector"
                or not type(model).__module__.endswith(".od_model")
                or type(head).__name__ != "ODRoIHeads"
                or not type(head).__module__.endswith(".od_model")):
            raise NotImplementedError("Only the audited plain ODAmodalDetector/ODRoIHeads topology is supported")
        if any(module.training for module in model.modules()):
            raise ValueError("Call model.eval() before entering deployment adapter")
        if (getattr(model, "visible_proposals", False)
                or getattr(model, "shallow_detail", False)
                or head.configuration.get("context_cascade", False)
                or head.configuration.get("mask_box_refinement", False)
                or getattr(head, "mask_geometry_head", None) is not None):
            raise NotImplementedError("Visible-proposal/detail/context/mask-geometry variants are unsupported")
        if head.final_nms != "dual_mask":
            raise NotImplementedError("This adapter requires the original exact dual-mask NMS policy")
        for name in ("_roi_predictions", "_decoded_boxes", "_final_nms", "_mask_predictions"):
            if name in head.__dict__:
                raise ValueError("Head already has an instance override: " + name)
        if any(parameter.is_floating_point() and parameter.dtype != torch.float32
               for parameter in model.parameters()):
            raise ValueError("Keep checkpoint parameters FP32; request mixed precision using caller autocast")
        if self.skip_dead_amodal:
            mask_head = head.od_mask_head
            if head.use_amodal_refinement is not False:
                raise ValueError("Dead-A bypass requires explicit use_amodal_refinement=False")
            if (type(mask_head).__name__ != "OcclusionDecoupledROIMaskHead"
                    or not type(mask_head).__module__.endswith(".od_roi")
                    or mask_head.amodal_base_head is None
                    or mask_head.amodal_base_predictor is None
                    or type(mask_head.amodal_base_apple_index) is not int
                    or mask_head.amodal_base_apple_index < 0
                    or mask_head.hidden_tower_mode not in ("shared_visible", "independent")):
                raise ValueError("Dead-A bypass needs the genuine known V/H/legacy-A topology")
            # Repeated A-base evaluations are equal only for stateless eval
            # modules. Reject arbitrary/dropout/custom modules, even in eval.
            allowed = (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.Conv2d,
                       torch.nn.ConvTranspose2d, torch.nn.ReLU, torch.nn.Identity,
                       MaskRCNNHeads, MaskRCNNPredictor, Conv2dNormActivation)
            for module in list(mask_head.amodal_base_head.modules()) + list(mask_head.amodal_base_predictor.modules()):
                if type(module) not in allowed:
                    raise NotImplementedError("Unknown stateful A-base module; bypass cannot assert dead work")

    def _set(self, owner, name, value):
        self._patches.append((owner, name, owner.__dict__.get(name, _MISSING)))
        setattr(owner, name, value)

    def _restore(self):
        for owner, name, original in reversed(self._patches):
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        self._patches.clear()
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Adapter is already active")
        self._validate_model()
        head = self.model.roi_heads
        if getattr(head, "_strong_fast_adapter_active", False):
            raise RuntimeError("Another adapter is active on this model")
        try:
            self._set(head, "_strong_fast_adapter_active", True)
            if self.roi_chunk_size is not None:
                self._set(head, "roi_chunk_size", self.roi_chunk_size)
            original_roi = head._roi_predictions
            original_decode = head._decoded_boxes
            original_nms = head._final_nms
            adapter = self

            def roi_predictions(_head, features, proposals, image_shapes, base_logits):
                values, logits = original_roi(features, proposals, image_shapes, base_logits)
                # Native eval forward now computes softmax/sigmoid in FP32.
                return {key: value.float() for key, value in values.items()}, logits.float()

            def decoded_boxes(_head, deltas, proposals, image_shape):
                with _fp32(deltas.device):
                    return original_decode(deltas.float(), [p.float() for p in proposals], image_shape)

            def final_nms(_head, masks, boxes, scores, image_shape):
                with _fp32(scores.device):
                    boxes, scores = boxes.float(), scores.float()
                    masks = {key: value.float() for key, value in masks.items()}
                    before = len(scores)
                    if adapter.pre_nms_threshold is None:
                        selected = original_nms(masks, boxes, scores, image_shape)
                        after = before
                    else:
                        keep = torch.where(scores >= adapter.pre_nms_threshold)[0]
                        after = len(keep)
                        if after:
                            subset = {key: value[keep] for key, value in masks.items()}
                            selected = keep[original_nms(subset, boxes[keep], scores[keep], image_shape)]
                        else:
                            selected = torch.empty(0, dtype=torch.int64, device=scores.device)
                    adapter.last_stats.setdefault("nms", []).append({
                        "original_final_score_candidates": before,
                        "candidates_after_optional_cutoff": after,
                        "survivors_before_original_detection_cap": len(selected)})
                    return selected

            self._set(head, "_roi_predictions", MethodType(roi_predictions, head))
            self._set(head, "_decoded_boxes", MethodType(decoded_boxes, head))
            self._set(head, "_final_nms", MethodType(final_nms, head))
            if self.skip_dead_amodal:
                def masks_without_discarded_A(_head, pooled):
                    return _dead_amodal_prediction(_head, pooled)
                self._set(head, "_mask_predictions", MethodType(masks_without_discarded_A, head))
            # ROIAlign has sensitive coordinate interpolation. The convolutions
            # following these pools still inherit AMP in roi_precision=autocast.
            seen = set()
            for pool in (head.box_roi_pool, head.mask_roi_pool, getattr(head, "highres_roi_pool", None)):
                if pool is None or id(pool) in seen:
                    continue
                seen.add(id(pool))
                original_forward = pool.forward
                def pool_forward(_pool, features, boxes, shapes, _original=original_forward):
                    device = next(iter(features.values())).device
                    with _fp32(device):
                        return _original(_float_features(features), [b.float() for b in boxes], shapes)
                self._set(pool, "forward", MethodType(pool_forward, pool))
            self._active = True
            return self
        except BaseException:
            self._restore()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self._restore()
        return False

    @torch.inference_mode()
    def predict_views(self, images, pixel_mask=None):
        if not self._active:
            raise RuntimeError("Use FastDeploymentAdapter as a context manager")
        if (not isinstance(images, torch.Tensor) or not images.is_floating_point()
                or images.ndim != 4 or images.shape[0] not in (1, 2)
                or images.shape[1] != 3 or min(images.shape[-2:]) < 1):
            raise ValueError("Expected normalized floating RGB[B,3,H,W], B=1 or 2")
        if pixel_mask is not None and (tuple(pixel_mask.shape) != (images.shape[0], *images.shape[-2:])
                                       or not bool(pixel_mask.all())):
            raise ValueError("Only unpadded all-one pixel_mask[B,H,W] is supported")
        if self.model.training or self.model.roi_heads.training:
            raise ValueError("Model was switched to training while adapter is active")
        batch, _, height, width = images.shape
        shape = (int(height), int(width))
        padded_h, padded_w = ((height + 31) // 32 * 32, (width + 31) // 32 * 32)
        image_list = ImageList(F.pad(images, (0, padded_w - width, 0, padded_h - height)),
                               [shape] * batch)
        self.last_stats = {"batch": batch, "roi_precision": self.roi_precision,
                           "pre_nms_threshold": self.pre_nms_threshold,
                           "roi_chunk_size": self.model.roi_heads.roi_chunk_size,
                           "skip_dead_amodal": self.skip_dead_amodal,
                           "exact_original_dual_mask_nms": True, "nms": []}
        features = self.model.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        if not isinstance(features, Mapping) or any(value.shape[0] != batch for value in features.values()):
            raise ValueError("Backbone feature batch does not match view batch")
        proposals, losses = self.model.rpn(image_list, features, None)
        if losses or len(proposals) != batch:
            raise RuntimeError("RPN did not return independent eval proposals for each view")
        results = []
        # Never hand two images to original _roi_predictions, whose contract
        # explicitly requires len(proposals)==1. Pool indices start from zero
        # against the corresponding sliced feature batch.
        for index in range(batch):
            # Cache the FP32 ROI feature view once per image. ROIAlign wrappers
            # then see FP32 tensors and .float() is a no-copy operation, rather
            # than re-converting all half-precision FPN maps for every ROI chunk.
            # Autocast still applies to the heavy convolutions after pooling.
            current = OrderedDict((key, value[index:index + 1].float()) for key, value in features.items())
            scope = _fp32(images.device) if self.roi_precision == "fp32" else nullcontext()
            with scope:
                detections, detector_losses = self.model.roi_heads(
                    current, [proposals[index].float()], [shape], None)
            if detector_losses or len(detections) != 1:
                raise RuntimeError("Original ROI head did not return exactly one eval view")
            result = {key: value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
                      for key, value in detections[0].items()}
            if result.get("image_hw") != shape:
                raise RuntimeError("ROI result coordinates do not match this view's input shape")
            results.append(result)
        return results
