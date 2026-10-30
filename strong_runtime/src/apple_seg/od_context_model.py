"""Context ROI, supervised cascade and visible-foreground detector heads.

Mask coordinates always describe the final predicted ROI. Cascade geometry is
detached between stages, with each stage's explicit genuine-A localization loss.
"""
from collections import OrderedDict
import math

import torch
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign, box_iou

from .od_model import ODRoIHeads
from .od_context import (ContextROIFusion, ContextCascadeBoxHead,
                         expand_context_boxes, matched_box_refinement_loss)
from .od_foreground import (GlobalVisibleForegroundHead, ForegroundPrototypeFusion,
                            visible_foreground_losses)


class ContextODRoIHeads(ODRoIHeads):
    def __init__(self, source, *, context_expansion=2.2,
                 cascade_iou_thresholds=(.5, .6, .7), **options):
        if (isinstance(context_expansion, bool) or not isinstance(context_expansion, (int, float))
                or not math.isfinite(context_expansion) or not 1 < context_expansion <= 4):
            raise ValueError("context_expansion must be finite in (1,4]")
        if (not isinstance(cascade_iou_thresholds, (tuple, list))
                or len(cascade_iou_thresholds) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in cascade_iou_thresholds)
                or not .5 <= cascade_iou_thresholds[0] < cascade_iou_thresholds[1]
                       < cascade_iou_thresholds[2] <= .9):
            raise ValueError("Three increasing cascade IoU thresholds in [.5,.9] required")
        super().__init__(source, **options)
        self.context_expansion = float(context_expansion)
        self.cascade_iou_thresholds = tuple(float(v) for v in cascade_iou_thresholds)
        self.context_fusion = ContextROIFusion(256)
        self.context_cascade = ContextCascadeBoxHead(256, stages=3)
        self.foreground_head = GlobalVisibleForegroundHead(256, hidden=64)
        self.foreground_fusion = ForegroundPrototypeFusion(256)
        self.foreground_roi_pool = MultiScaleRoIAlign(["0"], self.mask_roi_pool.output_size[0], 2)
        self.configuration.update(context_cascade=True, context_expansion=self.context_expansion,
                                  cascade_iou_thresholds=list(self.cascade_iou_thresholds))

    def _with_foreground(self, features):
        result = OrderedDict(features)
        if self.training and self.classifier_only:
            with torch.no_grad():
                result["_visible_foreground"] = self.foreground_head(features)
        else:
            result["_visible_foreground"] = self.foreground_head(features)
        return result

    def _pooled_roi_features(self, features, proposals, image_shapes):
        native = self.mask_roi_pool(features, proposals, image_shapes)
        expanded = [expand_context_boxes(boxes, shape, self.context_expansion)
                    for boxes, shape in zip(proposals, image_shapes)]
        context = self.mask_roi_pool(features, expanded, image_shapes)
        return self.context_fusion(native, context)

    def _fuse_classification_evidence(self, features, proposals, image_shapes, pooled, visible_logits, logits):
        semantic = self.foreground_roi_pool({"0": features["_visible_foreground"]}, proposals, image_shapes)
        return self.foreground_fusion(pooled, visible_logits, semantic, logits)

    def _cascade_boxes(self, features, boxes, image_shapes, targets=None):
        if len(image_shapes) != 1:
            raise ValueError("Context detector requires one image")
        known = (targets is not None and targets[0]["amodal_masks"] is not None
                 and targets[0]["annotation_validity"]["amodal"] is True)
        if targets is not None and not known:
            # Unknown A never acquires a localization loss, including a zero edge.
            return boxes.detach(), {}
        current = boxes.detach()
        stage_losses = []
        for stage, threshold in enumerate(self.cascade_iou_thresholds):
            chunks = []
            for start in range(0, len(current), self.roi_chunk_size):
                regions = current[start:start + self.roi_chunk_size]
                support = expand_context_boxes(regions, image_shapes[0], self.context_expansion)
                pooled = self.mask_roi_pool(features, [support], image_shapes)
                chunks.append(self.context_cascade.forward_stage(pooled, regions, image_shapes[0], stage))
            refined = torch.cat(chunks) if chunks else current.clone()
            if known:
                gt = targets[0]["boxes"]
                if len(current) and len(gt):
                    with torch.no_grad():
                        overlap, matched = box_iou(current.float(), gt.float()).max(dim=1)
                        positive = overlap >= threshold
                    if bool(positive.any()):
                        stage_losses.append(matched_box_refinement_loss(
                            refined[positive], gt[matched[positive]].detach(), image_shapes[0]))
            current = refined.detach()
        if not known:
            return current, {}
        # Empty/no-positive samples train foreground/classification; they do not
        # attach a dummy A edge that would decay the cascade parameters.
        if not stage_losses:
            return current, {}
        return current, {"loss_context_box": torch.stack(stage_losses).mean()}

    def _classifier_training(self, features, proposals, image_shapes, targets):
        with torch.no_grad():
            pooled = self.box_roi_pool(features, proposals, image_shapes)
            base, deltas = self.box_predictor(self.box_head(pooled))
            boxes, valid = self._decoded_boxes(deltas, proposals, image_shapes[0])
            boxes, _ = self._cascade_boxes(features, boxes, image_shapes)
            base = base[valid]
            matched, labels = self.assign_targets_to_proposals(
                [boxes], [target["boxes"] for target in targets], [target["labels"] for target in targets])
            labels = self._classifier_supervision_labels(boxes, labels[0], targets[0])
            occlusion = self._candidate_visible_ratio(matched[0], targets[0])
            selected = self._classifier_samples(features, boxes, image_shapes, base, labels,
                                                occlusion=occlusion)
        if not len(selected):
            return [], {"loss_classifier": sum(p.sum() * 0 for p in self.gated_classifier.parameters())}
        _, logits = self._roi_predictions(features, [boxes[selected]], image_shapes, base[selected])
        return [], {"loss_classifier": self._classification_loss(
            logits, [labels[selected]], [matched[0][selected]], targets),
            **self._verifier_losses(logits, [labels[selected]])}

    def forward(self, features, proposals, image_shapes, targets=None):
        features = self._with_foreground(features)
        if self.training and self.classifier_only:
            return self._classifier_training(features, proposals, image_shapes, targets)
        if self.training:
            proposals, matched, labels, regression_targets = self.select_training_samples(proposals, targets)
        pooled = self.box_roi_pool(features, proposals, image_shapes)
        base, deltas = self.box_predictor(self.box_head(pooled))
        boxes, valid = self._decoded_boxes(deltas.detach() if self.training else deltas,
                                           proposals, image_shapes[0])
        refined, cascade_losses = self._cascade_boxes(features, boxes, image_shapes,
                                                      targets if self.training else None)
        if self.training:
            losses = dict(cascade_losses)
            original_labels = torch.cat(labels)
            if targets[0]["amodal_masks"] is not None and bool((original_labels > 0).any()):
                positive = torch.where(original_labels > 0)[0]
                losses["loss_box_reg"] = F.smooth_l1_loss(
                    deltas.reshape(-1, 2, 4)[positive, 1], torch.cat(regression_targets)[positive],
                    beta=1 / 9, reduction="sum") / max(len(original_labels), 1)
            if targets[0]["amodal_masks"] is None:
                # Retain genuine V-coordinate mask supervision; the decoded A
                # regressor and cascade are not supervision for unknown A.
                regions, selected_base = proposals, base
            else:
                regions, selected_base = [refined], base[valid]
                with torch.no_grad():
                    matched, labels = self.assign_targets_to_proposals(
                        regions, [t["boxes"] for t in targets], [t["labels"] for t in targets])
            prediction, logits = self._roi_predictions(features, regions, image_shapes, selected_base)
            losses["loss_classifier"] = self._classification_loss(logits, labels, matched, targets)
            losses.update(self._verifier_losses(logits, labels))
            losses.update(self._mask_losses(prediction, regions, labels, matched, targets))
            losses.update(visible_foreground_losses(features["_visible_foreground"], targets, image_shapes))
            return [], losses
        prediction, logits = self._roi_predictions(features, [refined], image_shapes, base[valid])
        scores = logits.softmax(-1)[:, 1]
        keep = torch.where(scores > self.score_thresh)[0]
        keep = keep[torch.argsort(scores[keep], descending=True, stable=True)[:self.pre_mask_nms_topk]]
        boxes, scores = refined[keep], scores[keep]
        masks = {key: value[keep].sigmoid() for key, value in prediction.items()}
        selected = self._final_nms(masks, boxes, scores, image_shapes[0])[:self.detections_per_img]
        return [{"boxes": boxes[selected], "scores": scores[selected],
                 "labels": torch.zeros(len(selected), dtype=torch.int64, device=boxes.device),
                 "roi_visible_masks": masks["visible_logits"][selected],
                 "roi_hidden_masks": masks["hidden_logits"][selected],
                 "roi_amodal_masks": masks["amodal_logits"][selected],
                 "image_hw": tuple(image_shapes[0])}], {}
