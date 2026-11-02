"""Experimental pre-mask selection on the immutable original OD models.

No GT is accepted here. Disabled selection follows the original head's eval
operation order; enabled selection deliberately changes the candidate set.
Only the instantiated forward method is patched, then restored on exit.
"""
from types import MethodType
import time

import torch

from deployment.strong_fast import FastDeploymentAdapter
from deployment.candidate_policy import select_candidates


class PreMaskDeploymentAdapter(FastDeploymentAdapter):
    def __init__(self, model, *, candidate_policy=None, capture_candidates=False,
                 profile_stages=False, **kwargs):
        super().__init__(model, **kwargs)
        self.candidate_policy = dict(candidate_policy or {})
        if set(self.candidate_policy) - {"score_threshold", "box_nms_iou", "topk"}:
            raise ValueError("Unknown pre-mask policy key")
        # Validate even when no image is ever submitted.
        select_candidates(torch.empty((0, 4)), torch.empty(0), **self.candidate_policy)
        self.capture_candidates = bool(capture_candidates)
        self.profile_stages = bool(profile_stages)
        self.traces = []

    def _tick(self, device):
        if self.profile_stages and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def __enter__(self):
        if "forward" in self.model.roi_heads.__dict__:
            raise ValueError("ROI forward is already overridden")
        super().__enter__()
        adapter = self

        def forward(head, features, proposals, image_shapes, targets=None):
            if head.training or targets is not None or len(proposals) != 1:
                raise ValueError("Pre-mask deployment is eval-only, one ROI view, no targets")
            device = proposals[0].device
            t0 = adapter._tick(device)
            box_features = head.box_head(head.box_roi_pool(features, proposals, image_shapes))
            base_logits, box_regression = head.box_predictor(box_features)
            boxes, valid = head._decoded_boxes(box_regression, proposals, image_shapes[0])
            base_logits = base_logits[valid]
            # Two-class foreground-vs-background softmax, never sigmoid(z1).
            initial_scores = base_logits.float().softmax(-1)[:, 1]
            t1 = adapter._tick(device)
            indices = select_candidates(boxes, initial_scores, **adapter.candidate_policy)
            candidate_boxes, candidate_logits = boxes[indices], base_logits[indices]
            t2 = adapter._tick(device)
            predictions, class_logits = head._roi_predictions(
                features, [candidate_boxes], image_shapes, candidate_logits)
            scores = class_logits.softmax(-1)[:, 1]
            t3 = adapter._tick(device)
            keep = torch.where(scores > head.score_thresh)[0]
            order = torch.argsort(scores[keep], descending=True, stable=True)
            keep = keep[order[:head.pre_mask_nms_topk]]
            final_boxes, final_scores = candidate_boxes[keep], scores[keep]
            masks = {key: value[keep].sigmoid() for key, value in predictions.items()}
            selected = head._final_nms(masks, final_boxes, final_scores, image_shapes[0])[:head.detections_per_img]
            result = {
                "boxes": final_boxes[selected], "scores": final_scores[selected],
                "labels": torch.zeros(len(selected), dtype=torch.int64, device=device),
                "roi_visible_masks": masks["visible_logits"][selected],
                "roi_hidden_masks": masks["hidden_logits"][selected],
                "roi_amodal_masks": masks["amodal_logits"][selected],
                "image_hw": tuple(image_shapes[0]),
            }
            t4 = adapter._tick(device)
            stats = {"decoded_candidates": len(boxes), "mask_candidates": len(indices),
                     "output_candidates": len(selected), "policy": adapter.candidate_policy}
            if adapter.profile_stages:
                stats["synchronized_ms"] = {
                    "box_head_and_decode": (t1-t0)*1000,
                    "pre_mask_selection": (t2-t1)*1000,
                    "mask_gate_verifier": (t3-t2)*1000,
                    "final_selection_nms": (t4-t3)*1000,
                }
            adapter.last_stats.setdefault("pre_mask", []).append(stats)
            if adapter.capture_candidates:
                # GPU tensors are copied by the caller outside deployment timing.
                adapter.traces.append({"decoded_boxes": boxes.detach(),
                    "initial_scores": initial_scores.detach(),
                    "mask_indices": indices.detach(), "final_scores": scores.detach(),
                    "output_indices": indices[keep[selected]].detach()})
            return [result], {}

        try:
            self._set(self.model.roi_heads, "forward", MethodType(forward, self.model.roi_heads))
            return self
        except BaseException:
            self._restore()
            raise

    def predict_views(self, images, pixel_mask=None):
        self.traces = []
        return super().predict_views(images, pixel_mask)
