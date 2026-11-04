"""A-mask geometry refinement followed by final-ROI mask prediction."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.ops import box_iou

from .od_context import expand_context_boxes, matched_box_refinement_loss
from .od_context_model import ContextODRoIHeads
from .od_model import project_original_masks
from .od_mask_geometry import GeometryAHead, MaskGeometryRefiner


class MaskGeometryODRoIHeads(ContextODRoIHeads):
    """Extend the existing cascade without reusing masks in a different frame."""

    def initialize_mask_geometry(self):
        if hasattr(self, 'mask_geometry_head'):
            raise ValueError('Mask geometry has already been initialized')
        source = self.od_mask_head
        if source.amodal_base_head is None or source.amodal_base_predictor is None:
            raise ValueError('Mask geometry requires the independent amodal base')
        self.mask_geometry_head = GeometryAHead(source.amodal_base_head,
            source.amodal_base_predictor, source.amodal_base_apple_index)
        self.mask_geometry_refiner = MaskGeometryRefiner()
        self.configuration['mask_box_refinement'] = True

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'classifier_only', False) and hasattr(self, 'mask_geometry_head'):
            self.mask_geometry_head.eval()
            self.mask_geometry_refiner.eval()
        return self

    def _cascade_boxes(self, features, boxes, image_shapes, targets=None):
        current, losses = super()._cascade_boxes(features, boxes, image_shapes, targets)
        known = (targets is not None and targets[0]['amodal_masks'] is not None
                 and targets[0]['annotation_validity']['amodal'] is True)
        if targets is not None and not known:
            # Unknown A never calls the new A head or creates even a zero edge.
            return current, losses
        if not len(current):
            return current, losses
        support = expand_context_boxes(current, image_shapes[0], self.context_expansion)
        final_chunks, raw_chunks, logit_chunks = [], [], []
        for start in range(0, len(current), self.roi_chunk_size):
            stop = min(start + self.roi_chunk_size, len(current))
            pooled = self.mask_roi_pool(features, [support[start:stop]], image_shapes)
            if self.training and self.checkpoint_roi and torch.is_grad_enabled():
                logits = checkpoint(self.mask_geometry_head, pooled, use_reentrant=False)
            else:
                logits = self.mask_geometry_head(pooled)
            refined, mask_boxes = self.mask_geometry_refiner(logits.float().sigmoid(),
                support[start:stop], current[start:stop], image_shapes[0])
            final_chunks.append(refined)
            raw_chunks.append(mask_boxes)
            logit_chunks.append(logits)
        final, raw_boxes, logits = map(torch.cat, (final_chunks, raw_chunks, logit_chunks))
        if known and len(targets[0]['boxes']):
            with torch.no_grad():
                overlap, matched = box_iou(current.float(), targets[0]['boxes'].float()).max(dim=1)
                positives = torch.where(overlap >= .5)[0]
            if len(positives):
                matched_GT = targets[0]['boxes'][matched[positives]].detach()
                losses['loss_mask_geometry_box'] = matched_box_refinement_loss(
                    raw_boxes[positives], matched_GT, image_shapes[0])
                losses['loss_mask_geometry_mix'] = matched_box_refinement_loss(
                    final[positives], matched_GT, image_shapes[0])
                original = project_original_masks(targets[0], support[positives], matched[positives],
                    int(logits.shape[-1]), branch='amodal', chunk_size=self.mask_projection_chunk_size)
                selected = logits[positives, 0].float()
                losses['loss_mask_geometry_bce'] = F.binary_cross_entropy_with_logits(selected, original)
                probability = selected.sigmoid()
                intersection = (probability * original).sum((-1, -2))
                mass = probability.sum((-1, -2)) + original.sum((-1, -2))
                losses['loss_mask_geometry_dice'] = (1 - (2 * intersection + 1) / (mass + 1)).mean()
        # The inherited caller re-pools and predicts all final V/H/A masks and
        # classification in these final coordinates. Support logits stay local
        # to the auxiliary head and are never returned with a different box.
        return final.detach(), losses
